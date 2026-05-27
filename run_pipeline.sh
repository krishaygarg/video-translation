#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Print usage instructions if the arguments are incorrect
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_spanish_video> <output_synced_video> [--speech-bubble] [--crop-upscale]"
    echo "Example: $0 data/spanish_video.mp4 results/translated_and_synced.mp4 --speech-bubble --crop-upscale"
    exit 1
fi

INPUT_VIDEO=$(realpath -m "$1")
OUTPUT_VIDEO=$(realpath -m "$2")
SPEECH_BUBBLE=false
CROP_AND_UPSCALE=false

shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --speech-bubble)
            SPEECH_BUBBLE=true
            shift
            ;;
        --crop-upscale)
            CROP_AND_UPSCALE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Starting Spanish to English Video Translation Pipeline"
echo "Input:  $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "========================================="

# Create directory for output if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# 1. Run Audio Translation & Voice Cloning (Spanish -> English)
echo ""
echo "[Step 1] Running Spanish to English translation & voice cloning..."
conda run --no-capture-output -n audio_pipeline python3 audio_pipeline/pipeline_spanishtoengish.py \
    --input "$INPUT_VIDEO" \
    --output audio_pipeline/lipsync/translated_audio.wav

# Check that the audio was successfully created
if [ ! -f "audio_pipeline/lipsync/translated_audio.wav" ]; then
    echo "Error: Translated audio file was not created by the pipeline."
    exit 1
fi
echo "Audio translation completed successfully."

# 2. Stage files for MuseTalk
echo ""
echo "[Step 2] Staging files for MuseTalk..."
mkdir -p MuseTalk/data/video
mkdir -p MuseTalk/data/audio

cp "$INPUT_VIDEO" MuseTalk/data/video/translated_input_video.mp4
cp audio_pipeline/lipsync/translated_audio.wav MuseTalk/data/audio/translated_audio.wav

# Generate temporary MuseTalk config
cat <<EOF > MuseTalk/configs/inference/run_pipeline_temp.yaml
task_0:
  video_path: "data/video/translated_input_video.mp4"
  audio_path: "data/audio/translated_audio.wav"
  bbox_shift: 0
EOF

# 3. Run MuseTalk Lip-Sync Inference
echo ""
echo "[Step 3] Running MuseTalk lip-sync generation..."
cd MuseTalk
CROP_ARGS=""
if [ "$CROP_AND_UPSCALE" = true ]; then
    CROP_ARGS="--crop_and_upscale"
fi

conda run --no-capture-output -n MuseTalk python3 -m scripts.inference \
    --inference_config ./configs/inference/run_pipeline_temp.yaml \
    --result_dir ../results/tmp_musetalk \
    --unet_model_path ./models/musetalkV15/unet.pth \
    --unet_config ./models/musetalkV15/musetalk.json \
    --version v15 \
    $CROP_ARGS
cd ..

# Check if final video was compiled
FINAL_VIDEO_PATH="results/tmp_musetalk/v15/translated_input_video_translated_audio.mp4"
if [ ! -f "$FINAL_VIDEO_PATH" ]; then
    echo "Error: MuseTalk final video compilation failed."
    exit 1
fi

# 4. Copy output or apply Speech Bubble Overlay
if [ "$SPEECH_BUBBLE" = true ]; then
    echo ""
    echo "[Step 4] Applying speech bubble transcription overlay to lip-synced video..."
    # Ensure speech_bubble environment exists
    if ! conda info --envs | grep -qE "(^|[[:space:]])speech_bubble([[:space:]]|$)"; then
        echo "[*] Creating 'speech_bubble' Conda environment..."
        conda create -n speech_bubble python=3.10 -y
    fi
    # Ensure dependencies are installed in the speech_bubble environment
    conda run --no-capture-output -n speech_bubble pip install -r speech_bubble_transcription/requirements.txt -q

    conda run --no-capture-output -n speech_bubble python3 speech_bubble_transcription/transcribe_bubble.py \
        --input "$FINAL_VIDEO_PATH" \
        --output "$OUTPUT_VIDEO" \
        --task transcribe \
        --model medium
else
    echo ""
    echo "[Step 4] Copying final video to output path..."
    cp "$FINAL_VIDEO_PATH" "$OUTPUT_VIDEO"
fi

# 5. Clean up temporary files
echo ""
echo "[Step 5] Cleaning up temporary files..."
rm -f MuseTalk/data/video/translated_input_video.mp4
rm -f MuseTalk/data/audio/translated_audio.wav
rm -f MuseTalk/configs/inference/run_pipeline_temp.yaml
rm -f audio_pipeline/lipsync/translated_audio.wav
rm -f audio_pipeline/lipsync/translated_audio.txt
rm -rf results/tmp_musetalk
rm -rf MuseTalk/results

echo "========================================="
echo "Pipeline finished successfully!"
echo "Final video saved to: $OUTPUT_VIDEO"
echo "========================================="
