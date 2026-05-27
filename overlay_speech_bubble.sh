#!/bin/bash
# Script to overlay speech bubbles on a video without running the entire pipeline.
set -e

# Find the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If run from inside MuseTalk, the root is one level up. Otherwise, it is SCRIPT_DIR.
if [[ "$SCRIPT_DIR" == */MuseTalk ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

TRANSCRIBE_SCRIPT="$PROJECT_ROOT/speech_bubble_transcription/transcribe_bubble.py"
REQUIREMENTS_FILE="$PROJECT_ROOT/speech_bubble_transcription/requirements.txt"

# Default input/output
# Search for any mp4 file in the root results folder (excluding files with 'with_bubbles' in name)
DEFAULT_INPUT=""
for f in "$PROJECT_ROOT/results"/*.mp4; do
    if [ -f "$f" ]; then
        if [[ "$(basename "$f")" != *"with_bubbles"* ]]; then
            DEFAULT_INPUT="$f"
            break
        fi
    fi
done

# Fallback paths if no video is found in the root results folder
if [ -z "$DEFAULT_INPUT" ] || [ ! -f "$DEFAULT_INPUT" ]; then
    DEFAULT_INPUT="$PROJECT_ROOT/results/tmp_musetalk/v15/translated_input_video_translated_audio.mp4"
fi
if [ ! -f "$DEFAULT_INPUT" ]; then
    DEFAULT_INPUT="$PROJECT_ROOT/MuseTalk/results/test/v15/translated_input_video_translated_audio.mp4"
fi
DEFAULT_OUTPUT="$PROJECT_ROOT/results/output_video_with_bubbles.mp4"

# Parse arguments
INPUT_VIDEO=""
OUTPUT_VIDEO=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)
            INPUT_VIDEO="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$INPUT_VIDEO" ]; then
    if [ -f "$DEFAULT_INPUT" ]; then
        INPUT_VIDEO="$DEFAULT_INPUT"
        echo "[*] No input video specified. Using default: $INPUT_VIDEO"
    else
        echo "Error: No input video specified and default video not found."
        echo "Usage: $0 -i <input_video> -o <output_video> [additional_whisper_options]"
        exit 1
    fi
fi

if [ -z "$OUTPUT_VIDEO" ]; then
    OUTPUT_VIDEO="$DEFAULT_OUTPUT"
    echo "[*] No output video specified. Using default: $OUTPUT_VIDEO"
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

echo "========================================="
echo "Speech Bubble Overlay Standalone Generator"
echo "Input:  $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "========================================="

# Check and setup Conda environment if available
if command -v conda &> /dev/null; then
    if conda info --envs | grep -qE "(^|[[:space:]])speech_bubble([[:space:]]|$)"; then
        echo "[*] Conda environment 'speech_bubble' already exists."
    else
        echo "[*] Conda environment 'speech_bubble' not found. Creating it..."
        conda create -n speech_bubble python=3.10 -y
    fi
    echo "[*] Ensuring dependencies are installed in 'speech_bubble' environment..."
    conda run --no-capture-output -n speech_bubble pip install -r "$REQUIREMENTS_FILE" -q

    # Run the overlay script
    echo "[*] Running speech bubble transcription inside 'speech_bubble' environment..."
    conda run --no-capture-output -n speech_bubble python3 "$TRANSCRIBE_SCRIPT" \
        --input "$INPUT_VIDEO" \
        --output "$OUTPUT_VIDEO" \
        --task transcribe \
        --model medium \
        "${OTHER_ARGS[@]}"
else
    # Fallback to system python if Conda is not available
    echo "[!] Conda not found. Falling back to system python3..."
    echo "[*] Checking and installing dependencies..."
    pip install -r "$REQUIREMENTS_FILE" -q

    # Run the overlay script
    echo "[*] Running speech bubble transcription..."
    python3 "$TRANSCRIBE_SCRIPT" \
        --input "$INPUT_VIDEO" \
        --output "$OUTPUT_VIDEO" \
        --task transcribe \
        --model medium \
        "${OTHER_ARGS[@]}"
fi

echo "========================================="
echo "Done! Output saved to $OUTPUT_VIDEO"
echo "========================================="
