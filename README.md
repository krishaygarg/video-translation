# Video Translation and Vocal Tone Preservation Pipeline

This repository is the ACM AI Spring 2026 Project. The pipeline translates spoken-English video into Spanish (or vice versa) while preserving the speaker's vocal tone and synchronizing the video's lip movements using MuseTalk.

## Project Structure
The repository consists of three main components:
1. **`audio_pipeline/`** — Handles speech transcripts, translation (English to Spanish, Spanish to English), and vocal tone analysis/classification.
2. **`MuseTalk/`** — A high-fidelity real-time lip-syncing model (supporting versions 1.0 and 1.5) that synchronizes the output video's mouth movements with the translated audio.
3. **`speech_bubble_transcription/`** — Transcribes/translates video audio using OpenAI Whisper, tracks the speaker's face using MediaPipe, and overlays styled, animated tracking speech bubbles.

---

## 🛠️ Setup Instructions

### 1. Pre-requisites
- **Operating System:** Linux with GPU (CUDA support).
- **Miniconda/Conda** installed on the system.
- **FFmpeg** installed and accessible in the system PATH.

### 2. Audio Pipeline Setup
For detailed setup of tone analysis and translation:
1. Navigate to the `audio_pipeline` directory:
   ```bash
   cd audio_pipeline
   ```
2. Read the instructions in [audio_pipeline/README.md](audio_pipeline/README.md) to set up the necessary packages and dependencies.

### 3. MuseTalk Setup
To prepare the Python environment and download the pretrained weights for the lip-sync component:

#### A. Create the Conda Environment
Use the dedicated `MuseTalk` environment:
```bash
conda create -n MuseTalk python==3.10
conda activate MuseTalk
```

#### B. Install PyTorch and Basic Dependencies
```bash
# Install PyTorch 2.0.1 with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Navigate to MuseTalk and install requirements
cd MuseTalk
pip install -r requirements.txt
```

#### C. Install MMLab Ecosystem
```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

#### D. Download Model Weights
We have configured a download script that bypasses the system's global CLI wrappers. Run the following command from the `MuseTalk/` directory:
```bash
bash download_weights.sh
```
This script downloads all required model weights:
- MuseTalk & MuseTalk V1.5 UNet weights
- StabilityAI's `sd-vae-ft-mse`
- OpenAI's `whisper-tiny`
- DWPose
- LatentSync syncnet weights

The model weights will be arranged automatically in `MuseTalk/models/`.

> [!IMPORTANT]
> The dependencies require `huggingface-hub` to be strictly within `<1.0, >=0.19.3` due to a transformers constraint. The setup will pin and run with `huggingface-hub==0.36.2` to ensure compatibility.

---

## 🚀 Running Inference
To run a lip-syncing task using the `MuseTalk` conda environment, navigate to the `MuseTalk/` directory and execute:

```bash
conda run -n MuseTalk bash inference.sh v1.5 normal
```

This runs inference with MuseTalk v1.5 using the config file at `configs/inference/test.yaml`.

### Customizing Tasks
To add or modify tasks, update the YAML configuration file `MuseTalk/configs/inference/test.yaml`:
```yaml
task_0:
  video_path: "data/video/yongen.mp4"
  audio_path: "data/audio/yongen.wav"
  bbox_shift: 0
task_1:
  video_path: "data/video/yongen.mp4"
  audio_path: "data/audio/eng.wav"
  bbox_shift: 0
```
Output videos are compiled and saved in `MuseTalk/results/test/v15/`.

---

## 🔄 End-to-End Spanish to English Translation & Lip-Sync Pipeline
We provide a master orchestration script `run_pipeline.sh` at the top level of the repository. This script takes an input video containing Spanish speech, translates the speech to English, clones the speaker's vocal tone to generate the English audio, runs MuseTalk to produce a lip-synced video, and optionally overlays tracking speech bubbles.

### Usage
From the root of the repository, execute:
```bash
./run_pipeline.sh <input_spanish_video> <output_synced_video> [options]
```

### Options
* `--speech-bubble`: Transcribes the final lip-synced video using OpenAI Whisper, tracks the speaker's face, and overlays styled, animated tracking speech bubbles.
* `--crop-upscale`: Crops the speaker's face region, upscales it using super-resolution (FSRCNN) or Lanczos4 scaling, and pastes it back onto the original frame post-inference for higher resolution mouth movements.

**Example with all options enabled:**
```bash
./run_pipeline.sh Video1.mp4 results/output_video_with_bubbles.mp4 --speech-bubble --crop-upscale
```

This script automatically:
1. Activates the `audio_pipeline` Conda environment to run transcription, translation, and OpenVoice cloning.
2. Stages the files and generates a temporary config for MuseTalk.
3. Activates the `MuseTalk` Conda environment to run the lip-sync generation (with crop/upscale if enabled).
4. Runs the speech bubble overlay generator inside the `speech_bubble` Conda environment (if enabled).
5. Copies the final video to the destination path and cleans up all intermediate temporary files.

---

## 💬 Standalone Speech Bubble Overlay
If you already have a lip-synced video (or want to add subtitles to any video) and want to overlay tracking speech bubbles without running the entire translation and lip-sync pipeline, you can use the standalone wrapper script:

### Usage
```bash
./overlay_speech_bubble.sh -i <input_video> -o <output_video> [options]
```

**Example:**
```bash
./overlay_speech_bubble.sh -i results/output_video.mp4 -o results/output_video_with_bubbles.mp4 --task transcribe --model medium
```

### Options
* `-i`, `--input` (Required): Path to the input video.
* `-o`, `--output` (Default: `results/output_video_with_bubbles.mp4`): Path for the output video.
* `-a`, `--audio`: Custom audio track to merge (e.g. translated/cloned audio).
* `--task` (Default: `translate`): Whisper task (`translate` to translate Spanish to English text, or `transcribe` to transcribe matching audio).
* `--model` (Default: `medium`): Whisper model size (`tiny`, `base`, `small`, `medium`, `large`).
* See [speech_bubble_transcription/README.md](speech_bubble_transcription/README.md) for more customization options (such as bubble colors, font scales, tracking parameters, offsets, and deadzones).

---

## 🎥 Demo Videos
* **Original Spanish Video:** [Video1.mp4](Video1.mp4)
* **Final Synced & Bubble-Overlay Video:** [results/output_video_with_bubbles.mp4](results/output_video_with_bubbles.mp4)
