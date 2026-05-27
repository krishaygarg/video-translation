# Speech Bubble Overlay Generator

This component transcribes/translates video audio using OpenAI Whisper, tracks the speaker's nose using MediaPipe Face Landmarker, and overlays styled, animated speech bubbles that track the speaker.

---

## 🛠️ Setup Instructions

### Pre-requisites
- **Python 3.8+**
- **FFmpeg** installed on your system.

### Install Dependencies
You can install the dependencies inside your existing `audio_pipeline` Conda environment:

```bash
conda activate audio_pipeline
pip install -r requirements.txt
```

Alternatively, you can run the helper script which automatically sets up dependencies and executes the script inside the `audio_pipeline` environment:
```bash
chmod +x run_speech_bubble.sh
```

---

## 🚀 Usage

### Command Line Interface
You can run the python script directly:
```bash
python transcribe_bubble.py --input <input_video> --output <output_video> [options]
```

Or use the helper script:
```bash
./run_speech_bubble.sh --input <input_video> --output <output_video> [options]
```

### Options
- `--input`, `-i` (Required): Path to the input video file.
- `--output`, `-o` (Default: `translated_speech_bubble.mp4`): Path for the output video.
- `--audio`, `-a`: Path to a custom audio track (e.g. translated/cloned audio). If not specified, the script extracts and uses the original video's audio.
- `--no-audio`: Keep the output video silent.
- `--model`, `-m` (Default: `medium`): Whisper model size (`tiny`, `base`, `small`, `medium`, `large`).
- `--task` (Default: `translate`): Whisper task (`translate` to translate Spanish to English text, or `transcribe` to transcribe matching video audio).
- `--landmarker`, `-l` (Default: `face_landmarker.task`): Path to the face landmarker task model. If not present, it will download automatically.
- `--max-chars`, `-c` (Default: `22`): Maximum characters per line in the speech bubble.
- `--smoothing` (Default: `0.08`): Smoothing factor for face tracking (lower = smoother tracking, higher = more responsive).
- `--deadzone` (Default: `8.0`): Distance in pixels below which nose movement is ignored to prevent bubble jitter.
- `--offset-x` (Default: `0.18`): Horizontal bubble offset from the nose (as fraction of video width).
- `--offset-y` (Default: `0.18`): Vertical bubble offset from the nose (as fraction of video height).
- `--bubble-color` (Default: `255,255,255`): Bubble fill color (comma-separated RGB e.g. `255,255,255` or hex `#ffffff`).
- `--border-color` (Default: `0,0,0`): Bubble border color (comma-separated RGB e.g. `0,0,0` or hex `#000000`).
- `--text-color` (Default: `0,0,0`): Bubble text color (comma-separated RGB e.g. `0,0,0` or hex `#000000`).
- `--font-scale-mult` (Default: `1.0`): Scaling multiplier for text.
- `--thickness-mult` (Default: `1.0`): Thickness multiplier for text/borders.

---

## 💡 Examples

### 1. Spanish to English Translation Standalone
Translate a Spanish video's speech to English text in the bubbles and keep the original Spanish audio:
```bash
./run_speech_bubble.sh --input Video1.mp4 --output results/video_with_bubbles.mp4 --task translate
```

### 2. Transcribing English Video
Transcribe an English video's speech to English text in the bubbles and keep the original English audio:
```bash
./run_speech_bubble.sh --input EnglishVideo.mp4 --output results/video_with_bubbles.mp4 --task transcribe
```

### 3. Merging Cloned Audio (Translated Audio)
If you have a cloned/translated audio track from the translation pipeline, you can merge it with the video:
```bash
./run_speech_bubble.sh --input Video1.mp4 --output results/video_with_bubbles.mp4 --audio audio_pipeline/lipsync/translated_audio.wav --task transcribe
```
