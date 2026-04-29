# Tone Analysis Pipeline

Speech emotion recognition pipeline that classifies the vocal tone of audio clips. Component of the larger `video-translation` project.

## What it does

Takes any number of audio files (`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`) and classifies each into one of four emotions: **neutral, happy, angry, sad**. Outputs both a per-file CSV with all four scores and a combined summary CSV.

Replaces the deprecated MeaningCloud Sentiment API.

## Stack

- **Language:** Python 3.12 (PyTorch on Intel Mac requires 3.12, not 3.13+)
- **Model:** [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er) — Hugging Face speech emotion recognition (~1.2 GB, runs locally on CPU, no API key)
- **Audio decoding:** `soundfile` for wav/flac/ogg, `ffmpeg` fallback for mp3/m4a
- **Resampling:** `scipy.signal.resample_poly` (target 16 kHz mono)

## Setup

```bash
brew install python@3.12 ffmpeg
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "transformers==4.45.2" "numpy<2" soundfile scipy pandas torch
```

## Usage

1. Drop audio files into `../transcripts/`.
2. Run from the project root:

```bash
   python tone_analysis/analyze_all_transcripts.py
```

3. First run downloads the model (~1.2 GB) and caches it; later runs are fast.

## Output

Written to `tone_analysis/output/`:
- `<filename>_audio_emotions.csv` — per-file detailed scores
- `all_audio_emotions_summary.csv` — one row per file

Columns: `file, neutral, happy, angry, sad, dominant_emotion`.

## Notes

- Pinned to `transformers==4.45.2` and `numpy<2` because PyTorch 2.2 (latest wheel for Intel Mac + Python 3.12) is incompatible with NumPy 2 and the latest transformers.
- The audio model expects 16 kHz mono. The pipeline downmixes stereo to mono and resamples automatically.