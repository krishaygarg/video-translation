# Video Translation — Tone Analysis Pipeline

A pipeline that detects the vocal tone (emotion) of spoken English audio clips using a pre-trained speech emotion recognition model.

## What it does

Takes any number of audio files (`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`) and classifies each one into one of four emotions: **neutral, happy, angry, sad**. Outputs both a per-file CSV with all four scores and a combined summary CSV across all files.

Built as a replacement for the deprecated MeaningCloud Sentiment API.

## Stack

- **Language:** Python 3.12 (PyTorch on Intel Mac requires 3.12, not 3.13+)
- **Model:** [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er) — Hugging Face speech emotion recognition (~1.2 GB, runs locally on CPU, no API key)
- **Audio decoding:** `soundfile` for wav/flac/ogg, `ffmpeg` fallback for mp3/m4a
- **Resampling:** `scipy.signal.resample_poly` (target 16 kHz mono, what the model expects)

## Folder layout

```
video-translation/
├── tone_analysis/
│   ├── analyze_all_transcripts.py   # main pipeline
│   └── output/                      # generated CSVs land here
├── transcripts/                     # drop your audio files here
├── venv/                            # Python virtual environment (gitignored)
├── .env                             # gitignored
└── README.md
```

## Setup

You need Python 3.12 and ffmpeg.

```bash
# 1. Install Python 3.12 if you don't have it
brew install python@3.12

# 2. Install ffmpeg (used to decode mp3 audio)
brew install ffmpeg

# 3. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install "transformers==4.45.2" "numpy<2" soundfile scipy pandas torch
```

## Usage

1. Drop your audio files into the `transcripts/` folder.
2. Run:

```bash
   python tone_analysis/analyze_all_transcripts.py
```

3. The first run downloads the model (~1.2 GB) and caches it locally; subsequent runs are fast.

## Output

For each input audio file, the pipeline writes:

- `tone_analysis/output/<filename>_audio_emotions.csv` — detailed per-file scores
- `tone_analysis/output/all_audio_emotions_summary.csv` — one row per file

Each row contains: `file, neutral, happy, angry, sad, dominant_emotion`.

## Notes

- Pinned to `transformers==4.45.2` and `numpy<2` because PyTorch 2.2 (the latest wheel for Intel Mac + Python 3.12) is incompatible with NumPy 2 and with the latest transformers.
- The audio model expects 16 kHz mono. The pipeline downmixes stereo to mono and resamples automatically.