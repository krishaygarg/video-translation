"""
analyze_all_transcripts.py
Loads every audio file in ../transcripts/, runs Speech Emotion Recognition,
and writes results to ./output/.
"""

import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from transformers import pipeline


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
AUDIO_DIR = PROJECT_ROOT / "transcripts"
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "superb/hubert-large-superb-er"
TARGET_SR = 16000

print(f"Loading audio emotion model: {MODEL_NAME}")
print("(First run downloads ~1.2 GB - be patient.)")
classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
    top_k=None,
)

LABEL_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def _load_with_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(target_sr), "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "soundfile couldn't decode this file, and ffmpeg isn't installed. "
            "Install ffmpeg with:  brew install ffmpeg"
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return _load_with_ffmpeg(path, target_sr)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def analyze_audio(path: Path) -> dict:
    audio = load_audio(path)
    raw = classifier({"array": audio, "sampling_rate": TARGET_SR})
    scores = {LABEL_MAP.get(s["label"], s["label"]): round(float(s["score"]), 4)
              for s in raw}
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores


def main():
    extensions = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")
    audio_files = []
    for ext in extensions:
        audio_files.extend(AUDIO_DIR.glob(ext))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) in {AUDIO_DIR}\n")
    summary_rows = []

    for path in audio_files:
        try:
            result = analyze_audio(path)
        except Exception as e:
            print(f"[error] {path.name}: {e}")
            continue

        per_file_path = OUTPUT_DIR / f"{path.stem}_audio_emotions.csv"
        pd.DataFrame([result]).to_csv(per_file_path, index=False)

        print(f"[done] {path.name:20s} -> {result['dominant_emotion']:8s} "
              f"(confidence {result[result['dominant_emotion']]:.2f})")

        summary_rows.append({"file": path.name, **result})

    summary_path = OUTPUT_DIR / "all_audio_emotions_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote per-file reports and summary to: {OUTPUT_DIR}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()