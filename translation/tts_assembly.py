"""
tts_assembly.py

Reads a lipsync CSV produced by lipsync_translate.py and assembles a time-accurate
Spanish audio track:

  1. Generate TTS for each phrase at natural speaking rate (gTTS by default).
  2. Time-stretch the phrase (pitch-preserving) to fit exactly phrase_duration.
  3. Place the phrase at phrase_start, preserving original silence gaps.

The result matches the total runtime of the original English audio.

Usage:
  python tts_assembly.py --input output/english6_phrases_lipsync.csv \\
                         --output output/english6_spanish.wav

  # Filter to one clip when using the all_lipsync_summary.csv:
  python tts_assembly.py --input output/all_lipsync_summary.csv \\
                         --output output/english9_spanish.wav \\
                         --clip english9_phrases

Dependencies (install once):
  pip install gtts librosa soundfile
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 22050  # working sample rate for all processing
# Clamp stretch ratio so extreme phrases don't distort beyond recognition.
MIN_RATE = 0.5   # slowest allowed (0.5x = 2x duration)
MAX_RATE = 2.0   # fastest allowed (2x = 0.5x duration)


# ---------------------------------------------------------------------------
# TTS back-end
# ---------------------------------------------------------------------------

def _tts_gtts(text: str, lang: str = "es") -> tuple[np.ndarray, int]:
    """Generate audio with gTTS; returns (float32 mono array, sample_rate)."""
    try:
        from gtts import gTTS
    except ImportError:
        sys.exit("gTTS not found. Install it: pip install gtts")
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


# Swap this function to use ElevenLabs, Coqui, etc.
generate_tts = _tts_gtts


# ---------------------------------------------------------------------------
# Time-stretching (phase vocoder via scipy STFT — no extra deps beyond scipy)
# ---------------------------------------------------------------------------

def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Pitch-preserving time-stretch using a phase vocoder built on scipy STFT.
    rate > 1.0  → speed up  (shorter output)
    rate < 1.0  → slow down (longer output)
    """
    from scipy.signal import stft, istft

    if abs(rate - 1.0) < 0.005:
        return audio

    n_fft = 2048
    hop = 512
    win = "hann"

    _, _, Z = stft(audio, nperseg=n_fft, noverlap=n_fft - hop, window=win)
    mag = np.abs(Z)                       # (n_freqs, n_frames)
    phase = np.angle(Z)
    n_freqs, n_frames = mag.shape

    target_frames = max(2, int(round(n_frames / rate)))

    # Source frame indices for each output frame (fractional)
    src_idx = np.linspace(0, n_frames - 1, target_frames)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, n_frames - 1)
    frac = src_idx - lo                   # (target_frames,) in [0, 1)

    # Linearly interpolate magnitude
    mag_out = (1 - frac) * mag[:, lo] + frac * mag[:, hi]   # (n_freqs, target_frames)

    # Phase: accumulate instantaneous frequency to stay coherent across frames.
    phase_advance = 2 * np.pi * hop * np.arange(n_freqs) / n_fft  # (n_freqs,)
    phase_acc = phase[:, 0].copy()
    phase_out = np.empty_like(mag_out)
    phase_out[:, 0] = phase_acc

    for i in range(1, target_frames):
        # True phase delta at source transition
        dp = phase[:, hi[i]] - phase[:, lo[i]] - phase_advance
        dp -= 2 * np.pi * np.round(dp / (2 * np.pi))   # wrap to [-π, π]
        phase_acc += phase_advance + dp * frac[i]
        phase_out[:, i] = phase_acc

    Z_out = mag_out * np.exp(1j * phase_out)
    _, audio_out = istft(Z_out, nperseg=n_fft, noverlap=n_fft - hop, window=win)
    return audio_out.astype(np.float32)


def _fit_to_duration(audio: np.ndarray, sr: int, target_dur: float) -> np.ndarray:
    """Stretch/compress audio so it is exactly target_dur seconds long."""
    if target_dur <= 0 or len(audio) == 0:
        return audio
    current_dur = len(audio) / sr
    rate = np.clip(current_dur / target_dur, MIN_RATE, MAX_RATE)
    stretched = _time_stretch(audio, rate)
    # Hard-trim or zero-pad to exact sample count after stretching
    target_samples = int(round(target_dur * sr))
    if len(stretched) >= target_samples:
        return stretched[:target_samples]
    return np.pad(stretched, (0, target_samples - len(stretched)))


# ---------------------------------------------------------------------------
# Resampling (uses scipy — no librosa needed)
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(df: pd.DataFrame, output_path: Path, lang: str = "es") -> None:
    if df.empty:
        print("No rows to process.")
        return

    total_dur = float(df["phrase_end"].max()) + 0.25   # small trailing buffer
    n_samples = int(round(total_dur * SAMPLE_RATE))
    output = np.zeros(n_samples, dtype=np.float32)

    n = len(df)
    for row_num, (_, row) in enumerate(df.iterrows(), 1):
        phrase = str(row["best_spanish"])
        target_dur = float(row["phrase_duration"])
        phrase_start = float(row["phrase_start"])
        phrase_id = row.get("phrase_id", f"phrase_{row_num:03d}")

        print(f"  [{row_num}/{n}] {phrase_id}  target={target_dur:.2f}s  '{phrase[:50]}'")

        try:
            raw_audio, raw_sr = generate_tts(phrase, lang=lang)
        except Exception as exc:
            print(f"    ERROR: TTS failed — {exc}")
            continue

        # Normalise to working sample rate
        audio = _resample(raw_audio, raw_sr, SAMPLE_RATE)

        natural_dur = len(audio) / SAMPLE_RATE
        audio = _fit_to_duration(audio, SAMPLE_RATE, target_dur)

        rate_applied = natural_dur / target_dur if target_dur > 0 else 1.0
        print(f"    natural={natural_dur:.2f}s → fit={target_dur:.2f}s  "
              f"(stretch rate={rate_applied:.2f}x)")

        # Place at correct position (preserves silence gaps from phrase_start)
        start_sample = int(round(phrase_start * SAMPLE_RATE))
        end_sample = start_sample + len(audio)
        if end_sample > len(output):
            output = np.pad(output, (0, end_sample - len(output)))
        output[start_sample:end_sample] += audio

    # Normalise loudness, guard against silence-only output
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), output, SAMPLE_RATE, subtype="PCM_16")
    print(f"\nWrote: {output_path}  ({total_dur:.2f}s, {SAMPLE_RATE} Hz)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble time-accurate Spanish TTS audio from a lipsync CSV."
    )
    parser.add_argument("--input", required=True,
                        help="Path to lipsync CSV (single clip or all_lipsync_summary.csv)")
    parser.add_argument("--output", required=True,
                        help="Output WAV file path")
    parser.add_argument("--clip", default=None,
                        help="clip_id to process (required when CSV contains multiple clips)")
    parser.add_argument("--lang", default="es",
                        help="BCP-47 language code passed to gTTS (default: es)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.clip:
        df = df[df["clip_id"] == args.clip].reset_index(drop=True)
        if df.empty:
            sys.exit(f"No rows found for clip_id='{args.clip}' in {csv_path}")
    elif "clip_id" in df.columns and df["clip_id"].nunique() > 1:
        ids = df["clip_id"].unique().tolist()
        sys.exit(
            f"CSV contains {len(ids)} clips: {ids}\n"
            f"Re-run with --clip <clip_id> to select one."
        )

    print(f"Processing {len(df)} phrase(s) from {csv_path.name}")
    assemble(df, Path(args.output), lang=args.lang)


if __name__ == "__main__":
    main()
