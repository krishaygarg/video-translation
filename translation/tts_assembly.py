"""
tts_assembly.py

Reads a lipsync CSV produced by lipsync_translate.py and assembles a time-accurate
Spanish audio track.

Two providers are supported:

  openvoice (target)
    Uses MeloTTS for Spanish synthesis and OpenVoice ToneColorConverter to clone
    the speaker's voice from the original English audio. Pacing is handled natively
    via MeloTTS's `speed` parameter (driven by tts_rate_multiplier in the CSV), so
    no post-processing time-stretch is needed. The `emotion` column maps to OpenVoice
    style embeddings.

  gtts (fallback, no API key or GPU required)
    Google TTS at natural rate, then pitch-preserving time-stretch (phase vocoder via
    scipy) to fit phrase_duration. Does not clone voice or apply emotion.

Usage:
  # OpenVoice — target pipeline
  python tts_assembly.py \\
      --provider openvoice \\
      --input  output/english6_phrases_lipsync.csv \\
      --output output/english6_spanish.wav \\
      --reference ../transcripts/english6.mp3

  # gTTS — quick local test, no GPU needed
  python tts_assembly.py \\
      --provider gtts \\
      --input  output/english6_phrases_lipsync.csv \\
      --output output/english6_spanish.wav

  # Filter one clip from the combined summary CSV
  python tts_assembly.py --provider gtts \\
      --input output/all_lipsync_summary.csv \\
      --output output/english9_spanish.wav \\
      --clip english9_phrases

Dependencies:
  # gTTS fallback only:
  pip install gtts

  # OpenVoice (target):
  pip install git+https://github.com/myshell-ai/OpenVoice.git
  pip install git+https://github.com/myshell-ai/MeloTTS.git
  # Download checkpoints into checkpoints_v2/ per OpenVoice README.

Both providers require: soundfile scipy numpy pandas  (already in project venv)
"""

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 22050

# Clamp stretch ratio for the gTTS fallback path.
_MIN_RATE = 0.5
_MAX_RATE = 2.0

# Maps the emotion labels produced by lipsync_translate.py to OpenVoice style names.
# OpenVoice v1 built-in styles: neutral, happy, sad, angry, surprised, whispering.
EMOTION_TO_STYLE: dict[str, str] = {
    "joy":      "happy",
    "happiness":"happy",
    "sadness":  "sad",
    "anger":    "angry",
    "fear":     "angry",   # closest available proxy
    "surprise": "surprised",
    "disgust":  "neutral",
    "neutral":  "neutral",
}


# ---------------------------------------------------------------------------
# Provider: OpenVoice  (MeloTTS + ToneColorConverter)
# ---------------------------------------------------------------------------

def build_openvoice_provider(reference_audio: str,
                              ckpt_dir: str = "checkpoints_v2",
                              device: str = "cpu"):
    """
    Load OpenVoice models once and return a generate() callable.

    reference_audio: path to the original English mp3/wav — used for voice cloning.
    ckpt_dir:        directory containing OpenVoice v2 checkpoint folders
                     (base_speakers/ES and converter).
    """
    try:
        import torch
        from melo.api import TTS
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
    except ImportError:
        sys.exit(
            "OpenVoice / MeloTTS not installed.\n"
            "  pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
            "  pip install git+https://github.com/myshell-ai/MeloTTS.git"
        )

    ckpt_base      = Path(ckpt_dir) / "base_speakers" / "ES"
    ckpt_converter = Path(ckpt_dir) / "converter"

    print(f"Loading MeloTTS (ES, device={device})…")
    tts_model   = TTS(language="ES", device=device)
    speaker_ids = tts_model.hps.data.spk2id

    print("Loading ToneColorConverter…")
    converter = ToneColorConverter(str(ckpt_converter / "config.json"), device=device)
    converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))

    print(f"Extracting voice style from reference: {reference_audio}")
    target_se, _ = se_extractor.get_se(reference_audio, converter, vad=True)

    # Source style embedding shipped with the ES base speaker checkpoint
    source_se = torch.load(str(ckpt_base / "es_default_se.pth"), map_location=device)

    def generate(text: str, speed: float = 1.0, emotion: str = "neutral") -> tuple[np.ndarray, int]:
        style = EMOTION_TO_STYLE.get(emotion.lower(), "neutral")
        with (tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_base,
              tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out):
            base_path = f_base.name
            out_path  = f_out.name
        try:
            # MeloTTS generates Spanish at the requested speed
            tts_model.tts_to_file(text, speaker_ids["ES"], base_path, speed=speed)
            # ToneColorConverter clones the source speaker's voice
            converter.convert(
                audio_src_path=base_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_path,
                tau=0.3,
                style=style,
            )
            data, sr = sf.read(out_path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            return data, sr
        finally:
            os.unlink(base_path)
            os.unlink(out_path)

    return generate


# ---------------------------------------------------------------------------
# Provider: gTTS  (fallback — no GPU, no API key)
# ---------------------------------------------------------------------------

def build_gtts_provider(lang: str = "es"):
    """Return a generate() callable that uses Google TTS."""
    try:
        from gtts import gTTS
    except ImportError:
        sys.exit("gTTS not found. Install it: pip install gtts")

    def generate(text: str, speed: float = 1.0, emotion: str = "neutral") -> tuple[np.ndarray, int]:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, sr

    return generate


# ---------------------------------------------------------------------------
# Time-stretching (gTTS fallback only — phase vocoder via scipy STFT)
# ---------------------------------------------------------------------------

def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Pitch-preserving time-stretch.
    rate > 1.0 → speed up (shorter output), rate < 1.0 → slow down (longer output).
    """
    from scipy.signal import stft, istft

    if abs(rate - 1.0) < 0.005:
        return audio

    n_fft, hop, win = 2048, 512, "hann"
    _, _, Z = stft(audio, nperseg=n_fft, noverlap=n_fft - hop, window=win)
    mag = np.abs(Z)
    phase = np.angle(Z)
    n_freqs, n_frames = mag.shape

    target_frames = max(2, int(round(n_frames / rate)))
    src_idx = np.linspace(0, n_frames - 1, target_frames)
    lo  = np.floor(src_idx).astype(int)
    hi  = np.minimum(lo + 1, n_frames - 1)
    frac = src_idx - lo

    mag_out   = (1 - frac) * mag[:, lo] + frac * mag[:, hi]
    phase_adv = 2 * np.pi * hop * np.arange(n_freqs) / n_fft
    phase_acc = phase[:, 0].copy()
    phase_out = np.empty_like(mag_out)
    phase_out[:, 0] = phase_acc

    for i in range(1, target_frames):
        dp = phase[:, hi[i]] - phase[:, lo[i]] - phase_adv
        dp -= 2 * np.pi * np.round(dp / (2 * np.pi))
        phase_acc += phase_adv + dp * frac[i]
        phase_out[:, i] = phase_acc

    _, audio_out = istft(mag_out * np.exp(1j * phase_out),
                         nperseg=n_fft, noverlap=n_fft - hop, window=win)
    return audio_out.astype(np.float32)


def _trim_or_pad(audio: np.ndarray, sr: int, target_dur: float) -> np.ndarray:
    """Trim or zero-pad to exact sample count without changing pitch."""
    n = int(round(target_dur * sr))
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio)))


def _fit_to_duration(audio: np.ndarray, sr: int, target_dur: float) -> np.ndarray:
    """Time-stretch then trim/pad to hit target_dur exactly (gTTS path)."""
    if target_dur <= 0 or len(audio) == 0:
        return audio
    rate = np.clip(len(audio) / sr / target_dur, _MIN_RATE, _MAX_RATE)
    return _trim_or_pad(_time_stretch(audio, rate), sr, target_dur)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(df: pd.DataFrame,
             output_path: Path,
             provider,
             native_speed: bool) -> None:
    """
    provider:     callable(text, speed, emotion) -> (np.ndarray, sr)
    native_speed: True for OpenVoice (speed is already baked in, only trim/pad);
                  False for gTTS (needs post-hoc time-stretch).
    """
    if df.empty:
        print("No rows to process.")
        return

    total_dur = float(df["phrase_end"].max()) + 0.25
    output = np.zeros(int(round(total_dur * SAMPLE_RATE)), dtype=np.float32)
    n = len(df)

    for row_num, (_, row) in enumerate(df.iterrows(), 1):
        phrase      = str(row["best_spanish"])
        target_dur  = float(row["phrase_duration"])
        phrase_start = float(row["phrase_start"])
        speed       = float(row["tts_rate_multiplier"]) if "tts_rate_multiplier" in row else 1.0
        emotion     = str(row.get("emotion", "neutral") or "neutral")
        phrase_id   = row.get("phrase_id", f"phrase_{row_num:03d}")

        print(f"  [{row_num}/{n}] {phrase_id}  target={target_dur:.2f}s  "
              f"speed={speed:.2f}x  emotion={emotion}")
        print(f"    '{phrase[:60]}'")

        try:
            raw, raw_sr = provider(phrase, speed=speed, emotion=emotion)
        except Exception as exc:
            print(f"    ERROR: TTS failed — {exc}")
            continue

        audio = _resample(raw, raw_sr, SAMPLE_RATE)

        if native_speed:
            # OpenVoice already respected speed; just align to exact duration.
            natural_dur = len(audio) / SAMPLE_RATE
            audio = _trim_or_pad(audio, SAMPLE_RATE, target_dur)
            print(f"    OpenVoice: natural={natural_dur:.2f}s → trimmed/padded to {target_dur:.2f}s")
        else:
            # gTTS: stretch to fit.
            natural_dur = len(audio) / SAMPLE_RATE
            audio = _fit_to_duration(audio, SAMPLE_RATE, target_dur)
            print(f"    gTTS: natural={natural_dur:.2f}s → stretched to {target_dur:.2f}s")

        start_sample = int(round(phrase_start * SAMPLE_RATE))
        end_sample   = start_sample + len(audio)
        if end_sample > len(output):
            output = np.pad(output, (0, end_sample - len(output)))
        output[start_sample:end_sample] += audio

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
    parser.add_argument("--provider", choices=["openvoice", "gtts"], default="gtts",
                        help="TTS back-end (default: gtts)")
    parser.add_argument("--input",    required=True,
                        help="Lipsync CSV (single clip or all_lipsync_summary.csv)")
    parser.add_argument("--output",   required=True,
                        help="Output WAV file path")
    parser.add_argument("--clip",     default=None,
                        help="clip_id to process (needed when CSV has multiple clips)")
    parser.add_argument("--reference", default=None,
                        help="[openvoice] Path to original English audio for voice cloning")
    parser.add_argument("--ckpt-dir", default="checkpoints_v2",
                        help="[openvoice] OpenVoice checkpoint directory (default: checkpoints_v2)")
    parser.add_argument("--device",   default="cpu",
                        help="[openvoice] PyTorch device: cpu or cuda (default: cpu)")
    parser.add_argument("--lang",     default="es",
                        help="[gtts] BCP-47 language code (default: es)")
    args = parser.parse_args()

    if args.provider == "openvoice" and not args.reference:
        sys.exit("--reference is required when using --provider openvoice")

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.clip:
        df = df[df["clip_id"] == args.clip].reset_index(drop=True)
        if df.empty:
            sys.exit(f"No rows for clip_id='{args.clip}' in {csv_path}")
    elif "clip_id" in df.columns and df["clip_id"].nunique() > 1:
        ids = df["clip_id"].unique().tolist()
        sys.exit(
            f"CSV contains {len(ids)} clips: {ids}\n"
            "Re-run with --clip <clip_id> to select one."
        )

    if args.provider == "openvoice":
        provider     = build_openvoice_provider(args.reference, args.ckpt_dir, args.device)
        native_speed = True
    else:
        provider     = build_gtts_provider(lang=args.lang)
        native_speed = False

    print(f"Provider: {args.provider}  |  {len(df)} phrase(s)  |  {csv_path.name}")
    assemble(df, Path(args.output), provider, native_speed)


if __name__ == "__main__":
    main()
