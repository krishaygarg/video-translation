#!/usr/bin/env python3
"""
realtime_pipeline.py
====================
Real-Time Video Translation & Lip-Sync Pipeline — Streaming Engine Entrypoint.

Architecture
------------
Runs a sentence-chunked, 4-GPU streaming translation pipeline:
  - GPU 0 : Video prep (face tracking & VAE mouth latent extraction)
  - GPU 1 : Audio prep (Whisper STT, 10-beam NMT, OpenVoice TTS/clone)
  - GPU 2 : MuseTalk render (frames 1..N/2)
  - GPU 3 : MuseTalk render (frames N/2+1..N)

All models are released from VRAM at session end via shared_gpu_session.

Usage
-----
  python realtime_pipeline.py \\
      --input  data/test_spanish.mp4 \\
      --output results/realtime_out.mp4 \\
      --gpus   0,1,2,3

Note: The original batch pipeline (run_pipeline_optimized.py) is unchanged.
"""

import os
import sys
import warnings

# Suppress verbose C++/library warnings before importing torch/tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Auto-Environment Redispatch:
# If executed with an unconfigured python (e.g. base python 3.13),
# automatically re-execute under the pre-configured MuseTalk conda environment.
# ---------------------------------------------------------------------------
CONDA_PYTHON = "/home/arykun47/.conda/envs/MuseTalk/bin/python"
if sys.executable != CONDA_PYTHON and os.path.exists(CONDA_PYTHON):
    try:
        import soundfile
        import torch
    except ImportError:
        os.execv(CONDA_PYTHON, [CONDA_PYTHON] + sys.argv)

import asyncio
import logging
import time
import argparse
from typing import List

import cv2
import numpy as np
import soundfile as sf
import torch
import transformers

transformers.logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# realtime_engine imports (all modular, no side-effects on import)
# ---------------------------------------------------------------------------
from realtime_engine.utils.gpu_guard import shared_gpu_session
from realtime_engine.audio_service.audio_processor import AudioProcessor
from realtime_engine.video_service.face_processor import FaceProcessor
from realtime_engine.renderer.distributed_renderer import DistributedRenderer
from realtime_engine.orchestrator.session_manager import StreamingSession
from realtime_engine.orchestrator.live_session import LiveStreamSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Video Translation & Lip-Sync Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,  help="Input Spanish video file path")
    parser.add_argument("--output", "-o", required=True,  help="Output translated video file path")
    parser.add_argument("--gpus",   default="0,1,2,3",    help="Comma-separated GPU device IDs")
    parser.add_argument("--source-lang",  default="es",   help="Source language code for Whisper")
    parser.add_argument("--num-beams",    type=int, default=10,
                        help="Beam width for NMT translation")
    parser.add_argument("--max-chunk-sec", type=float, default=6.0,
                        help="Max sentence duration before forced split (seconds)")
    parser.add_argument("--mode", choices=["live", "batch"], default="live",
                        help="Execution mode: 'live' (concurrent frame/audio streaming) or 'batch' (precomputed face/audio pass)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(args: argparse.Namespace, gpu_ids: List[int]) -> dict:
    """Main streaming pipeline execution for one input video."""
    input_video = os.path.abspath(args.input)
    output_video = os.path.abspath(args.output)

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # --- Read video metadata (fps, width, height) ---
    cap = cv2.VideoCapture(input_video)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_dim = max(width, height)
    if max_dim > 720:
        scale  = 720.0 / max_dim
        width  = int(round(width * scale)) & ~1
        height = int(round(height * scale)) & ~1

    video_duration = total_frames / fps
    log.info(
        "[pipeline] Input: %s  |  %d frames @ %.1ffps  (%.2fs)",
        input_video, total_frames, fps, video_duration,
    )

    # --- Extract source audio for transcription ---
    import subprocess
    tmp_audio_path = input_video.rsplit(".", 1)[0] + "_src_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp_audio_path,
    ], check=True, stderr=subprocess.DEVNULL)

    # --- GPU assignment:
    #   gpu_ids[0] = Face Preprocessing  (DWPose + FaceParser + SD-VAE)
    # --- GPU assignment:
    #   gpu_ids[0] = Face Preprocessing  (DWPose + FaceParser + SD-VAE)
    #   gpu_ids[1] = Audio Pipeline      (Whisper + MarianMT + OpenVoice + ToneColor + Whisper feat)
    #   gpu_ids[2] = MuseTalk Renderer A (UNet + VAE Decoder)
    #   gpu_ids[3] = MuseTalk Renderer B (UNet + VAE Decoder)
    # ---
    render_gpu_ids = gpu_ids[2:]  # GPU 2 and 3 for 57 FPS rendering

    audio_proc = AudioProcessor(
        device_id=gpu_ids[1],
        source_lang=args.source_lang,
        num_beams=args.num_beams,
    )
    face_proc = FaceProcessor(
        device_id=gpu_ids[0],
        render_gpu_ids=render_gpu_ids,
    )
    renderer = DistributedRenderer(
        gpu_ids=render_gpu_ids,
        batch_size=16,
    )

    # --- Load all models ---
    log.info("[pipeline] Loading models onto GPUs %s...", gpu_ids)
    t_load = time.perf_counter()
    audio_proc.load_models()
    audio_proc.extract_speaker_embedding(tmp_audio_path)
    face_proc.load_models()
    renderer.load_models()
    load_dur = time.perf_counter() - t_load
    log.info("[pipeline] All models loaded in %.2fs.", load_dur)

    # --- Run Session (Live or Batch) ---
    if args.mode == "live":
        session = LiveStreamSession(
            audio_processor=audio_proc,
            face_processor=face_proc,
            renderer=renderer,
            fps=fps,
            width=width,
            height=height,
            output_path=output_video,
            max_chunk_sec=args.max_chunk_sec,
        )
        live_stats = session.run(input_video=input_video, input_audio_path=tmp_audio_path)
        stream_dur = live_stats["streaming_dur"]
        total_dur = live_stats["total_dur"]
        first_latency = live_stats["first_sentence_latency"]
        total_sentences = live_stats["total_sentences"]
    else:
        # Fallback batch mode
        cap = cv2.VideoCapture(input_video)
        src_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_dim > 720:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            src_frames.append(frame)
        cap.release()

        # Standardize to 25.0 FPS for native 50Hz Whisper integer token alignment
        if abs(fps - 25.0) > 0.1 and len(src_frames) > 1:
            target_count = max(1, int(round(len(src_frames) * (25.0 / fps))))
            indices = np.linspace(0, len(src_frames) - 1, target_count).round().astype(int)
            all_frames = [src_frames[idx] for idx in indices]
            log.info("[pipeline] Resampled %d frames (%.1f FPS) → %d frames @ 25.0 FPS for integer audio sync.", len(src_frames), fps, len(all_frames))
        else:
            all_frames = src_frames
        fps = 25.0

        session = StreamingSession(
            audio_processor=audio_proc,
            face_processor=face_proc,
            renderer=renderer,
            fps=fps,
            width=width,
            height=height,
            output_path=output_video,
        )
        t_total_start = time.perf_counter()
        stream_dur = session.run(all_frames, tmp_audio_path)
        session.finalize()
        total_dur = time.perf_counter() - t_total_start
        first_latency = 0.0
        total_sentences = 0

    # --- Unload models and release VRAM ---
    audio_proc.unload_models()
    face_proc.unload_models()
    renderer.unload_models()

    # --- Cleanup temp audio ---
    try:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
    except OSError:
        pass

    return {
        "total_dur": total_dur,
        "stream_dur": stream_dur,
        "video_duration": video_duration,
        "total_frames": total_frames,
        "load_dur": load_dur,
        "first_latency": first_latency,
        "total_sentences": total_sentences,
        "mode": args.mode,
    }


def main():
    args = _parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    if len(gpu_ids) < 4:
        log.warning(
            "[pipeline] Fewer than 4 GPUs specified (%s). "
            "Render GPUs will map to last available: %d.", gpu_ids, gpu_ids[-1]
        )
        while len(gpu_ids) < 4:
            gpu_ids.append(gpu_ids[-1])

    log.info("=" * 60)
    log.info(" Real-Time Streaming Video Translation Pipeline (Mode: %s)", args.mode.upper())
    log.info(" Input   : %s", args.input)
    log.info(" Output  : %s", args.output)
    log.info(" GPUs    : Face=%d  Audio=%d  Render=%s", gpu_ids[0], gpu_ids[1], gpu_ids[2:])
    log.info(" Beams   : %d  |  MaxChunk: %.1fs", args.num_beams, args.max_chunk_sec)
    log.info("=" * 60)

    with shared_gpu_session(gpu_ids=gpu_ids):
        stats = _run_pipeline(args, gpu_ids)

    fps_rate = stats["total_frames"] / max(0.01, stats["stream_dur"])
    rt_factor = stats["video_duration"] / max(0.01, stats["stream_dur"])

    log.info("=" * 60)
    log.info(" Mode                      : %s", stats["mode"].upper())
    if stats["mode"] == "live":
        log.info(" First Sentence Latency    : %.2fs (viewer interactive delay)", stats["first_latency"])
        log.info(" Total Sentences Processed : %d", stats["total_sentences"])
    log.info(" Streaming Phase Duration  : %.2fs", stats["stream_dur"])
    log.info(" Video Duration            : %.2fs", stats["video_duration"])
    log.info(" Real-Time Factor          : %.2fx  (%.1f FPS)", rt_factor, fps_rate)
    log.info(" Total Wall Time           : %.2fs  (incl. ingestion + finalize)", stats["total_dur"])
    log.info(" Model Load Time           : %.2fs  (server startup, excluded from RT)", stats["load_dur"])
    log.info(" Output                    : %s", args.output)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

