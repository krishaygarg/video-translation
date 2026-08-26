"""
livestream_feeder.py
====================
Simulates a live camera and microphone by reading a video file and feeding
frames and audio in real-time slices to their respective queues.

FrameFeeder: reads frames one-by-one as fast as possible and pushes to
             frame_queue. (Demo mode: no sleep, lets GPU 0 build lookahead.)
AudioFeeder: reads a pre-extracted WAV in 0.5s chunks and pushes to
             audio_chunk_queue at real-time pace (simulates a microphone).

In a real livestream:
  - FrameFeeder would read from a camera device instead of a file.
  - AudioFeeder would read from a microphone input buffer.
  The rest of the pipeline is identical.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

AUDIO_CHUNK_SEC = 0.5   # seconds per audio chunk pushed to queue
AUDIO_SR        = 16_000  # expected Whisper sample rate (Hz)


# ---------------------------------------------------------------------------
# Data carrier types
# ---------------------------------------------------------------------------

@dataclass
class VideoFrame:
    """One raw BGR frame from the camera / video file."""
    frame: np.ndarray  # HxWx3 BGR uint8
    frame_idx: int     # 0-based sequential frame index
    timestamp: float   # nominal capture time in seconds (frame_idx / fps)


@dataclass
class AudioChunk:
    """One chunk of raw float32 PCM at AUDIO_SR Hz from the microphone."""
    samples: np.ndarray  # shape (N,) float32 at 16kHz mono
    t_start: float       # seconds from session start
    t_end: float         # seconds from session start


# ---------------------------------------------------------------------------
# FrameFeeder
# ---------------------------------------------------------------------------

class FrameFeeder(threading.Thread):
    """
    Reads frames from a video file and pushes VideoFrame objects to frame_queues.
    Dispatches round-robin across worker queues for parallel multi-GPU ingestion.
    """

    def __init__(
        self,
        video_path: str,
        frame_queue,
        fps: float,
        downscale_to: int = 720,
    ):
        super().__init__(daemon=True, name="FrameFeeder")
        self.video_path = video_path
        self.frame_queues = frame_queue if isinstance(frame_queue, list) else [frame_queue]
        self.fps = fps
        self.downscale_to = downscale_to
        self._stop_event = threading.Event()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        t0 = time.perf_counter()
        log.info("[feeder] FrameFeeder started -> %s (%d worker queues)", self.video_path, len(self.frame_queues))

        n_workers = len(self.frame_queues)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.downscale_to:
                scale = self.downscale_to / max_dim
                nw = int(round(w * scale))
                nh = int(round(h * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

            target_q = self.frame_queues[frame_idx % n_workers]
            target_q.put(VideoFrame(
                frame=frame,
                frame_idx=frame_idx,
                timestamp=frame_idx / self.fps,
            ))
            frame_idx += 1

        cap.release()
        for q in self.frame_queues:
            q.put(None)  # sentinel for each worker
        log.info("[feeder] FrameFeeder done: %d frames dispatched in %.2fs.", frame_idx, time.perf_counter() - t0)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# AudioFeeder
# ---------------------------------------------------------------------------

class AudioFeeder(threading.Thread):
    """
    Reads a pre-extracted WAV file and pushes 0.5-second AudioChunk objects
    to audio_chunk_queue at real-time pace (simulates a live microphone).

    Pushes None as a sentinel when all audio is consumed.
    """

    def __init__(
        self,
        audio_path: str,
        audio_chunk_queue: queue.Queue,
        pacing: bool = False,
    ):
        super().__init__(daemon=True, name="AudioFeeder")
        self.audio_path = audio_path
        self.audio_chunk_queue = audio_chunk_queue
        self.pacing = pacing
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("[feeder] AudioFeeder started (pacing=%s) -> %s", self.pacing, self.audio_path)
        samples, sr = sf.read(self.audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]

        if sr != AUDIO_SR:
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sr, target_sr=AUDIO_SR)
            except ImportError:
                factor = AUDIO_SR / sr
                n_out = int(len(samples) * factor)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, n_out),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.float32)

        chunk_size = int(AUDIO_CHUNK_SEC * AUDIO_SR)
        total_audio_sec = len(samples) / AUDIO_SR
        n_chunks = (len(samples) + chunk_size - 1) // chunk_size
        log.info("[feeder] AudioFeeder: %.2fs audio -> %d chunks of %.1fs.", total_audio_sec, n_chunks, AUDIO_CHUNK_SEC)

        t_wall_start = time.perf_counter()

        for i in range(n_chunks):
            if self._stop_event.is_set():
                break

            t_start = i * AUDIO_CHUNK_SEC
            t_end   = min((i + 1) * AUDIO_CHUNK_SEC, total_audio_sec)
            chunk   = samples[i * chunk_size : (i + 1) * chunk_size].copy()

            self.audio_chunk_queue.put(AudioChunk(
                samples=chunk,
                t_start=t_start,
                t_end=t_end,
            ))

            if self.pacing:
                target_wall = t_wall_start + t_end
                wait = target_wall - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

        self.audio_chunk_queue.put(None)  # sentinel
        log.info("[feeder] AudioFeeder done.")

    def stop(self) -> None:
        self._stop_event.set()
