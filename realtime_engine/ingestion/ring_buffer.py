"""
ring_buffer.py
==============
Fixed-capacity CPU-RAM ring buffer for continuous audio/video ingestion
from a live WebRTC / RTSP stream.

Why a ring buffer?
------------------
- Audio packets arrive every ~20 ms; video frames every ~33 ms.
- Pipeline stages (Whisper STT, MuseTalk) process *sentence-level chunks*
  (typically 1.5 – 4 seconds), not individual frames.
- A ring buffer holds the last N seconds of media in memory and allows
  time-aligned slicing for any [t_start, t_end] window.
- Old data beyond the capacity window is automatically overwritten, keeping
  memory usage bounded during long-running streams.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AudioPacket:
    """One chunk of PCM audio samples with an arrival timestamp."""
    samples: np.ndarray          # float32, mono, 16 kHz
    timestamp: float             # seconds since epoch (time.monotonic())
    sample_rate: int = 16_000


@dataclass
class VideoFrame:
    """One BGR video frame with an arrival timestamp."""
    frame: np.ndarray            # uint8 BGR, shape (H, W, 3)
    timestamp: float             # seconds since epoch (time.monotonic())
    frame_index: int = 0


class MediaRingBuffer:
    """
    Thread-safe ring buffer storing the last `max_seconds` of audio and video.

    Parameters
    ----------
    max_seconds : float
        How many seconds of media to retain.  Older data is discarded.
    audio_sample_rate : int
        Expected sample rate for audio packets (default 16 kHz).
    """

    def __init__(self, max_seconds: float = 6.0, audio_sample_rate: int = 16_000):
        self.max_seconds = max_seconds
        self.audio_sample_rate = audio_sample_rate
        self._lock = threading.Lock()

        # Deques store (timestamp, data) pairs ordered by arrival time
        self._audio: deque = deque()   # deque of AudioPacket
        self._video: deque = deque()   # deque of VideoFrame

        self._video_frame_counter = 0

    # ------------------------------------------------------------------
    # Push methods (called by the ingestion / WebRTC receive loop)
    # ------------------------------------------------------------------

    def push_audio(self, samples: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append a chunk of PCM audio samples to the ring buffer.

        Parameters
        ----------
        samples : np.ndarray
            float32 mono audio array at 16 kHz.
        timestamp : float, optional
            Monotonic timestamp of the packet. Defaults to now.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        packet = AudioPacket(samples=samples.astype(np.float32), timestamp=ts)
        with self._lock:
            self._audio.append(packet)
            self._evict_old_audio()

    def push_video(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append a raw BGR video frame to the ring buffer.

        Parameters
        ----------
        frame : np.ndarray
            uint8 BGR image array of shape (H, W, 3).
        timestamp : float, optional
            Monotonic timestamp of the frame. Defaults to now.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        vf = VideoFrame(frame=frame, timestamp=ts, frame_index=self._video_frame_counter)
        self._video_frame_counter += 1
        with self._lock:
            self._video.append(vf)
            self._evict_old_video()

    # ------------------------------------------------------------------
    # Slice methods (called by the orchestrator on sentence boundary)
    # ------------------------------------------------------------------

    def slice_audio(self, t_start: float, t_end: float) -> np.ndarray:
        """
        Return concatenated PCM audio samples between t_start and t_end.

        Parameters
        ----------
        t_start, t_end : float
            Monotonic timestamps (seconds) defining the slice window.

        Returns
        -------
        np.ndarray
            Concatenated float32 audio samples, or empty array if none found.
        """
        with self._lock:
            chunks = [
                p.samples for p in self._audio
                if t_start <= p.timestamp <= t_end
            ]
        if not chunks:
            log.warning("[ring_buffer] No audio packets in window [%.3f, %.3f]", t_start, t_end)
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def slice_video(self, t_start: float, t_end: float) -> List[np.ndarray]:
        """
        Return ordered list of BGR frames whose timestamps fall in [t_start, t_end].

        Parameters
        ----------
        t_start, t_end : float
            Monotonic timestamps (seconds) defining the slice window.

        Returns
        -------
        list[np.ndarray]
            List of BGR uint8 frames, may be empty.
        """
        with self._lock:
            frames = [
                vf.frame for vf in self._video
                if t_start <= vf.timestamp <= t_end
            ]
        if not frames:
            log.warning("[ring_buffer] No video frames in window [%.3f, %.3f]", t_start, t_end)
        return frames

    def latest_audio_timestamp(self) -> Optional[float]:
        """Return the timestamp of the most recently received audio packet."""
        with self._lock:
            return self._audio[-1].timestamp if self._audio else None

    def latest_video_timestamp(self) -> Optional[float]:
        """Return the timestamp of the most recently received video frame."""
        with self._lock:
            return self._video[-1].timestamp if self._video else None

    # ------------------------------------------------------------------
    # Private eviction helpers
    # ------------------------------------------------------------------

    def _evict_old_audio(self) -> None:
        """Drop audio packets older than max_seconds. Must hold _lock."""
        if not self._audio:
            return
        cutoff = self._audio[-1].timestamp - self.max_seconds
        while self._audio and self._audio[0].timestamp < cutoff:
            self._audio.popleft()

    def _evict_old_video(self) -> None:
        """Drop video frames older than max_seconds. Must hold _lock."""
        if not self._video:
            return
        cutoff = self._video[-1].timestamp - self.max_seconds
        while self._video and self._video[0].timestamp < cutoff:
            self._video.popleft()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"MediaRingBuffer("
                f"audio_packets={len(self._audio)}, "
                f"video_frames={len(self._video)}, "
                f"max_seconds={self.max_seconds})"
            )
