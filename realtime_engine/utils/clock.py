"""
clock.py
========
Audio Master Clock utilities for the real-time streaming pipeline.

Problem solved: after OpenVoice adjusts speech speed the translated audio
duration differs from the source video frame count. Accumulating across
many sentence chunks causes progressive audio-video desync.

Solution: designate the *translated audio* as the master clock.  After TTS
synthesis, compute the exact frame count needed for that audio duration and
duplicate or drop video frames to align the two streams.
"""

import logging
from typing import List, TypeVar

import numpy as np

log = logging.getLogger(__name__)

T = TypeVar("T")


def align_frames_to_audio(
    frames: List[T],
    audio_duration_sec: float,
    fps: float,
    max_dup_run: int = 3,
) -> List[T]:
    """
    Align a list of video frames to match a target audio duration exactly.

    The frame list is either extended by duplicating tail frames or trimmed
    by dropping evenly-spaced interior frames so that:
        len(output) == round(audio_duration_sec * fps)

    Parameters
    ----------
    frames : list
        Ordered list of video frames (any type: np.ndarray, PIL Image, etc.).
    audio_duration_sec : float
        Target duration in seconds as determined by the translated audio clip.
    fps : float
        Video frames-per-second.
    max_dup_run : int
        Maximum consecutive duplicate frames allowed when padding.
        Capped to prevent overly static mouth regions at chunk ends.

    Returns
    -------
    list
        Resampled frame list whose length matches the audio duration exactly.
    """
    target_n = max(1, round(audio_duration_sec * fps))
    current_n = len(frames)

    if current_n == target_n:
        return frames

    if target_n > current_n:
        # Need more frames — duplicate tail frames (comfort padding)
        needed = target_n - current_n
        # Cap each duplication run to max_dup_run to avoid frozen mouth
        tail_frame = frames[-1]
        dup_count = min(needed, max_dup_run)
        padding = [tail_frame] * dup_count

        if needed > max_dup_run:
            # Distribute remaining duplicates evenly from the end of the list
            extra = needed - dup_count
            step = max(1, current_n // (extra + 1))
            indices = [min(current_n - 1 - i * step, current_n - 1) for i in range(extra)]
            distributed = [frames[i] for i in sorted(indices)]
            result = frames + distributed + padding
        else:
            result = frames + padding

        log.debug(
            "[clock] Padded %d → %d frames (audio %.3fs @ %.1f fps)",
            current_n, len(result), audio_duration_sec, fps,
        )
        return result[:target_n]

    else:
        # Need fewer frames — drop evenly-spaced frames
        indices = np.round(np.linspace(0, current_n - 1, target_n)).astype(int)
        result = [frames[i] for i in indices]
        log.debug(
            "[clock] Trimmed %d → %d frames (audio %.3fs @ %.1f fps)",
            current_n, target_n, audio_duration_sec, fps,
        )
        return result


def compute_speed_ratio(
    generated_duration: float,
    original_duration: float,
    min_ratio: float = 0.80,
    max_ratio: float = 1.30,
) -> float:
    """
    Compute TTS playback speed ratio to match original audio duration.

    Clamped to [min_ratio, max_ratio] for perceptual quality. If the true
    ratio falls outside bounds, the remaining mismatch is handled by the
    audio master clock frame alignment instead of distorting the voice.

    Parameters
    ----------
    generated_duration : float
        Duration (s) of synthesized TTS audio at speed 1.0x.
    original_duration : float
        Duration (s) of the source audio segment.
    min_ratio, max_ratio : float
        Bounds for the speed ratio to keep voice sounding natural.

    Returns
    -------
    float
        Clamped TTS speed ratio.
    """
    if original_duration <= 0:
        return 1.0
    ratio = generated_duration / original_duration
    clamped = max(min_ratio, min(max_ratio, ratio))
    if abs(clamped - ratio) > 0.01:
        log.debug(
            "[clock] Speed ratio %.3f clamped to %.3f — frame alignment will cover remainder.",
            ratio, clamped,
        )
    return clamped
