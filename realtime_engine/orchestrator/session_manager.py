"""
session_manager.py
==================
Async assembly-line session manager for the real-time pipeline.

Coordinates five concurrent async workers via bounded asyncio.Queue channels:

  [Ingestion] --> prep_queue --> [AudioProc || VideoProc]
                                         |
                              render_queue (after both prep workers finish)
                                         |
                              [DistributedRenderer]
                                         |
                              output_queue --> [Muxer / WebRTC push]

Backpressure:
  Each queue has a max capacity.  If a downstream stage is slower than
  upstream, the oldest item is dropped (with a warning) rather than blocking
  indefinitely or accumulating unbounded memory.

Session lifecycle:
  1. Call ``start()`` to load all models and begin worker coroutines.
  2. Feed media via ``push_audio()`` / ``push_video()``.
  3. Call ``stop()`` to drain queues and unload all models.
  4. Models are always released inside shared_gpu_session context in
     ``realtime_pipeline.py`` so VRAM returns to 0 MB regardless.
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
import soundfile as sf

from realtime_engine.video_service.face_processor import FaceGeometryFrame, VideoChunkGeometry

log = logging.getLogger(__name__)

# Queue depth limits
_PREP_QUEUE_DEPTH   = 32  # Max sentence chunks waiting for audio/video prep
_RENDER_QUEUE_DEPTH = 32  # Max chunks waiting for MuseTalk render
_OUTPUT_QUEUE_DEPTH = 32  # Max rendered chunk bundles waiting for mux


@dataclass
class _PrepItem:
    """One sentence chunk dispatched to the prep workers."""
    audio_samples: np.ndarray
    source_text: str
    video_frames: List[np.ndarray]
    t_start: float
    t_end: float
    fps: float
    chunk_id: int
    frame_start_idx: Optional[int] = None


@dataclass
class _RenderItem:
    """Prep outputs ready for MuseTalk rendering."""
    audio_result: object   # AudioChunkResult
    geometry: object       # VideoChunkGeometry
    whisper_chunks: list
    chunk_id: int


async def _enqueue_with_drop(q: asyncio.Queue, item, label: str) -> None:
    """
    Put item into queue, dropping the oldest entry if the queue is full.
    Prevents blocking and accumulating unbounded memory under live stream load.
    """
    if q.full():
        try:
            dropped = q.get_nowait()
            log.warning(
                "[session] BACKPRESSURE: %s queue full — dropped chunk %s",
                label, getattr(dropped, "chunk_id", "?"),
            )
        except asyncio.QueueEmpty:
            pass
    await q.put(item)


class SessionManager:
    """
    Orchestrates the full real-time pipeline for one streaming session.

    Parameters
    ----------
    audio_processor : AudioProcessor
        Loaded audio service (GPU 1).
    face_processor : FaceProcessor
        Loaded video prep service (GPU 0).
    renderer : DistributedRenderer
        Loaded split-GPU MuseTalk renderer (GPUs 2 & 3).
    ring_buffer : MediaRingBuffer
        Shared media ring buffer fed by the ingestion loop.
    output_callback : coroutine
        Async callable invoked with (frames, audio_path, chunk_id) for
        each completed rendered chunk (e.g., WebRTC push or file write).
    max_chunk_duration : float
        Hard timeout in seconds to force-flush a chunk even with no
        sentence boundary (prevents unbounded latency).
    """

    def __init__(
        self,
        audio_processor,
        face_processor,
        renderer,
        ring_buffer,
        output_callback,
        max_chunk_duration: float = 3.5,
    ):
        self._audio_proc = audio_processor
        self._face_proc = face_processor
        self._renderer = renderer
        self._ring = ring_buffer
        self._output_cb = output_callback
        self.max_chunk_duration = max_chunk_duration

        # Bounded async queues between pipeline stages
        self._prep_queue: asyncio.Queue = asyncio.Queue(maxsize=_PREP_QUEUE_DEPTH)
        self._render_queue: asyncio.Queue = asyncio.Queue(maxsize=_RENDER_QUEUE_DEPTH)
        self._output_queue: asyncio.Queue = asyncio.Queue(maxsize=_OUTPUT_QUEUE_DEPTH)

        self._running = False
        self._chunk_counter = 0
        self._completed_count = 0
        self._tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load models and start all pipeline worker coroutines."""
        log.info("[session] Loading all models...")
        loop = asyncio.get_event_loop()

        # Load models in thread pool to avoid blocking the event loop
        await loop.run_in_executor(None, self._audio_proc.load_models)
        await loop.run_in_executor(None, self._face_proc.load_models)
        await loop.run_in_executor(None, self._renderer.load_models)
        log.info("[session] All models loaded. Starting pipeline workers.")

        self._running = True
        self._tasks = [
            asyncio.create_task(self._prep_worker()),
            asyncio.create_task(self._render_worker()),
            asyncio.create_task(self._output_worker()),
        ]

    async def stop(self) -> None:
        """Signal workers to drain queues, then unload all models."""
        log.info("[session] Stopping pipeline...")
        self._running = False

        # Drain sentinel values through the pipeline
        await self._prep_queue.put(None)
        await self._render_queue.put(None)
        await self._output_queue.put(None)

        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                task.cancel()

        # Unload all models — gpu_guard.shared_gpu_session will do final VRAM cleanup
        self._audio_proc.unload_models()
        self._face_proc.unload_models()
        self._renderer.unload_models()
        log.info("[session] Pipeline stopped and models unloaded.")

    def dispatch_chunk(
        self,
        audio_samples: np.ndarray,
        source_text: str,
        video_frames: List[np.ndarray],
        t_start: float,
        t_end: float,
        fps: float,
        frame_start_idx: Optional[int] = None,
    ) -> None:
        """
        Dispatch a sentence chunk into the prep queue (non-blocking).

        Called from the ingestion/VAD loop when a sentence boundary is detected.
        """
        item = _PrepItem(
            audio_samples=audio_samples,
            source_text=source_text,
            video_frames=video_frames,
            t_start=t_start,
            t_end=t_end,
            fps=fps,
            chunk_id=self._chunk_counter,
            frame_start_idx=frame_start_idx,
        )
        self._chunk_counter += 1
        # Fire-and-forget schedule into prep_queue
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.ensure_future(
                _enqueue_with_drop(self._prep_queue, item, "prep")
            )
        )

    # ------------------------------------------------------------------
    # Pipeline worker coroutines
    # ------------------------------------------------------------------

    async def _prep_worker(self) -> None:
        """
        Async worker that runs AudioProcessor and FaceProcessor concurrently
        (in thread pool executors) for each queued sentence chunk.
        """
        loop = asyncio.get_event_loop()
        while self._running:
            item: Optional[_PrepItem] = await self._prep_queue.get()
            if item is None:
                break

            log.info("[session] Prep worker: processing chunk %d", item.chunk_id)
            t0 = time.perf_counter()

            # FaceProcessor never changes the frame count — it computes geometry
            # FOR the frames passed in.  So target_n_frames is known upfront,
            # allowing audio and video prep to run concurrently.
            target_n_frames = len(item.video_frames)

            audio_future = loop.run_in_executor(
                None,
                self._audio_proc.process_chunk,
                item.audio_samples,
                item.source_text,
                item.t_start,
                item.t_end,
                target_n_frames,
                item.fps,
                item.chunk_id,
                None,   # original_audio_path → extracted from audio_samples
            )
            video_future = loop.run_in_executor(
                None,
                self._face_proc.process_chunk,
                item.video_frames,
                item.fps,
                item.frame_start_idx,
            )

            try:
                audio_result, geometry = await asyncio.gather(audio_future, video_future)
            except Exception as exc:
                log.error("[session] Prep failed for chunk %d: %s", item.chunk_id, exc)
                continue

            render_item = _RenderItem(
                audio_result=audio_result,
                geometry=geometry,
                whisper_chunks=audio_result.whisper_chunks,
                chunk_id=item.chunk_id,
            )
            await _enqueue_with_drop(self._render_queue, render_item, "render")
            log.info(
                "[session] Chunk %d prepped in %.3fs → queued for rendering.",
                item.chunk_id, time.perf_counter() - t0,
            )

    async def _render_worker(self) -> None:
        """
        Async worker that submits queued prep items to the DistributedRenderer.
        """
        loop = asyncio.get_event_loop()
        while self._running:
            item: Optional[_RenderItem] = await self._render_queue.get()
            if item is None:
                break

            log.info("[session] Render worker: chunk %d", item.chunk_id)
            t0 = time.perf_counter()

            try:
                result = await loop.run_in_executor(
                    None,
                    self._renderer.render_chunk,
                    item.whisper_chunks,
                    item.geometry,
                    item.audio_result.audio_path,
                    getattr(item.audio_result, "speech_frames", None),
                )
            except Exception as exc:
                log.error("[session] Render failed for chunk %d: %s", item.chunk_id, exc)
                continue

            output = (result.frames, item.audio_result.audio_path, item.chunk_id)
            await _enqueue_with_drop(self._output_queue, output, "output")
            log.info(
                "[session] Chunk %d rendered in %.3fs → queued for output.",
                item.chunk_id, time.perf_counter() - t0,
            )

    async def wait_until_complete(self, timeout: float = 600.0) -> None:
        """
        Block until all dispatched sentence chunks have completed rendering and output.
        """
        log.info("[session] Waiting for %d chunks to complete processing...", self._chunk_counter)
        t0 = time.time()
        while self._completed_count < self._chunk_counter:
            if time.time() - t0 > timeout:
                log.warning("[session] wait_until_complete timed out after %.1fs.", timeout)
                break
            await asyncio.sleep(0.2)
        log.info(
            "[session] All %d chunks finished processing (%d completed).",
            self._chunk_counter, self._completed_count,
        )

    async def _output_worker(self) -> None:
        """
        Async worker that calls the output_callback for each completed chunk.
        Maintains chunk ordering by chunk_id.
        """
        pending = {}  # chunk_id -> (frames, audio_path)
        next_expected = 0

        while self._running:
            item = await self._output_queue.get()
            if item is None:
                break

            frames, audio_path, chunk_id = item
            pending[chunk_id] = (frames, audio_path)

            # Flush in order
            while next_expected in pending:
                f, ap = pending.pop(next_expected)
                try:
                    await self._output_cb(f, ap, next_expected)
                except Exception as exc:
                    log.error("[session] Output callback failed for chunk %d: %s", next_expected, exc)
                self._completed_count += 1
                next_expected += 1

    # ------------------------------------------------------------------
    # Note: _extract_whisper_features removed — Whisper is now resident on
    # GPU 1 inside AudioProcessor. Features are extracted inside process_chunk()
    # and returned as AudioChunkResult.whisper_chunks (zero cold-start per chunk).
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# StreamingSession — Continuous sentence-by-sentence pipeline
# ---------------------------------------------------------------------------

class StreamingSession:
    """
    Runs the continuous streaming pipeline for one video session.

    Execution model:
      1. transcribe_and_segment() — one-time Whisper pass (GPU 1).
      2. preprocess_all_frames()  — one-time face geometry pass (GPU 0).
      3. Sentence streaming loop:
           For each sentence i:
             - Submit audio processing for sentence i to GPU 1 executor.
             - (Simultaneously, GPU 2+3 are rendering sentence i-1.)
             - get_sentence_geometry() — instant Python slice (no GPU).
             - Wait for audio future to complete.
             - Submit render_chunk for sentence i to GPU 2+3 executor.
             - Wait for render future (if needed for in-order output).
             - Collect rendered frames + audio into output list.
      4. finalize() — concatenate all chunk videos + audio → final MP4.

    This gives true sentence-level pipelining:
      GPU 0 finishes before the loop → not in the critical path at all.
      GPU 1 (audio) and GPUs 2+3 (render) overlap continuously.
    """

    def __init__(
        self,
        audio_processor,
        face_processor,
        renderer,
        fps: float,
        width: int,
        height: int,
        output_path: str,
    ):
        if isinstance(audio_processor, list):
            self._audio_processors = audio_processor
        else:
            self._audio_processors = [audio_processor]
        self._audio = self._audio_processors[0]
        self._face = face_processor
        self._renderer = renderer
        self.fps = fps
        self.width = width
        self.height = height
        self.output_path = output_path

        # Results collected per sentence: list of (frames, audio_path, idx)
        self._results: List[tuple] = []

    def run(
        self,
        all_frames: List[np.ndarray],
        audio_path: str,
    ) -> float:
        """
        Run the full unthrottled pipelined streaming pipeline:
        All 4 GPUs overlap concurrently from millisecond 0:
          - GPUs 0 & 1: Multi-GPU parallel voice synthesis (Sentences 0,2,4 on GPU 1, 1,3,5 on GPU 0)
          - GPU 0: Concurrent rolling face geometry prep
          - GPUs 2 & 3: Distributed MuseTalk rendering at 57 FPS as soon as each chunk arrives
        """
        import concurrent.futures
        import threading

        t_stream_start = time.perf_counter()
        n_total = len(all_frames)

        # Thread-safe geometry buffer populated by GPU 0 face worker
        geo_buffer: Dict[int, FaceGeometryFrame] = {}
        geo_lock = threading.Lock()
        geo_condition = threading.Condition(geo_lock)
        max_geo_frame_idx = [-1]
        face_done = [False]
        face_err = []

        def face_worker_thread():
            try:
                log.info("[stream] GPU 0 Face Worker active: streaming %d frames with incremental batched VAE...", n_total)
                def handle_batch(batch_geos):
                    with geo_condition:
                        for fi, gf in batch_geos:
                            geo_buffer[fi] = gf
                            if fi > max_geo_frame_idx[0]:
                                max_geo_frame_idx[0] = fi
                        geo_condition.notify_all()

                self._face.preprocess_all_frames(
                    all_frames, self.fps, keyframe_step=12, on_batch_ready=handle_batch
                )
            except Exception as e:
                log.exception("[stream] Error in face_worker_thread: %s", e)
                face_err.append(e)
            finally:
                with geo_condition:
                    face_done[0] = True
                    geo_condition.notify_all()

        t_face = threading.Thread(target=face_worker_thread, name="StreamFaceWorker", daemon=True)
        t_face.start()

        # Phase 1: Fast Whisper transcription + sentence segmentation on GPU 1
        log.info("[stream] Transcribing and segmenting audio on GPU %d...", self._audio.device_id)
        segments = self._audio.transcribe_and_segment(audio_path, self.fps)
        if not segments:
            log.error("[stream] No segments returned from transcription. Aborting.")
            return 0.0

        for seg in segments:
            seg.frame_end = min(seg.frame_end, n_total)
            seg.n_frames = max(1, seg.frame_end - seg.frame_start)

        log.info(
            "[stream] Overlapping Voice Cloning across %d GPUs %s and MuseTalk Render on GPUs %s for %d sentences...",
            len(self._audio_processors), [ap.device_id for ap in self._audio_processors], self._renderer.gpu_ids, len(segments)
        )

        n_audio_workers = len(self._audio_processors)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_audio_workers) as audio_exec, \
             concurrent.futures.ThreadPoolExecutor(max_workers=1) as render_exec:

            # Pre-submit ALL sentences across available audio GPUs in parallel!
            pending_audio = {}
            for k, seg in enumerate(segments):
                ap = self._audio_processors[k % n_audio_workers]
                pending_audio[seg.idx] = audio_exec.submit(
                    ap.process_sentence, seg, self.fps
                )

            pending_render: Optional[concurrent.futures.Future] = None
            prev_render_idx: int = -1

            for i, seg in enumerate(segments):
                t_sent_start = time.perf_counter()

                # Wait for GPU 0 to finish geometry for this sentence's frames
                with geo_condition:
                    while max_geo_frame_idx[0] < (seg.frame_end - 1) and not face_done[0]:
                        geo_condition.wait(timeout=0.05)

                if face_err:
                    raise face_err[0]

                # Extract geometry slice from geo_buffer
                frame_slice_geos = [geo_buffer[fi] for fi in range(seg.frame_start, seg.frame_end)]
                s_h, s_w = all_frames[0].shape[:2]
                geo = VideoChunkGeometry(
                    coord_list=[f.coord for f in frame_slice_geos],
                    mask_list=[f.mask for f in frame_slice_geos],
                    crop_box_list=[f.crop_box for f in frame_slice_geos],
                    frame_list=[all_frames[fi] for fi in range(seg.frame_start, seg.frame_end)],
                    input_latent_list=[f.latent for f in frame_slice_geos],
                    width=s_w,
                    height=s_h,
                    fps=self.fps,
                )

                # Wait for this sentence's audio from GPU 1
                audio_result = pending_audio.pop(seg.idx).result()

                # Wait for previous render chunk to complete
                if pending_render is not None:
                    prev_result = pending_render.result()
                    self._results.append((prev_result.frames, None, prev_render_idx))

                # Submit chunk render to GPUs 2 & 3
                pending_render = render_exec.submit(
                    self._renderer.render_chunk,
                    audio_result.whisper_chunks,
                    geo,
                    audio_result.speech_mask,
                    seg.idx,
                )
                prev_render_idx = seg.idx

                self._results_audio = getattr(self, "_results_audio", {})
                self._results_audio[seg.idx] = audio_result.audio_path

                t_sent = time.perf_counter() - t_sent_start
                log.info(
                    "[stream] Sentence %d/%d rendered chunk ready in %.2fs (audio=%.2fs) | '%s'",
                    i + 1, len(segments), t_sent, audio_result.audio_duration,
                    audio_result.translated_text[:60],
                )

            if pending_render is not None:
                prev_result = pending_render.result()
                self._results.append((prev_result.frames, None, prev_render_idx))

        t_face.join()
        stream_dur = time.perf_counter() - t_stream_start
        log.info(
            "[stream] Unthrottled streaming loop complete: %d sentences in %.2fs.",
            len(segments), stream_dur,
        )
        return stream_dur

    def finalize(self) -> None:
        """
        Concatenate all rendered sentence chunks into the final zero-drift H.264 output MP4.
        """
        import subprocess
        import tempfile
        import shutil

        if not self._results:
            log.warning("[stream] No rendered chunks to finalize.")
            return

        audio_map = getattr(self, "_results_audio", {})
        ordered = sorted(self._results, key=lambda x: x[2])

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="stream_final_")
        chunk_videos: List[Tuple[int, str, Optional[str]]] = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for frames, _, chunk_idx in ordered:
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            chunk_silent = os.path.join(tmp_dir, f"chunk_{chunk_idx:04d}_silent.mp4")

            writer = cv2.VideoWriter(chunk_silent, fourcc, self.fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()
            chunk_videos.append((chunk_idx, chunk_silent, audio_map.get(chunk_idx)))

        if not chunk_videos:
            log.warning("[stream] No valid video frames to write.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # 1. Concat video stream to standard H.264 yuv420p video
        video_concat_txt = os.path.join(tmp_dir, "video_concat.txt")
        with open(video_concat_txt, "w") as f:
            for _, v_path, _ in chunk_videos:
                f.write(f"file '{v_path}'\n")

        merged_video = os.path.join(tmp_dir, "merged_video.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", video_concat_txt,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-crf", "18",
            merged_video,
        ], check=True, stderr=subprocess.DEVNULL)

        # 2. Concat audio stream losslessly as raw PCM WAV
        audio_concat_txt = os.path.join(tmp_dir, "audio_concat.txt")
        valid_audio = [a_path for _, _, a_path in chunk_videos if a_path and os.path.exists(a_path)]

        if valid_audio:
            with open(audio_concat_txt, "w") as f:
                for a_path in valid_audio:
                    f.write(f"file '{a_path}'\n")

            merged_audio = os.path.join(tmp_dir, "merged_audio.wav")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", audio_concat_txt,
                "-c", "copy",
                merged_audio,
            ], check=True, stderr=subprocess.DEVNULL)

            # 3. Final Mux: merged H.264 video + merged AAC audio
            subprocess.run([
                "ffmpeg", "-y",
                "-i", merged_video,
                "-i", merged_audio,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                self.output_path,
            ], check=True, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", merged_video,
                "-c:v", "copy",
                self.output_path,
            ], check=True, stderr=subprocess.DEVNULL)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("[stream] Finalized output written to %s", self.output_path)
        log.info("[stream] Final video written to: %s", self.output_path)
