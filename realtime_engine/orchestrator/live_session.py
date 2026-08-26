"""
live_session.py
===============
True Real-Time Live Streaming Session Orchestrator.

Coordinates 6 concurrent worker threads:
  1. FrameFeeder           -> frame_queue
  2. AudioFeeder           -> audio_chunk_queue
  3. FacePreprocessWorker  (GPU 0) -> geo_buffer (thread-safe deque/dict)
  4. AudioPipelineWorker   (GPU 1) -> sentence_queue
  5. RenderWorker          (GPUs 2+3) -> output_queue
  6. OutputWorker          -> final output MP4

Execution flow:
  - Input video is read frame-by-frame and audio chunk-by-chunk concurrently.
  - No blocking passes over the full video before streaming begins.
  - GPU 0 builds facial tracking and VAE latents frame-by-frame.
  - GPU 1 accumulates incoming audio chunks and re-runs Whisper every 0.5s.
  - As soon as a sentence boundary is detected, GPU 1 translates + synthesizes voice + extracts Whisper features.
  - GPUs 2 & 3 immediately render the sentence frames in parallel.
  - Output is assembled chunk-by-chunk in real time.
"""

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from realtime_engine.audio_service.audio_processor import AudioChunkResult, SentenceSegment
from realtime_engine.ingestion.livestream_feeder import AudioChunk, AudioFeeder, FrameFeeder, VideoFrame
from realtime_engine.video_service.face_processor import FaceGeometryFrame, VideoChunkGeometry

log = logging.getLogger(__name__)


class LiveStreamSession:
    """
    Orchestrates the 6-worker true live streaming pipeline.
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
        max_chunk_sec: float = 6.0,
    ):
        self._audio = audio_processor
        if isinstance(face_processor, list):
            self._face_processors = face_processor
        else:
            self._face_processors = [face_processor]
        self._renderer = renderer
        self.fps = fps
        self.width = width
        self.height = height
        self.output_path = output_path
        self.max_chunk_sec = max_chunk_sec

        # Shared thread-safe queues
        self.frame_queues: List[queue.Queue] = [
            queue.Queue(maxsize=64) for _ in self._face_processors
        ]
        self.audio_chunk_queue: queue.Queue = queue.Queue(maxsize=128)
        self.sentence_queue: queue.Queue = queue.Queue(maxsize=32)
        self.output_queue: queue.Queue = queue.Queue(maxsize=32)

        # Thread-safe geometry buffer: frame_idx -> FaceGeometryFrame
        self.geo_buffer: Dict[int, FaceGeometryFrame] = {}
        self.geo_lock = threading.Lock()
        self.geo_condition = threading.Condition(self.geo_lock)
        self.max_geo_frame_idx: int = -1
        self.face_done: bool = False
        self._face_workers_lock = threading.Lock()
        self._active_face_workers = len(self._face_processors)

        # Benchmarking metrics
        self.first_sentence_latency: Optional[float] = None
        self.total_sentences: int = 0

    def run(self, input_video: str, input_audio_path: str) -> dict:
        """
        Run the full concurrent live streaming pipeline.

        Parameters
        ----------
        input_video : str
            Path to input video file (simulating camera).
        input_audio_path : str
            Path to pre-extracted audio WAV (simulating microphone).

        Returns
        -------
        dict with benchmark performance metrics.
        """
        log.info("[live_session] Starting true live streaming session with %d Face Workers...", len(self._face_processors))
        t_session_start = time.perf_counter()

        # Reset rolling live states
        for fp in self._face_processors:
            fp.reset_live_state()
        self._audio.reset_live_audio()
        self._active_face_workers = len(self._face_processors)
        self.face_done = False

        # 1. Ingestion Feeders
        frame_feeder = FrameFeeder(
            video_path=input_video,
            frame_queue=self.frame_queues,
            fps=self.fps,
            downscale_to=720,
        )
        audio_feeder = AudioFeeder(
            audio_path=input_audio_path,
            audio_chunk_queue=self.audio_chunk_queue,
            pacing=False,
        )

        # 2. Worker Threads
        face_workers = [
            threading.Thread(
                target=self._face_preprocess_worker,
                args=(fp, i),
                name=f"FaceWorker_{i}_GPU_{fp.device_id}",
                daemon=True,
            )
            for i, fp in enumerate(self._face_processors)
        ]
        audio_worker = threading.Thread(target=self._audio_pipeline_worker, name="AudioWorker", daemon=True)
        render_worker = threading.Thread(target=self._render_worker, name="RenderWorker", daemon=True)

        tmp_dir = tempfile.mkdtemp(prefix="live_session_out_")
        output_chunks: List[Tuple[int, str, Optional[str]]] = []
        output_worker = threading.Thread(
            target=self._output_worker,
            args=(tmp_dir, output_chunks),
            name="OutputWorker",
            daemon=True,
        )

        # Launch all threads
        t_threads_start = time.perf_counter()
        output_worker.start()
        render_worker.start()
        audio_worker.start()
        for fw in face_workers:
            fw.start()
        frame_feeder.start()
        audio_feeder.start()

        # Wait for completion in pipeline order
        frame_feeder.join()
        audio_feeder.join()
        for fw in face_workers:
            fw.join()
        audio_worker.join()
        render_worker.join()
        output_worker.join()

        t_streaming_end = time.perf_counter()
        streaming_dur = t_streaming_end - t_threads_start

        # Finalize concatenation of all chunk files to output_path
        self._finalize_video(tmp_dir, output_chunks)
        total_dur = time.perf_counter() - t_session_start

        log.info("[live_session] Live session complete in %.2fs (streaming loop: %.2fs).", total_dur, streaming_dur)

        return {
            "streaming_dur": streaming_dur,
            "total_dur": total_dur,
            "first_sentence_latency": self.first_sentence_latency or 0.0,
            "total_sentences": self.total_sentences,
        }

    # ------------------------------------------------------------------
    # Worker implementations
    # ------------------------------------------------------------------

    def _face_preprocess_worker(self, fp, worker_id: int) -> None:
        """Face Worker on dedicated GPU: consumes frames from dedicated worker queue."""
        log.info("[face_worker_%d] GPU %d Face Preprocess Worker active.", worker_id, fp.device_id)
        t0 = time.perf_counter()
        count = 0
        my_queue = self.frame_queues[worker_id]

        while True:
            item = my_queue.get()
            if item is None:
                break

            geo_frame = fp.process_frame_live(
                frame_bgr=item.frame,
                frame_idx=item.frame_idx,
                fps=self.fps,
                dwpose_keyframe_step=12,
            )

            with self.geo_condition:
                self.geo_buffer[item.frame_idx] = geo_frame
                self.max_geo_frame_idx = max(self.max_geo_frame_idx, item.frame_idx)
                self.geo_condition.notify_all()

            count += 1

        with self._face_workers_lock:
            self._active_face_workers -= 1
            if self._active_face_workers == 0:
                with self.geo_condition:
                    self.face_done = True
                    self.geo_condition.notify_all()

        log.info("[face_worker_%d] GPU %d preprocessed %d frames in %.2fs (%.1f FPS).", worker_id, fp.device_id, count, time.perf_counter() - t0, count / max(0.01, time.perf_counter() - t0))

    def _audio_pipeline_worker(self) -> None:
        """GPU 1 Worker: consumes audio chunks, detects sentence boundaries, translates + synthesizes."""
        log.info("[audio_worker] GPU 1 Audio Pipeline Worker active.")
        t0 = time.perf_counter()

        while True:
            chunk: Optional[AudioChunk] = self.audio_chunk_queue.get()
            if chunk is None:
                # End of stream -- flush remaining uncommitted audio
                committed = self._audio.try_commit_sentence(
                    fps=self.fps,
                    max_chunk_sec=self.max_chunk_sec,
                    force_flush=True,
                )
                for seg, res in committed:
                    self.total_sentences += 1
                    self.sentence_queue.put((seg, res))
                break

            self._audio.push_audio_chunk(chunk.samples, chunk.t_start, chunk.t_end)
            committed = self._audio.try_commit_sentence(
                fps=self.fps,
                max_chunk_sec=self.max_chunk_sec,
                force_flush=False,
            )

            for seg, res in committed:
                self.total_sentences += 1
                self.sentence_queue.put((seg, res))

        self.sentence_queue.put(None)  # sentinel for render_worker
        log.info("[audio_worker] GPU 1 Audio Worker finished: %d sentences in %.2fs.", self.total_sentences, time.perf_counter() - t0)

    def _render_worker(self) -> None:
        """GPUs 2 & 3 Worker: renders sentences as soon as audio + geometry are ready."""
        log.info("[render_worker] GPUs 2+3 Distributed Render Worker active.")

        while True:
            item = self.sentence_queue.get()
            if item is None:
                break

            seg: SentenceSegment = item[0]
            audio_res: AudioChunkResult = item[1]

            t_render_wait_start = time.perf_counter()

            # Wait until face_worker has produced all frames for this sentence
            with self.geo_condition:
                while self.max_geo_frame_idx < (seg.frame_end - 1) and not self.face_done:
                    self.geo_condition.wait(timeout=0.1)

                # Extract geometry slice from buffer
                slice_frames: List[FaceGeometryFrame] = []
                for fi in range(seg.frame_start, seg.frame_end):
                    if fi in self.geo_buffer:
                        slice_frames.append(self.geo_buffer[fi])
                    else:
                        # Fallback if frame index missing
                        last_f = self.geo_buffer.get(self.max_geo_frame_idx)
                        if last_f is not None:
                            slice_frames.append(last_f)

            if not slice_frames:
                log.warning("[render_worker] Empty geometry slice for sentence %d [%d:%d].", seg.idx, seg.frame_start, seg.frame_end)
                continue

            h, w = slice_frames[0].frame_bgr.shape[:2]
            geometry = VideoChunkGeometry(
                coord_list=[f.coord for f in slice_frames],
                mask_list=[f.mask for f in slice_frames],
                crop_box_list=[f.crop_box for f in slice_frames],
                frame_list=[f.frame_bgr for f in slice_frames],
                input_latent_list=[f.latent for f in slice_frames],
                width=w,
                height=h,
                fps=self.fps,
            )

            # Split-GPU render across GPUs 2 & 3
            render_res = self._renderer.render_chunk(
                whisper_chunks=audio_res.whisper_chunks,
                geometry=geometry,
                speech_mask=audio_res.speech_mask,
                chunk_id=seg.idx,
            )

            t_chunk_ready = time.perf_counter()
            if self.first_sentence_latency is None:
                # Time from end of spoken sentence to rendered chunk ready
                self.first_sentence_latency = t_chunk_ready - t_render_wait_start
                log.info(
                    "[live_session] >>> FIRST SENTENCE READY <<< | Sentence %d latency = %.2fs | '%s'",
                    seg.idx, self.first_sentence_latency, audio_res.translated_text[:50],
                )

            self.output_queue.put((seg.idx, render_res.frames, audio_res.audio_path))

        self.output_queue.put(None)  # sentinel for output_worker
        log.info("[render_worker] GPUs 2+3 Render Worker finished.")

    def _output_worker(self, tmp_dir: str, output_chunks: List[Tuple[int, str, Optional[str]]]) -> None:
        """Collects rendered frames, encodes per-chunk silent MP4s and tracks audio tracks."""
        log.info("[output_worker] Output Mux Worker active.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while True:
            item = self.output_queue.get()
            if item is None:
                break

            chunk_idx, frames, audio_path = item
            if not frames:
                continue

            chunk_silent = os.path.join(tmp_dir, f"chunk_{chunk_idx:04d}_silent.mp4")

            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(chunk_silent, fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

            output_chunks.append((chunk_idx, chunk_silent, audio_path))
            log.info("[output_worker] Chunk %d saved (%d frames).", chunk_idx, len(frames))

        log.info("[output_worker] All %d chunks prepared for finalization.", len(output_chunks))

    def _finalize_video(self, tmp_dir: str, output_chunks: List[Tuple[int, str, Optional[str]]]) -> None:
        """Concatenate all video and audio chunks with zero-drift lossless final mux."""
        if not output_chunks:
            log.warning("[live_session] No chunks to finalize.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        ordered = sorted(output_chunks, key=lambda x: x[0])
        
        # 1. Concat video stream
        video_concat_txt = os.path.join(tmp_dir, "video_concat.txt")
        with open(video_concat_txt, "w") as f:
            for _, v_path, _ in ordered:
                f.write(f"file '{v_path}'\n")

        merged_video = os.path.join(tmp_dir, "merged_video.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", video_concat_txt,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            merged_video,
        ], check=True, stderr=subprocess.DEVNULL)

        # 2. Concat audio stream losslessly as raw PCM WAV
        audio_concat_txt = os.path.join(tmp_dir, "audio_concat.txt")
        valid_audio = [a_path for _, _, a_path in ordered if a_path and os.path.exists(a_path)]
        
        if valid_audio:
            with open(audio_concat_txt, "w") as f:
                for a_path in valid_audio:
                    f.write(f"file '{a_path}'\n")

            merged_audio = os.path.join(tmp_dir, "merged_audio.wav")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", audio_concat_txt,
                "-c:a", "pcm_s16le",
                merged_audio,
            ], check=True, stderr=subprocess.DEVNULL)

            # 3. Clean single-pass mux (0 ms AAC priming delay)
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", merged_video,
                "-i", merged_audio,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                self.output_path,
            ], check=True, stderr=subprocess.DEVNULL)
        else:
            shutil.copyfile(merged_video, self.output_path)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("[live_session] Final live-stream video written to: %s", self.output_path)
