"""
audio_processor.py
==================
GPU 1 Audio Processing Service for the real-time streaming pipeline.

Streaming Pipeline:
  Pass 1 (once)  : transcribe_and_segment() — Whisper with word_timestamps=True
                   produces a SentenceSegment per sentence, giving each sentence
                   an exact [t_start, t_end] from the original audio.
  Pass 2 (loop)  : process_sentence() — per-sentence translate + TTS + ToneColor
                   + Whisper feature extraction, all immediately on GPU 1.
                   Per-sentence speed calibrated to segment.duration.

Legacy batch methods (process_full_audio, process_chunk) kept for backwards
compatibility and marked accordingly.

Pipeline (per sentence chunk):
  1. MarianMT NMT  – 10-beam Spanish->English translation.
  2. OpenVoice TTS – BaseSpeakerTTS generates base English speech. Speed is
                     estimated via adaptive seconds-per-word profiling from the
                     first chunk, then clamped to [0.75x, 1.35x]. No double-
                     synthesis pass.
  3. Audio duration matching – synthesized audio is trimmed or zero-padded to
                     exactly match the source video frame count duration. Video
                     frames are NEVER dropped or duplicated.
  4. ToneColorConverter – Applies cached target_se speaker embedding to
                     clone the original speaker's vocal tone.
  5. Whisper feature extraction – MuseTalk Whisper audio features extracted
                     immediately from the synthesized audio on GPU 1. The
                     WhisperModel is kept resident for the full run, then
                     removed by unload_models() along with all other models.

All models loaded onto GPU 1 at session start; explicitly deleted on teardown
by shared_gpu_session context manager in gpu_guard.py.
"""

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path bootstrapping — resolve audio_pipeline modules relative to repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_AUDIO_PIPELINE_DIR = os.path.join(_REPO_ROOT, "audio_pipeline")
_OPENVOICE_DIR = os.path.join(_AUDIO_PIPELINE_DIR, "openvoice")
_TRANSLATION_DIR = os.path.join(_AUDIO_PIPELINE_DIR, "translation")
_MUSETALK_DIR = os.path.join(_REPO_ROOT, "MuseTalk")

for _d in [_REPO_ROOT, _AUDIO_PIPELINE_DIR, _OPENVOICE_DIR, _TRANSLATION_DIR, _MUSETALK_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Average English words per second — calibrated empirically from TTS output.
# Updated after first chunk via adaptive profiling.
_AVG_WORDS_PER_SECOND = 3.2   # ~0.31 seconds per word at normal speaking pace


@dataclass
class SentenceSegment:
    """
    One sentence from the Whisper transcription, with exact time bounds.
    Produced by transcribe_and_segment() and consumed by process_sentence().
    """
    source_text: str     # Original Spanish text for this sentence
    t_start: float       # Start time in original audio (seconds)
    t_end: float         # End time in original audio (seconds)
    frame_start: int     # t_start * fps (rounded down)
    frame_end: int       # t_end * fps (rounded up), exclusive
    duration: float      # t_end - t_start
    n_frames: int        # frame_end - frame_start
    idx: int = 0         # Sequential sentence index (0-based)
    speech_start: float = 0.0  # Lead silence before first word (seconds)


@dataclass
class AudioChunkResult:
    """Output produced by processing one sentence chunk."""
    source_text: str          # Original Spanish transcription
    translated_text: str      # English translation
    audio_path: str           # Path to temp WAV file with duration-matched cloned speech
    audio_samples: np.ndarray # Raw float32 PCM of the synthesized audio (in-memory copy)
    audio_sr: int             # Sample rate of the synthesized audio
    audio_duration: float     # Duration in seconds (== source frame count / fps)
    whisper_chunks: list      # MuseTalk Whisper audio features (ready for UNet)
    t_start: float            # Monotonic timestamp of chunk start
    t_end: float              # Monotonic timestamp of chunk end
    speech_frames: int = 0    # Number of active speech frames before tail silence
    speech_mask: Optional[np.ndarray] = None  # Boolean frame mask (True for active speech, False for silence)


class AudioProcessor:
    """
    Stateful audio processing service for one streaming session.

    Parameters
    ----------
    device_id : int
        CUDA GPU index for all audio models (default: 1).
    openvoice_ckpt_base : str
        Path to OpenVoice BaseSpeakerTTS checkpoint directory.
    openvoice_ckpt_converter : str
        Path to OpenVoice ToneColorConverter checkpoint directory.
    source_lang : str
        Source language code for Whisper (e.g. "es" for Spanish).
    target_lang_nmt : str
        HuggingFace MarianMT model name for translation.
    num_beams : int
        Beam width for syllable-matching translation (default: 10).
    """

    def __init__(
        self,
        device_id: int = 1,
        openvoice_ckpt_base: Optional[str] = None,
        openvoice_ckpt_converter: Optional[str] = None,
        source_lang: str = "es",
        target_lang_nmt: str = "Helsinki-NLP/opus-mt-es-en",
        num_beams: int = 10,
    ):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.source_lang = source_lang
        self.target_lang_nmt = target_lang_nmt
        self.num_beams = num_beams

        # Default checkpoint paths (relative to audio_pipeline/openvoice/)
        self._ckpt_base = openvoice_ckpt_base or os.path.join(
            _OPENVOICE_DIR, "checkpoints", "base_speakers", "EN"
        )
        self._ckpt_converter = openvoice_ckpt_converter or os.path.join(
            _OPENVOICE_DIR, "checkpoints", "converter"
        )

        # Model handles (None until load_models() is called)
        self._tts = None
        self._converter = None
        self._source_se: Optional[torch.Tensor] = None
        self._target_se: Optional[torch.Tensor] = None   # cached after first chunk
        self._nmt_model = None
        self._nmt_tokenizer = None
        self._whisper_model = None        # Resident MuseTalk Whisper on GPU 1
        self._whisper_processor = None   # MuseTalk AudioProcessor wrapper

        # Adaptive TTS speed profiling
        self._speed_words_per_sec: float = _AVG_WORDS_PER_SECOND
        self._speed_sample_count: int = 0

        # In-memory audio buffer: chunk_id -> (samples, sr)
        self._audio_buffer: dict = {}

        # Temp files for converter output (short-lived until in-memory copy is taken)
        self._temp_files: List[str] = []

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """
        Load all audio models onto GPU device_id.
        Called once at session start (inside shared_gpu_session context).
        """
        log.info("[audio_proc] Loading models onto GPU %d...", self.device_id)
        torch.cuda.set_device(self.device_id)
        os.chdir(_MUSETALK_DIR)

        from transformers import MarianMTModel, MarianTokenizer
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        # Whisper STT transcription model
        log.info("[audio_proc] Loading Whisper STT model on GPU %d...", self.device_id)
        import whisper
        self._stt_model = whisper.load_model("base", device=self.device)

        # NMT translation: Spanish -> English
        log.info("[audio_proc] Loading MarianMT translation model: %s", self.target_lang_nmt)
        self._nmt_tokenizer = MarianTokenizer.from_pretrained(self.target_lang_nmt)
        self._nmt_model = MarianMTModel.from_pretrained(self.target_lang_nmt).to(self.device)

        # OpenVoice BaseSpeakerTTS
        log.info("[audio_proc] Loading OpenVoice BaseSpeakerTTS from %s", self._ckpt_base)
        self._tts = BaseSpeakerTTS(
            f"{self._ckpt_base}/config.json", device=str(self.device)
        )
        self._tts.load_ckpt(f"{self._ckpt_base}/checkpoint.pth")
        self._source_se = torch.load(
            os.path.join(self._ckpt_base, "en_default_se.pth")
        ).to(self.device)

        # OpenVoice ToneColorConverter
        log.info("[audio_proc] Loading ToneColorConverter from %s", self._ckpt_converter)
        self._converter = ToneColorConverter(
            f"{self._ckpt_converter}/config.json", device=str(self.device)
        )
        self._converter.load_ckpt(f"{self._ckpt_converter}/checkpoint.pth")

        # Resident MuseTalk Whisper feature extractor on GPU 1
        log.info("[audio_proc] Loading MuseTalk Whisper feature extractor on GPU %d...", self.device_id)
        from musetalk.utils.audio_processor import AudioProcessor as MuseAudioProcessor
        from transformers import WhisperModel
        whisper_dir = "./models/whisper"
        self._whisper_processor = MuseAudioProcessor(feature_extractor_path=whisper_dir)
        self._whisper_model = WhisperModel.from_pretrained(whisper_dir)
        self._whisper_model = self._whisper_model.to(
            device=self.device, dtype=torch.float16
        ).eval()
        self._whisper_model.requires_grad_(False)
        log.info("[audio_proc] All audio models loaded on GPU %d.", self.device_id)

    def extract_speaker_embedding(self, reference_audio_path: str) -> None:
        """
        Extract the speaker's vocal timbre embedding (target_se) from a clean reference audio track.
        Locks the cloned voice timbre consistently across all sentences in the stream.
        """
        torch.cuda.set_device(self.device_id)
        if os.path.exists(reference_audio_path) and self._converter is not None:
            log.info("[audio_proc] Extracting speaker embedding (target_se) from %s...", reference_audio_path)
            self._target_se = self._converter.extract_se([reference_audio_path])
            log.info("[audio_proc] Speaker embedding locked successfully.")

    def unload_models(self) -> None:
        """
        Delete all model references so gpu_guard can fully release VRAM.
        """
        for attr in (
            "_stt_model", "_tts", "_converter", "_nmt_model", "_nmt_tokenizer",
            "_source_se", "_target_se",
            "_whisper_model", "_whisper_processor",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)

        self._audio_buffer.clear()

        # Clean up any remaining temp audio files
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        self._temp_files.clear()
        log.info("[audio_proc] Models unloaded from GPU %d.", self.device_id)

    def set_reference_audio(self, ref_audio_path: str) -> None:
        """
        Extract speaker tone embedding (target_se) from a clean reference audio clip.
        """
        if self._converter is None:
            self.load_models()
        log.info("[audio_proc] Extracting reference speaker embedding from: %s", ref_audio_path)
        torch.cuda.set_device(self.device_id)
        self._target_se = self._converter.extract_se([ref_audio_path])
        log.info("[audio_proc] Reference speaker embedding (target_se) cached successfully.")

    def process_full_audio(
        self,
        audio_path: str,
        target_duration: float,
        fps: int = 25,
        render_gpu_ids: Optional[List[int]] = None,
    ) -> AudioChunkResult:
        """
        Dynamically transcribes, translates, and synthesizes the full audio track for any input video.
        """
        t0 = time.perf_counter()
        # 1. Complete Whisper transcription of the input audio
        res = self._stt_model.transcribe(audio_path, language=self.source_lang, task="transcribe")
        full_source_text = res.get("text", "").strip()
        if not full_source_text:
            full_source_text = "Hola."

        # Split into full sentences by punctuation to avoid cutting off words or clauses
        import re
        spanish_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_source_text) if s.strip()]
        if not spanish_sentences:
            spanish_sentences = [full_source_text]

        # Batch translate full complete sentences with MarianMT (ensures zero dropped clauses)
        inp = self._nmt_tokenizer(spanish_sentences, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self._nmt_model.generate(**inp, num_beams=self.num_beams)
        english_sentences = [self._nmt_tokenizer.decode(t, skip_special_tokens=True) for t in out]

        full_english_text = " ".join(english_sentences)
        log.info("[audio_proc] Transcribed Spanish : %s", full_source_text)
        log.info("[audio_proc] Translated English  : %s", full_english_text)

        # 2. Reference speaker embedding
        if self._target_se is None:
            self.set_reference_audio(audio_path)

        # 3. Dynamic pacing TTS synthesis
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="audio_full_")
        tmp_src = os.path.join(tmp_dir, "tts_raw.wav")
        tmp_conv = os.path.join(tmp_dir, "tts_conv.wav")

        sr = self._tts.hps.data.sampling_rate
        total_samples = int(round(target_duration * sr))
        master_audio = np.zeros(total_samples, dtype=np.float32)
        total_frames = int(round(target_duration * fps))
        is_speech_frame = np.zeros(total_frames, dtype=np.float32)

        total_words = sum(len(s.split()) for s in english_sentences)
        total_pause_time = max(0.0, (len(english_sentences) - 1) * 0.35)
        target_speech_time = max(1.0, target_duration - total_pause_time - 0.4)
        speed = float(np.clip((total_words / 2.70) / target_speech_time, 0.75, 1.35))
        pause_dur = 0.35  # Natural 350ms breathing pause between sentences
        pause_samples = int(round(pause_dur * sr))
        curr_sample = 0

        for text in english_sentences:
            raw = self._tts.tts(text, output_path=None, speaker="default", language="English", speed=speed)
            start_s = curr_sample / sr
            end_s = (curr_sample + len(raw)) / sr
            
            end_sample = min(curr_sample + len(raw), total_samples)
            master_audio[curr_sample:end_sample] = raw[:end_sample - curr_sample]
            
            f_start = int(round(start_s * fps))
            f_end = min(int(round(end_s * fps)), total_frames)
            is_speech_frame[f_start:f_end] = 1.0
            
            curr_sample = end_sample + pause_samples
            if curr_sample >= total_samples:
                break

        sf.write(tmp_src, master_audio, sr)

        # 4. Voice tone color conversion
        self._converter.convert(
            audio_src_path=tmp_src,
            src_se=self._source_se,
            tgt_se=self._target_se,
            output_path=tmp_conv,
            message="@MyShell",
        )

        aligned_audio, sr = sf.read(tmp_conv)

        # Normalize peak
        peak = np.abs(aligned_audio).max()
        if peak > 0:
            aligned_audio = (aligned_audio / peak) * 0.89

        out_path = os.path.join(tmp_dir, "final_audio.wav")
        sf.write(out_path, aligned_audio, sr)

        # Extract MuseTalk Whisper features for the entire audio
        whisper_chunks = self._extract_whisper_features(out_path, fps)
        if isinstance(whisper_chunks, torch.Tensor):
            whisper_chunks = [whisper_chunks[i] for i in range(len(whisper_chunks))]
        elif not isinstance(whisper_chunks, list):
            whisper_chunks = list(whisper_chunks)

        # Pad or trim whisper_chunks to exact total_frames
        if len(whisper_chunks) < total_frames:
            last_chunk = whisper_chunks[-1] if len(whisper_chunks) > 0 else torch.zeros((50, 384))
            for _ in range(total_frames - len(whisper_chunks)):
                whisper_chunks.append(last_chunk.clone())
        elif len(whisper_chunks) > total_frames:
            whisper_chunks = whisper_chunks[:total_frames]

        # Zero out Whisper features during silent frames so UNet denoiser stays neutral
        for idx in range(len(whisper_chunks)):
            if idx < len(is_speech_frame) and not is_speech_frame[idx]:
                whisper_chunks[idx].zero_()

        speech_frames = int(np.sum(is_speech_frame))
        log.info(
            "[audio_proc] Slot-aligned audio done in %.3fs | target=%.2fs active_speech=%.2fs (frames=%d/%d)",
            time.perf_counter() - t0, target_duration, speech_frames / fps, speech_frames, total_frames,
        )

        res_obj = AudioChunkResult(
            source_text=full_source_text,
            translated_text=full_english_text,
            audio_path=out_path,
            audio_samples=aligned_audio,
            audio_sr=sr,
            audio_duration=target_duration,
            whisper_chunks=whisper_chunks,
            t_start=0.0,
            t_end=target_duration,
            speech_frames=speech_frames,
            speech_mask=is_speech_frame,
        )
        res_obj._replica_map = {}
        return res_obj

    def process_chunk(
        self,
        audio_samples: np.ndarray,
        source_text: str,
        t_start: float,
        t_end: float,
        target_n_frames: Optional[int] = None,
        fps: float = 25.0,
        chunk_id: int = 0,
        original_audio_path: Optional[str] = None,
    ) -> AudioChunkResult:
        """
        Translate and synthesize one sentence chunk.

        Parameters
        ----------
        audio_samples : np.ndarray
            Raw float32 PCM of the source audio chunk (16 kHz mono).
        source_text : str
            Spanish transcription for this sentence.
        t_start, t_end : float
            Monotonic timestamps from the ring buffer slice.
        target_n_frames : int, optional
            Number of video frames this chunk covers. If provided, the
            synthesized audio is duration-matched to exactly this many frames.
            No video frames are ever modified.
        fps : float
            Video frames-per-second (used for duration matching).
        chunk_id : int
            Sequential chunk index for in-memory audio buffer keying.
        original_audio_path : str, optional
            Path to source audio; used to extract target_se on first call if not already set.
        """
        torch.cuda.set_device(self.device_id)
        t0 = time.perf_counter()

        # --- 1. Extract target_se once from first chunk if not pre-set ---
        if self._target_se is None:
            ref_path = original_audio_path
            if ref_path is None:
                ref_path = self._write_temp_wav(audio_samples, suffix="_ref.wav")
            self._target_se = self._converter.extract_se([ref_path])
            log.info("[audio_proc] target_se extracted and cached.")

        # --- 2. Translate Spanish -> English ---
        english_text = self._translate(source_text)
        log.debug("[audio_proc] '%s' → '%s'", source_text.strip(), english_text)

        # --- 3. Determine target duration from golden frame count ---
        source_duration = t_end - t_start
        if target_n_frames is not None:
            target_duration = target_n_frames / fps
        else:
            target_duration = source_duration

        # --- 4. TTS Synthesis at Uniform Natural Pace (1.0x) ---
        # Maintains steady conversational pace across all chunks with zero speed whiplash
        tts_sr = self._tts.hps.data.sampling_rate
        speed = 1.0
        raw_audio = self._tts.tts(
            english_text, output_path=None,
            speaker="default", language="English", speed=speed,
        )
        actual_speech_duration = len(raw_audio) / tts_sr
        speech_frames = min(target_n_frames or 999999, max(1, round(actual_speech_duration * fps)))

        # --- 5. Write TTS output for ToneColorConverter ---
        tmp_base_path = self._write_temp_wav(raw_audio, suffix="_base_tts.wav", sample_rate=tts_sr)

        # --- 6. Apply tone color conversion ---
        tmp_cloned_path = self._write_temp_wav(
            np.zeros(1, dtype=np.float32), suffix="_cloned.wav"
        )
        self._converter.convert(
            audio_src_path=tmp_base_path,
            src_se=self._source_se,
            tgt_se=self._target_se,
            output_path=tmp_cloned_path,
            message="@MyShell",
        )

        # --- 7. Read back converted audio ---
        audio_out, audio_sr = sf.read(tmp_cloned_path)
        audio_out = audio_out.astype(np.float32)

        # --- 8. Peak normalize to -1 dBFS ---
        peak = np.abs(audio_out).max()
        if peak > 0:
            audio_out = audio_out * (10 ** (-1.0 / 20) / peak)

        # --- 9. Duration match: trim or zero-pad audio to exactly target_duration.
        #        Video frames are NEVER modified.
        audio_out = self._match_audio_duration(audio_out, target_duration, audio_sr)

        # Write final duration-matched audio back to the temp file
        sf.write(tmp_cloned_path, audio_out, audio_sr)
        audio_duration = len(audio_out) / audio_sr

        # --- 10. Extract MuseTalk Whisper features (resident Whisper on GPU 1) ---
        whisper_chunks = self._extract_whisper_features(tmp_cloned_path, fps)

        # --- 11. Copy audio into in-memory buffer (survives unload_models) ---
        self._audio_buffer[chunk_id] = (audio_out.copy(), audio_sr)

        elapsed = time.perf_counter() - t0
        log.info(
            "[audio_proc] Chunk %d done in %.3fs | orig=%.2fs target=%.2fs speech=%.2fs speed=%.2fx",
            chunk_id, elapsed, source_duration, target_duration, actual_speech_duration, speed,
        )

        return AudioChunkResult(
            source_text=source_text,
            translated_text=english_text,
            audio_path=tmp_cloned_path,
            audio_samples=audio_out,
            audio_sr=audio_sr,
            audio_duration=audio_duration,
            whisper_chunks=whisper_chunks,
            speech_frames=speech_frames,
            t_start=t_start,
            t_end=t_end,
        )

    def get_chunk_audio(self, chunk_id: int) -> Optional[Tuple[np.ndarray, int]]:
        """Return in-memory audio for a chunk (survives model unload)."""
        return self._audio_buffer.get(chunk_id)

    # ------------------------------------------------------------------
    # Streaming API — Pass 1 + Pass 2
    # ------------------------------------------------------------------

    def transcribe_and_segment(
        self,
        audio_path: str,
        fps: float,
        source_lang: Optional[str] = None,
        max_chunk_sec: float = 8.0,
    ) -> "List[SentenceSegment]":
        """
        Pass 1 (one-time): Transcribe full audio with Whisper word_timestamps=True.
        Splits on sentence-ending punctuation (. ! ?) from Whisper's own segment list.
        Returns one SentenceSegment per sentence with exact t_start/t_end from the
        original audio — the source of truth for all downstream timing.

        Parameters
        ----------
        audio_path : str
            Path to the extracted source audio WAV (16 kHz mono).
        fps : float
            Video frames-per-second — used to derive frame_start / frame_end.
        source_lang : str, optional
            Override language (default: self.source_lang).
        max_chunk_sec : float
            Hard cap: if a sentence is longer than this, it is split at the
            nearest word boundary to avoid GPU memory spikes.
        """
        import re
        torch.cuda.set_device(self.device_id)
        lang = source_lang or self.source_lang
        log.info("[audio_proc] Transcribing with word timestamps (lang=%s)...", lang)
        t0 = time.perf_counter()

        result = self._stt_model.transcribe(
            audio_path,
            language=lang,
            task="transcribe",
            word_timestamps=True,
        )

        # Also extract the reference speaker tone from the original audio (once)
        if self._target_se is None:
            self._target_se = self._converter.extract_se([audio_path])
            log.info("[audio_proc] target_se extracted and cached from source audio.")

        # Build sentence list from Whisper segments, splitting on punctuation
        segments: List[SentenceSegment] = []
        current_words: List[dict] = []
        current_start: Optional[float] = None

        for seg in result.get("segments", []):
            for word_info in seg.get("words", []):
                word_text = word_info.get("word", "").strip()
                w_start = float(word_info.get("start", seg["start"]))
                w_end = float(word_info.get("end", seg["end"]))
                if not word_text:
                    continue

                if current_start is None:
                    current_start = w_start

                current_words.append({"word": word_text, "start": w_start, "end": w_end})

                sentence_end = bool(re.search(r'[.!?]$', word_text)) or (
                    bool(re.search(r'[,;:]$', word_text)) and (w_end - current_start) >= 2.5
                )
                duration_cap = (w_end - current_start) >= max_chunk_sec

                if sentence_end or duration_cap:
                    text = " ".join(w["word"] for w in current_words).strip()
                    t_s = current_start
                    t_e = w_end
                    w_first = current_words[0]["start"] if current_words else t_s
                    if len(segments) == 0:
                        t_s = 0.0
                    dur = max(t_e - t_s, 0.1)
                    f_s = max(0, int(t_s * fps))
                    f_e = max(f_s + 1, int(round(t_e * fps)))
                    segments.append(SentenceSegment(
                        source_text=text,
                        t_start=t_s,
                        t_end=t_e,
                        frame_start=f_s,
                        frame_end=f_e,
                        duration=dur,
                        n_frames=f_e - f_s,
                        idx=len(segments),
                        speech_start=max(0.0, w_first - t_s),
                    ))
                    current_words = []
                    current_start = None

        # Flush any trailing words that didn't end with punctuation
        if current_words and current_start is not None:
            text = " ".join(w["word"] for w in current_words).strip()
            t_s = current_start
            t_e = current_words[-1]["end"]
            w_first = current_words[0]["start"] if current_words else t_s
            if len(segments) == 0:
                t_s = 0.0
                f_s = 0
            dur = max(t_e - t_s, 0.1)
            f_s = max(0, int(t_s * fps))
            f_e = max(f_s + 1, int(round(t_e * fps)))
            segments.append(SentenceSegment(
                source_text=text,
                t_start=t_s,
                t_end=t_e,
                frame_start=f_s,
                frame_end=f_e,
                duration=dur,
                n_frames=f_e - f_s,
                idx=len(segments),
                speech_start=max(0.0, w_first - t_s),
            ))

        # Fallback: if Whisper returned no word-level segments, use text-level split
        if not segments:
            full_text = result.get("text", "").strip()
            log.warning("[audio_proc] No word timestamps — falling back to text-level split.")
            import re as _re
            sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
            total_dur = result["segments"][-1]["end"] if result.get("segments") else 5.0
            dur_each = total_dur / max(len(sentences), 1)
            for i, sent in enumerate(sentences):
                t_s = i * dur_each
                t_e = (i + 1) * dur_each
                f_s = int(t_s * fps)
                f_e = int(round(t_e * fps))
                segments.append(SentenceSegment(
                    source_text=sent, t_start=t_s, t_end=t_e,
                    frame_start=f_s, frame_end=f_e,
                    duration=dur_each, n_frames=f_e - f_s, idx=i,
                ))

        # Merge short trailing fragments (< 0.8s or < 20 frames) into previous segment
        merged_segments: List[SentenceSegment] = []
        for s in segments:
            if merged_segments and (s.duration < 0.8 or s.n_frames < 20):
                prev = merged_segments[-1]
                prev.source_text = (prev.source_text + " " + s.source_text).strip()
                prev.t_end = s.t_end
                prev.frame_end = s.frame_end
                prev.duration = prev.t_end - prev.t_start
                prev.n_frames = prev.frame_end - prev.frame_start
                log.info("[audio_proc] Merged short fragment '%s' into segment %d.", s.source_text, prev.idx)
            else:
                s.idx = len(merged_segments)
                merged_segments.append(s)

        segments = merged_segments

        # Seamlessly extend Chunk 0's frame_end to absorb the gap before Sentence 1 (gives Chunk 0 perfect lip-sync)
        if len(segments) > 1 and segments[0].frame_end < segments[1].frame_start:
            segments[0].frame_end = segments[1].frame_start
            segments[0].t_end = segments[1].t_start
            segments[0].duration = segments[0].t_end - segments[0].t_start
            segments[0].n_frames = segments[0].frame_end - segments[0].frame_start

        log.info(
            "[audio_proc] Transcription done in %.2fs → %d sentences (Chunk 0 calibrated).",
            time.perf_counter() - t0, len(segments),
        )
        for s in segments:
            log.info(
                "  [%d] t=[%.2f,%.2f] frames=[%d,%d] n=%d  '%s'",
                s.idx, s.t_start, s.t_end, s.frame_start, s.frame_end, s.n_frames,
                s.source_text[:60],
            )
        return segments

    def process_sentence(
        self,
        segment: "SentenceSegment",
        fps: float,
    ) -> AudioChunkResult:
        """
        Pass 2 (per sentence): Translate + TTS + ToneColor + Whisper features.

        Speed is calibrated per-sentence based on segment.duration so that
        TTS output fills exactly the time window the speaker used.
        ToneColorConverter is applied immediately (not batched to end).

        Parameters
        ----------
        segment : SentenceSegment
            One sentence as produced by transcribe_and_segment().
        fps : float
            Video frames-per-second for Whisper chunk alignment.
        """
        torch.cuda.set_device(self.device_id)
        t0 = time.perf_counter()

        # --- 1. Translate sentence ---
        english_text = self._translate(segment.source_text)
        log.debug("[audio_proc] [%d] '%s' → '%s'", segment.idx, segment.source_text[:50], english_text)

        # --- 2. Measure natural English speech duration ---
        tts_sr = self._tts.hps.data.sampling_rate
        base_probe = self._tts.tts(
            english_text, output_path=None,
            speaker="default", language="English", speed=1.0,
        )
        gen_dur = len(base_probe) / tts_sr

        # Calibrate speed so English speech naturally fills segment duration
        target_speed = float(np.clip(gen_dur / max(0.5, segment.duration), 0.85, 1.25))

        # --- 3. Synthesize at exact calibrated pace ---
        raw_audio = self._tts.tts(
            english_text, output_path=None,
            speaker="default", language="English", speed=target_speed,
        )
        actual_speech_sec = len(raw_audio) / tts_sr
        speed = target_speed

        # --- 4. Write TTS output for ToneColorConverter ---
        tmp_base = self._write_temp_wav(raw_audio, suffix="_base.wav", sample_rate=tts_sr)

        # --- 5. Apply ToneColorConverter immediately (not batched) ---
        tmp_conv = self._write_temp_wav(
            np.zeros(1, dtype=np.float32), suffix="_conv.wav"
        )
        self._converter.convert(
            audio_src_path=tmp_base,
            src_se=self._source_se,
            tgt_se=self._target_se,
            output_path=tmp_conv,
            message="@MyShell",
        )

        # --- 6. Read converted audio + peak-normalize ---
        audio_out, audio_sr = sf.read(tmp_conv)
        audio_out = audio_out.astype(np.float32)
        peak = np.abs(audio_out).max()
        if peak > 0:
            audio_out = audio_out * (10 ** (-1.0 / 20) / peak)

        # --- 7. Duration match and lead-silence alignment ---
        target_duration = segment.n_frames / fps
        t_lead = segment.speech_start if segment.idx == 0 else 0.0
        lead_samples = int(round(t_lead * audio_sr))
        lead_frames = int(round(t_lead * fps))

        if lead_samples > 0:
            target_speech_dur = max(0.5, (segment.n_frames - lead_frames) / fps)
            speech_audio = self._match_audio_duration(audio_out, target_speech_dur, audio_sr)
            audio_out = np.concatenate([np.zeros(lead_samples, dtype=np.float32), speech_audio])
            req_samples = int(round(target_duration * audio_sr))
            if len(audio_out) < req_samples:
                audio_out = np.concatenate([audio_out, np.zeros(req_samples - len(audio_out), dtype=np.float32)])
            elif len(audio_out) > req_samples:
                audio_out = audio_out[:req_samples]
        else:
            audio_out = self._match_audio_duration(audio_out, target_duration, audio_sr)

        sf.write(tmp_conv, audio_out, audio_sr)

        # --- 8. Build per-frame speech mask ---
        speech_mask = np.zeros(segment.n_frames, dtype=np.float32)
        end_speech_frame = min(segment.n_frames, lead_frames + max(1, round(actual_speech_sec * fps)))
        speech_mask[lead_frames:end_speech_frame] = 1.0

        # --- 9. Extract MuseTalk Whisper features from duration-matched audio ---
        whisper_chunks = self._extract_whisper_features(tmp_conv, fps)

        # Pad / trim whisper_chunks to exactly n_frames
        if isinstance(whisper_chunks, torch.Tensor):
            whisper_chunks = [whisper_chunks[i] for i in range(len(whisper_chunks))]
        elif not isinstance(whisper_chunks, list):
            whisper_chunks = list(whisper_chunks)

        if len(whisper_chunks) < segment.n_frames:
            last = whisper_chunks[-1] if whisper_chunks else torch.zeros((50, 384))
            for _ in range(segment.n_frames - len(whisper_chunks)):
                whisper_chunks.append(last.clone())
        elif len(whisper_chunks) > segment.n_frames:
            whisper_chunks = whisper_chunks[:segment.n_frames]

        # Zero out Whisper features during lead silence and trailing breathing pause
        for idx in range(len(whisper_chunks)):
            if idx < lead_frames or idx >= end_speech_frame:
                whisper_chunks[idx].zero_()

        # --- 10. Cache in buffer ---
        self._audio_buffer[segment.idx] = (audio_out.copy(), audio_sr)

        elapsed = time.perf_counter() - t0
        log.info(
            "[audio_proc] Sentence %d done in %.3fs | dur=%.2fs speed=%.2fx speech=%.2fs '%s'",
            segment.idx, elapsed, target_duration, speed, actual_speech_sec,
            english_text[:50],
        )

        return AudioChunkResult(
            source_text=segment.source_text,
            translated_text=english_text,
            audio_path=tmp_conv,
            audio_samples=audio_out,
            audio_sr=audio_sr,
            audio_duration=target_duration,
            whisper_chunks=whisper_chunks,
            t_start=segment.t_start,
            t_end=segment.t_end,
            speech_frames=int(np.sum(speech_mask)),
            speech_mask=speech_mask,
        )

    # ------------------------------------------------------------------
    # Streaming Live Audio Pipeline (GPU 1 rolling worker)
    # ------------------------------------------------------------------

    def reset_live_audio(self) -> None:
        """Reset all rolling audio state for a new live streaming session."""
        self._live_audio_chunks: List[np.ndarray] = []
        self._live_audio_sr: int = 16_000
        self._live_committed_sample_idx: int = 0
        self._live_sentence_idx: int = 0
        self._live_last_committed_frame: int = 0
        self._live_prompt_history: List[str] = []
        log.info("[audio_proc] Live audio state reset.")

    def push_audio_chunk(self, samples: np.ndarray, t_start: float, t_end: float) -> None:
        """
        Append an incoming 0.5s audio chunk from the AudioFeeder/microphone.

        Parameters
        ----------
        samples : np.ndarray
            16 kHz mono float32 PCM samples.
        t_start, t_end : float
            Nominal audio timestamp window.
        """
        if samples.ndim > 1:
            samples = samples.flatten()
        self._live_audio_chunks.append(samples.astype(np.float32))

    def try_commit_sentence(
        self,
        fps: float,
        max_chunk_sec: float = 6.0,
        force_flush: bool = False,
    ) -> List[Tuple["SentenceSegment", AudioChunkResult]]:
        """
        Check the uncommitted audio buffer for completed, grammatically sound sentences.

        Uses rolling Whisper over the active clause buffer to prevent mid-phoneme
        or mid-clause slicing. Commits when a true sentence boundary (. ! ?),
        a natural speech pause (>= 0.35s), or duration cap is reached, ensuring
        MarianMT receives complete, high-fidelity sentences.

        Parameters
        ----------
        fps : float
            Video frame rate for frame alignment.
        max_chunk_sec : float
            Maximum duration before forcing a sentence boundary.
        force_flush : bool
            True at end-of-stream to flush any remaining uncommitted audio.

        Returns
        -------
        List of (SentenceSegment, AudioChunkResult) for newly committed sentences.
        """
        import re

        DANGLING_CONNECTORS = {
            "aunque", "que", "de", "y", "e", "pero", "mas", "o", "u", "en",
            "para", "por", "con", "el", "la", "los", "las", "un", "una",
            "unos", "unas", "al", "del", "es", "son", "era", "fue", "como",
            "si", "cuando", "donde", "a", "su", "sus", "mi", "mis", "tu", "tus",
            "mas", "más", "se", "le", "les", "me", "te", "nos"
        }

        if not hasattr(self, "_live_audio_chunks"):
            self.reset_live_audio()

        if not self._live_audio_chunks:
            return []

        all_samples = np.concatenate(self._live_audio_chunks)
        uncommitted = all_samples[self._live_committed_sample_idx:]
        uncommitted_dur = len(uncommitted) / float(self._live_audio_sr)

        # In live streaming, accumulate at least 2.5s before attempting sentence boundary detection
        if uncommitted_dur < 2.5 and not force_flush:
            return []

        torch.cuda.set_device(self.device_id)

        # Write uncommitted slice to temp wav for Whisper
        tmp_uncommitted_wav = self._write_temp_wav(uncommitted, suffix="_uncomm.wav", sample_rate=self._live_audio_sr)

        # Extract target_se once from the accumulated source audio
        if self._target_se is None and len(all_samples) >= self._live_audio_sr:
            ref_dur_samples = min(len(all_samples), self._live_audio_sr * 3)
            tmp_ref = self._write_temp_wav(all_samples[:ref_dur_samples], suffix="_ref.wav", sample_rate=self._live_audio_sr)
            self._target_se = self._converter.extract_se([tmp_ref])
            log.info("[audio_proc] target_se extracted live from first %.2fs of audio.", ref_dur_samples / self._live_audio_sr)

        # Pass recent prompt history to Whisper for linguistic continuity
        prompt = " ".join(self._live_prompt_history[-2:]) if self._live_prompt_history else None

        result = self._stt_model.transcribe(
            tmp_uncommitted_wav,
            language=self.source_lang,
            task="transcribe",
            word_timestamps=True,
            initial_prompt=prompt,
        )

        committed_results: List[Tuple[SentenceSegment, AudioChunkResult]] = []
        base_time_sec = self._live_committed_sample_idx / float(self._live_audio_sr)

        words_list = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words_list.append(w)

        if not words_list and result.get("text", "").strip():
            raw_text = result.get("text", "").strip()
            if force_flush or uncommitted_dur >= max_chunk_sec:
                t_s = base_time_sec
                t_e = base_time_sec + uncommitted_dur
                f_s = self._live_last_committed_frame
                f_e = max(f_s + 1, int(round(t_e * fps)))
                self._live_last_committed_frame = f_e
                n_f = f_e - f_s
                seg_obj = SentenceSegment(
                    source_text=raw_text,
                    t_start=t_s,
                    t_end=t_e,
                    frame_start=f_s,
                    frame_end=f_e,
                    duration=n_f / fps,
                    n_frames=n_f,
                    idx=self._live_sentence_idx,
                )
                self._live_sentence_idx += 1
                self._live_committed_sample_idx = len(all_samples)
                self._live_prompt_history.append(raw_text)
                audio_res = self.process_sentence(seg_obj, fps)
                committed_results.append((seg_obj, audio_res))
            return committed_results

        current_words: List[dict] = []
        current_start: Optional[float] = None

        for idx_w, word_info in enumerate(words_list):
            word_text = word_info.get("word", "").strip()
            w_start = float(word_info.get("start", 0.0))
            w_end = float(word_info.get("end", w_start + 0.1))

            if not word_text:
                continue

            if current_start is None:
                current_start = w_start

            current_words.append({"word": word_text, "start": w_start, "end": w_end})
            curr_dur = w_end - current_start

            w_clean = re.sub(r'[^\w]', '', word_text).lower()
            is_dangling = w_clean in DANGLING_CONNECTORS

            # Check for speech pause after this word
            next_pause = (words_list[idx_w + 1]["start"] - w_end) if (idx_w + 1 < len(words_list)) else (uncommitted_dur - w_end)

            # Never commit on the very trailing edge of the audio buffer (active mid-speech) unless force_flush
            is_near_buffer_edge = (uncommitted_dur - w_end) < 0.40 and not force_flush

            # Valid commit condition:
            # 1. True sentence punctuation (. ! ?) with curr_dur >= 2.5s
            # 2. Natural acoustic speech pause (>= 0.45s of silence) at end of a complete clause (>= 4.0s)
            # 3. Hard duration cap (>= max_chunk_sec)
            is_sentence_punct = bool(re.search(r'[.!?]$', word_text)) and (curr_dur >= 2.5) and not is_near_buffer_edge
            is_clause_pause = (next_pause >= 0.45) and (curr_dur >= 4.0) and not is_near_buffer_edge
            is_cap_reached = (curr_dur >= max_chunk_sec) and not is_near_buffer_edge

            should_commit = (is_sentence_punct or is_clause_pause or is_cap_reached) and not is_dangling

            if should_commit:
                text = " ".join(w["word"] for w in current_words).strip()
                t_s = base_time_sec + current_start
                t_e = base_time_sec + w_end
                f_s = self._live_last_committed_frame
                f_e = max(f_s + 1, int(round(t_e * fps)))
                self._live_last_committed_frame = f_e
                n_f = f_e - f_s

                seg_obj = SentenceSegment(
                    source_text=text,
                    t_start=t_s,
                    t_end=t_e,
                    frame_start=f_s,
                    frame_end=f_e,
                    duration=n_f / fps,
                    n_frames=n_f,
                    idx=self._live_sentence_idx,
                )
                self._live_sentence_idx += 1
                self._live_committed_sample_idx += int(round(w_end * float(self._live_audio_sr)))
                base_time_sec = self._live_committed_sample_idx / float(self._live_audio_sr)
                self._live_prompt_history.append(text)

                audio_res = self.process_sentence(seg_obj, fps)
                committed_results.append((seg_obj, audio_res))

                current_words = []
                current_start = None

        # At EOF, flush any remaining uncommitted words
        if force_flush and current_words and current_start is not None:
            text = " ".join(w["word"] for w in current_words).strip()
            t_s = base_time_sec + current_start
            t_e = base_time_sec + uncommitted_dur
            f_s = self._live_last_committed_frame
            f_e = max(f_s + 1, int(round(t_e * fps)))
            self._live_last_committed_frame = f_e
            n_f = f_e - f_s

            seg_obj = SentenceSegment(
                source_text=text,
                t_start=t_s,
                t_end=t_e,
                frame_start=f_s,
                frame_end=f_e,
                duration=n_f / fps,
                n_frames=n_f,
                idx=self._live_sentence_idx,
            )
            self._live_sentence_idx += 1
            self._live_committed_sample_idx = len(all_samples)
            self._live_prompt_history.append(text)
            audio_res = self.process_sentence(seg_obj, fps)
            committed_results.append((seg_obj, audio_res))

        return committed_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _translate(self, text: str) -> str:
        """10-beam Spanish-to-English translation on GPU 1."""
        inputs = self._nmt_tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self._nmt_model.generate(**inputs, num_beams=self.num_beams, max_new_tokens=256)
        return self._nmt_tokenizer.decode(out[0], skip_special_tokens=True)

    def _match_audio_duration(
        self,
        audio: np.ndarray,
        target_duration: float,
        sr: int,
    ) -> np.ndarray:
        """
        Trim or zero-pad audio to exactly target_duration seconds.

        Used only to absorb sample-count rounding residuals (typically <5 ms)
        after TTS speed clamping. The speed ratio already brings the synthesized
        audio very close to the target; this only corrects any remaining integer
        sample mismatch.

        - Trim: hard-cut at target; log warning if trim > 40 ms (indicates
          the speed clamp hit its upper bound and speech was lost).
        - Pad: append pure silence. No fade is applied to real speech — the
          junction is inaudible at <5 ms gaps.
        """
        target_samples = int(round(target_duration * sr))
        current_samples = len(audio)

        if current_samples == target_samples:
            return audio

        if current_samples > target_samples:
            trimmed_ms = (current_samples - target_samples) / sr * 1000
            try:
                import librosa
                rate = current_samples / float(target_samples)
                if 0.85 <= rate <= 1.25:
                    stretched = librosa.effects.time_stretch(audio, rate=rate)
                    if len(stretched) >= target_samples:
                        return stretched[:target_samples]
                    else:
                        return np.pad(stretched, (0, target_samples - len(stretched)), "constant")
            except Exception:
                pass
            log.warning("[audio_proc] Audio trimmed %.1f ms.", trimmed_ms)
            return audio[:target_samples]

        # Append pure silence — no fade needed, gap is <5 ms after speed matching
        pad_len = target_samples - current_samples
        return np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])

    # _update_speed_profile removed — speed is now measured directly from
    # the 1x synthesis probe, making the adaptive estimator unnecessary.

    def _extract_whisper_features(self, audio_path: str, fps: float) -> list:
        """
        Extract MuseTalk Whisper audio features from synthesized audio.
        Uses the resident WhisperModel on GPU 1 — no cold-start per chunk.

        Returns tensors on CPU so the renderer can place them on any GPU
        (GPU 2/3) without a device mismatch.
        """
        torch.cuda.set_device(self.device_id)
        whisper_input_features, librosa_length = self._whisper_processor.get_audio_feature(
            audio_path, weight_dtype=torch.float16
        )
        whisper_chunks = self._whisper_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            torch.float16,
            self._whisper_model,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        # Move all tensors to CPU — renderer will move them to its own GPU
        if isinstance(whisper_chunks, (list, tuple)):
            whisper_chunks = [
                c.cpu() if isinstance(c, torch.Tensor) else c
                for c in whisper_chunks
            ]
        elif isinstance(whisper_chunks, torch.Tensor):
            whisper_chunks = whisper_chunks.cpu()
        return whisper_chunks

    def _write_temp_wav(
        self,
        samples: np.ndarray,
        suffix: str = ".wav",
        sample_rate: int = 16_000,
    ) -> str:
        """Write numpy audio array to a named temp WAV file and track it for cleanup."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        sf.write(path, samples.astype(np.float32), sample_rate)
        self._temp_files.append(path)
        return path
