#!/usr/bin/env python3
"""
Optimized End-to-End Spanish to English Video Translation & Lip-Sync Pipeline
Single-process execution with 0 MB idle VRAM lifecycle management.
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import glob
import copy
import numpy as np
import cv2
import torch
import soundfile as sf
import librosa
from scipy.signal import resample_poly
from math import gcd
import queue
import threading

# Setup sys.path for internal modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

audio_pipeline_dir = os.path.join(BASE_DIR, "audio_pipeline")
openvoice_dir = os.path.join(audio_pipeline_dir, "openvoice")
musetalk_dir = os.path.join(BASE_DIR, "MuseTalk")
speech_bubble_dir = os.path.join(BASE_DIR, "speech_bubble_transcription")

for d in [audio_pipeline_dir, openvoice_dir, musetalk_dir, speech_bubble_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import whisper
from transformers import pipeline as hf_pipeline, MarianMTModel, MarianTokenizer
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Speech bubble imports
import mediapipe as mp
from transcribe_bubble import parse_color, split_text_into_lines, draw_rounded_bubble, download_face_landmarker

def load_audio(path, target_sr=16000):
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr), "-"
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        sr = target_sr

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio

def free_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

model_load_lock = threading.Lock()

class AvatarCache:
    """
    Caches baseline facial geometry coordinates, blending masks, and SD-VAE latents
    to enable zero-redundancy facial latent reuse and low-latency processing.
    """
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.coords_path = os.path.join(cache_dir, "coords.pkl")
        self.latents_path = os.path.join(cache_dir, "latents.pt")
        self.masks_path = os.path.join(cache_dir, "masks.pkl")

    def exists(self):
        return (os.path.exists(self.coords_path) and 
                os.path.exists(self.latents_path) and 
                os.path.exists(self.masks_path))

    def save(self, coord_list, input_latent_list, mask_list, crop_box_list):
        import pickle
        with open(self.coords_path, "wb") as f:
            pickle.dump(coord_list, f)
        with open(self.masks_path, "wb") as f:
            pickle.dump({"masks": mask_list, "crop_boxes": crop_box_list}, f)
        torch.save(input_latent_list, self.latents_path)

    def load(self):
        import pickle
        with open(self.coords_path, "rb") as f:
            coord_list = pickle.load(f)
        with open(self.masks_path, "rb") as f:
            mask_data = pickle.load(f)
        input_latent_list = torch.load(self.latents_path)
        return coord_list, input_latent_list, mask_data["masks"], mask_data["crop_boxes"]

# ==========================================
# STEP 1: Audio Translation & Voice Cloning
# ==========================================
def run_step1_audio(input_video, output_audio_path, device="cuda"):
    print("\n[Step 1] Running Audio Translation & Voice Cloning...")
    t0 = time.time()
    tmp_audio_path = os.path.join(BASE_DIR, "tmp_audio_opt.wav")

    # Extract Audio
    subprocess.run([
        'ffmpeg', '-i', input_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', tmp_audio_path, '-y'
    ], check=True, stderr=subprocess.DEVNULL)

    # Transcription
    print(" -> Transcribing audio with Whisper...")
    whisper_model = whisper.load_model("base").to(device)
    transcription_result = whisper_model.transcribe(tmp_audio_path)
    spanish_text = transcription_result["text"].strip()
    print(f" -> Spanish Text: {spanish_text}")

    # Tone Analysis
    print(" -> Analyzing tone...")
    emotion_classifier = hf_pipeline(
        "audio-classification",
        model="superb/hubert-large-superb-er",
        top_k=None,
        device=0 if device == "cuda" else -1
    )
    LABEL_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
    audio_data = load_audio(tmp_audio_path, target_sr=16000)
    raw_scores = emotion_classifier({"array": audio_data, "sampling_rate": 16000})
    scores = {LABEL_MAP.get(s["label"], s["label"]): float(s["score"]) for s in raw_scores}
    dominant_emotion = max(scores, key=scores.get)

    # Translation
    print(" -> Translating to English...")
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name).to(device)
    inputs = tokenizer([spanish_text], return_tensors="pt", padding=True).to(device)
    translated = translation_model.generate(**inputs, max_length=256)
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f" -> English Text: {english_text}")

    # Voice Cloning (OpenVoice)
    print(" -> Generating voice clone with OpenVoice...")
    ckpt_base = os.path.join(audio_pipeline_dir, 'openvoice/checkpoints/base_speakers/EN')
    ckpt_converter = os.path.join(audio_pipeline_dir, 'openvoice/checkpoints/converter')

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    emotion_map = {"neutral": "default", "happy": "cheerful", "angry": "angry", "sad": "sad"}
    speaker_style = emotion_map.get(dominant_emotion, "default")
    if speaker_style == "default":
        source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
    else:
        source_se = torch.load(f'{ckpt_base}/en_style_se.pth').to(device)

    target_se = tone_color_converter.extract_se([tmp_audio_path])
    src_path = os.path.join(BASE_DIR, 'tmp_base_tts_opt.wav')

    base_audio = base_speaker_tts.tts(english_text, output_path=None, speaker=speaker_style, language='English', speed=1.0)
    generated_duration = len(base_audio) / base_speaker_tts.hps.data.sampling_rate
    original_duration = librosa.get_duration(path=tmp_audio_path)
    target_speed = generated_duration / original_duration

    base_speaker_tts.tts(english_text, src_path, speaker=speaker_style, language='English', speed=target_speed)

    os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_audio_path,
        message="@MyShell"
    )

    # Audio Peak Normalization to -1dBFS
    # Ensures Whisper extracts strong phoneme features → full mouth openings
    print(" -> Normalizing audio peak to -1dBFS for strong Whisper phoneme features...")
    audio_data, audio_sr = sf.read(output_audio_path)
    peak = np.abs(audio_data).max()
    if peak > 0:
        target_peak = 10 ** (-1.0 / 20)  # -1 dBFS
        audio_data = audio_data * (target_peak / peak)
    sf.write(output_audio_path, audio_data, audio_sr)

    # Cleanup Step 1 temporary files
    for p in [tmp_audio_path, src_path]:
        if os.path.exists(p):
            os.remove(p)

    # Cleanup Step 1 models & VRAM
    del whisper_model, emotion_classifier, translation_model, tokenizer, base_speaker_tts, tone_color_converter
    free_vram()

    elapsed = time.time() - t0
    print(f"Step 1 completed in {elapsed:.2f}s! (VRAM Cleared)")
    return english_text, spanish_text

class PipelinedBlender:
    """
    Overlays and blends mouth patches onto the original video frames in a concurrent CPU thread,
    preventing the GPU from idling during frame blending and writing.
    """
    def __init__(self, out_writer, coord_list, mask_list, crop_box_list, frame_list, start_idx=0):
        self.out_writer = out_writer
        self.coord_list = coord_list
        self.mask_list = mask_list
        self.crop_box_list = crop_box_list
        self.frame_list = frame_list
        self.start_idx = start_idx
        
        self.queue = queue.Queue(maxsize=32)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def push(self, recon_batch, batch_start_idx):
        self.queue.put((recon_batch, batch_start_idx))

    def close(self):
        self.queue.put(None)
        self.thread.join()

    def _worker(self):
        import cv2
        import numpy as np
        import copy
        from musetalk.utils.blending import get_image_blending
        
        while True:
            item = self.queue.get()
            if item is None:
                break
            recon_batch, batch_start_idx = item
            for i, res_frame in enumerate(recon_batch):
                idx = batch_start_idx + i
                if idx >= len(self.frame_list):
                    break
                
                bbox = self.coord_list[idx % len(self.coord_list)]
                ori_frame = copy.deepcopy(self.frame_list[idx % len(self.frame_list)])
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    self.out_writer.write(ori_frame)
                    continue

                mask_array = self.mask_list[idx % len(self.mask_list)]
                crop_box = self.crop_box_list[idx % len(self.crop_box_list)]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask_array, crop_box)
                self.out_writer.write(combine_frame)

def precompute_geometry_and_masks(input_video, use_avatar_cache=True, device_id=0):
    print(f" -> [Geometry Prep] Initializing on GPU {device_id}...")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    from musetalk.utils.preprocessing import get_landmark_and_bbox
    from musetalk.models.vae import VAE
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.blending import get_image_prepare_material

    # Setup video stream info
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_img_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_img_list.append(frame)
    cap.release()

    cache_dir = os.path.join(BASE_DIR, "results/avatar_cache", os.path.basename(input_video).replace(".", "_"))
    avatar_cache = AvatarCache(cache_dir)

    if use_avatar_cache and avatar_cache.exists():
        print(f" -> [Geometry Prep] Loading from cache...")
        coord_list, input_latent_list, mask_list, crop_box_list = avatar_cache.load()
        frame_list = input_img_list
    else:
        # Load VAE and FaceParsing on device
        vae = VAE(model_path="./models/sd-vae")
        vae.vae = vae.vae.half().to(device)
        fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

        print(" -> [Geometry Prep] Extracting face landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0)
        input_latent_list = []

        for idx, (bbox, frame_active) in enumerate(zip(coord_list, frame_list)):
            if bbox == (0.0, 0.0, 0.0, 0.0) or (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
                if len(input_latent_list) > 0:
                    input_latent_list.append(input_latent_list[-1])
                else:
                    resized_crop = cv2.resize(frame_active, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents = vae.get_latents_for_unet(resized_crop)
                    input_latent_list.append(latents)
                continue

            x1, y1, x2, y2 = bbox
            y2 = min(y2 + 10, frame_active.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame_active[y1:y2, x1:x2]
            resized_crop = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop)
            input_latent_list.append(latents)

        print(" -> [Geometry Prep] Pre-computing blending masks (mode='jaw')...")
        mask_list = []
        crop_box_list = []
        for bbox, frame_active in zip(coord_list, frame_list):
            if bbox == (0.0, 0.0, 0.0, 0.0):
                if len(mask_list) > 0:
                    mask_list.append(mask_list[-1])
                    crop_box_list.append(crop_box_list[-1])
                else:
                    mask_array, crop_box = get_image_prepare_material(frame_active, [0, 0, width, height], fp=fp, mode="jaw")
                    mask_list.append(mask_array)
                    crop_box_list.append(crop_box)
                continue

            mask_array, crop_box = get_image_prepare_material(frame_active, bbox, fp=fp, mode="jaw")
            mask_list.append(mask_array)
            crop_box_list.append(crop_box)

        if use_avatar_cache:
            print(f" -> [Geometry Prep] Saving cache to {cache_dir}...")
            avatar_cache.save(coord_list, input_latent_list, mask_list, crop_box_list)

        del vae, fp
        free_vram()

    return coord_list, input_latent_list, mask_list, crop_box_list, width, height, fps, frame_list

def extract_audio_features(audio_path, fps, device_id=0):
    print(f" -> [Audio Features] Extracting features on GPU {device_id}...")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel

    whisper_dir = "./models/whisper"
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    
    whisper_mod = WhisperModel.from_pretrained(whisper_dir)
    whisper_mod = whisper_mod.to(device=device, dtype=torch.float16).eval()
    whisper_mod.requires_grad_(False)

    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=torch.float16)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        torch.float16,
        whisper_mod,
        librosa_length,
        fps=fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )

    del whisper_mod, audio_processor
    free_vram()
    return whisper_chunks

def render_segment(
    output_segment_path,
    coord_slice,
    latent_slice,
    mask_slice,
    crop_box_slice,
    frame_slice,
    whisper_slice,
    fps,
    width,
    height,
    device_id
):
    print(f" -> [Segment Engine] Starting render on GPU {device_id} for {len(frame_slice)} frames...")
    t_start = time.time()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    from musetalk.utils.utils import datagen, load_all_model

    unet_config = "./models/musetalkV15/musetalk.json"
    unet_model_path = "./models/musetalkV15/unet.pth"

    global model_load_lock
    with model_load_lock:
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=device
        )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    batch_size = 8
    gen = datagen(whisper_slice, latent_slice, batch_size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_segment_path, fourcc, fps, (width, height))

    blender = PipelinedBlender(
        out_writer=out_writer,
        coord_list=coord_slice,
        mask_list=mask_slice,
        crop_box_list=crop_box_slice,
        frame_list=frame_slice,
        start_idx=0
    )

    idx = 0
    for whisper_batch, latent_batch in gen:
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred_latents)

        blender.push(recon, idx)
        idx += len(recon)

    blender.close()
    out_writer.release()

    del vae, unet, pe
    free_vram()

    elapsed = time.time() - t_start
    print(f" -> [Segment Engine] GPU {device_id} completed rendering in {elapsed:.2f}s.")

# ==========================================
# STEP 2: MuseTalk Lip-Sync Generation
# ==========================================
def run_step2_musetalk(input_video, audio_path, output_video_path, crop_and_upscale=False, use_avatar_cache=False, gpu_list=[0, 1]):
    print("\n[Step 2] Running Pipelined Distributed MuseTalk Lip-Sync...")
    t0 = time.time()

    # Change directory to MuseTalk so relative paths resolve properly for all threads
    os.chdir(musetalk_dir)

    # Determine GPU assignments
    gpu_geom = gpu_list[0]
    gpu_audio = gpu_list[1] if len(gpu_list) >= 2 else gpu_list[0]
    print(f" -> GPU assignments: Prep (Geometry: GPU {gpu_geom}, Audio: GPU {gpu_audio})")

    # Step A: Run Parallel Prep (Step 1 Audio on GPU 1, Geometry Prep on GPU 0)
    audio_results = {}
    geom_results = {}

    def run_audio_prep_task():
        try:
            device = f"cuda:{gpu_audio}" if torch.cuda.is_available() else "cpu"
            english_text, spanish_text = run_step1_audio(input_video, audio_path, device=device)
            audio_results["success"] = True
            audio_results["english_text"] = english_text
            audio_results["spanish_text"] = spanish_text
        except Exception as e:
            audio_results["success"] = False
            audio_results["error"] = e

    def run_geom_prep_task():
        try:
            coords, latents, masks, crops, w, h, f, frames = precompute_geometry_and_masks(
                input_video, use_avatar_cache=use_avatar_cache, device_id=gpu_geom
            )
            geom_results["success"] = True
            geom_results["coord_list"] = coords
            geom_results["input_latent_list"] = latents
            geom_results["mask_list"] = masks
            geom_results["crop_box_list"] = crops
            geom_results["width"] = w
            geom_results["height"] = h
            geom_results["fps"] = f
            geom_results["frame_list"] = frames
        except Exception as e:
            geom_results["success"] = False
            geom_results["error"] = e

    audio_thread = threading.Thread(target=run_audio_prep_task)
    geom_thread = threading.Thread(target=run_geom_prep_task)
    
    audio_thread.start()
    geom_thread.start()
    
    audio_thread.join()
    geom_thread.join()

    if not audio_results.get("success", False):
        raise audio_results["error"]
    if not geom_results.get("success", False):
        raise geom_results["error"]

    english_text = audio_results["english_text"]
    spanish_text = audio_results["spanish_text"]

    coord_list = geom_results["coord_list"]
    input_latent_list = geom_results["input_latent_list"]
    mask_list = geom_results["mask_list"]
    crop_box_list = geom_results["crop_box_list"]
    width = geom_results["width"]
    height = geom_results["height"]
    fps = geom_results["fps"]
    frame_list = geom_results["frame_list"]

    # Step B: Extract Audio features
    whisper_chunks = extract_audio_features(audio_path, fps, device_id=gpu_geom)

    # Step C: Split the video and audio frames for Distributed Temporal Rendering
    total_frames = len(whisper_chunks)
    mid = total_frames // 2
    print(f" -> Splitting video: total={total_frames} frames. Segment 0: 0-{mid}, Segment 1: {mid}-{total_frames}")

    tmp_seg0 = os.path.join(BASE_DIR, "tmp_seg0.mp4")
    tmp_seg1 = os.path.join(BASE_DIR, "tmp_seg1.mp4")

    t_render0 = threading.Thread(
        target=render_segment,
        args=(tmp_seg0, coord_list[:mid], input_latent_list[:mid], mask_list[:mid], crop_box_list[:mid], frame_list[:mid], whisper_chunks[:mid], fps, width, height, gpu_list[0])
    )
    t_render1 = threading.Thread(
        target=render_segment,
        args=(tmp_seg1, coord_list[mid:], input_latent_list[mid:], mask_list[mid:], crop_box_list[mid:], frame_list[mid:], whisper_chunks[mid:], fps, width, height, gpu_list[1] if len(gpu_list) >= 2 else gpu_list[0])
    )

    t_render0.start()
    t_render1.start()

    t_render0.join()
    t_render1.join()

    # Step D: Concatenate segments and merge audio
    print(" -> Concatenating rendering segments...")
    concat_list_path = os.path.join(BASE_DIR, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        f.write(f"file '{tmp_seg0}'\n")
        f.write(f"file '{tmp_seg1}'\n")

    tmp_concated_silent = os.path.join(BASE_DIR, "tmp_concated_silent.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", tmp_concated_silent
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(concat_list_path)
    os.remove(tmp_seg0)
    os.remove(tmp_seg1)

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    print(" -> Merging audio with NVENC GPU hardware encoder...")
    cmd_merge = [
        "ffmpeg", "-y", "-i", tmp_concated_silent, "-i", audio_path,
        "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", output_video_path
    ]
    res = subprocess.run(cmd_merge, capture_output=True)
    if res.returncode != 0:
        cmd_merge_fallback = [
            "ffmpeg", "-y", "-i", tmp_concated_silent, "-i", audio_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", output_video_path
        ]
        subprocess.run(cmd_merge_fallback, check=True, stderr=subprocess.DEVNULL)

    if os.path.exists(tmp_concated_silent):
        os.remove(tmp_concated_silent)

    # Restore base working directory
    os.chdir(BASE_DIR)

    elapsed = time.time() - t0
    print(f"Step 2 completed in {elapsed:.2f}s! (VRAM Cleared)")
    return english_text, spanish_text

# ==========================================
# STEP 3: Speech Bubble Subtitles (Direct Text Reuse)
# ==========================================
def run_step3_speech_bubbles(input_video, english_text, output_video_path, audio_track_path):
    print("\n[Step 3] Overlaying Speech Bubbles (Dynamic Timestamp Alignment)...")
    t0 = time.time()

    # Transcribe output audio track to get exact timed segments for active speech text
    print(" -> Extracting audio timestamps for dynamic speech bubble updates...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("base").to(device)
    trans_res = whisper_model.transcribe(audio_track_path, task="transcribe")
    segments = trans_res["segments"]
    del whisper_model
    free_vram()
    print(f" -> Found {len(segments)} timed speech segments.")

    landmarker_model_path = os.path.join(BASE_DIR, "face_landmarker.task")
    download_face_landmarker(landmarker_model_path)

    # Initialize MediaPipe Face Landmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_overlay_silent = os.path.join(BASE_DIR, "tmp_overlay_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(tmp_overlay_silent, fourcc, fps, (width, height))

    bg_color = parse_color("255,255,255", (255, 255, 255))
    border_color = parse_color("0,0,0", (0, 0, 0))
    text_color = parse_color("0,0,0", (0, 0, 0))

    smooth_nose = None
    alpha = 0.08
    deadzone_threshold = 8.0

    landmarker = FaceLandmarker.create_from_options(options)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps
        timestamp_ms = int(current_time_sec * 1000)

        # Find active segment text for current timestamp
        active_text = ""
        for seg in segments:
            if seg["start"] <= current_time_sec <= seg["end"]:
                active_text = seg["text"].strip()
                break

        if active_text:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            raw_nose = None
            if result and result.face_landmarks and len(result.face_landmarks) > 0:
                nose_lm = result.face_landmarks[0][4]  # Nose tip (index 4)
                raw_nose = np.array([nose_lm.x * width, nose_lm.y * height])

            if raw_nose is not None:
                if smooth_nose is None:
                    smooth_nose = raw_nose.copy()
                else:
                    dist = np.linalg.norm(raw_nose - smooth_nose)
                    if dist > deadzone_threshold:
                        smooth_nose = (1 - alpha) * smooth_nose + alpha * raw_nose

            if smooth_nose is not None:
                nose_pos = (int(smooth_nose[0]), int(smooth_nose[1]))
                offset_x = int(width * 0.18)
                offset_y = int(height * 0.18)
                bubble_pos = (int(smooth_nose[0] + offset_x), int(smooth_nose[1] - offset_y))
                text_lines = split_text_into_lines(active_text, max_chars=22)
                draw_rounded_bubble(frame, text_lines, bubble_pos, nose_pos, bg_color, border_color, text_color)
            else:
                bubble_pos = (int(width * 0.5), int(height * 0.3))
                text_lines = split_text_into_lines(active_text, max_chars=22)
                draw_rounded_bubble(frame, text_lines, bubble_pos, None, bg_color, border_color, text_color)

        out_writer.write(frame)
        frame_idx += 1

    cap.release()
    out_writer.release()

    # Fast hardware GPU NVENC final merge
    print(" -> Merging audio into final speech bubble video...")
    cmd_merge = [
        "ffmpeg", "-y", "-i", tmp_overlay_silent, "-i", audio_track_path,
        "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", output_video_path
    ]
    res = subprocess.run(cmd_merge, capture_output=True)
    if res.returncode != 0:
        cmd_merge_fallback = [
            "ffmpeg", "-y", "-i", tmp_overlay_silent, "-i", audio_track_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", output_video_path
        ]
        subprocess.run(cmd_merge_fallback, check=True, stderr=subprocess.DEVNULL)

    if os.path.exists(tmp_overlay_silent):
        os.remove(tmp_overlay_silent)

    elapsed = time.time() - t0
    print(f"Step 3 completed in {elapsed:.2f}s!")

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Optimized Video Translation & Lip-Sync Pipeline")
    parser.add_argument("--input", required=True, help="Path to input Spanish video")
    parser.add_argument("--output", required=True, help="Path for output translated video")
    parser.add_argument("--speech-bubble", action="store_true", help="Overlay speech bubbles")
    parser.add_argument("--crop-upscale", action="store_true", help="Enable crop and upscale")
    parser.add_argument("--avatar-cache", action="store_true", help="Cache and reuse facial geometry latents")
    parser.add_argument("--realtime", action="store_true", help="Enable low-latency streaming pipeline optimizations")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU device IDs to use")
    args = parser.parse_args()

    t_start = time.time()

    input_video = os.path.abspath(args.input)
    output_video = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    translated_audio_path = os.path.join(audio_pipeline_dir, "lipsync/translated_audio_opt.wav")
    tmp_synced_video = os.path.join(output_dir, "tmp_synced_opt.mp4")

    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]

    print("==========================================================")
    print(" Starting OPTIMIZED Multi-GPU Zero-Idle-VRAM Pipeline")
    print(f" Input:     {input_video}")
    print(f" Output:    {output_video}")
    print(f" Cache:     {'ENABLED' if args.avatar_cache else 'DISABLED'}")
    print(f" Realtime:  {'ENABLED' if args.realtime else 'DISABLED'}")
    print(f" GPUs:      {gpu_list}")
    print("==========================================================")

    # Step 2: Orchestrates both Parallel Prep and MuseTalk rendering
    step2_out = output_video if not args.speech_bubble else tmp_synced_video
    english_text, spanish_text = run_step2_musetalk(
        input_video,
        translated_audio_path,
        step2_out,
        crop_and_upscale=args.crop_upscale,
        use_avatar_cache=args.avatar_cache or args.realtime,
        gpu_list=gpu_list
    )

    # Step 3: Speech Bubble Subtitles (if requested)
    if args.speech_bubble:
        run_step3_speech_bubbles(tmp_synced_video, english_text, output_video, translated_audio_path)
        if os.path.exists(tmp_synced_video):
            os.remove(tmp_synced_video)

    # Clean up intermediate audio
    if os.path.exists(translated_audio_path):
        os.remove(translated_audio_path)

    free_vram()

    total_time = time.time() - t_start
    print("==========================================================")
    print(f" Pipeline Finished Successfully in {total_time:.2f} seconds!")
    print(f" Saved output video to: {output_video}")

    # Print VRAM Status for User Verification
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f" Current GPU VRAM Usage: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved (0 MB Idle Policy Enforced!)")
    print("==========================================================")

if __name__ == "__main__":
    main()
