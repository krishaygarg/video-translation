import argparse
import subprocess
import os
import torch
import whisper
import pandas as pd
from transformers import pipeline as hf_pipeline, MarianMTModel, MarianTokenizer

# Tone Analysis imports (we copy the necessary parts instead of importing to avoid relative path hell)
import soundfile as sf
from scipy.signal import resample_poly
import numpy as np
from math import gcd
import librosa

# OpenVoice imports
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

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

def main():
    parser = argparse.ArgumentParser(description="Audio Translation & Cloning Pipeline")
    parser.add_argument("--input", required=True, help="Input video or audio file")
    parser.add_argument("--output", default="final_output.wav", help="Output Spanish audio file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Audio Extraction
    print("\n--- 1. Extracting Audio ---")
    audio_path = "tmp_audio.wav"
    subprocess.run([
        'ffmpeg', '-i', args.input, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y'
    ], check=True, stderr=subprocess.DEVNULL)
    print("Audio extracted successfully.")

    # 2. Transcription
    print("\n--- 2. Transcribing Audio ---")
    whisper_model = whisper.load_model("base").to(device)
    transcription_result = whisper_model.transcribe(audio_path)
    english_text = transcription_result["text"].strip()
    print(f"Transcribed Text: {english_text}")

    # 3. Tone Analysis
    print("\n--- 3. Analyzing Tone ---")
    print("Loading emotion model...")
    emotion_classifier = hf_pipeline(
        "audio-classification",
        model="superb/hubert-large-superb-er",
        top_k=None,
        device=0 if device == "cuda" else -1
    )
    
    LABEL_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
    audio_data = load_audio(audio_path, target_sr=16000)
    raw_scores = emotion_classifier({"array": audio_data, "sampling_rate": 16000})
    scores = {LABEL_MAP.get(s["label"], s["label"]): float(s["score"]) for s in raw_scores}
    dominant_emotion = max(scores, key=scores.get)
    print(f"Detected Emotion: {dominant_emotion} (Scores: {scores})")

    # 4. Translation
    print("\n--- 4. Translating to Spanish ---")
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name).to(device)
    
    inputs = tokenizer([english_text], return_tensors="pt", padding=True).to(device)
    translated = translation_model.generate(**inputs, max_length=256)
    spanish_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Spanish Text: {spanish_text}")

    # 5. OpenVoice Synthesize & Clone
    print("\n--- 5. Generating Voice with OpenVoice ---")
    ckpt_base = 'openvoice/checkpoints/base_speakers/EN'
    ckpt_converter = 'openvoice/checkpoints/converter'
    
    print("Loading Base Speaker TTS...")
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    print("Loading Tone Color Converter...")
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    
    emotion_map = {
        "neutral": "default",
        "happy": "cheerful",
        "angry": "angry",
        "sad": "sad"
    }
    speaker_style = emotion_map.get(dominant_emotion, "default")
    
    if speaker_style == "default":
        source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
    else:
        source_se = torch.load(f'{ckpt_base}/en_style_se.pth').to(device)
        
    print(f"Extracting target speaker Tone Color from {audio_path}...")
    target_se = tone_color_converter.extract_se([audio_path])
    
    src_path = 'tmp_base_tts.wav'
    print(f"Synthesizing base TTS with style '{speaker_style}' to calculate duration...")
    # Generate base audio in memory at speed 1.0 to get its duration
    base_audio = base_speaker_tts.tts(spanish_text, output_path=None, speaker=speaker_style, language='English', speed=1.0)
    generated_duration = len(base_audio) / base_speaker_tts.hps.data.sampling_rate
    
    # Calculate required speed to match original duration
    original_duration = librosa.get_duration(path=audio_path)
    target_speed = generated_duration / original_duration
    print(f"Original duration: {original_duration:.2f}s, Generated duration at 1.0x: {generated_duration:.2f}s")
    print(f"Adjusting TTS speed to {target_speed:.2f}x to match original audio length.")
    
    # Generate base audio again with the adjusted speed
    base_speaker_tts.tts(spanish_text, src_path, speaker=speaker_style, language='English', speed=target_speed)
    
    print("Converting Tone Color...")
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=args.output,
        message="@MyShell"
    )
    
    # Save the translation and transcription to a text file
    output_base, _ = os.path.splitext(args.output)
    text_output_path = f"{output_base}.txt"
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(f"--- English Transcription ---\n{english_text}\n\n")
        f.write(f"--- Spanish Translation ---\n{spanish_text}\n")
    print(f"Saved text to {text_output_path}")
    
    print(f"\nPipeline finished successfully! Output saved to {args.output}")
    
    # Cleanup tmp files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(src_path):
        os.remove(src_path)

if __name__ == "__main__":
    main()
