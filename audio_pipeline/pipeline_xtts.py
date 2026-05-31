import argparse
import subprocess
import os
import torch

# PyTorch 2.6 backwards compatibility for Coqui TTS
original_torch_load = torch.load
def legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = legacy_load
import whisper
import librosa
import numpy as np
import soundfile as sf
from transformers import MarianMTModel, MarianTokenizer

try:
    from TTS.api import TTS
except ImportError:
    print("Warning: TTS package not found. Please wait for installation to finish.")
    TTS = None

def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 Zero-Shot Voice Cloning Pipeline")
    parser.add_argument("--input", required=True, help="Input video or audio file")
    parser.add_argument("--output", default="xtts_output.wav", help="Output translated audio file")
    args = parser.parse_args()

    # Force CPU to avoid Apple Silicon MPS backend crashes with XTTS
    device = "cpu"
    print(f"Using device: {device}")

    audio_path = "tmp_xtts_audio.wav"
    
    # 1. Extract Audio
    print("\n--- 1. Extracting Audio ---")
    subprocess.run([
        'ffmpeg', '-i', args.input, '-vn', '-acodec', 'pcm_s16le', '-ar', '24000', '-ac', '1', audio_path, '-y'
    ], check=True, stderr=subprocess.DEVNULL)
    # Note: XTTS-v2 generates 24kHz audio, so we'll use 24kHz throughout.
    
    # Load the full original audio for segment extraction
    y_full, sr_full = librosa.load(audio_path, sr=24000)

    # 2. Transcribe & Segment Audio (using Whisper on 16kHz via internal resample)
    print("\n--- 2. Transcribing Audio (Whisper) ---")
    whisper_device = "cpu" if device == "mps" else device
    whisper_model = whisper.load_model("base").to(whisper_device)
    transcription_result = whisper_model.transcribe(audio_path, language="es")
    segments = transcription_result["segments"]
    print(f"Found {len(segments)} segments.")

    # 3. Load Translation Model (Spanish -> English)
    print("\n--- 3. Loading Translation Model ---")
    mt_model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    translation_model = MarianMTModel.from_pretrained(mt_model_name).to(device)

    # 4. Load XTTS-v2
    print("\n--- 4. Loading XTTS-v2 Model ---")
    if TTS is None:
        print("Error: TTS package is missing. Aborting.")
        return
    
    # XTTS needs a GPU for reasonable speed. If MPS is available, try it, but TTS library might fallback to CPU.
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sr_out = 24000

    # Generate segments
    print("\n--- 5. Processing Segments ---")
    final_audio_segments = []
    current_time = 0.0
    
    full_spanish_text = []
    full_english_text = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        orig_text = seg["text"].strip()
        segment_duration = end - start

        # Add silence if there's a gap before this segment
        if start > current_time:
            silence_duration = start - current_time
            num_silence_samples = int(silence_duration * sr_out)
            final_audio_segments.append(np.zeros(num_silence_samples, dtype=np.float32))
            current_time = start

        # Translate
        inputs = tokenizer([orig_text], return_tensors="pt", padding=True).to(device)
        translated = translation_model.generate(**inputs, max_length=256)
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        print(f"Segment {i+1}: [{start:.2f}s - {end:.2f}s]")
        print(f"  Orig: {orig_text}")
        print(f"  Transl: {english_text}")

        full_spanish_text.append(orig_text)
        full_english_text.append(english_text)

        # Extract specific segment audio to use as speaker prompt!
        start_sample = int(start * sr_full)
        end_sample = int(end * sr_full)
        segment_audio = y_full[start_sample:end_sample]
        
        # XTTS requires at least 3 seconds of audio for a good prompt. 
        # If segment is too short, we expand it slightly or just pass what we have.
        prompt_path = f"tmp_prompt_{i}.wav"
        sf.write(prompt_path, segment_audio, sr_full)

        # Synthesize with XTTS
        try:
            # XTTS tts() returns a list of floats, convert to numpy array
            generated_wav = tts.tts(text=english_text, speaker_wav=prompt_path, language="en")
            generated_wav = np.array(generated_wav, dtype=np.float32)
            generated_duration = len(generated_wav) / sr_out
            
            # Time stretch to match exact duration
            target_speed = generated_duration / segment_duration
            target_speed = max(0.5, min(target_speed, 2.0))
            
            # Using ffmpeg atempo to time stretch (bypassing librosa/numpy hell)
            tmp_in = f"tmp_in_{i}.wav"
            tmp_out = f"tmp_out_{i}.wav"
            sf.write(tmp_in, generated_wav, sr_out)
            
            subprocess.run([
                'ffmpeg', '-i', tmp_in, '-filter:a', f'atempo={target_speed}', tmp_out, '-y'
            ], check=True, stderr=subprocess.DEVNULL)
            
            generated_wav_adjusted, _ = sf.read(tmp_out)
            
            # Clean up stretch temp files
            if os.path.exists(tmp_in): os.remove(tmp_in)
            if os.path.exists(tmp_out): os.remove(tmp_out)
            
            final_audio_segments.append(generated_wav_adjusted)
            current_time += len(generated_wav_adjusted) / sr_out
        except Exception as e:
            print(f"  Warning: XTTS failed for segment ({e}). Adding silence instead.")
            final_audio_segments.append(np.zeros(int(segment_duration * sr_out), dtype=np.float32))
            current_time += segment_duration
            
        if os.path.exists(prompt_path):
            os.remove(prompt_path)

    # Concatenate all audio segments
    print("\n--- 6. Concatenating Audio ---")
    if len(final_audio_segments) == 0:
        print("Error: No segments processed.")
        return

    concatenated_audio = np.concatenate(final_audio_segments)
    sf.write(args.output, concatenated_audio, sr_out)

    # Save the text to a file
    output_base, _ = os.path.splitext(args.output)
    text_output_path = f"{output_base}.txt"
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write("--- Full Spanish Transcription ---\n")
        f.write(" ".join(full_spanish_text) + "\n\n")
        f.write("--- Full English Translation ---\n")
        f.write(" ".join(full_english_text) + "\n")
    print(f"Saved text to {text_output_path}")

    print(f"\nPipeline finished successfully! Output saved to {args.output}")

    # Cleanup tmp files
    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    main()
