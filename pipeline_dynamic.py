import argparse
import subprocess
import os
import torch
import whisper
import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline as hf_pipeline, MarianMTModel, MarianTokenizer

# OpenVoice imports
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

def main():
    parser = argparse.ArgumentParser(description="Dynamic Emotion & Segment-Paced Pipeline")
    parser.add_argument("--input", required=True, help="Input video or audio file")
    parser.add_argument("--output", default="dynamic_output.wav", help="Output translated audio file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    audio_path = "tmp_dynamic_audio.wav"
    
    # 1. Extract Audio
    print("\n--- 1. Extracting Audio ---")
    subprocess.run([
        'ffmpeg', '-i', args.input, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y'
    ], check=True, stderr=subprocess.DEVNULL)
    print("Audio extracted successfully.")

    # 2. Transcribe & Segment Audio
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

    # 4. Load Emotion Classifier
    print("\n--- 4. Loading Emotion Classifier ---")
    emotion_classifier = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False, device=device)

    # 5. Load OpenVoice Base Speaker & Tone Color Converter
    print("\n--- 5. Loading OpenVoice Models ---")
    ckpt_base = 'openvoice/checkpoints/base_speakers/EN'
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter('openvoice/checkpoints/converter/config.json', device=device)
    tone_color_converter.load_ckpt('openvoice/checkpoints/converter/checkpoint.pth')
    
    source_se = torch.load(f'{ckpt_base}/en_style_se.pth').to(device)
    target_se = tone_color_converter.extract_se([audio_path])

    # Emotion mapping to OpenVoice styles
    emotion_to_style = {
        'anger': 'angry',
        'disgust': 'angry',
        'fear': 'terrified',
        'joy': 'cheerful',
        'sadness': 'sad',
        'surprise': 'cheerful',
        'neutral': 'default'
    }

    # Generate segments
    print("\n--- 6. Processing Segments ---")
    final_audio_segments = []
    current_time = 0.0
    sr = base_speaker_tts.hps.data.sampling_rate
    
    full_spanish_text = []
    full_english_text = []
    segment_details = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        orig_text = seg["text"].strip()
        segment_duration = end - start

        # Add silence if there's a gap before this segment
        if start > current_time:
            silence_duration = start - current_time
            num_silence_samples = int(silence_duration * sr)
            final_audio_segments.append(np.zeros(num_silence_samples, dtype=np.float32))
            current_time = start

        # Translate
        inputs = tokenizer([orig_text], return_tensors="pt", padding=True).to(device)
        translated = translation_model.generate(**inputs, max_length=256)
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        # Analyze Emotion
        emotion_result = emotion_classifier(english_text)[0]
        detected_emotion = emotion_result['label']
        speaker_style = emotion_to_style.get(detected_emotion, 'default')

        print(f"Segment {i+1}: [{start:.2f}s - {end:.2f}s]")
        print(f"  Orig: {orig_text}")
        print(f"  Transl: {english_text}")
        print(f"  Emotion: {detected_emotion} -> Style: {speaker_style}")
        
        full_spanish_text.append(orig_text)
        full_english_text.append(english_text)
        segment_details.append(f"Segment {i+1} [{start:.2f}s - {end:.2f}s] | Emotion: {detected_emotion} (Style: {speaker_style})\nSpanish: {orig_text}\nEnglish: {english_text}\n")

        # Synthesize base audio to get default duration
        try:
            base_audio = base_speaker_tts.tts(english_text, output_path=None, speaker=speaker_style, language='English', speed=1.0)
            generated_duration = len(base_audio) / sr
            
            # Calculate required speed
            target_speed = generated_duration / segment_duration
            # Clamp speed to avoid extreme distortions
            target_speed = max(0.5, min(target_speed, 2.0))
            
            # Regenerate with target speed
            base_audio_adjusted = base_speaker_tts.tts(english_text, output_path=None, speaker=speaker_style, language='English', speed=target_speed)
            
            final_audio_segments.append(base_audio_adjusted)
            current_time += len(base_audio_adjusted) / sr
        except Exception as e:
            print(f"  Warning: TTS failed for segment ({e}). Adding silence instead.")
            final_audio_segments.append(np.zeros(int(segment_duration * sr), dtype=np.float32))
            current_time += segment_duration

    # Concatenate all audio segments
    print("\n--- 7. Concatenating and Converting Tone Color ---")
    if len(final_audio_segments) == 0:
        print("Error: No segments processed.")
        return

    concatenated_base_audio = np.concatenate(final_audio_segments)
    tmp_concat_path = "tmp_concat_base.wav"
    sf.write(tmp_concat_path, concatenated_base_audio, sr)

    tone_color_converter.convert(
        audio_src_path=tmp_concat_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=args.output,
        message="@MyShell"
    )

    # Save the text to a file
    output_base, _ = os.path.splitext(args.output)
    text_output_path = f"{output_base}.txt"
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write("--- Segment-by-Segment Emotion Tracking ---\n")
        f.write("\n".join(segment_details) + "\n")
        f.write("--- Full Spanish Transcription ---\n")
        f.write(" ".join(full_spanish_text) + "\n\n")
        f.write("--- Full English Translation ---\n")
        f.write(" ".join(full_english_text) + "\n")
    print(f"Saved text to {text_output_path}")

    print(f"\nPipeline finished successfully! Output saved to {args.output}")

    # Cleanup tmp files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(tmp_concat_path):
        os.remove(tmp_concat_path)

if __name__ == "__main__":
    main()
