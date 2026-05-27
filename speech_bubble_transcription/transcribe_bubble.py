#!/usr/bin/env python3
"""
Speech Bubble Overlay Generator
Uses OpenAI Whisper to transcribe/translate audio and MediaPipe to track faces
and overlay animated speech bubbles.
"""

import os
import sys
import argparse
import urllib.request
import subprocess
import shutil
import cv2
import numpy as np
import whisper
import torch
import mediapipe as mp

def download_face_landmarker(dest_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if os.path.exists(dest_path):
        print(f"[*] Face landmarker model already exists at: {dest_path}")
        return
    print(f"[*] Downloading face landmarker model from {url} to {dest_path}...")
    try:
        # Simple download progress reporter
        def report(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 100.0 / total_size
                sys.stdout.write(f"\r    Downloading: {percent:.1f}%")
            else:
                sys.stdout.write(f"\r    Downloading: {read_so_far} bytes")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report)
        print("\n[*] Download complete.")
    except Exception as e:
        print(f"\n[!] Failed to download face landmarker: {e}")
        raise e

def parse_color(color_str, default_color):
    if not color_str:
        return default_color
    
    # Hex format
    if color_str.startswith("#"):
        hex_color = color_str.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join([c*2 for c in hex_color])
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (b, g, r)  # OpenCV uses BGR
            except ValueError:
                pass
    
    # RGB comma-separated format
    try:
        parts = [int(p.strip()) for p in color_str.split(",")]
        if len(parts) == 3:
            return (parts[2], parts[1], parts[0])  # Convert RGB to BGR for OpenCV
    except Exception:
        pass
        
    print(f"[!] Warning: Could not parse color '{color_str}'. Using default.")
    return default_color

def split_text_into_lines(text, max_chars):
    """Greedy line splitting logic to cleanly wrap text into up to 2 rows."""
    words = text.split()
    line1 = ""
    line2 = ""
    for word in words:
        if len(line1 + " " + word) <= max_chars:
            line1 = (line1 + " " + word).strip()
        else:
            line2 = (line2 + " " + word).strip()

    lines = [line1] if line1 else []
    if line2:
        lines.append(line2)
    return lines

def draw_rounded_bubble(img, text_lines, bubble_pos, nose_pos,
                        bg_color, border_color, text_color,
                        font_scale_mult=1.0, thickness_mult=1.0):
    if not text_lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Resolution Agnostic Scaling relative to video height
    base_height = img.shape[0]
    font_scale = (base_height / 1080.0) * 0.95 * font_scale_mult
    thickness = max(1, int((base_height / 1080.0) * 2 * thickness_mult))
    padding = max(10, int((base_height / 1080.0) * 25))
    line_spacing = max(5, int((base_height / 1080.0) * 12))
    radius = max(10, int((base_height / 1080.0) * 30))
    tail_width = max(10, int((base_height / 1080.0) * 25))

    max_w = 0
    line_heights = []
    total_text_h = 0

    for line in text_lines:
        size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        max_w = max(max_w, size[0])
        line_heights.append(size[1])
        total_text_h += size[1]

    if len(text_lines) > 1:
        total_text_h += line_spacing * (len(text_lines) - 1)

    rect_w = max_w + (padding * 2)
    rect_h = total_text_h + (padding * 2)

    x, y = bubble_pos
    y_top = y - rect_h

    # Boundary safety checks
    x = max(20, min(x, img.shape[1] - rect_w - 20))
    y_top = max(20, min(y_top, img.shape[0] - rect_h - 20))
    y = y_top + rect_h

    # Draw Background shapes
    cv2.rectangle(img, (x + radius, y_top), (x + rect_w - radius, y), bg_color, -1)
    cv2.rectangle(img, (x, y_top + radius), (x + rect_w, y - radius), bg_color, -1)
    for corner in [(x+radius, y_top+radius), (x+rect_w-radius, y_top+radius),
                   (x+radius, y-radius), (x+rect_w-radius, y-radius)]:
        cv2.circle(img, corner, radius, bg_color, -1)

    # Draw Pointer Tail pointing towards nose
    if nose_pos is not None:
        nose_x, nose_y = nose_pos
        cx = x + rect_w // 2
        cy = y_top + rect_h // 2
        dx = nose_x - cx
        dy = nose_y - cy

        # Determine which edge of the bubble to attach the tail to
        if abs(dy) > abs(dx):
            if dy > 0:  # Nose is below bubble: attach to bottom edge
                p1_x = max(x + radius, min(cx - tail_width // 2, x + rect_w - radius))
                p2_x = max(x + radius, min(cx + tail_width // 2, x + rect_w - radius))
                P1 = (p1_x, y)
                P2 = (p2_x, y)
            else:  # Nose is above bubble: attach to top edge
                p1_x = max(x + radius, min(cx - tail_width // 2, x + rect_w - radius))
                p2_x = max(x + radius, min(cx + tail_width // 2, x + rect_w - radius))
                P1 = (p1_x, y_top)
                P2 = (p2_x, y_top)
        else:
            if dx > 0:  # Nose is to the right: attach to right edge
                p1_y = max(y_top + radius, min(cy - tail_width // 2, y - radius))
                p2_y = max(y_top + radius, min(cy + tail_width // 2, y - radius))
                P1 = (x + rect_w, p1_y)
                P2 = (x + rect_w, p2_y)
            else:  # Nose is to the left: attach to left edge
                p1_y = max(y_top + radius, min(cy - tail_width // 2, y - radius))
                p2_y = max(y_top + radius, min(cy + tail_width // 2, y - radius))
                P1 = (x, p1_y)
                P2 = (x, p2_y)

        P3 = (int(nose_x), int(nose_y))

        # Draw filled tail
        cv2.drawContours(img, [np.array([P1, P2, P3], dtype=np.int32)], 0, bg_color, -1)
        # Draw border lines of the tail
        border_thickness = max(1, int(thickness * 1.5))
        cv2.line(img, P1, P3, border_color, border_thickness)
        cv2.line(img, P2, P3, border_color, border_thickness)

    # Draw Rounded Bubble Border
    border_thickness = max(1, int(thickness * 1.5))
    cv2.line(img, (x + radius, y_top), (x + rect_w - radius, y_top), border_color, border_thickness)
    cv2.line(img, (x + radius, y), (x + rect_w - radius, y), border_color, border_thickness)
    cv2.line(img, (x, y_top + radius), (x, y - radius), border_color, border_thickness)
    cv2.line(img, (x + rect_w, y_top + radius), (x + rect_w, y - radius), border_color, border_thickness)
    cv2.ellipse(img, (x + radius, y_top + radius), (radius, radius), 180, 0, 90, border_color, border_thickness)
    cv2.ellipse(img, (x + rect_w - radius, y_top + radius), (radius, radius), 270, 0, 90, border_color, border_thickness)
    cv2.ellipse(img, (x + radius, y - radius), (radius, radius), 90, 0, 90, border_color, border_thickness)
    cv2.ellipse(img, (x + rect_w - radius, y - radius), (radius, radius), 0, 0, 90, border_color, border_thickness)

    # Output Text
    current_y = y_top + padding + line_heights[0]
    for line in text_lines:
        cv2.putText(img, line, (x + padding, current_y), font, font_scale, text_color, thickness)
        current_y += line_heights[0] + line_spacing

def merge_audio(silent_video, audio_source, input_video, output_video, no_audio=False):
    if no_audio:
        print("[*] Skipping audio merging as requested. Re-encoding to H.264...")
        cmd = [
            "ffmpeg", "-y", "-i", silent_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_video
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            shutil.copy(silent_video, output_video)
        return

    temp_audio = None
    if audio_source is None:
        temp_audio = "_temp_extracted_audio.wav"
        print("[*] Extracting original audio from input video...")
        cmd = [
            "ffmpeg", "-y", "-i", input_video, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_audio
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("[!] Warning: No audio track detected or extraction failed. Saving silent video.")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            audio_source = None
        else:
            audio_source = temp_audio
            
    if audio_source and os.path.exists(audio_source):
        print(f"[*] Merging audio from {audio_source} and encoding video to H.264...")
        cmd = [
            "ffmpeg", "-y", "-i", silent_video, "-i", audio_source,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", output_video
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("[!] Error merging audio. Saving silent video.")
            shutil.copy(silent_video, output_video)
    else:
        print("[*] Saving video without audio, encoding to H.264...")
        cmd = [
            "ffmpeg", "-y", "-i", silent_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_video
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            shutil.copy(silent_video, output_video)
        
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)

def main():
    parser = argparse.ArgumentParser(description="Add tracking speech bubbles to video.")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", default="translated_speech_bubble.mp4", help="Output video file path")
    parser.add_argument("--audio", "-a", default=None, help="Custom audio track to merge (optional)")
    parser.add_argument("--no-audio", action="store_true", help="Do not merge any audio track into the output")
    parser.add_argument("--model", "-m", default="medium", help="Whisper model size (tiny, base, small, medium, large, etc.)")
    parser.add_argument("--landmarker", "-l", default="face_landmarker.task", help="Path to face landmarker model file")
    parser.add_argument("--max-chars", "-c", type=int, default=22, help="Max characters per line in bubble text")
    parser.add_argument("--smoothing", type=float, default=0.08, help="Smoothing factor (0.01 - 1.0) for nose tracking")
    parser.add_argument("--deadzone", type=float, default=8.0, help="Deadzone threshold in pixels to prevent jitter")
    parser.add_argument("--offset-x", type=float, default=0.18, help="Horizontal bubble offset from nose (as fraction of width)")
    parser.add_argument("--offset-y", type=float, default=0.18, help="Vertical bubble offset from nose (as fraction of height)")
    parser.add_argument("--task", default="translate", choices=["translate", "transcribe"], help="Whisper task: 'translate' or 'transcribe'")
    parser.add_argument("--device", default=None, help="Device to run Whisper ('cuda', 'cpu', etc.)")
    parser.add_argument("--bubble-color", default="255,255,255", help="Bubble fill color (RGB comma-separated or hex #ffffff)")
    parser.add_argument("--border-color", default="0,0,0", help="Bubble border color (RGB comma-separated or hex #000000)")
    parser.add_argument("--text-color", default="0,0,0", help="Bubble text color (RGB comma-separated or hex #000000)")
    parser.add_argument("--font-scale-mult", type=float, default=1.0, help="Font scale multiplier")
    parser.add_argument("--thickness-mult", type=float, default=1.0, help="Font thickness multiplier")
    args = parser.parse_args()

    # Determine Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device for Whisper: {device}")

    # Ensure Face Landmarker model exists
    download_face_landmarker(args.landmarker)

    # Parse Colors
    bg_color = parse_color(args.bubble_color, (255, 255, 255))
    border_color = parse_color(args.border_color, (0, 0, 0))
    text_color = parse_color(args.text_color, (0, 0, 0))

    # 1. Run Whisper
    print(f"[*] Loading Whisper model '{args.model}' and processing audio ({args.task})...")
    whisper_model = whisper.load_model(args.model).to(device)
    
    # Whisper expects a path, let's extract audio if the input is a video to speed up or pass directly
    # Whisper can transcribe video files directly if ffmpeg is installed, which it is!
    result = whisper_model.transcribe(
        args.input,
        task=args.task
    )
    segments = result["segments"]
    print(f"[*] Whisper processing complete. Found {len(segments)} timed segments.")

    # 2. Setup Face Landmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.landmarker),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Video Setup
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[!] Error: Could not open input video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary silent output path
    temp_silent_video = "_temp_silent_output.mp4"
    out_video = cv2.VideoWriter(temp_silent_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Tracking State
    smoothed_nose_x, smoothed_nose_y = None, None

    print("[*] Processing frames and overlaying speech bubbles...")
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = frame_index / fps
            timestamp_ms = int(1000 * current_time_sec)

            # Check if any segment belongs in this frame
            active_text = ""
            for segment in segments:
                if segment["start"] <= current_time_sec <= segment["end"]:
                    active_text = segment["text"].strip()
                    break

            if active_text:
                # Convert BGR to RGB for MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                nose_pos = None
                if result.face_landmarks:
                    # Index 4 is tip of the nose
                    nose = result.face_landmarks[0][4]
                    nose_px = (int(nose.x * width), int(nose.y * height))
                    nose_pos = nose_px

                    # Smooth nose position
                    if smoothed_nose_x is None or smoothed_nose_y is None:
                        smoothed_nose_x, smoothed_nose_y = nose_px[0], nose_px[1]
                    else:
                        dist = np.sqrt((nose_px[0] - smoothed_nose_x)**2 + (nose_px[1] - smoothed_nose_y)**2)
                        if dist > args.deadzone:
                            smoothed_nose_x = (1 - args.smoothing) * smoothed_nose_x + args.smoothing * nose_px[0]
                            smoothed_nose_y = (1 - args.smoothing) * smoothed_nose_y + args.smoothing * nose_px[1]
                    
                    # Target bubble position with proportionate offset
                    offset_x_px = int(width * args.offset_x)
                    offset_y_px = int(height * args.offset_y)

                    sx = int(smoothed_nose_x + offset_x_px)
                    sy = int(smoothed_nose_y - offset_y_px)
                    bubble_pos = (sx, sy)
                else:
                    # If face is lost but we have text, use last known position or center
                    if smoothed_nose_x is not None and smoothed_nose_y is not None:
                        bubble_pos = (int(smoothed_nose_x + width * args.offset_x), int(smoothed_nose_y - height * args.offset_y))
                        nose_pos = (int(smoothed_nose_x), int(smoothed_nose_y))
                    else:
                        bubble_pos = (int(width * 0.5), int(height * 0.3))
                        nose_pos = None

                # Format and Draw Bubble
                lines = split_text_into_lines(active_text, args.max_chars)
                draw_rounded_bubble(frame, lines, bubble_pos, nose_pos,
                                    bg_color, border_color, text_color,
                                    args.font_scale_mult, args.thickness_mult)

            out_video.write(frame)
            frame_index += 1
            
            if frame_index % 50 == 0 or frame_index == total_frames:
                percent = (frame_index / total_frames) * 100
                print(f"    Progress: {frame_index}/{total_frames} frames ({percent:.1f}%)")

        cap.release()
        out_video.release()

    print("[*] Processing complete. Finalizing output file...")
    
    # 3. Merge Audio Track using FFmpeg
    merge_audio(temp_silent_video, args.audio, args.input, args.output, args.no_audio)

    # Cleanup temp silent file
    if os.path.exists(temp_silent_video):
        os.remove(temp_silent_video)

    print(f"[*] Success! Final video with speech bubbles saved to: {args.output}")

if __name__ == "__main__":
    main()
