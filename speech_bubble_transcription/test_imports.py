#!/usr/bin/env python3
"""
Diagnostic script to test Python version and verify imports for speech bubble transcription.
"""

import sys

print("=========================================")
print("Speech Bubble Transcription - Diagnostic")
print("=========================================")
print(f"Python Version: {sys.version}")
print("-----------------------------------------")

modules = ["cv2", "numpy", "whisper", "torch", "mediapipe"]
all_ok = True

for mod in modules:
    try:
        __import__(mod)
        print(f"[+] {mod:<12} : IMPORT OK")
    except ImportError as e:
        print(f"[-] {mod:<12} : FAILED ({e})")
        all_ok = False

print("-----------------------------------------")
if all_ok:
    print("[*] All required packages are imported successfully!")
    
    # Check CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"[*] PyTorch CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"[*] CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[!] Error checking CUDA status: {e}")
else:
    print("[!] Some dependencies are missing. Please run:")
    print("    pip install -r requirements.txt")
print("=========================================")
