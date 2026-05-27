#!/bin/bash
# Helper script to run the speech bubble transcription generator

# Exit immediately if any command exits with a non-zero status
set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure we run from the correct directory or absolute paths are used
cd "$SCRIPT_DIR"

echo "========================================="
2: echo "Speech Bubble Overlay Generator"
3: echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[!] conda command not found. Please make sure Conda is installed and in your PATH."
    echo "[*] Falling back to system python3..."
    pip3 install -r requirements.txt -q
    python3 transcribe_bubble.py "$@"
else
    # Check if speech_bubble environment exists
    if ! conda info --envs | grep -qE "(^|[[:space:]])speech_bubble([[:space:]]|$)"; then
        echo "[*] Creating 'speech_bubble' Conda environment..."
        conda create -n speech_bubble python=3.10 -y
    fi
    echo "[*] Installing dependencies into 'speech_bubble' environment..."
    conda run --no-capture-output -n speech_bubble pip install -r requirements.txt -q
    
    echo "[*] Running transcribe_bubble.py inside the 'speech_bubble' environment..."
    conda run --no-capture-output -n speech_bubble python3 transcribe_bubble.py "$@"
fi
