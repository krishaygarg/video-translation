#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Print usage instructions if the arguments are incorrect
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_spanish_video> <output_synced_video> [--speech-bubble] [--crop-upscale] [--realtime] [--avatar-cache]"
    echo "Example: $0 data/Video1.mp4 results/optimized_synced.mp4 --speech-bubble --realtime"
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_VIDEO="$2"
shift 2

PYTHON_BIN="/home/arykun47/.conda/envs/MuseTalk/bin/python"
SCRIPT_PATH="$(dirname "$(realpath "$0")")/run_pipeline_optimized.py"

exec "$PYTHON_BIN" "$SCRIPT_PATH" --input "$INPUT_VIDEO" --output "$OUTPUT_VIDEO" "$@"
