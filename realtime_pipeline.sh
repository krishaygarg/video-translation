#!/bin/bash
# ==============================================================================
# realtime_pipeline.sh
# Runs the Real-Time Video Translation & Lip-Sync Pipeline out-of-the-box
# Uses the pre-configured MuseTalk conda environment.
# ==============================================================================

set -e

# Silence TensorFlow C++ and PyTorch deprecation warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS=ignore

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_spanish_video> <output_synced_video> [--gpus GPUS] [--source-lang LANG] [--num-beams BEAMS]"
    echo "Example: $0 data/Video1.mp4 results/realtime_out.mp4 --gpus 0,1,2,3"
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_VIDEO="$2"
shift 2

PYTHON_BIN="/home/arykun47/.conda/envs/MuseTalk/bin/python"
SCRIPT_PATH="$(dirname "$(realpath "$0")")/realtime_pipeline.py"

if [ ! -f "$PYTHON_BIN" ]; then
    # Fallback to current python if conda path differs
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_PATH" --input "$INPUT_VIDEO" --output "$OUTPUT_VIDEO" "$@"
