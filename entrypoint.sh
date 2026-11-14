#!/usr/bin/env bash
set -e

echo "=================================================================="
echo " Starting FLL-Vision Production System (Single-Container Edition)"
echo " Target: NVIDIA DGX Spark (ARM64 Grace CPU + Blackwell GB10 GPU)"
echo "=================================================================="

# 1. Check GPU availability
python3 -c "import torch; print(f'[GPU Check] CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')" || true

# 2. Initialize SQLite Database & Users if DB does not exist yet and FLL_ADMIN_PASSWORD is set
DB_PATH="${FLL_DB_PATH:-/workspace/data/db/fll_vision.sqlite3}"
if [ ! -f "$DB_PATH" ]; then
    if [ -n "${FLL_ADMIN_PASSWORD:-}" ]; then
        echo "[DB Setup] Initializing SQLite database at $DB_PATH..."
        CMD=(python3 /workspace/scripts/setup_db.py --non-interactive --db-path "$DB_PATH" --admin-user "${FLL_ADMIN_USER:-admin}" --admin-password "${FLL_ADMIN_PASSWORD}")
        if [ -n "${FLL_VOLUNTEERS_PASSWORD:-}" ]; then
            CMD+=(--volunteers-user "${FLL_VOLUNTEERS_USER:-volunteers}" --volunteers-password "${FLL_VOLUNTEERS_PASSWORD}")
        fi
        if [ -n "${FLL_ENGINEERS_PASSWORD:-}" ]; then
            CMD+=(--engineers-user "${FLL_ENGINEERS_USER:-engineers}" --engineers-password "${FLL_ENGINEERS_PASSWORD}")
        fi
        "${CMD[@]}"
    else
        echo "[DB Warning] No database found at $DB_PATH and FLL_ADMIN_PASSWORD is not set."
        echo "[DB Warning] Please run './setup.sh' on the host or 'docker compose exec fll-vision python3 scripts/setup_db.py' to set your admin password."
    fi
else
    echo "[DB Setup] Existing SQLite database found at $DB_PATH"
fi

# 3. Ensure a valid fallback video exists in /workspace/data/videos/match_input.mp4
#    in case the default USB camera (/dev/video0) is not plugged in yet
DEFAULT_VIDEO="/workspace/data/videos/match_input.mp4"
if [ ! -f "$DEFAULT_VIDEO" ]; then
    echo "[Video Check] Generating fallback FLL video at $DEFAULT_VIDEO..."
    python3 /workspace/scripts/generate_demo_video.py --output "$DEFAULT_VIDEO" --frames 150
fi

# 4. Native NVIDIA TensorRT Engine (.engine) compilation on first startup on DGX Spark
PT_MODEL="/workspace/models/fll_yolo11s.pt"
TRT_ENGINE="/workspace/models/fll_yolo11s.engine"

if [ -f "$PT_MODEL" ] && [ ! -f "$TRT_ENGINE" ]; then
    echo "[TensorRT] Found PyTorch weights ($PT_MODEL) without compiled TensorRT engine."
    echo "[TensorRT] Compiling FP16 TensorRT Engine natively for this GPU..."
    python3 /workspace/scripts/export_trt.py --weights "$PT_MODEL" --imgsz 960 --half || echo "[Warning] TensorRT export skipped (falls back to PyTorch weights)."
elif [ -f "$TRT_ENGINE" ]; then
    echo "[TensorRT] Pre-compiled TensorRT engine ready: $TRT_ENGINE"
else
    echo "[Model Info] No custom model in /workspace/models/ yet. Using base YOLO11s until training is run."
fi

exec "$@"
