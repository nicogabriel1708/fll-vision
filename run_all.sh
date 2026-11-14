#!/usr/bin/env bash
set -e

echo "=================================================================="
echo " FLL-Vision: 1-Click Automated Training & TensorRT Pipeline"
echo " Target: NVIDIA DGX Spark (GB10 Grace Blackwell)"
echo "=================================================================="
echo "Steps executed automatically:"
echo "  1. Discover all videos in data/training_videos/ (1 long or multiple short)"
echo "  2. Mask out unused table regions & surrounding room"
echo "  3. Auto-Label 'robot' (0) and 'hand' (1) via YOLO-World + Motion"
echo "  4. Split dataset into train/val (85% / 15%)"
echo "  5. Train YOLO11s on DGX Spark GPU"
echo "  6. Compile FP16 TensorRT Engine (.engine)"
echo "=================================================================="

install_runtime_deps() {
    docker compose exec fll-vision pip install --no-cache-dir --root-user-action=ignore -q \
        --extra-index-url https://pypi.ngc.nvidia.com \
        polars pyarrow matplotlib pillow psutil tqdm ultralytics-thop ultralytics-platform \
        onnx-graphsurgeon onnxruntime lief polygraphy "nvidia-modelopt[onnx]"
}

# Case 1: PyTorch weights (.pt) already exist, only TensorRT (.engine) is missing
if [ -f "models/fll_yolo11s.pt" ] && [ ! -f "models/fll_yolo11s.engine" ] && [ -t 0 ]; then
    echo ""
    echo "[Info] Trainiertes PyTorch-Modell (models/fll_yolo11s.pt) gefunden, aber TensorRT-Engine fehlt noch."
    read -r -p "Nur den TensorRT-Export (.pt -> .engine) ausführen statt komplett neu zu trainieren? [Y/n]: " ONLY_EXPORT_ANS
    case "$ONLY_EXPORT_ANS" in
        ""|[yY]|[yY][eE][sS]|[jJ]|[jJ][aA])
            install_runtime_deps
            docker compose exec -e YOLO_AUTOINSTALL=False fll-vision \
                python3 scripts/export_trt.py --weights /workspace/models/fll_yolo11s.pt --imgsz 960 --half
            docker compose restart fll-vision
            exit 0
            ;;
    esac
fi

# Case 2: Both .pt and .engine already exist
if [ -f "models/fll_yolo11s.engine" ] && [ -t 0 ]; then
    echo ""
    echo "[Info] Ein fertig trainiertes & kompiliertes Modell existiert bereits in ./models/:"
    ls -lh models/fll_yolo11s.* 2>/dev/null || true
    read -r -p "Möchtest du trotzdem neu trainieren? [y/N]: " RETRAIN_ANS
    case "$RETRAIN_ANS" in
        [yY]|[yY][eE][sS]|[jJ]|[jJ][aA])
            echo "[OK] Starte neues Training..."
            ;;
        *)
            echo "[Abgebrochen] Bestehendes Modell in ./models/ bleibt aktiv."
            exit 0
            ;;
    esac
fi

if docker ps --format '{{.Names}}' | grep -q '^fll-vision$'; then
    echo "[Container aktiv] Führe Training direkt im laufenden fll-vision Container aus..."
    install_runtime_deps
    docker compose exec -e YOLO_AUTOINSTALL=False fll-vision python3 scripts/auto_pipeline.py \
        --config /workspace/config/config.yaml \
        --videos-dir /workspace/data/training_videos \
        --target-frames 600 \
        --epochs 80 \
        --batch 32 "$@"
    echo "[OK] Starte fll-vision Container neu, um die neue TensorRT-Engine live zu aktivieren..."
    docker compose restart fll-vision
else
    echo "[Container gestoppt] Starte Trainings-Container ohne Port-Bindung (--no-deps)..."
    docker compose run --rm --no-deps -e YOLO_AUTOINSTALL=False fll-vision bash -c \
        "pip install --no-cache-dir --root-user-action=ignore -q --extra-index-url https://pypi.ngc.nvidia.com polars pyarrow matplotlib pillow psutil tqdm ultralytics-thop ultralytics-platform onnx-graphsurgeon onnxruntime lief polygraphy 'nvidia-modelopt[onnx]' && python3 scripts/auto_pipeline.py --config /workspace/config/config.yaml --videos-dir /workspace/data/training_videos --target-frames 600 --epochs 80 --batch 32 $*"
fi
