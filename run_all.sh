#!/usr/bin/env bash
set -e

echo "=================================================================="
echo " FLL-Vision: 1-Click Automated Training & TensorRT Pipeline"
echo " Target: NVIDIA DGX Spark (GB10 Grace Blackwell)"
echo "=================================================================="
echo "Steps executed automatically:"
echo "  1. Discover all videos in data/training_videos/ (1 long or multiple short)"
echo "  2. Mask out unused half of the FLL table & surrounding room"
echo "  3. Auto-Label 'robot' (0) and 'hand' (1) via YOLO-World + Motion"
echo "  4. Split dataset into train/val (85% / 15%)"
echo "  5. Train YOLO11s on DGX Spark GPU"
echo "  6. Compile FP16 TensorRT Engine (.engine)"
echo "=================================================================="

# Falls bereits ein fertig trainiertes Modell in models/ liegt, vorab interaktiv fragen:
if [ -f "models/fll_yolo11s.engine" ] || [ -f "models/fll_yolo11s.pt" ]; then
    echo ""
    echo "[Info] Ein trainiertes Modell existiert bereits in ./models/:"
    ls -lh models/fll_yolo11s.* 2>/dev/null || true
    if [ -t 0 ]; then
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
fi

# Sicherstellen, dass 'polars' im Container installiert ist (wird von Ultralytics beim Checkpoint-Speichern benötigt)
if docker ps --format '{{.Names}}' | grep -q '^fll-vision$'; then
    echo "[Container aktiv] Führe Training direkt im laufenden fll-vision Container aus (kein Port-8080-Konflikt)..."
    docker compose exec fll-vision pip install --no-cache-dir -q polars matplotlib pillow psutil tqdm ultralytics-thop
    docker compose exec fll-vision python3 scripts/auto_pipeline.py \
        --config /workspace/config/config.yaml \
        --videos-dir /workspace/data/training_videos \
        --target-frames 600 \
        --epochs 80 \
        --batch 32 "$@"
    echo "[OK] Starte fll-vision Container neu, um die neue TensorRT-Engine live zu aktivieren..."
    docker compose restart fll-vision
else
    echo "[Container gestoppt] Starte Trainings-Container ohne Port-Bindung (--no-deps)..."
    docker compose run --rm --no-deps fll-vision bash -c "pip install --no-cache-dir -q polars matplotlib pillow psutil tqdm ultralytics-thop && python3 scripts/auto_pipeline.py --config /workspace/config/config.yaml --videos-dir /workspace/data/training_videos --target-frames 600 --epochs 80 --batch 32 $*"
fi