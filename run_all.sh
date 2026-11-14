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

docker compose run --rm --service-ports fll-vision \
  python3 scripts/auto_pipeline.py \
  --config /workspace/config/config.yaml \
  --videos-dir /workspace/data/training_videos \
  --target-frames 600 \
  --epochs 80 \
  --batch 32
