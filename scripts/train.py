"""
Training script for FLL-Vision on NVIDIA DGX Spark (GB10 Grace Blackwell).
Trains a multi-class YOLO11 detector ('robot', 'hand', 'attachment_jig') with
augmentations tailored to overhead FLL table cameras.
"""

import argparse
import os

from ultralytics import YOLO


def main():
	parser = argparse.ArgumentParser(description="Train FLL-Vision YOLO11 Detector on DGX Spark")
	parser.add_argument("--data", type=str, default="/workspace/config/dataset.yaml")
	parser.add_argument("--base-model", type=str, default="yolo11s.pt")
	parser.add_argument("--epochs", type=int, default=120)
	parser.add_argument("--imgsz", type=int, default=960)
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--device", type=str, default="0")
	args = parser.parse_args()

	model = YOLO(args.base_model)

	model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=min(16, max(4, (os.cpu_count() or 8) - 2)),
		project="/workspace/models",
		name="fll_yolo11s_run",
		exist_ok=True,
		pretrained=True,
		optimizer="AdamW",
		lr0=0.001,
		cos_lr=True,
		# Augmentations tuned for overhead FLL Robot Game tables
		degrees=15.0,
		translate=0.1,
		scale=0.4,
		fliplr=0.5,
		flipud=0.2,
		mosaic=1.0,
		mixup=0.1,
		hsv_h=0.015,
		hsv_s=0.5,
		hsv_v=0.4,
	)


if __name__ == "__main__":
	main()
