"""
Compiles trained PyTorch weights (.pt) into an NVIDIA TensorRT Engine (.engine)
natively on the DGX Spark's GB10 Blackwell GPU (SM 12.1).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
	parser = argparse.ArgumentParser(description="Export YOLO weights to NVIDIA TensorRT .engine")
	parser.add_argument("--weights", type=str, default="/workspace/models/fll_yolo11s.pt")
	parser.add_argument("--imgsz", type=int, default=960)
	parser.add_argument("--half", action="store_true", default=True)
	parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace size in GB")
	args = parser.parse_args()

	weights_path = Path(args.weights)
	if not weights_path.exists():
		raise FileNotFoundError(f"Weights file not found: {weights_path}")

	model = YOLO(str(weights_path))
	exported_path = model.export(
		format="engine",
		imgsz=args.imgsz,
		quantize="fp16" if args.half else "fp32",
		simplify=True,
		workspace=args.workspace,
		device="cuda:0",
	)

	target_path = weights_path.with_suffix(".engine")
	if Path(exported_path) != target_path:
		shutil.move(str(exported_path), str(target_path))
	print(f"[Success] TensorRT engine saved to: {target_path}")


if __name__ == "__main__":
	main()
