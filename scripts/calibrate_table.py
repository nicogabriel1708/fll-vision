"""
Utility script to extract a reference frame from a video and verify the 4 FLL table corners
by generating a preview Bird's-Eye View image.
"""

import argparse

import cv2

from src.config_loader import load_config
from src.homography import TableHomography


def main():
	parser = argparse.ArgumentParser(description="Verify FLL Table 4-Point Homography Calibration")
	parser.add_argument("--config", type=str, default="/workspace/config/config.yaml")
	parser.add_argument("--output", type=str, default="/workspace/data/outputs/calibration_preview.jpg")
	args = parser.parse_args()

	cfg = load_config(args.config)
	cap = cv2.VideoCapture(cfg.video.source)
	ret, frame = cap.read()
	cap.release()

	if not ret:
		raise RuntimeError(f"Could not read frame from {cfg.video.source}")

	homography = TableHomography(cfg.table, minimap_scale=0.5)
	warped = homography.warp_to_minimap(frame)
	cv2.imwrite(args.output, warped)
	print(f"Saved warped Bird's-Eye View preview to {args.output}")


if __name__ == "__main__":
	main()
