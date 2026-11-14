#!/usr/bin/env python3
"""
Generates a synthetic 1920x1080 FLL Table video (`data/videos/match_input.mp4`)
showing the full FLL table with:
  - Left Half (Active Mat): Launch area + moving LEGO robot driving a sortie loop
  - Right Half (Unused Half): Simulated distractions (stationary bottle / hand box)
    to verify that Active Half-Table Masking strictly ignores the unused half.
Used automatically on first container startup if no input video file exists yet.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_video(output_path: str, num_frames: int = 120, fps: int = 30):
	out_file = Path(output_path)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	w, h = 1920, 1080
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

	# Table perspective quad matching config/config.yaml default src_corners_px
	table_pts = np.array([[210, 140], [1710, 140], [1860, 960], [60, 960]], dtype=np.int32)

	for idx in range(num_frames):
		frame = np.full((h, w, 3), 28, dtype=np.uint8)

		# Draw FLL Table Mat surface
		cv2.fillPoly(frame, [table_pts], (55, 75, 95))
		cv2.polylines(frame, [table_pts], isClosed=True, color=(235, 235, 235), thickness=8)

		# Draw center line dividing Left Half (Active Mat) and Right Half (Unused Half)
		cv2.line(frame, (960, 140), (960, 960), (140, 140, 140), 3)

		# Draw Launch Area quarter-circle on Left Half
		cv2.ellipse(frame, (160, 920), (220, 220), 0, 270, 360, (255, 255, 255), 3)

		# Draw simulated robot moving across the ACTIVE LEFT HALF (x between 280 and 820)
		t = (idx / max(1, num_frames - 1)) * 2.0 * np.pi
		rx = int(520 + 220 * np.sin(t))
		ry = int(540 + 180 * np.cos(t))
		cv2.rectangle(frame, (rx - 38, ry - 38), (rx + 38, ry + 38), (0, 215, 255), -1)
		cv2.rectangle(frame, (rx - 38, ry - 38), (rx + 38, ry + 38), (20, 20, 20), 3)
		cv2.putText(frame, "ROBOT", (rx - 30, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

		# Draw simulated distraction on the UNUSED RIGHT HALF (must be ignored by mask!)
		bx = int(1350 + 60 * np.cos(t * 2))
		by = 500
		cv2.circle(frame, (bx, by), 45, (180, 105, 255), -1)
		cv2.putText(frame, "UNUSED HALF", (1280, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

		writer.write(frame)

	writer.release()
	print(f"[OK] Generated demo FLL video at {out_file} ({num_frames} frames)")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate synthetic FLL demo video")
	parser.add_argument("--output", type=str, default="/workspace/data/videos/match_input.mp4")
	parser.add_argument("--frames", type=int, default=120)
	args = parser.parse_args()
	generate_video(args.output, num_frames=args.frames)
