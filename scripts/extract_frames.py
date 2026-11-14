"""
Smart frame extractor for building the FLL-Vision training dataset.
Uses frame-differencing so only frames with actual movement on the FLL table are saved.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract training frames from FLL match videos")
    parser.add_argument("--video", type=str, required=True, help="Path to input FLL video")
    parser.add_argument("--out-dir", type=str, default="/workspace/data/dataset/raw_frames")
    parser.add_argument("--every-n", type=int, default=15, help="Minimum frame interval")
    parser.add_argument("--motion-thresh", type=float, default=2.5, help="Mean pixel diff threshold")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    prev_gray = None
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.every_n != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff < args.motion_thresh:
                continue

        prev_gray = gray
        out_file = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_file), frame)
        saved_count += 1

    cap.release()
    print(f"Extracted {saved_count} active frames to {out_dir}")


if __name__ == "__main__":
    main()
