"""
FLL-Vision v4: Automated Training Pipeline (Extract -> Auto-Label -> Split -> Train -> TensorRT Export)
Supports:
  - Both one single long video and multiple short videos in `data/training_videos/` (regardless of filename)
  - Strict Active Half-Table Masking: ignores the unused half of the table (hands, bottles, etc.)
    and everything outside the table border during both frame extraction/auto-labeling and training.
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except ImportError:
	YOLO = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.homography import TableHomography

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fll-vision.auto-pipeline")

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}


def discover_training_videos(videos_dir: Path, fallback_video: Path | None = None) -> List[Path]:
	"""Finds all video files in `data/training_videos/` regardless of filename or extension case."""
	found = []
	if videos_dir.exists():
		for p in sorted(videos_dir.rglob("*")):
			if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
				found.append(p)
	if not found and fallback_video and fallback_video.exists():
		logger.info(f"No videos in {videos_dir}; falling back to configured source video: {fallback_video}")
		found.append(fallback_video)
	return found


def xyxy_to_yolo(box: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
	x1, y1, x2, y2 = box
	x1, x2 = np.clip([x1, x2], 0, img_w)
	y1, y2 = np.clip([y1, y2], 0, img_h)
	bw = (x2 - x1) / img_w
	bh = (y2 - y1) / img_h
	cx = ((x1 + x2) / 2.0) / img_w
	cy = ((y1 + y2) / 2.0) / img_h
	return float(cx), float(cy), float(bw), float(bh)


def box_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	inter = max(0.0, xB - xA) * max(0.0, yB - yA)
	if inter <= 0:
		return 0.0
	areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	return float(inter / (areaA + areaB - inter + 1e-6))


def step1_extract_and_autolabel(
	videos: List[Path],
	homography: TableHomography,
	raw_dir: Path,
	target_total_frames: int = 600,
	device: str = "cuda:0",
) -> List[Tuple[Path, Path]]:
	"""
	Adaptively samples frames across either 1 long video or N short videos,
	masks out the unused half of the FLL table and outside surroundings,
	and generates YOLO labels for 'robot' (0) and 'hand' (1).
	"""
	img_dir = raw_dir / "images"
	lbl_dir = raw_dir / "labels"
	img_dir.mkdir(parents=True, exist_ok=True)
	lbl_dir.mkdir(parents=True, exist_ok=True)

	# Compute adaptive frame stride so both 1 long video and many short videos yield balanced datasets
	total_video_frames = 0
	for vp in videos:
		cap = cv2.VideoCapture(str(vp))
		total_video_frames += max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		cap.release()

	adaptive_stride = max(3, int(total_video_frames / max(100, target_total_frames)))
	logger.info(
		f"Discovered {len(videos)} video(s) ({total_video_frames} total frames). "
		f"Using adaptive sampling stride: every {adaptive_stride} frames."
	)

	logger.info("Loading Zero-Shot YOLO-World model (yolov8x-worldv2.pt) for automatic labeling...")
	world_model = YOLO("yolov8x-worldv2.pt")
	world_model.set_classes(
		[
			"small wheeled lego robot",
			"toy vehicle robot",
			"human hand",
			"person arm fingers",
		]
	)

	samples: List[Tuple[Path, Path]] = []
	sample_idx = 0

	for video_path in videos:
		logger.info(f"Processing training video: {video_path.name}")
		cap = cv2.VideoCapture(str(video_path))
		bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=False)
		frame_idx = 0

		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame_idx += 1
			if frame_idx == 1 and homography.cfg.auto_detect_borders:
				homography.auto_detect_corners(frame, apply_immediately=True)

			# Strictly mask out the unused half of the table + outside surroundings!
			masked = homography.apply_roi_mask(frame)
			fg_mask = bg_sub.apply(masked)

			if frame_idx % adaptive_stride != 0 or frame_idx < 20:
				continue

			h, w = masked.shape[:2]

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
			clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
			clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)
			contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			motion_boxes = []
			for cnt in contours:
				area = cv2.contourArea(cnt)
				if 800 < area < (w * h * 0.22):
					x, y, bw, bh = cv2.boundingRect(cnt)
					motion_boxes.append(np.array([x, y, x + bw, y + bh], dtype=float))

			results = world_model.predict(
				source=masked,
				imgsz=960,
				conf=0.20,
				iou=0.45,
				device=device,
				verbose=False,
			)

			yolo_annotations: List[str] = []

			if results and results[0].boxes is not None and len(results[0].boxes) > 0:
				boxes = results[0].boxes.xyxy.cpu().numpy()
				cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
				confs = results[0].boxes.conf.cpu().numpy()

				for box, cid, conf in zip(boxes, cls_ids, confs):
					cx_px = 0.5 * (box[0] + box[2])
					cy_px = 0.5 * (box[1] + box[3])
					x_mm, y_mm = homography.px_to_mm(cx_px, cy_px)
					# Double-check detection lies inside the active half of the table
					if not homography.is_mm_in_active_zone(x_mm, y_mm):
						continue

					target_cls = 0 if cid in (0, 1) else 1
					has_motion = any(box_iou(box, mbox) > 0.10 for mbox in motion_boxes)
					if conf >= 0.45 or (conf >= 0.22 and has_motion):
						cx, cy, bw, bh = xyxy_to_yolo(box, w, h)
						yolo_annotations.append(f"{target_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

			has_hand = any(line.startswith("1 ") for line in yolo_annotations)
			has_robot = any(line.startswith("0 ") for line in yolo_annotations)
			if not has_robot and not has_hand and len(motion_boxes) == 1:
				mbox = motion_boxes[0]
				mcx = 0.5 * (mbox[0] + mbox[2])
				mcy = 0.5 * (mbox[1] + mbox[3])
				x_mm, y_mm = homography.px_to_mm(mcx, mcy)
				in_base, _ = homography.is_in_launch_area(x_mm, y_mm)
				if not in_base and homography.is_mm_in_active_zone(x_mm, y_mm):
					cx, cy, bw, bh = xyxy_to_yolo(mbox, w, h)
					yolo_annotations.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

			is_static_background = len(motion_boxes) == 0 and len(yolo_annotations) == 0 and random.random() < 0.15
			if yolo_annotations or is_static_background:
				sample_idx += 1
				stem = f"fll_auto_{sample_idx:06d}"
				img_path = img_dir / f"{stem}.jpg"
				lbl_path = lbl_dir / f"{stem}.txt"
				# Save the active-half MASKED image so training only ever sees the relevant half!
				cv2.imwrite(str(img_path), masked)
				with open(lbl_path, "w", encoding="utf-8") as f:
					f.write("\n".join(yolo_annotations))
				samples.append((img_path, lbl_path))

		cap.release()

	logger.info(f"Auto-labeled {len(samples)} active-half frames.")
	return samples


def step2_split_dataset(samples: List[Tuple[Path, Path]], dataset_dir: Path, val_ratio: float = 0.15):
	random.shuffle(samples)
	n_val = max(1, int(len(samples) * val_ratio))
	val_samples = samples[:n_val]
	train_samples = samples[n_val:]

	for split_name, split_list in [("train", train_samples), ("val", val_samples)]:
		s_img_dir = dataset_dir / "images" / split_name
		s_lbl_dir = dataset_dir / "labels" / split_name
		s_img_dir.mkdir(parents=True, exist_ok=True)
		s_lbl_dir.mkdir(parents=True, exist_ok=True)

		for img_p, lbl_p in split_list:
			shutil.copy2(img_p, s_img_dir / img_p.name)
			shutil.copy2(lbl_p, s_lbl_dir / lbl_p.name)

	logger.info(f"Dataset split complete: {len(train_samples)} train, {len(val_samples)} val images.")


def step3_train_and_export_trt(
	dataset_yaml: str,
	base_model: str,
	epochs: int,
	imgsz: int,
	batch: int,
	target_pt: Path,
	target_engine: Path,
	device: str = "0",
):
	logger.info(f"Starting YOLO11 training for {epochs} epochs at imgsz={imgsz}...")
	model = YOLO(base_model)
	model.train(
		data=dataset_yaml,
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		device=device,
		workers=min(16, max(4, (os.cpu_count() or 8) - 2)),
		project="/workspace/models",
		name="fll_auto_train",
		exist_ok=True,
		pretrained=True,
		optimizer="AdamW",
		lr0=0.001,
		cos_lr=True,
		degrees=15.0,
		translate=0.1,
		scale=0.4,
		fliplr=0.5,
		flipud=0.2,
		mosaic=1.0,
	)

	best_pt = Path("/workspace/models/fll_auto_train/weights/best.pt")
	if not best_pt.exists():
		raise FileNotFoundError(f"Trained weights not found at {best_pt}")

	target_pt.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(best_pt, target_pt)
	logger.info(f"Copied best PyTorch weights to {target_pt}")

	logger.info("Compiling FP16 NVIDIA TensorRT Engine (.engine) for DGX Spark GPU...")
	trained_model = YOLO(str(target_pt))
	exported = trained_model.export(
		format="engine",
		imgsz=imgsz,
		quantize="fp16",
		simplify=True,
		workspace=4,
		device="cuda:0",
	)
	if Path(exported) != target_engine:
		shutil.move(str(exported), str(target_engine))
	logger.info(f"TensorRT Engine ready at: {target_engine}")


def main():
	parser = argparse.ArgumentParser(
		description="End-to-End Automated Pipeline: Extract -> Auto-Label -> Train -> TensorRT"
	)
	parser.add_argument("--config", type=str, default="/workspace/config/config.yaml")
	parser.add_argument("--videos-dir", type=str, default=None)
	parser.add_argument("--target-frames", type=int, default=600)
	parser.add_argument("--epochs", type=int, default=80)
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--run-tracker-after", action="store_true", default=False)
	args = parser.parse_args()

	cfg = load_config(args.config)
	homography = TableHomography(cfg.table)

	videos_dir = Path(args.videos_dir if args.videos_dir else cfg.video.training_videos_dir)
	fallback_vid = Path(str(cfg.video.source)) if not str(cfg.video.source).isdigit() else None
	videos = discover_training_videos(videos_dir, fallback_video=fallback_vid)

	if not videos:
		raise RuntimeError(
			f"No training videos found in {videos_dir}! Place one long video or multiple short videos there."
		)

	raw_dir = Path("/workspace/data/dataset/auto_raw")
	dataset_dir = Path("/workspace/data/dataset")

	samples = step1_extract_and_autolabel(
		videos=videos,
		homography=homography,
		raw_dir=raw_dir,
		target_total_frames=args.target_frames,
		device=cfg.inference.device,
	)
	if len(samples) < 10:
		raise RuntimeError("Too few active frames extracted. Check video or active table half configuration.")

	step2_split_dataset(samples, dataset_dir=dataset_dir, val_ratio=0.15)

	step3_train_and_export_trt(
		dataset_yaml="/workspace/config/dataset.yaml",
		base_model="yolo11s.pt",
		epochs=args.epochs,
		imgsz=cfg.inference.imgsz,
		batch=args.batch,
		target_pt=Path(cfg.inference.weights_pt),
		target_engine=Path(cfg.inference.engine_trt),
		device="0",
	)

	if args.run_tracker_after:
		from src.main import run_pipeline

		run_pipeline(args.config)


if __name__ == "__main__":
	main()
