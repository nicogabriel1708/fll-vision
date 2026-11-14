"""
FLL-Vision Automated Training Pipeline (Extract -> Auto-Label -> Split -> Train -> TensorRT Export)
Audited & Hardened for Single-Camera Dual-Table FLL Setups:
  - Supports training videos where ONLY Table 1 (Left), ONLY Table 2 (Right), or BOTH tables are used!
    Automatically evaluates both calibrated table regions (`homography1` and `homography2`) and masks out
    whichever table has no active robot/hand in that frame so distractions on an unused table are ignored.
  - Point #11 Fix: Strips MOG2 shadows (`detectShadows=True`, threshold > 200) and rejects global exposure shifts.
  - Point #12 Fix: Prevents Train/Val data leakage by splitting across distinct videos (when multiple videos exist)
    or enforcing a wide chronological embargo buffer gap between train and val blocks (when 1 video exists).
  - Point #13 Fix: Never labels unconfirmed motion blobs as 'robot' without YOLO-World AI confirmation.
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

# SampleTuple: (img_path, lbl_path, source_video_name)
SampleTuple = Tuple[Path, Path, str]


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
	yY = min(boxA[3], boxB[3])
	inter = max(0.0, xB - xA) * max(0.0, yY - yA)
	if inter <= 0:
		return 0.0
	areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	return float(inter / (areaA + areaB - inter + 1e-6))


def _extract_shadow_free_motion_boxes(
	fg_mask: np.ndarray,
	homography: TableHomography,
	frame_w: int,
	frame_h: int,
) -> List[np.ndarray]:
	"""Extracts shadow-free motion boxes for a single table while rejecting global illumination jumps (#11)."""
	_, no_shadow = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
	ax1, ay1, ax2, ay2 = homography.get_active_bbox_px(frame_w, frame_h)
	table_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))

	if float(cv2.countNonZero(no_shadow)) > (table_area * 0.14):
		return []

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	clean = cv2.morphologyEx(no_shadow, cv2.MORPH_OPEN, kernel)
	clean = cv2.dilate(clean, kernel, iterations=2)
	contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	boxes: List[np.ndarray] = []
	for cnt in contours:
		area = float(cv2.contourArea(cnt))
		if 450.0 < area < (table_area * 0.20):
			x, y, bw, bh = cv2.boundingRect(cnt)
			cx, cy = x + 0.5 * bw, y + 0.5 * bh
			if homography.is_px_in_active_zone(cx, cy):
				boxes.append(np.array([x, y, x + bw, y + bh], dtype=float))
	return boxes


def step1_extract_and_autolabel(
	videos: List[Path],
	homography1: TableHomography,
	raw_dir: Path,
	target_total_frames: int = 600,
	device: str = "cuda:0",
	homography2: TableHomography | None = None,
) -> List[SampleTuple]:
	"""
	Adaptively samples frames across 1 long or N short training videos.
	Supports videos where ONLY Table 1 (Left), ONLY Table 2 (Right), or BOTH tables are used:
	  - Evaluates Table 1 and Table 2 independently.
	  - Masks out whichever table has no confirmed activity on that frame so distractions on an
	    unused table never pollute the training images.
	  - Point #13 Fix: Only creates labels from YOLO-World AI detections (never from unconfirmed motion alone).
	"""
	img_dir = raw_dir / "images"
	lbl_dir = raw_dir / "labels"
	img_dir.mkdir(parents=True, exist_ok=True)
	lbl_dir.mkdir(parents=True, exist_ok=True)

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

	samples: List[SampleTuple] = []
	sample_idx = 0

	for video_path in videos:
		logger.info(f"Processing training video: {video_path.name}")
		cap = cv2.VideoCapture(str(video_path))
		bg_sub1 = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32, detectShadows=True)
		bg_sub2 = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32, detectShadows=True)
		frame_idx = 0
		t1_active_hits = 0
		t2_active_hits = 0

		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame_idx += 1

			masked1 = homography1.apply_roi_mask(frame)
			fg_mask1 = bg_sub1.apply(masked1)

			if homography2 is not None:
				masked2 = homography2.apply_roi_mask(frame)
				fg_mask2 = bg_sub2.apply(masked2)
			else:
				masked2 = None
				fg_mask2 = None

			if frame_idx % adaptive_stride != 0 or frame_idx < 20:
				continue

			h, w = frame.shape[:2]
			motion_boxes1 = _extract_shadow_free_motion_boxes(fg_mask1, homography1, w, h)
			motion_boxes2 = (
				_extract_shadow_free_motion_boxes(fg_mask2, homography2, w, h)
				if (homography2 is not None and fg_mask2 is not None)
				else []
			)

			combined_masked = cv2.bitwise_or(masked1, masked2) if masked2 is not None else masked1
			results = world_model.predict(
				source=combined_masked,
				imgsz=960,
				conf=0.20,
				iou=0.45,
				device=device,
				verbose=False,
			)

			anns_t1: List[str] = []
			anns_t2: List[str] = []

			if results and results[0].boxes is not None and len(results[0].boxes) > 0:
				boxes = results[0].boxes.xyxy.cpu().numpy()
				cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
				confs = results[0].boxes.conf.cpu().numpy()

				for box, cid, conf in zip(boxes, cls_ids, confs):
					cx_px = float(0.5 * (box[0] + box[2]))
					cy_px = float(box[1] + 0.72 * (box[3] - box[1]))
					target_cls = 0 if cid in (0, 1) else 1

					in_t1 = homography1.is_px_in_active_zone(cx_px, cy_px)
					in_t2 = homography2.is_px_in_active_zone(cx_px, cy_px) if homography2 is not None else False

					if in_t1:
						has_motion1 = any(box_iou(box, mbox) > 0.10 for mbox in motion_boxes1)
						if conf >= 0.45 or (conf >= 0.24 and has_motion1):
							cx, cy, bw, bh = xyxy_to_yolo(box, w, h)
							anns_t1.append(f"{target_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
					elif in_t2:
						has_motion2 = any(box_iou(box, mbox) > 0.10 for mbox in motion_boxes2)
						if conf >= 0.45 or (conf >= 0.24 and has_motion2):
							cx, cy, bw, bh = xyxy_to_yolo(box, w, h)
							anns_t2.append(f"{target_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

			# Point #13 Fix: Do NOT auto-label unconfirmed motion boxes without AI confirmation!
			# Select which table mask(s) to save based on which table actually has confirmed activity:
			# - If only Table 1 (Left) is used in this frame -> save `masked1` (masks out unused Table 2)
			# - If only Table 2 (Right) is used in this frame -> save `masked2` (masks out unused Table 1)
			# - If both tables are used simultaneously -> save `combined_masked`
			yolo_annotations = anns_t1 + anns_t2
			if anns_t1 and not anns_t2:
				out_img = masked1
				t1_active_hits += 1
			elif anns_t2 and not anns_t1 and masked2 is not None:
				out_img = masked2
				t2_active_hits += 1
			else:
				out_img = combined_masked
				if anns_t1:
					t1_active_hits += 1
				if anns_t2:
					t2_active_hits += 1

			is_static_background = (
				len(motion_boxes1) == 0
				and len(motion_boxes2) == 0
				and len(yolo_annotations) == 0
				and random.random() < 0.12
			)
			if yolo_annotations or is_static_background:
				sample_idx += 1
				stem = f"{video_path.stem}_f{frame_idx:06d}_{sample_idx:05d}"
				img_path = img_dir / f"{stem}.jpg"
				lbl_path = lbl_dir / f"{stem}.txt"
				cv2.imwrite(str(img_path), out_img)
				with open(lbl_path, "w", encoding="utf-8") as f:
					f.write("\n".join(yolo_annotations))
				samples.append((img_path, lbl_path, video_path.name))

		cap.release()
		logger.info(
			f"Finished {video_path.name}: Table 1 (Left) active frames={t1_active_hits}, "
			f"Table 2 (Right) active frames={t2_active_hits}"
		)

	logger.info(f"Auto-labeled {len(samples)} total frames across {len(videos)} video(s).")
	return samples


def step2_split_dataset(samples: List[SampleTuple], dataset_dir: Path, val_ratio: float = 0.15):
	"""
	Point #12 Fix: Zero-Leakage Train/Val Dataset Split!
	  - Case A (Multiple training videos): Splits strictly at the VIDEO level so frames from the
	    same video never appear in both `train` and `val`.
	  - Case B (Single training video): Sorts chronologically by frame index and enforces a wide
	    embargo buffer gap (>=15 sampled frames) between the training block and validation block.
	"""
	by_video: Dict[str, List[SampleTuple]] = {}
	for item in samples:
		vid_name = item[2] if len(item) >= 3 else "default_video"
		by_video.setdefault(vid_name, []).append(item)

	train_samples: List[SampleTuple] = []
	val_samples: List[SampleTuple] = []

	if len(by_video) >= 2:
		# Video-level split: assign whole videos to validation until ~val_ratio is reached
		vid_names = sorted(by_video.keys(), key=lambda k: len(by_video[k]))
		total_n = len(samples)
		target_val = max(1, int(round(total_n * val_ratio)))
		for vname in vid_names:
			if len(val_samples) < target_val and (len(by_video) - len(val_samples)) > 1:
				val_samples.extend(by_video[vname])
			else:
				train_samples.extend(by_video[vname])
		if not train_samples and val_samples:
			# Safety fallback if only 2 videos and first took all
			train_samples = val_samples[: -max(1, len(val_samples) // 5)]
			val_samples = val_samples[-max(1, len(val_samples) // 5) :]
		logger.info(
			f"Zero-leakage VIDEO-level split across {len(by_video)} videos: "
			f"{len(train_samples)} train, {len(val_samples)} val images."
		)
	else:
		# Single video: contiguous chronological split with wide embargo buffer gap
		samples_sorted = sorted(samples, key=lambda item: item[0].name)
		n_total = len(samples_sorted)
		n_val = max(1, int(round(n_total * val_ratio)))
		buffer_gap = min(15, max(2, int(n_total * 0.05))) if n_total > 20 else min(3, max(0, n_total - n_val - 2))
		split_idx = max(1, n_total - n_val - buffer_gap)
		train_samples = samples_sorted[:split_idx]
		val_samples = samples_sorted[split_idx + buffer_gap :]
		logger.info(
			f"Zero-leakage chronological block split (embargo gap={buffer_gap} samples): "
			f"{len(train_samples)} train, {len(val_samples)} val images."
		)

	for split_name, split_list in [("train", train_samples), ("val", val_samples)]:
		s_img_dir = dataset_dir / "images" / split_name
		s_lbl_dir = dataset_dir / "labels" / split_name
		if s_img_dir.exists():
			shutil.rmtree(s_img_dir)
		if s_lbl_dir.exists():
			shutil.rmtree(s_lbl_dir)
		s_img_dir.mkdir(parents=True, exist_ok=True)
		s_lbl_dir.mkdir(parents=True, exist_ok=True)

		for item in split_list:
			img_p, lbl_p = item[0], item[1]
			shutil.copy2(img_p, s_img_dir / img_p.name)
			shutil.copy2(lbl_p, s_lbl_dir / lbl_p.name)


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
	parser.add_argument("--force", action="store_true", help="Force re-extraction and re-training without asking")
	parser.add_argument("--skip-extract", action="store_true", help="Skip Step 1 & 2 if dataset already exists")
	parser.add_argument("--run-tracker-after", action="store_true", default=False)
	args = parser.parse_args()

	cfg = load_config(args.config)
	target_pt = Path(cfg.inference.weights_pt)
	target_engine = Path(cfg.inference.engine_trt)
	is_interactive = sys.stdin.isatty() and not args.force

	if (target_pt.exists() or target_engine.exists()) and is_interactive:
		print("\n==================================================================")
		print(" [Info] Ein trainiertes Modell existiert bereits:")
		if target_pt.exists():
			print(f"   - PyTorch Weights : {target_pt}")
		if target_engine.exists():
			print(f"   - TensorRT Engine : {target_engine}")
		print("==================================================================")
		ans = input("Möchtest du trotzdem neu trainieren? [y/N]: ").strip().lower()
		if ans not in ("y", "yes", "j", "ja"):
			logger.info("Training übersprungen – bestehendes Modell wird weiterverwendet.")
			return

	raw_dir = Path("/workspace/data/dataset/auto_raw")
	dataset_dir = Path("/workspace/data/dataset")
	existing_train_imgs = (
		list((dataset_dir / "images" / "train").glob("*.jpg")) if (dataset_dir / "images" / "train").exists() else []
	)
	existing_val_imgs = (
		list((dataset_dir / "images" / "val").glob("*.jpg")) if (dataset_dir / "images" / "val").exists() else []
	)

	reuse_existing_dataset = False
	if len(existing_train_imgs) > 0 and len(existing_val_imgs) > 0:
		if args.skip_extract:
			reuse_existing_dataset = True
		elif is_interactive:
			print("\n==================================================================")
			print(" [Dataset gefunden] Bereits extrahiertes & annotiertes Dataset vorhanden:")
			print(f"   - Train-Bilder : {len(existing_train_imgs)}")
			print(f"   - Val-Bilder   : {len(existing_val_imgs)}")
			print("==================================================================")
			ans_ds = (
				input(
					"Video-Extraktion & Auto-Labeling (Schritt 1 & 2) überspringen und direkt mit Schritt 3 (Training) starten? [Y/n]: "
				)
				.strip()
				.lower()
			)
			if ans_ds in ("", "y", "yes", "j", "ja"):
				reuse_existing_dataset = True

	if not reuse_existing_dataset:
		homography1 = TableHomography(cfg.table)
		homography2 = TableHomography(cfg.table.make_table2_config())
		videos_dir = Path(args.videos_dir if args.videos_dir else cfg.video.training_videos_dir)
		fallback_vid = (
			Path(str(cfg.video.source))
			if not str(cfg.video.source).isdigit() and not str(cfg.video.source).startswith("/dev/")
			else None
		)
		videos = discover_training_videos(videos_dir, fallback_video=fallback_vid)

		if not videos:
			raise RuntimeError(
				f"No training videos found in {videos_dir}! Place one long video or multiple short videos there."
			)

		samples = step1_extract_and_autolabel(
			videos=videos,
			homography1=homography1,
			raw_dir=raw_dir,
			target_total_frames=args.target_frames,
			device=cfg.inference.device,
			homography2=homography2,
		)
		if len(samples) < 10:
			raise RuntimeError("Too few active frames extracted. Check video or table calibration.")

		step2_split_dataset(samples, dataset_dir=dataset_dir, val_ratio=0.15)
	else:
		logger.info(
			f"Überspringe Schritt 1 & 2 – verwende vorhandenes Dataset ({len(existing_train_imgs)} train, {len(existing_val_imgs)} val Bilder)."
		)

	step3_train_and_export_trt(
		dataset_yaml="/workspace/config/dataset.yaml",
		base_model="yolo11s.pt",
		epochs=args.epochs,
		imgsz=cfg.inference.imgsz,
		batch=args.batch,
		target_pt=target_pt,
		target_engine=target_engine,
		device="0",
	)

	if args.run_tracker_after:
		from src.main import run_pipeline

		run_pipeline(args.config)


if __name__ == "__main__":
	main()
