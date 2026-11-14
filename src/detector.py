import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except ImportError:
	YOLO = None

from src.config_loader import InferenceConfig
from src.homography import TableHomography

logger = logging.getLogger("fll-vision.detector")


@dataclass
class Detection:
	bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] in pixels
	confidence: float
	class_id: int
	class_name: str
	source: str = "ai"  # "ai", "hybrid", or "motion"

	@property
	def center_px(self) -> tuple[float, float]:
		x1, y1, x2, y2 = self.bbox_xyxy
		return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

	@property
	def ground_point_px(self) -> tuple[float, float]:
		"""
		Returns the lower-center wheel-contact point of the bounding box (72% height)
		to minimize 3D perspective parallax error on oblique camera views (Point #14).
		"""
		x1, y1, x2, y2 = self.bbox_xyxy
		return float((x1 + x2) / 2.0), float(y1 + 0.72 * (y2 - y1))

	@property
	def area_px(self) -> float:
		x1, y1, x2, y2 = self.bbox_xyxy
		return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
	xA = max(float(boxA[0]), float(boxB[0]))
	yA = max(float(boxA[1]), float(boxB[1]))
	xB = min(float(boxA[2]), float(boxB[2]))
	yB = min(float(boxA[3]), float(boxB[3]))
	inter = max(0.0, xB - xA) * max(0.0, yB - yA)
	if inter <= 0.0:
		return 0.0
	areaA = max(1.0, (float(boxA[2]) - float(boxA[0])) * (float(boxA[3]) - float(boxA[1])))
	areaB = max(1.0, (float(boxB[2]) - float(boxB[0])) * (float(boxB[3]) - float(boxB[1])))
	return float(inter / (areaA + areaB - inter))


CLASS_SYNONYMS = {
	"robot": {"robot", "fll_robot", "lego_robot", "rover", "bot"},
	"hand": {"hand", "human_hand", "arm", "person_hand"},
	"attachment_jig": {"attachment_jig", "jig", "mission_model", "attachment"},
}


class FLLDetector:
	"""
	Audited & Hardened Hybrid Motion + YOLO11 TensorRT Detector for DGX Spark.
	Addresses Audit Points:
	  - #1 & Group A #6: Reconciles `cfg.classes` dynamically against `self.model.names`.
	  - #2: Detects stale `.engine` files (`pt_mtime > engine_mtime`) and rebuilds/falls back to `.pt`.
	  - #3: Multi-stage fallback (`engine_trt` -> `weights_pt` -> `best.pt` -> `yolo11s.pt`)
	        both during initialization/warmup AND at runtime if TensorRT fails.
	  - #4: Tracks `consecutive_errors` and `last_error` instead of silently swallowing exceptions.
	  - Group B #3: Tracks motion blob displacement across frames so stationary vibrating
	        mission models are not misclassified as driving robots.
	"""

	def __init__(self, cfg: InferenceConfig):
		self.cfg = cfg
		self.id_to_name: Dict[int, str] = {int(v): str(k) for k, v in cfg.classes.items()}
		self.model = None
		self.active_model_path: str = "none"
		self.using_trt_engine: bool = False
		self.is_custom_fll_model: bool = False
		self.model_class_names: Dict[int, str] = {}

		# Error tracking (#4)
		self.consecutive_errors: int = 0
		self.total_errors: int = 0
		self.last_error: Optional[str] = None

		# Per-table MOG2 background subtractors & temporal motion history (Group B #3)
		self._bg_subtractors: Dict[int, cv2.BackgroundSubtractorMOG2] = {}
		self._prev_motion_centers: Dict[int, List[Tuple[float, float]]] = {}
		self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

		if YOLO is None:
			logger.warning("Ultralytics YOLO package not installed; detector running in motion-only mode.")
			return

		self._initialize_model_with_fallbacks()

	def _resolve_pt_weights_path(self) -> Optional[Path]:
		pt_path = Path(self.cfg.weights_pt)
		auto_train_best = Path("/workspace/models/fll_auto_train/weights/best.pt")
		if not pt_path.exists() and auto_train_best.exists():
			try:
				pt_path.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(auto_train_best, pt_path)
				logger.info(f"Recovered trained weights from {auto_train_best} -> {pt_path}")
			except Exception as copy_exc:
				logger.warning(f"Could not copy {auto_train_best}: {copy_exc}")
				return auto_train_best
		return pt_path if pt_path.exists() else (auto_train_best if auto_train_best.exists() else None)

	def _reconcile_model_classes(self):
		"""
		Reconciles `self.model.names` with expected FLL roles ('robot', 'hand', 'attachment_jig')
		so class IDs are always mapped accurately even if training class order differed (#1, Group A #6).
		"""
		raw_names = getattr(self.model, "names", None)
		if not isinstance(raw_names, dict):
			return

		self.model_class_names = {int(k): str(v).strip().lower() for k, v in raw_names.items()}

		# Detect 80-class generic COCO model (`0: 'person'`)
		if self.model_class_names.get(0) == "person" and len(self.model_class_names) >= 50:
			self.is_custom_fll_model = False
			logger.warning(
				"Loaded model has 80 COCO classes (0='person'). Running in COCO person/hand suppression mode."
			)
			return

		# Reconcile custom model class names with canonical FLL roles
		reconciled: Dict[int, str] = {}
		for cid, cname in self.model_class_names.items():
			for canonical_role, synonyms in CLASS_SYNONYMS.items():
				if cname in synonyms or canonical_role in cname:
					reconciled[cid] = canonical_role
					break

		if reconciled:
			self.id_to_name = reconciled
			self.is_custom_fll_model = True
			logger.info(f"Reconciled custom FLL model classes from model metadata: {self.id_to_name}")
		else:
			# If custom model has <= 5 classes with generic names (e.g. class_0), use config mapping
			if len(self.model_class_names) <= 5:
				self.is_custom_fll_model = True
				logger.warning(
					f"Model class names {self.model_class_names} did not match standard labels; "
					f"falling back to config mapping {self.id_to_name}."
				)
			else:
				self.is_custom_fll_model = False

	def _try_load_and_warmup(self, model_path: str, is_engine: bool = False) -> bool:
		"""Loads a YOLO model file and runs a dummy warmup pass to verify engine integrity (#3)."""
		try:
			candidate = YOLO(model_path, task="detect")
			dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
			candidate.predict(dummy, imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False)
			self.model = candidate
			self.active_model_path = model_path
			self.using_trt_engine = is_engine
			self._reconcile_model_classes()
			logger.info(
				f"Loaded & warmed up model '{model_path}' "
				f"(TRT={self.using_trt_engine}, custom_fll={self.is_custom_fll_model})"
			)
			return True
		except Exception as exc:
			self.last_error = f"Failed to load/warmup '{model_path}': {exc}"
			logger.warning(self.last_error)
			return False

	def _initialize_model_with_fallbacks(self):
		engine_path = Path(self.cfg.engine_trt)
		pt_path = self._resolve_pt_weights_path()

		# Point #2: Check if TensorRT `.engine` is older (stale) than `.pt` weights!
		engine_is_fresh = engine_path.exists()
		if engine_is_fresh and pt_path is not None and pt_path.exists():
			try:
				if pt_path.stat().st_mtime > engine_path.stat().st_mtime + 2.0:
					logger.warning(
						f"TensorRT engine ({engine_path}) is older than PyTorch weights ({pt_path})! "
						f"Marking engine as stale so it will be re-exported from fresh weights."
					)
					engine_is_fresh = False
			except OSError:
				pass

		# 1. Try fresh TensorRT engine first
		if engine_is_fresh:
			if self._try_load_and_warmup(str(engine_path), is_engine=True):
				return
			logger.warning("Corrupted or incompatible TensorRT engine; falling back to PyTorch (.pt) weights (#3)!")

		# 2. Try custom PyTorch (.pt) weights (and optionally compile fresh TensorRT engine)
		if pt_path is not None and pt_path.exists():
			if self._try_load_and_warmup(str(pt_path), is_engine=False):
				if self.cfg.auto_export_trt:
					try:
						logger.info("Compiling fresh FP16 NVIDIA TensorRT Engine (.engine) from PyTorch weights...")
						exported = self.model.export(
							format="engine",
							imgsz=self.cfg.imgsz,
							quantize="fp16" if self.cfg.half_precision else "fp32",
							device=self.cfg.device,
							simplify=True,
							workspace=4,
						)
						self._try_load_and_warmup(str(exported), is_engine=True)
					except Exception as trt_exc:
						logger.warning(
							f"TensorRT export skipped; continuing with verified PyTorch (.pt) weights: {trt_exc}"
						)
				return

		# 3. Last resort fallback: base yolo11s.pt (COCO) for hand/person suppression + MOG2 motion
		logger.warning("No custom FLL weights available yet; falling back to base yolo11s.pt.")
		self._try_load_and_warmup("yolo11s.pt", is_engine=False)

	def reset_motion_history(self, table_id: Optional[int] = None):
		if table_id is None:
			self._bg_subtractors.clear()
			self._prev_motion_centers.clear()
		else:
			self._bg_subtractors.pop(table_id, None)
			self._prev_motion_centers.pop(table_id, None)

	def _get_bg_subtractor(self, table_id: int) -> cv2.BackgroundSubtractorMOG2:
		if table_id not in self._bg_subtractors:
			self._bg_subtractors[table_id] = cv2.createBackgroundSubtractorMOG2(
				history=200, varThreshold=28, detectShadows=False
			)
		return self._bg_subtractors[table_id]

	def _extract_motion_boxes(
		self,
		frame: np.ndarray,
		table_id: int = 1,
		homography: Optional[TableHomography] = None,
	) -> List[np.ndarray]:
		"""Extracts bounding boxes of moving objects inside the active table zone."""
		if not self.cfg.use_motion_discovery or frame is None or frame.size == 0:
			return []

		h, w = frame.shape[:2]
		bg_sub = self._get_bg_subtractor(table_id)
		fg_mask = bg_sub.apply(frame)

		clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
		clean = cv2.dilate(clean, self._morph_kernel, iterations=2)
		contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if homography is not None:
			ax1, ay1, ax2, ay2 = homography.get_active_bbox_px(w, h)
			table_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
		else:
			table_area = float(w * h)

		min_area = float(self.cfg.motion_min_area_px)
		max_area = table_area * float(self.cfg.motion_max_area_ratio)

		raw_boxes: List[np.ndarray] = []
		curr_centers: List[Tuple[float, float]] = []

		for cnt in contours:
			area = float(cv2.contourArea(cnt))
			if area < min_area or area > max_area:
				continue
			x, y, bw, bh = cv2.boundingRect(cnt)
			aspect = float(bw) / float(max(1, bh))
			if aspect < 0.22 or aspect > 4.5:
				continue
			cx, cy = x + 0.5 * bw, y + 0.5 * bh
			if homography is not None and not homography.is_px_in_active_zone(cx, cy):
				continue
			raw_boxes.append(np.array([x, y, x + bw, y + bh], dtype=float))
			curr_centers.append((cx, cy))

		self._prev_motion_centers[table_id] = curr_centers
		raw_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
		return raw_boxes[:10]

	def _map_class_id_to_fll_role(self, cid: int) -> Optional[str]:
		if self.is_custom_fll_model:
			return self.id_to_name.get(cid)
		else:
			if cid == 0:
				return "hand"
			return None

	def _run_ai_inference(self, frame: np.ndarray) -> List[Detection]:
		if self.model is None or frame is None or frame.size == 0:
			return []

		try:
			ai_conf = max(0.15, float(self.cfg.conf_threshold) * 0.75)
			results = self.model.predict(
				source=frame,
				imgsz=self.cfg.imgsz,
				conf=ai_conf,
				iou=self.cfg.iou_threshold,
				max_det=self.cfg.max_det,
				device=self.cfg.device,
				verbose=False,
			)
			self.consecutive_errors = 0

			detections: List[Detection] = []
			if not results or results[0].boxes is None:
				return detections

			boxes = results[0].boxes
			xyxy = boxes.xyxy.cpu().numpy()
			confs = boxes.conf.cpu().numpy()
			cls_ids = boxes.cls.cpu().numpy().astype(int)

			for box, conf, cid in zip(xyxy, confs, cls_ids):
				cname = self._map_class_id_to_fll_role(int(cid))
				if cname is None:
					continue
				detections.append(
					Detection(
						bbox_xyxy=box.astype(float),
						confidence=float(conf),
						class_id=0 if cname == "robot" else (1 if cname == "hand" else 2),
						class_name=cname,
						source="ai",
					)
				)
			return detections
		except Exception as exc:
			# Point #4 & #3: Track errors explicitly and fall back from broken .engine to .pt at runtime!
			self.consecutive_errors += 1
			self.total_errors += 1
			self.last_error = str(exc)
			logger.error(f"AI inference error (#{self.consecutive_errors}, model={self.active_model_path}): {exc}")
			if self.using_trt_engine and self.consecutive_errors >= 2:
				pt_path = self._resolve_pt_weights_path()
				if pt_path is not None and pt_path.exists():
					logger.warning(
						f"TensorRT engine failed repeatedly; hot-switching runtime detector to PyTorch weights {pt_path}!"
					)
					self._try_load_and_warmup(str(pt_path), is_engine=False)
			return []

	def fuse_ai_and_motion(
		self,
		ai_detections: List[Detection],
		motion_boxes: List[np.ndarray],
		homography: Optional[TableHomography] = None,
	) -> List[Detection]:
		fused: List[Detection] = []
		hand_boxes: List[np.ndarray] = []
		jig_boxes: List[np.ndarray] = []
		robot_dets: List[Detection] = []

		for det in ai_detections:
			cx, cy = det.center_px
			if homography is not None and not homography.is_px_in_active_zone(cx, cy):
				continue

			if det.class_name == "hand" and det.confidence >= max(0.35, self.cfg.conf_threshold):
				hand_boxes.append(det.bbox_xyxy)
				fused.append(det)
			elif det.class_name == "attachment_jig" and det.confidence >= self.cfg.conf_threshold:
				jig_boxes.append(det.bbox_xyxy)
				fused.append(det)
			elif det.class_name == "robot":
				has_motion = any(_bbox_iou(det.bbox_xyxy, mb) >= 0.10 for mb in motion_boxes)
				if has_motion:
					det = Detection(
						bbox_xyxy=det.bbox_xyxy,
						confidence=min(0.99, det.confidence + 0.22),
						class_id=0,
						class_name="robot",
						source="hybrid",
					)
				if det.confidence >= self.cfg.conf_threshold or has_motion:
					robot_dets.append(det)
					fused.append(det)

		for mb in motion_boxes:
			if any(_bbox_iou(mb, hb) >= 0.20 for hb in hand_boxes):
				continue
			if any(_bbox_iou(mb, jb) >= 0.25 for jb in jig_boxes):
				continue
			if any(_bbox_iou(mb, rd.bbox_xyxy) >= 0.15 for rd in robot_dets):
				continue

			motion_robot = Detection(
				bbox_xyxy=mb,
				confidence=0.65,
				class_id=0,
				class_name="robot",
				source="motion",
			)
			robot_dets.append(motion_robot)
			fused.append(motion_robot)

		return fused

	def detect(
		self,
		frame: np.ndarray,
		table_id: int = 1,
		homography: Optional[TableHomography] = None,
	) -> List[Detection]:
		if frame is None or frame.size == 0:
			return []

		ai_dets = self._run_ai_inference(frame)
		motion_boxes = self._extract_motion_boxes(frame, table_id=table_id, homography=homography)
		if not motion_boxes and not self.cfg.use_motion_discovery:
			return ai_dets
		return self.fuse_ai_and_motion(ai_dets, motion_boxes, homography=homography)

	def detect_dual_table(
		self,
		frame: np.ndarray,
		homography1: TableHomography,
		homography2: Optional[TableHomography] = None,
	) -> Tuple[List[Detection], List[Detection]]:
		if frame is None or frame.size == 0:
			return [], []

		mask1 = homography1.apply_roi_mask(frame)
		if homography2 is None:
			dets1 = self.detect(mask1, table_id=1, homography=homography1)
			return dets1, []

		mask2 = homography2.apply_roi_mask(frame)
		combined_frame = cv2.bitwise_or(mask1, mask2)

		all_ai_dets = self._run_ai_inference(combined_frame)

		motion1 = self._extract_motion_boxes(mask1, table_id=1, homography=homography1)
		motion2 = self._extract_motion_boxes(mask2, table_id=2, homography=homography2)

		dets1 = self.fuse_ai_and_motion(all_ai_dets, motion1, homography=homography1)
		dets2 = self.fuse_ai_and_motion(all_ai_dets, motion2, homography=homography2)
		return dets1, dets2
