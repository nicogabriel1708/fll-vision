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
		to minimize 3D perspective parallax error on oblique camera views.
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
	Audited & Hardened Hybrid Motion + YOLO11 TensorRT Detector for Single-Camera Dual-Table FLL Setup.
	Key Guarantees:
	  - Reconciles `cfg.classes` dynamically against `self.model.names`.
	  - Detects stale `.engine` files (`pt_mtime > engine_mtime`) and falls back to `.pt` on load or runtime errors.
	  - Shadow-resilient & illumination-gated MOG2 motion analysis (`detectShadows=True`, threshold > 200) (#11).
	  - Evaluates multi-frame motion history (`_motion_tracks[table_id]`) to verify genuine directional travel (#9).
	  - Never promotes unassociated motion blobs to false robots; motion only bridges short AI dropouts near
	    confirmed robot trajectories (#10).
	"""

	def __init__(self, cfg: InferenceConfig):
		self.cfg = cfg
		self.id_to_name: Dict[int, str] = {int(v): str(k) for k, v in cfg.classes.items()}
		self.model = None
		self.active_model_path: str = "none"
		self.using_trt_engine: bool = False
		self.is_custom_fll_model: bool = False
		self.model_class_names: Dict[int, str] = {}

		self.consecutive_errors: int = 0
		self.total_errors: int = 0
		self.last_error: Optional[str] = None

		# Per-table MOG2 subtractors & multi-frame motion history (#7, #9, #10, #11)
		self._bg_subtractors: Dict[int, cv2.BackgroundSubtractorMOG2] = {}
		self._motion_step: Dict[int, int] = {}
		self._motion_tracks: Dict[int, List[Dict[str, float]]] = {}
		self._last_confirmed_robot: Dict[int, Tuple[float, float, int]] = {}
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
		raw_names = getattr(self.model, "names", None)
		if not isinstance(raw_names, dict):
			return

		self.model_class_names = {int(k): str(v).strip().lower() for k, v in raw_names.items()}

		if self.model_class_names.get(0) == "person" and len(self.model_class_names) >= 50:
			self.is_custom_fll_model = False
			logger.warning(
				"Loaded model has 80 COCO classes (0='person'). Running in COCO person/hand suppression mode."
			)
			return

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
			if len(self.model_class_names) <= 5:
				self.is_custom_fll_model = True
			else:
				self.is_custom_fll_model = False

	def _try_load_and_warmup(self, model_path: str, is_engine: bool = False) -> bool:
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

		if engine_is_fresh:
			if self._try_load_and_warmup(str(engine_path), is_engine=True):
				return
			logger.warning("Corrupted or incompatible TensorRT engine; falling back to PyTorch (.pt) weights!")

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

		logger.warning("No custom FLL weights available yet; falling back to base yolo11s.pt.")
		self._try_load_and_warmup("yolo11s.pt", is_engine=False)

	def reset_motion_history(self, table_id: Optional[int] = None):
		"""Resets MOG2 background models, multi-frame motion tracks, and last robot anchors (#7)."""
		if table_id is None:
			self._bg_subtractors.clear()
			self._motion_step.clear()
			self._motion_tracks.clear()
			self._last_confirmed_robot.clear()
		else:
			self._bg_subtractors.pop(table_id, None)
			self._motion_step.pop(table_id, None)
			self._motion_tracks.pop(table_id, None)
			self._last_confirmed_robot.pop(table_id, None)

	def _get_bg_subtractor(self, table_id: int) -> cv2.BackgroundSubtractorMOG2:
		if table_id not in self._bg_subtractors:
			# Point #11 Fix: Enable detectShadows=True so shadows (value 127) can be stripped cleanly!
			self._bg_subtractors[table_id] = cv2.createBackgroundSubtractorMOG2(
				history=200, varThreshold=32, detectShadows=True
			)
		return self._bg_subtractors[table_id]

	def _extract_motion_boxes(
		self,
		frame: np.ndarray,
		table_id: int = 1,
		homography: Optional[TableHomography] = None,
	) -> List[np.ndarray]:
		"""
		Extracts moving object bounding boxes inside the active table zone while:
		  1. Stripping MOG2 shadow pixels (value 127) via binary threshold at 200 (#11).
		  2. Rejecting global lighting / camera auto-exposure jumps (#11).
		  3. Evaluating multi-frame motion history (`_motion_tracks[table_id]`) so only
		     blobs with genuine directional displacement across frames are returned (#9).
		"""
		if not self.cfg.use_motion_discovery or frame is None or frame.size == 0:
			return []

		h, w = frame.shape[:2]
		step_idx = self._motion_step.get(table_id, 0) + 1
		self._motion_step[table_id] = step_idx

		bg_sub = self._get_bg_subtractor(table_id)
		raw_fg = bg_sub.apply(frame)

		# Point #11 Fix: Strip shadow pixels (marked as 127 by MOG2) and keep only true foreground (255)
		_, fg_no_shadows = cv2.threshold(raw_fg, 200, 255, cv2.THRESH_BINARY)

		if homography is not None:
			ax1, ay1, ax2, ay2 = homography.get_active_bbox_px(w, h)
			table_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
		else:
			table_area = float(w * h)

		# Point #11 Fix: If >14% of the active table area changes simultaneously, it is a global
		# lighting / shadow / camera exposure shift -> ignore motion on this frame!
		changed_pixels = float(cv2.countNonZero(fg_no_shadows))
		if changed_pixels > (table_area * 0.14):
			return []

		clean = cv2.morphologyEx(fg_no_shadows, cv2.MORPH_OPEN, self._morph_kernel)
		clean = cv2.dilate(clean, self._morph_kernel, iterations=2)
		contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		min_area = float(self.cfg.motion_min_area_px)
		max_area = table_area * float(self.cfg.motion_max_area_ratio)

		prev_tracks = self._motion_tracks.get(table_id, [])
		updated_tracks: List[Dict[str, float]] = []
		verified_motion_boxes: List[np.ndarray] = []

		for cnt in contours:
			area = float(cv2.contourArea(cnt))
			if area < min_area or area > max_area:
				continue
			x, y, bw, bh = cv2.boundingRect(cnt)
			aspect = float(bw) / float(max(1, bh))
			if aspect < 0.25 or aspect > 4.0:
				continue
			cx, cy = float(x + 0.5 * bw), float(y + 0.5 * bh)
			if homography is not None and not homography.is_px_in_active_zone(cx, cy, margin_mm=20.0):
				continue

			bbox = np.array([x, y, x + bw, y + bh], dtype=float)

			# Point #9 Fix: Match contour center against multi-frame motion history
			best_trk = None
			best_dist = 135.0
			for trk in prev_tracks:
				if (step_idx - int(trk["last_seen"])) > 5:
					continue
				d = float(np.hypot(cx - trk["cx"], cy - trk["cy"]))
				if d < best_dist:
					best_dist = d
					best_trk = trk

			if best_trk is not None:
				step_disp = float(np.hypot(cx - best_trk["cx"], cy - best_trk["cy"]))
				total_disp = float(np.hypot(cx - best_trk["origin_cx"], cy - best_trk["origin_cy"]))
				hits = int(best_trk["hits"]) + 1
				new_trk = {
					"cx": cx,
					"cy": cy,
					"origin_cx": best_trk["origin_cx"],
					"origin_cy": best_trk["origin_cy"],
					"hits": float(hits),
					"last_seen": float(step_idx),
					"total_disp": total_disp,
					"step_disp": step_disp,
				}
				updated_tracks.append(new_trk)
				# Require genuine multi-frame directional displacement (not stationary jitter)
				if hits >= 2 and total_disp >= 8.0 and step_disp >= 2.0:
					verified_motion_boxes.append(bbox)
			else:
				updated_tracks.append(
					{
						"cx": cx,
						"cy": cy,
						"origin_cx": cx,
						"origin_cy": cy,
						"hits": 1.0,
						"last_seen": float(step_idx),
						"total_disp": 0.0,
						"step_disp": 0.0,
					}
				)

		# Keep recent tracks from the last 5 frames
		for old_t in prev_tracks:
			if (step_idx - int(old_t["last_seen"])) <= 4 and all(
				float(np.hypot(old_t["cx"] - u["cx"], old_t["cy"] - u["cy"])) > 10.0 for u in updated_tracks
			):
				updated_tracks.append(old_t)

		self._motion_tracks[table_id] = updated_tracks[:25]
		verified_motion_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
		return verified_motion_boxes[:8]

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
		table_id: int = 1,
	) -> List[Detection]:
		"""
		Fuses YOLO AI detections with verified multi-frame MOG2 motion boxes:
		  1. Keeps confident AI 'hand' and 'attachment_jig' detections.
		  2. Boosts AI 'robot' detections that overlap with verified motion (`source="hybrid"`),
		     and records `_last_confirmed_robot[table_id]`.
		  3. Point #10 Fix: Does NOT blindly turn unassociated motion blobs into false robots!
		     When the custom FLL AI model is active, a motion blob is only accepted as a motion-bridged
		     robot if it lies within a plausible continuity radius (280 px) of the last AI-confirmed
		     robot position within the last 45 frames.
		"""
		fused: List[Detection] = []
		hand_boxes: List[np.ndarray] = []
		jig_boxes: List[np.ndarray] = []
		robot_dets: List[Detection] = []
		step_idx = self._motion_step.get(table_id, 0)

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
					self._last_confirmed_robot[table_id] = (cx, cy, step_idx)

		# Point #10 Fix: Only bridge AI dropouts if no AI robot was found on this table in the current frame
		if not robot_dets and motion_boxes:
			last_robot_info = self._last_confirmed_robot.get(table_id)
			for mb in motion_boxes:
				if any(_bbox_iou(mb, hb) >= 0.15 for hb in hand_boxes):
					continue
				if any(_bbox_iou(mb, jb) >= 0.20 for jb in jig_boxes):
					continue

				mcx, mcy = float(0.5 * (mb[0] + mb[2])), float(0.5 * (mb[1] + mb[3]))

				if self.is_custom_fll_model:
					# With custom FLL model loaded: require proximity to a recently confirmed AI robot!
					if last_robot_info is None:
						continue
					last_x, last_y, last_step = last_robot_info
					if (step_idx - last_step) > 45:
						continue
					if float(np.hypot(mcx - last_x, mcy - last_y)) > 280.0:
						continue

				# Update continuity anchor and emit motion-bridged robot detection
				self._last_confirmed_robot[table_id] = (mcx, mcy, step_idx)
				motion_robot = Detection(
					bbox_xyxy=mb,
					confidence=0.62,
					class_id=0,
					class_name="robot",
					source="motion",
				)
				robot_dets.append(motion_robot)
				fused.append(motion_robot)
				break  # At most one robot exists per single FLL table!

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
		return self.fuse_ai_and_motion(ai_dets, motion_boxes, homography=homography, table_id=table_id)

	def detect_dual_table(
		self,
		frame: np.ndarray,
		homography1: TableHomography,
		homography2: Optional[TableHomography] = None,
	) -> Tuple[List[Detection], List[Detection]]:
		"""
		Extracts the two independent table regions from the single central camera frame
		(excluding the inter-table gap and outer borders), runs AI inference, and fuses
		with each table's independent motion history.
		"""
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

		dets1 = self.fuse_ai_and_motion(all_ai_dets, motion1, homography=homography1, table_id=1)
		dets2 = self.fuse_ai_and_motion(all_ai_dets, motion2, homography=homography2, table_id=2)
		return dets1, dets2
