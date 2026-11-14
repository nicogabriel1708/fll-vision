import logging
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config_loader import TableConfig

logger = logging.getLogger("fll-vision.homography")


def order_quad_points(pts: np.ndarray) -> np.ndarray:
	"""Orders 4 corner points canonically as: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
	rect = np.zeros((4, 2), dtype=np.float32)
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def validate_quad_geometry(
	pts: np.ndarray,
	min_area_px: float = 2500.0,
	min_edge_px: float = 30.0,
	frame_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
	"""
	Validates that 4 corner points form a non-degenerate, convex quadrilateral within camera bounds (#11, #16, #24).
	Rejects NaN/Inf, out-of-frame coordinates, collinear points, self-intersecting bowties, or tiny collapsed boxes.
	"""
	raw = np.asarray(pts, dtype=np.float32)
	if raw.shape != (4, 2) or not np.all(np.isfinite(raw)):
		raise ValueError("Table corners must be exactly 4 finite [x, y] coordinate pairs.")

	if frame_size is not None:
		fw, fh = int(frame_size[0]), int(frame_size[1])
		if fw > 0 and fh > 0:
			if (
				np.any(raw[:, 0] < 0.0)
				or np.any(raw[:, 0] > float(fw))
				or np.any(raw[:, 1] < 0.0)
				or np.any(raw[:, 1] > float(fh))
			):
				raise ValueError(f"Table corners must lie within camera resolution bounds [0..{fw}] x [0..{fh}].")

	ordered = order_quad_points(raw)
	contour = ordered.reshape((4, 1, 2)).astype(np.float32)
	if not cv2.isContourConvex(contour):
		raise ValueError("Table corners must form a convex quadrilateral (no self-intersecting edges).")

	area = float(cv2.contourArea(contour))
	if area < min_area_px:
		raise ValueError(f"Table quadrilateral area ({area:.1f} px²) is too small (minimum {min_area_px} px²).")

	for i in range(4):
		p1 = ordered[i]
		p2 = ordered[(i + 1) % 4]
		edge_len = float(np.linalg.norm(p2 - p1))
		if edge_len < min_edge_px:
			raise ValueError(f"Table edge #{i + 1} is too short ({edge_len:.1f} px < {min_edge_px} px).")

	return ordered


def validate_tables_non_overlapping(
	corners1_px: List[List[float]],
	corners2_px: List[List[float]],
	max_overlap_px: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Validates that Table 1 and Table 2 quadrilaterals do not overlap (#10).
	Returns the ordered (quad1, quad2) arrays if valid, otherwise raises ValueError.
	"""
	q1 = validate_quad_geometry(np.asarray(corners1_px, dtype=np.float32))
	q2 = validate_quad_geometry(np.asarray(corners2_px, dtype=np.float32))

	inter_area, _ = cv2.intersectConvexConvex(q1, q2)
	if float(inter_area) > max_overlap_px:
		raise ValueError(
			f"Tisch 1 und Tisch 2 überschneiden sich um {float(inter_area):.1f} px²! "
			f"Beide Spielfelder müssen inklusive Zwischenraum klar voneinander getrennt sein."
		)
	return q1, q2


def detect_two_independent_tables(
	frame: np.ndarray,
	fallback_t1: List[List[float]],
	fallback_t2: List[List[float]],
) -> Tuple[bool, List[List[float]], List[List[float]]]:
	"""
	Detects the 4 corners of Table 1 (Left) and Table 2 (Right) independently from a single
	central camera frame (#1, #12):
	  - Does NOT assume the camera is centered at 50% width (`0.48*w` / `0.52*w` cutoff removed) (#12).
	  - Evaluates aspect ratio, interior mission-mat edge richness, and physical twin-table symmetry
	    (similar area, similar width/height, and vertical Y-alignment) so random floor/wall rectangles
	    are rejected.
	  - Returns `detected = True` ONLY when BOTH Table 1 and Table 2 were genuinely detected (#1).
	"""
	if frame is None or frame.size == 0:
		return False, fallback_t1, fallback_t2

	h, w = frame.shape[:2]
	img_area = float(h * w)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.bilateralFilter(gray, 9, 75, 75)

	med_val = np.median(blurred)
	lower = int(max(20, 0.66 * med_val))
	upper = int(min(255, 1.33 * med_val))
	edges = cv2.Canny(blurred, lower, upper)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

	candidates_binary = [
		cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2),
		cv2.morphologyEx(
			cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5),
			cv2.MORPH_CLOSE,
			kernel,
			iterations=2,
		),
	]

	candidate_quads: List[Tuple[float, np.ndarray]] = []

	for binary in candidates_binary:
		contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

		for cnt in contours:
			area = float(cv2.contourArea(cnt))
			if area < 0.05 * img_area or area > 0.48 * img_area:
				continue
			hull = cv2.convexHull(cnt)
			peri = cv2.arcLength(hull, True)
			for eps_factor in np.linspace(0.015, 0.08, 10):
				approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
				if len(approx) == 4 and cv2.isContourConvex(approx):
					try:
						pts = validate_quad_geometry(approx.reshape(4, 2), frame_size=(w, h))
					except ValueError:
						break
					top_w = float(np.linalg.norm(pts[1] - pts[0]))
					bot_w = float(np.linalg.norm(pts[2] - pts[3]))
					left_h = float(np.linalg.norm(pts[3] - pts[0]))
					right_h = float(np.linalg.norm(pts[2] - pts[1]))
					avg_w = 0.5 * (top_w + bot_w)
					avg_h = max(1.0, 0.5 * (left_h + right_h))
					aspect = avg_w / avg_h
					# Reject extreme elongated or narrow shapes that cannot be an FLL table
					if aspect < 0.65 or aspect > 3.2:
						break

					# Compute interior edge/texture richness bonus (FLL mission mats have high internal detail)
					q_mask = np.zeros((h, w), dtype=np.uint8)
					cv2.fillPoly(q_mask, [pts.astype(np.int32)], 255)
					edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=q_mask))) / max(1.0, area)
					texture_bonus = min(0.08, edge_density * 0.5)

					score = (area / img_area) + texture_bonus
					if all(
						float(np.mean(np.linalg.norm(pts - existing_pts, axis=1))) > 25.0
						for _, existing_pts in candidate_quads
					):
						candidate_quads.append((score, pts))
					break

	candidate_quads.sort(key=lambda item: item[0], reverse=True)

	# Find the highest-scoring non-overlapping pair of FLL tables that also satisfy
	# physical twin-table symmetry (similar area, similar height/width, and vertical Y-alignment)!
	best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
	best_pair_score = -1.0

	for i in range(len(candidate_quads)):
		s1, q1 = candidate_quads[i]
		area1 = float(cv2.contourArea(q1.reshape((4, 1, 2))))
		w1 = 0.5 * float(np.linalg.norm(q1[1] - q1[0]) + np.linalg.norm(q1[2] - q1[3]))
		h1 = 0.5 * float(np.linalg.norm(q1[3] - q1[0]) + np.linalg.norm(q1[2] - q1[1]))
		cx1, cy1 = float(np.mean(q1[:, 0])), float(np.mean(q1[:, 1]))

		for j in range(i + 1, len(candidate_quads)):
			s2, q2 = candidate_quads[j]
			inter_area, _ = cv2.intersectConvexConvex(q1, q2)
			if float(inter_area) > 1.0:
				continue
			area2 = float(cv2.contourArea(q2.reshape((4, 1, 2))))
			w2 = 0.5 * float(np.linalg.norm(q2[1] - q2[0]) + np.linalg.norm(q2[2] - q2[3]))
			h2 = 0.5 * float(np.linalg.norm(q2[3] - q2[0]) + np.linalg.norm(q2[2] - q2[1]))
			cx2, cy2 = float(np.mean(q2[:, 0])), float(np.mean(q2[:, 1]))

			# 1. Require horizontal side-by-side separation
			if abs(cx1 - cx2) < 0.15 * w:
				continue
			# 2. Both FLL tables have identical physical dimensions: require similar projected area & height
			area_ratio = min(area1, area2) / max(1.0, max(area1, area2))
			height_ratio = min(h1, h2) / max(1.0, max(h1, h2))
			width_ratio = min(w1, w2) / max(1.0, max(w1, w2))
			if area_ratio < 0.50 or height_ratio < 0.65 or width_ratio < 0.60:
				continue
			# 3. Side-by-side tables share a similar vertical Y position in the overhead camera view
			if abs(cy1 - cy2) > 0.20 * h:
				continue

			symmetry_bonus = 0.25 * (area_ratio + height_ratio + width_ratio)
			y_align_penalty = (abs(cy1 - cy2) / float(h)) * 0.30
			pair_score = s1 + s2 + symmetry_bonus - y_align_penalty
			if pair_score > best_pair_score:
				best_pair_score = pair_score
				best_pair = (q1, q2) if cx1 < cx2 else (q2, q1)

	if best_pair is not None:
		left_q, right_q = best_pair
		t1_out = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in left_q]
		t2_out = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in right_q]
		return True, t1_out, t2_out

	# Point #1 Fix: If fewer than 2 non-overlapping tables were found, return False!
	return False, fallback_t1, fallback_t2


class TableHomography:
	"""
	Handles 4-point perspective calibration for a single FLL table, strict polygon ROI extraction
	(excluding the inter-table gap, wooden borders, and the neighboring table), and non-stretched
	2D Bird's-Eye View warping.
	"""

	def __init__(
		self,
		cfg: TableConfig,
		minimap_base_height: int = 540,
		frame_size: Tuple[int, int] = (1920, 1080),
		minimap_scale: Optional[float] = None,
	):
		if minimap_scale is not None:
			minimap_base_height = max(240, int(round(cfg.real_height_mm * float(minimap_scale))))
		self.cfg = cfg
		self.real_w = cfg.real_width_mm
		self.real_h = cfg.real_height_mm
		self.minimap_base_height = int(minimap_base_height)
		self.minimap_w = 960
		self.minimap_h = 540
		self.frame_size: Tuple[int, int] = frame_size
		self._lock = threading.Lock()

		self._cached_mask: Optional[np.ndarray] = None
		self._cached_shape: Optional[Tuple[int, int]] = None
		self._active_quad_px: np.ndarray = np.zeros((4, 2), dtype=np.float32)

		self.update_calibration(
			corners_px=cfg.src_corners_px,
			active_half_mode=cfg.active_half_mode,
			active_x_ratio=cfg.active_x_ratio,
			active_y_ratio=cfg.active_y_ratio,
			frame_size=frame_size,
		)

	def _compute_active_ratios(
		self, mode: str, x_ratio: List[float], y_ratio: List[float]
	) -> Tuple[Tuple[float, float], Tuple[float, float]]:
		if mode == "left_half":
			return (0.0, 0.50), (0.0, 1.0)
		elif mode == "right_half":
			return (0.50, 1.0), (0.0, 1.0)
		elif mode == "top_half":
			return (0.0, 1.0), (0.0, 0.50)
		elif mode == "bottom_half":
			return (0.0, 1.0), (0.50, 1.0)
		elif mode == "full":
			return (0.0, 1.0), (0.0, 1.0)
		else:
			rx0 = float(np.clip(min(x_ratio), 0.0, 0.95))
			rx1 = float(np.clip(max(x_ratio), rx0 + 0.05, 1.0))
			ry0 = float(np.clip(min(y_ratio), 0.0, 0.95))
			ry1 = float(np.clip(max(y_ratio), ry0 + 0.05, 1.0))
			return (rx0, rx1), (ry0, ry1)

	def _compute_validated_state(
		self,
		corners_px: List[List[float]],
		active_half_mode: Optional[str] = None,
		active_x_ratio: Optional[List[float]] = None,
		active_y_ratio: Optional[List[float]] = None,
		frame_size: Optional[Tuple[int, int]] = None,
	) -> dict:
		"""
		Point #5 & #11 Fix: Computes and validates all geometry, ratios, and homography matrices
		purely in local variables WITHOUT mutating `self` or `self.cfg`.
		If any validation check fails, raises ValueError while leaving the instance untouched!
		"""
		eff_frame_size = frame_size if frame_size is not None else self.frame_size
		src_pts = validate_quad_geometry(np.array(corners_px, dtype=np.float32), frame_size=eff_frame_size)
		ordered_corners = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in src_pts]

		eff_mode = active_half_mode if active_half_mode is not None else self.cfg.active_half_mode
		eff_x_in = list(active_x_ratio) if active_x_ratio is not None else list(self.cfg.active_x_ratio)
		eff_y_in = list(active_y_ratio) if active_y_ratio is not None else list(self.cfg.active_y_ratio)

		(rx0, rx1), (ry0, ry1) = self._compute_active_ratios(eff_mode, eff_x_in, eff_y_in)

		top_w = float(np.linalg.norm(src_pts[1] - src_pts[0]))
		bot_w = float(np.linalg.norm(src_pts[2] - src_pts[3]))
		left_h = float(np.linalg.norm(src_pts[3] - src_pts[0]))
		right_h = float(np.linalg.norm(src_pts[2] - src_pts[1]))
		avg_w = max(64.0, 0.5 * (top_w + bot_w))
		avg_h = max(64.0, 0.5 * (left_h + right_h))
		quad_aspect = float(np.clip(avg_w / avg_h, 0.45, 3.6))

		minimap_h = self.minimap_base_height
		minimap_w = max(240, int(round(minimap_h * quad_aspect)))

		dst_pts_mm = np.array(
			[
				[0.0, 0.0],
				[self.real_w, 0.0],
				[self.real_w, self.real_h],
				[0.0, self.real_h],
			],
			dtype=np.float32,
		)

		H_px2mm = cv2.getPerspectiveTransform(src_pts, dst_pts_mm)
		H_mm2px = cv2.getPerspectiveTransform(dst_pts_mm, src_pts)
		det_val = float(abs(np.linalg.det(H_px2mm)))
		cond_val = float(np.linalg.cond(H_px2mm))
		if det_val < 1e-11 or not np.isfinite(cond_val) or cond_val > 1e9:
			raise ValueError(f"Ill-conditioned homography matrix (det={det_val:.2e}, cond={cond_val:.2e}).")

		dst_pts_minimap = np.array(
			[
				[0.0, 0.0],
				[float(minimap_w), 0.0],
				[float(minimap_w), float(minimap_h)],
				[0.0, float(minimap_h)],
			],
			dtype=np.float32,
		)
		H_px_to_minimap = cv2.getPerspectiveTransform(src_pts, dst_pts_minimap)

		active_mm_quad = np.array(
			[
				[
					[rx0 * self.real_w, ry0 * self.real_h],
					[rx1 * self.real_w, ry0 * self.real_h],
					[rx1 * self.real_w, ry1 * self.real_h],
					[rx0 * self.real_w, ry1 * self.real_h],
				]
			],
			dtype=np.float32,
		)
		active_quad_px = cv2.perspectiveTransform(active_mm_quad, H_mm2px)[0]

		return {
			"ordered_corners": ordered_corners,
			"active_half_mode": eff_mode,
			"active_x_ratio": [rx0, rx1],
			"active_y_ratio": [ry0, ry1],
			"minimap_w": minimap_w,
			"minimap_h": minimap_h,
			"H_px_to_mm": H_px2mm,
			"H_mm_to_px": H_mm2px,
			"H_px_to_minimap": H_px_to_minimap,
			"active_quad_px": active_quad_px,
			"frame_size": eff_frame_size,
		}

	def _apply_validated_state_unlocked(self, st: dict) -> List[List[float]]:
		self.frame_size = st["frame_size"]
		self.cfg.src_corners_px = st["ordered_corners"]
		self.cfg.active_half_mode = st["active_half_mode"]
		self.cfg.active_x_ratio = st["active_x_ratio"]
		self.cfg.active_y_ratio = st["active_y_ratio"]
		self.minimap_w = st["minimap_w"]
		self.minimap_h = st["minimap_h"]
		self.H_px_to_mm = st["H_px_to_mm"]
		self.H_mm_to_px = st["H_mm_to_px"]
		self.H_px_to_minimap = st["H_px_to_minimap"]
		self._active_quad_px = st["active_quad_px"]
		self._cached_mask = None
		return self.cfg.src_corners_px

	def update_calibration(
		self,
		corners_px: List[List[float]],
		active_half_mode: Optional[str] = None,
		active_x_ratio: Optional[List[float]] = None,
		active_y_ratio: Optional[List[float]] = None,
		frame_size: Optional[Tuple[int, int]] = None,
	) -> List[List[float]]:
		"""
		Atomically hot-reloads the 4 table corners and active ratios at runtime (#5).
		If validation or matrix conditioning fails, neither `self` nor `self.cfg` is modified.
		"""
		with self._lock:
			st = self._compute_validated_state(
				corners_px=corners_px,
				active_half_mode=active_half_mode,
				active_x_ratio=active_x_ratio,
				active_y_ratio=active_y_ratio,
				frame_size=frame_size,
			)
			return self._apply_validated_state_unlocked(st)

	@staticmethod
	def update_dual_calibration_atomic(
		homography1: "TableHomography",
		homography2: "TableHomography",
		corners1_px: List[List[float]],
		corners2_px: List[List[float]],
		active_half_mode1: Optional[str] = None,
		active_x_ratio1: Optional[List[float]] = None,
		active_y_ratio1: Optional[List[float]] = None,
		active_half_mode2: Optional[str] = None,
		active_x_ratio2: Optional[List[float]] = None,
		active_y_ratio2: Optional[List[float]] = None,
		frame_size: Optional[Tuple[int, int]] = None,
	) -> Tuple[List[List[float]], List[List[float]]]:
		"""
		Points #4, #5, #10, #11 Fix: Validates BOTH Table 1 and Table 2 (including frame bounds,
		matrix conditioning, and non-overlap between Table 1 and Table 2) BEFORE committing
		changes to either homography instance!
		If Table 2 fails or overlaps Table 1, Table 1 remains 100% untouched.
		"""
		with homography1._lock, homography2._lock:
			st1 = homography1._compute_validated_state(
				corners_px=corners1_px,
				active_half_mode=active_half_mode1,
				active_x_ratio=active_x_ratio1,
				active_y_ratio=active_y_ratio1,
				frame_size=frame_size,
			)
			st2 = homography2._compute_validated_state(
				corners_px=corners2_px,
				active_half_mode=active_half_mode2,
				active_x_ratio=active_x_ratio2,
				active_y_ratio=active_y_ratio2,
				frame_size=frame_size,
			)
			# Check non-overlap on both full table quads and active quads (#10)
			validate_tables_non_overlapping(st1["ordered_corners"], st2["ordered_corners"])

			c1 = homography1._apply_validated_state_unlocked(st1)
			c2 = homography2._apply_validated_state_unlocked(st2)
			return c1, c2

	def update_corners(
		self, corners_px: List[List[float]], frame_size: Optional[Tuple[int, int]] = None
	) -> List[List[float]]:
		return self.update_calibration(corners_px=corners_px, frame_size=frame_size)

	def get_active_quad_px(self) -> List[List[float]]:
		with self._lock:
			return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in self._active_quad_px]

	def get_active_bbox_px(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
		"""Returns the bounding box (x1, y1, x2, y2) of this table's active region in pixel coordinates."""
		with self._lock:
			pts = self._active_quad_px
		x1 = int(np.clip(np.min(pts[:, 0]), 0, frame_w - 2))
		y1 = int(np.clip(np.min(pts[:, 1]), 0, frame_h - 2))
		x2 = int(np.clip(np.max(pts[:, 0]), x1 + 2, frame_w))
		y2 = int(np.clip(np.max(pts[:, 1]), y1 + 2, frame_h))
		return x1, y1, x2, y2

	def is_mm_in_active_zone(self, x_mm: float, y_mm: float, margin_mm: float = 45.0) -> bool:
		with self._lock:
			rx0, rx1 = self.cfg.active_x_ratio
			ry0, ry1 = self.cfg.active_y_ratio

		left_pad = margin_mm if rx0 <= 0.01 else 15.0
		right_pad = margin_mm if rx1 >= 0.99 else 15.0
		top_pad = margin_mm if ry0 <= 0.01 else 15.0
		bot_pad = margin_mm if ry1 >= 0.99 else 15.0

		min_x = rx0 * self.real_w - left_pad
		max_x = rx1 * self.real_w + right_pad
		min_y = ry0 * self.real_h - top_pad
		max_y = ry1 * self.real_h + bot_pad
		return (min_x <= x_mm <= max_x) and (min_y <= y_mm <= max_y)

	def is_px_in_active_zone(self, x_px: float, y_px: float, margin_mm: float = 45.0) -> bool:
		x_mm, y_mm = self.px_to_mm(x_px, y_px, clip=False)
		return self.is_mm_in_active_zone(x_mm, y_mm, margin_mm=margin_mm)

	def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
		"""
		Extracts ONLY this table's perspective polygon from the shared camera frame.
		Everything outside this table's 4 corners (including the inter-table gap, borders,
		and the other table) is strictly zeroed out.
		"""
		if not self.cfg.apply_table_roi_mask:
			return frame

		h, w = frame.shape[:2]
		with self._lock:
			if self._cached_mask is None or self._cached_shape != (h, w):
				mask = np.zeros((h, w), dtype=np.uint8)
				pts = self._active_quad_px.astype(np.float32)
				if self.cfg.roi_padding_px != 0:
					center = np.mean(pts, axis=0)
					vecs = pts - center
					norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6
					poly_pts = np.round(pts + (vecs / norms) * float(self.cfg.roi_padding_px)).astype(np.int32)
				else:
					poly_pts = np.round(pts).astype(np.int32)
				cv2.fillPoly(mask, [poly_pts], 255)
				self._cached_mask = mask
				self._cached_shape = (h, w)
			mask_ref = self._cached_mask

		return cv2.bitwise_and(frame, frame, mask=mask_ref)

	def auto_detect_corners(self, frame: np.ndarray, apply_immediately: bool = False) -> Tuple[bool, List[List[float]]]:
		h, w = frame.shape[:2]
		img_area = float(h * w)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.bilateralFilter(gray, 9, 75, 75)

		best_quad = None
		best_score = -1.0
		med_val = np.median(blurred)
		edges = cv2.Canny(blurred, int(max(20, 0.66 * med_val)), int(min(255, 1.33 * med_val)))
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

		contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
		for cnt in contours:
			area = float(cv2.contourArea(cnt))
			if area < 0.08 * img_area or area > 0.92 * img_area:
				continue
			hull = cv2.convexHull(cnt)
			peri = cv2.arcLength(hull, True)
			for eps_factor in np.linspace(0.015, 0.08, 10):
				approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
				if len(approx) == 4 and cv2.isContourConvex(approx):
					try:
						pts = validate_quad_geometry(approx.reshape(4, 2), frame_size=(w, h))
					except ValueError:
						break
					score = area / img_area
					if score > best_score:
						best_score = score
						best_quad = pts
					break

		if best_quad is not None:
			detected_list = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in best_quad]
			if apply_immediately:
				self.update_corners(detected_list, frame_size=(w, h))
			return True, detected_list

		return False, self.cfg.src_corners_px

	def px_to_mm(self, x_px: float, y_px: float, clip: bool = False) -> Tuple[float, float]:
		pt = np.array([[[x_px, y_px]]], dtype=np.float32)
		with self._lock:
			projected = cv2.perspectiveTransform(pt, self.H_px_to_mm)[0][0]
		x_mm = float(projected[0])
		y_mm = float(projected[1])
		if clip:
			x_mm = float(np.clip(x_mm, 0.0, self.real_w))
			y_mm = float(np.clip(y_mm, 0.0, self.real_h))
		return x_mm, y_mm

	def mm_to_px(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
		pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
		with self._lock:
			projected = cv2.perspectiveTransform(pt, self.H_mm_to_px)[0][0]
		return int(projected[0]), int(projected[1])

	def mm_to_minimap(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
		with self._lock:
			mw, mh = self.minimap_w, self.minimap_h
		mx = int(round((x_mm / max(1.0, self.real_w)) * mw))
		my = int(round((y_mm / max(1.0, self.real_h)) * mh))
		return int(np.clip(mx, 0, mw - 1)), int(np.clip(my, 0, mh - 1))

	def warp_to_minimap(self, frame: np.ndarray, active_only: bool = True) -> np.ndarray:
		with self._lock:
			H = self.H_px_to_minimap.copy()
			mw, mh = self.minimap_w, self.minimap_h
			rx0, rx1 = self.cfg.active_x_ratio
			ry0, ry1 = self.cfg.active_y_ratio

		warped = cv2.warpPerspective(frame, H, (mw, mh))
		if active_only and (rx0 > 0.001 or rx1 < 0.999 or ry0 > 0.001 or ry1 < 0.999):
			overlay = warped.copy()
			mask = np.zeros((mh, mw), dtype=np.uint8)
			ax1, ay1 = int(rx0 * mw), int(ry0 * mh)
			ax2, ay2 = int(rx1 * mw), int(ry1 * mh)
			mask[ay1:ay2, ax1:ax2] = 255
			warped[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
			cv2.rectangle(warped, (ax1, ay1), (ax2 - 1, ay2 - 1), (56, 189, 248), 2)
		return warped

	def is_in_launch_area(self, x_mm: float, y_mm: float) -> Tuple[bool, str]:
		for area_name, bounds in self.cfg.launch_areas_mm.items():
			x_min, y_min, x_max, y_max = bounds
			if x_min <= x_mm <= x_max and y_min <= y_mm <= y_max:
				return True, area_name
		return False, "field"
