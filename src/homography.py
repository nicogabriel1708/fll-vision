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


def validate_quad_geometry(pts: np.ndarray, min_area_px: float = 2500.0, min_edge_px: float = 30.0) -> np.ndarray:
	"""
	Validates that 4 corner points form a non-degenerate, convex quadrilateral (#16, #24).
	Rejects NaN/Inf, collinear points, self-intersecting bowties, or tiny collapsed boxes.
	"""
	raw = np.asarray(pts, dtype=np.float32)
	if raw.shape != (4, 2) or not np.all(np.isfinite(raw)):
		raise ValueError("Table corners must be exactly 4 finite [x, y] coordinate pairs.")

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


def split_double_table_into_two_tables(
	double_corners_px: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
	"""
	Given the 4 outer corners of a Double-Table (2 FLL tables side-by-side),
	splits them along the center line into:
	  - table1_corners (Left Table / Red Team)
	  - table2_corners (Right Table / Blue Team)
	"""
	pts = order_quad_points(np.array(double_corners_px, dtype=np.float32))
	tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
	mid_top = 0.5 * (tl + tr)
	mid_bot = 0.5 * (bl + br)

	t1 = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in [tl, mid_top, mid_bot, bl]]
	t2 = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in [mid_top, tr, br, mid_bot]]
	return t1, t2


class TableHomography:
	"""
	Handles 4-point perspective calibration for a single FLL table (or half-table),
	automatic border detection, active zone masking, and aspect-ratio-preserving
	2D Bird's-Eye View Minimap warping (never stretches the image!).
	"""

	def __init__(self, cfg: TableConfig, minimap_base_height: int = 540):
		self.cfg = cfg
		self.real_w = cfg.real_width_mm
		self.real_h = cfg.real_height_mm
		self.minimap_base_height = int(minimap_base_height)
		self.minimap_w = 960
		self.minimap_h = 540
		self._lock = threading.Lock()

		self._cached_mask: Optional[np.ndarray] = None
		self._cached_shape: Optional[Tuple[int, int]] = None
		self._active_quad_px: np.ndarray = np.zeros((4, 2), dtype=np.float32)

		self.update_calibration(
			corners_px=cfg.src_corners_px,
			active_half_mode=cfg.active_half_mode,
			active_x_ratio=cfg.active_x_ratio,
			active_y_ratio=cfg.active_y_ratio,
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

	def update_calibration(
		self,
		corners_px: List[List[float]],
		active_half_mode: Optional[str] = None,
		active_x_ratio: Optional[List[float]] = None,
		active_y_ratio: Optional[List[float]] = None,
	) -> List[List[float]]:
		"""
		Hot-reloads the 4 table corners and active ratios at runtime.
		Computes a non-stretched minimap size whose aspect ratio matches the selected table quad.
		"""
		with self._lock:
			src_pts = validate_quad_geometry(np.array(corners_px, dtype=np.float32))
			self.cfg.src_corners_px = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in src_pts]

			if active_half_mode is not None:
				self.cfg.active_half_mode = active_half_mode
			if active_x_ratio is not None:
				self.cfg.active_x_ratio = list(active_x_ratio)
			if active_y_ratio is not None:
				self.cfg.active_y_ratio = list(active_y_ratio)

			(rx0, rx1), (ry0, ry1) = self._compute_active_ratios(
				self.cfg.active_half_mode, self.cfg.active_x_ratio, self.cfg.active_y_ratio
			)
			self.cfg.active_x_ratio = [rx0, rx1]
			self.cfg.active_y_ratio = [ry0, ry1]

			# Compute true visual aspect ratio of the selected table quad so the 2D minimap is NEVER stretched!
			top_w = float(np.linalg.norm(src_pts[1] - src_pts[0]))
			bot_w = float(np.linalg.norm(src_pts[2] - src_pts[3]))
			left_h = float(np.linalg.norm(src_pts[3] - src_pts[0]))
			right_h = float(np.linalg.norm(src_pts[2] - src_pts[1]))
			avg_w = max(64.0, 0.5 * (top_w + bot_w))
			avg_h = max(64.0, 0.5 * (left_h + right_h))
			quad_aspect = float(np.clip(avg_w / avg_h, 0.45, 3.6))

			self.minimap_h = self.minimap_base_height
			self.minimap_w = max(240, int(round(self.minimap_h * quad_aspect)))

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
			self.H_px_to_mm = H_px2mm
			self.H_mm_to_px = H_mm2px

			dst_pts_minimap = np.array(
				[
					[0.0, 0.0],
					[float(self.minimap_w), 0.0],
					[float(self.minimap_w), float(self.minimap_h)],
					[0.0, float(self.minimap_h)],
				],
				dtype=np.float32,
			)
			self.H_px_to_minimap = cv2.getPerspectiveTransform(src_pts, dst_pts_minimap)

			# Project the active table sub-rectangle from millimeter space back into camera pixel space
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
			self._active_quad_px = cv2.perspectiveTransform(active_mm_quad, self.H_mm_to_px)[0]

			self._cached_mask = None
			return self.cfg.src_corners_px

	def update_corners(self, corners_px: List[List[float]]) -> List[List[float]]:
		return self.update_calibration(corners_px=corners_px)

	def get_active_quad_px(self) -> List[List[float]]:
		with self._lock:
			return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in self._active_quad_px]

	def get_active_bbox_px(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
		"""Returns the bounding box (x1, y1, x2, y2) of the active table region in pixel coordinates."""
		with self._lock:
			pts = self._active_quad_px
		x1 = int(np.clip(np.min(pts[:, 0]), 0, frame_w - 2))
		y1 = int(np.clip(np.min(pts[:, 1]), 0, frame_h - 2))
		x2 = int(np.clip(np.max(pts[:, 0]), x1 + 2, frame_w))
		y2 = int(np.clip(np.max(pts[:, 1]), y1 + 2, frame_h))
		return x1, y1, x2, y2

	def is_mm_in_active_zone(self, x_mm: float, y_mm: float, margin_mm: float = 110.0) -> bool:
		"""
		Checks if a millimeter coordinate lies inside the active table zone.
		Includes a default `margin_mm` (110 mm) tolerance on outer table edges so tall 3D robots
		driving right along the walls (whose 2D bbox center projects slightly outside the border)
		are never excluded, while keeping a strict split on an internal half-table boundary.
		"""
		with self._lock:
			rx0, rx1 = self.cfg.active_x_ratio
			ry0, ry1 = self.cfg.active_y_ratio

		# Apply margin on outer table edges (0.0 or 1.0), keep internal split line (e.g. 0.50) exact
		left_pad = margin_mm if rx0 <= 0.01 else 25.0
		right_pad = margin_mm if rx1 >= 0.99 else 25.0
		top_pad = margin_mm if ry0 <= 0.01 else 25.0
		bot_pad = margin_mm if ry1 >= 0.99 else 25.0

		min_x = rx0 * self.real_w - left_pad
		max_x = rx1 * self.real_w + right_pad
		min_y = ry0 * self.real_h - top_pad
		max_y = ry1 * self.real_h + bot_pad
		return (min_x <= x_mm <= max_x) and (min_y <= y_mm <= max_y)

	def is_px_in_active_zone(self, x_px: float, y_px: float, margin_mm: float = 110.0) -> bool:
		x_mm, y_mm = self.px_to_mm(x_px, y_px, clip=False)
		return self.is_mm_in_active_zone(x_mm, y_mm, margin_mm=margin_mm)

	def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
		if not self.cfg.apply_table_roi_mask:
			return frame

		h, w = frame.shape[:2]
		with self._lock:
			if self._cached_mask is None or self._cached_shape != (h, w):
				mask = np.zeros((h, w), dtype=np.uint8)
				pts = self._active_quad_px.astype(np.float32)
				center = np.mean(pts, axis=0)
				vecs = pts - center
				norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6
				padded_pts = (pts + (vecs / norms) * self.cfg.roi_padding_px).astype(np.int32)
				cv2.fillPoly(mask, [padded_pts], 255)
				self._cached_mask = mask
				self._cached_shape = (h, w)
			mask_ref = self._cached_mask

		return cv2.bitwise_and(frame, frame, mask=mask_ref)

	def auto_detect_corners(self, frame: np.ndarray, apply_immediately: bool = False) -> Tuple[bool, List[List[float]]]:
		"""Automatically detects the 4 outer corners of the FLL table/double-table from a camera frame."""
		h, w = frame.shape[:2]
		img_area = float(h * w)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.bilateralFilter(gray, 9, 75, 75)

		best_quad = None
		best_score = -1.0
		candidates_binary = []

		med_val = np.median(blurred)
		lower = int(max(20, 0.66 * med_val))
		upper = int(min(255, 1.33 * med_val))
		edges = cv2.Canny(blurred, lower, upper)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		candidates_binary.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2))

		adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
		candidates_binary.append(cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2))

		_, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		candidates_binary.append(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2))

		for binary in candidates_binary:
			contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

			for cnt in contours:
				area = cv2.contourArea(cnt)
				if area < 0.15 * img_area or area > 0.95 * img_area:
					continue
				hull = cv2.convexHull(cnt)
				peri = cv2.arcLength(hull, True)
				for eps_factor in np.linspace(0.015, 0.08, 12):
					approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
					if len(approx) == 4 and cv2.isContourConvex(approx):
						pts = order_quad_points(approx.reshape(4, 2))
						top_w = np.linalg.norm(pts[1] - pts[0])
						bot_w = np.linalg.norm(pts[2] - pts[3])
						left_h = np.linalg.norm(pts[3] - pts[0])
						right_h = np.linalg.norm(pts[2] - pts[1])
						avg_w = 0.5 * (top_w + bot_w)
						avg_h = 0.5 * (left_h + right_h)
						if avg_h < 50 or avg_w < 100:
							continue
						aspect = avg_w / (avg_h + 1e-6)
						if 0.8 <= aspect <= 4.0:
							score = area / img_area
							if score > best_score:
								best_score = score
								best_quad = pts
						break

		if best_quad is not None:
			detected_list = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in best_quad]
			if apply_immediately:
				self.update_corners(detected_list)
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
		"""
		Warps the camera view into an aspect-ratio-preserving 2D Bird's-Eye View of the table.
		Never stretches the image!
		"""
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
