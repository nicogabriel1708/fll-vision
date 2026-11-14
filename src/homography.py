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


class TableHomography:
	"""
	Handles 4-point perspective calibration between camera pixels and real-world
	FLL table millimeter coordinates (2362mm x 1143mm), automatic FLL table
	border detection, and ACTIVE HALF-TABLE ROI MASKING.

	CRITICAL REQUIREMENT:
	In setups where the camera views the entire table (two halves side-by-side or one full table),
	only the active half containing the mission mat is used. The unused half (which may contain
	hands, water bottles, laptops, or spare parts) and everything outside the table are strictly
	masked out in both training and live inference.
	"""

	def __init__(self, cfg: TableConfig, minimap_scale: float = 0.4):
		self.cfg = cfg
		self.real_w = cfg.real_width_mm
		self.real_h = cfg.real_height_mm
		self.minimap_w = int(self.real_w * minimap_scale)
		self.minimap_h = int(self.real_h * minimap_scale)
		self.minimap_scale = minimap_scale
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
		Hot-reloads the 4 table corners and active half-table ratios at runtime.
		Recomputes homography matrices and the active half-table polygon in pixel space.
		"""
		with self._lock:
			raw_pts = np.array(corners_px, dtype=np.float32)
			if raw_pts.shape != (4, 2):
				raise ValueError("corners_px must contain exactly 4 [x, y] points.")
			src_pts = order_quad_points(raw_pts)
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

			dst_pts_mm = np.array(
				[
					[0.0, 0.0],
					[self.real_w, 0.0],
					[self.real_w, self.real_h],
					[0.0, self.real_h],
				],
				dtype=np.float32,
			)

			self.H_px_to_mm = cv2.getPerspectiveTransform(src_pts, dst_pts_mm)
			self.H_mm_to_px = cv2.getPerspectiveTransform(dst_pts_mm, src_pts)

			dst_pts_minimap = dst_pts_mm * self.minimap_scale
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
		"""Returns the 4 corners of the active table region in camera pixel coordinates."""
		with self._lock:
			return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in self._active_quad_px]

	def get_active_bbox_px(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
		"""Returns the bounding box (x1, y1, x2, y2) of the active half-table in pixel coordinates."""
		with self._lock:
			pts = self._active_quad_px
		x1 = int(np.clip(np.min(pts[:, 0]), 0, frame_w - 1))
		y1 = int(np.clip(np.min(pts[:, 1]), 0, frame_h - 1))
		x2 = int(np.clip(np.max(pts[:, 0]), x1 + 1, frame_w))
		y2 = int(np.clip(np.max(pts[:, 1]), y1 + 1, frame_h))
		return x1, y1, x2, y2

	def is_mm_in_active_zone(self, x_mm: float, y_mm: float) -> bool:
		"""Checks if a millimeter coordinate lies inside the configured active half of the table."""
		with self._lock:
			rx0, rx1 = self.cfg.active_x_ratio
			ry0, ry1 = self.cfg.active_y_ratio
		return (rx0 * self.real_w <= x_mm <= rx1 * self.real_w) and (ry0 * self.real_h <= y_mm <= ry1 * self.real_h)

	def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
		"""
		Strictly masks out everything outside the ACTIVE half of the FLL table.
		The unused half of the table (bottles, hands, laptops) and surrounding room are blacked out.
		"""
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

	def auto_detect_corners(self, frame: np.ndarray, apply_immediately: bool = True) -> Tuple[bool, List[List[float]]]:
		"""Automatically detects the 4 outer corners of the FLL table from a camera frame."""
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
				if area < 0.20 * img_area or area > 0.94 * img_area:
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
						if 1.15 <= aspect <= 3.6:
							aspect_penalty = abs(aspect - 2.06) * 0.15
							score = (area / img_area) - aspect_penalty
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

	def px_to_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
		pt = np.array([[[x_px, y_px]]], dtype=np.float32)
		with self._lock:
			projected = cv2.perspectiveTransform(pt, self.H_px_to_mm)[0][0]
		x_mm = float(np.clip(projected[0], 0.0, self.real_w))
		y_mm = float(np.clip(projected[1], 0.0, self.real_h))
		return x_mm, y_mm

	def mm_to_px(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
		pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
		with self._lock:
			projected = cv2.perspectiveTransform(pt, self.H_mm_to_px)[0][0]
		return int(projected[0]), int(projected[1])

	def mm_to_minimap(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
		return int(x_mm * self.minimap_scale), int(y_mm * self.minimap_scale)

	def warp_to_minimap(self, frame: np.ndarray, active_only: bool = True) -> np.ndarray:
		"""
		Warps the camera view into a top-down 2D Bird's-Eye View of the FLL mat.
		If active_only is True, visually shades the excluded table half so operators see the active zone clearly.
		"""
		with self._lock:
			H = self.H_px_to_minimap.copy()
			rx0, rx1 = self.cfg.active_x_ratio
			ry0, ry1 = self.cfg.active_y_ratio

		warped = cv2.warpPerspective(frame, H, (self.minimap_w, self.minimap_h))
		if active_only and (rx0 > 0.001 or rx1 < 0.999 or ry0 > 0.001 or ry1 < 0.999):
			overlay = warped.copy()
			# Darken unused portion of the table on the minimap
			mask = np.zeros((self.minimap_h, self.minimap_w), dtype=np.uint8)
			ax1, ay1 = int(rx0 * self.minimap_w), int(ry0 * self.minimap_h)
			ax2, ay2 = int(rx1 * self.minimap_w), int(ry1 * self.minimap_h)
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
