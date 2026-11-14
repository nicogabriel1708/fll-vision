import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

try:
	from filterpy.kalman import KalmanFilter
except ImportError:

	class KalmanFilter:
		"""Self-contained NumPy Linear Kalman Filter fallback."""

		def __init__(self, dim_x: int, dim_z: int):
			self.dim_x = dim_x
			self.dim_z = dim_z
			self.x = np.zeros((dim_x, 1), dtype=float)
			self.F = np.eye(dim_x, dtype=float)
			self.H = np.zeros((dim_z, dim_x), dtype=float)
			self.P = np.eye(dim_x, dtype=float)
			self.R = np.eye(dim_z, dtype=float)
			self.Q = np.eye(dim_x, dtype=float)

		def predict(self):
			self.x = self.F @ self.x
			self.P = self.F @ self.P @ self.F.T + self.Q

		def update(self, z: np.ndarray):
			y = z - (self.H @ self.x)
			S = self.H @ self.P @ self.H.T + self.R
			K = self.P @ self.H.T @ np.linalg.inv(S)
			self.x = self.x + (K @ y)
			I = np.eye(self.dim_x)
			self.P = (I - K @ self.H) @ self.P


from src.config_loader import TrackerConfig
from src.detector import Detection
from src.homography import TableHomography


class RobotState(str, Enum):
	UNINITIALIZED = "UNINITIALIZED"
	IN_BASE = "IN_BASE"
	AUTONOMOUS = "AUTONOMOUS"
	MANIPULATED_BY_HAND = "MANIPULATED_BY_HAND"
	OCCLUDED = "OCCLUDED"


@dataclass
class TrajectoryPoint:
	timestamp: float
	frame_idx: int
	x_mm: float
	y_mm: float
	vx_mm_s: float
	vy_mm_s: float
	speed_mm_s: float
	state: RobotState
	run_id: int
	table_id: int = 1
	is_predicted: bool = False


@dataclass
class SortieRun:
	run_id: int
	start_time: float
	table_id: int = 1
	end_time: Optional[float] = None
	points: List[TrajectoryPoint] = field(default_factory=list)
	interrupted_by_hand: bool = False
	max_plausible_speed_mm_s: float = 2200.0

	@property
	def duration_s(self) -> float:
		end = self.end_time if self.end_time is not None else time.time()
		return max(0.0, end - self.start_time)

	@property
	def total_distance_mm(self) -> float:
		"""
		Computes total autonomous travel distance in millimeters while rejecting
		unphysical position jumps (Point #20) and sub-millimeter sensor jitter.
		"""
		if len(self.points) < 2:
			return 0.0
		total = 0.0
		prev = self.points[0]
		for curr in self.points[1:]:
			if curr.is_predicted:
				continue
			dt = max(1e-3, float(curr.timestamp - prev.timestamp))
			step_mm = float(np.hypot(curr.x_mm - prev.x_mm, curr.y_mm - prev.y_mm))
			max_step = max(250.0, self.max_plausible_speed_mm_s * dt * 1.5)
			if 2.0 <= step_mm <= max_step:
				total += step_mm
			prev = curr
		return float(total)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	xA = max(float(box_a[0]), float(box_b[0]))
	yA = max(float(box_a[1]), float(box_b[1]))
	xB = min(float(box_a[2]), float(box_b[2]))
	yB = min(float(box_a[3]), float(box_b[3]))

	inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
	if inter_area <= 0.0:
		return 0.0
	area_a = max(1.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
	area_b = max(1.0, (float(box_b[2]) - box_b[0]) * (float(box_b[3]) - float(box_b[1])))
	return float(inter_area / (area_a + area_b - inter_area + 1e-6))


class FLLRobotTracker:
	"""
	Audited & Hardened Constant-Velocity Kalman + Nearest-Neighbor Position Tracker.
	Addresses all 7 + 25 + 13 Audit Findings:
	  - Real-time delta-t (`dt = now - last_ts`) and skipped-frame accounting (#8, #9)
	  - Real-time expiration (`max_age_seconds`) independent of camera FPS (#23)
	  - Clean Run Termination on long occlusion & automatic new Run creation on re-lock (Group A #5, Group B #6)
	  - Bounded Kinematic Gate during occlusion so distant false detections are never adopted (Group A #2)
	  - Spatial consistency check across consecutive initialization hits (Group B #5)
	  - Synchronized Position & Velocity under Stationary-Noise-Gate (Group B #7)
	  - Complete `reset()` method for camera source switching (Group B #1)
	"""

	def __init__(self, cfg: TrackerConfig, homography: TableHomography, fps: float = 30.0, table_id: int = 1):
		self.cfg = cfg
		self.homography = homography
		self.table_id = int(table_id)
		self.nominal_fps = max(1.0, float(fps))
		self.dt = 1.0 / self.nominal_fps

		self.kf = KalmanFilter(dim_x=4, dim_z=2)
		self.current_run_id: int = 0
		self.runs: List[SortieRun] = []
		self.all_points: List[TrajectoryPoint] = []
		self.reset(clear_runs=True)

	def reset(self, clear_runs: bool = False):
		"""
		Resets Kalman state, position lock, and initialization buffers (Group B #1).
		Closes any active run cleanly before resetting.
		"""
		now = time.time()
		if self.runs and self.runs[-1].end_time is None:
			self.runs[-1].end_time = getattr(self, "_last_hit_timestamp", None) or now

		self.dt = 1.0 / max(1.0, self.nominal_fps)
		self.kf = KalmanFilter(dim_x=4, dim_z=2)
		self._update_kalman_transition_matrix(self.dt)
		self.kf.H = np.array(
			[
				[1.0, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, 0.0],
			],
			dtype=float,
		)
		self.kf.R = np.eye(2, dtype=float) * 12.0
		self.kf.P = np.eye(4, dtype=float) * 80.0

		self.initialized = False
		self.state = RobotState.UNINITIALIZED
		self.consecutive_hits = 0
		self.missed_frames = 0
		self.hand_cooldown = 0

		self._init_candidate_xy_mm: Optional[Tuple[float, float]] = None
		self._last_timestamp: Optional[float] = None
		self._last_hit_timestamp: Optional[float] = None
		self._last_frame_idx: Optional[int] = None

		self.smooth_x_mm: float = 0.0
		self.smooth_y_mm: float = 0.0
		self.last_bbox_px: Optional[np.ndarray] = None

		if clear_runs:
			self.current_run_id = 0
			self.runs.clear()
			self.all_points.clear()

	def _update_kalman_transition_matrix(self, dt: float):
		dt = float(np.clip(dt, 1e-3, 1.0))
		self.dt = dt
		self.kf.F = np.array(
			[
				[1.0, 0.0, dt, 0.0],
				[0.0, 1.0, 0.0, dt],
				[0.0, 0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0, 1.0],
			],
			dtype=float,
		)
		q_pos = max(0.5, 60.0 * dt)
		q_vel = max(5.0, 600.0 * dt)
		self.kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

	@property
	def max_age_seconds(self) -> float:
		return max(0.8, float(self.cfg.max_age_frames) / self.nominal_fps)

	def _project_detection_mm(self, det: Detection) -> Tuple[float, float]:
		"""Projects detection to table mm WITHOUT clamping so out-of-table points are rejected (Group B #4, #14)."""
		gx, gy = getattr(det, "ground_point_px", det.center_px)
		x_mm, y_mm = self.homography.px_to_mm(gx, gy, clip=False)
		if not self.homography.is_mm_in_active_zone(x_mm, y_mm):
			cx, cy = det.center_px
			x_mm, y_mm = self.homography.px_to_mm(cx, cy, clip=False)
		return x_mm, y_mm

	def _select_best_robot_detection(
		self,
		robot_dets: List[Detection],
		elapsed_since_hit_s: float,
	) -> Optional[Tuple[Detection, float, float]]:
		"""
		Selects the best robot candidate:
		  - Unclamped active-zone check (Group B #4)
		  - Spatial consistency check during initialization (Group B #5)
		  - Strictly bounded kinematic gate even during long occlusions (Group A #2)
		  - Proximity-first scoring anchored to last position & Kalman prediction (#15, #25)
		"""
		if not robot_dets:
			return None

		min_conf = max(0.15, float(self.cfg.track_thresh) * 0.75)
		candidates: List[Tuple[Detection, float, float]] = []
		for det in robot_dets:
			if det.confidence < min_conf and det.source != "motion":
				continue
			x_mm, y_mm = self._project_detection_mm(det)
			# Group B #4: Check UNCLAMPED coordinates first!
			if not self.homography.is_mm_in_active_zone(x_mm, y_mm):
				continue
			x_clamped = float(np.clip(x_mm, 0.0, self.homography.real_w))
			y_clamped = float(np.clip(y_mm, 0.0, self.homography.real_h))
			candidates.append((det, x_clamped, y_clamped))

		# Point #5: Guard against empty candidate list
		if not candidates:
			return None

		if not self.initialized:
			# Group B #5: If we already have a tentative initialization hit, prefer candidates
			# that are spatially consistent (within 200mm) of `_init_candidate_xy_mm`!
			if self._init_candidate_xy_mm is not None:
				ix, iy = self._init_candidate_xy_mm
				consistent = [c for c in candidates if float(np.hypot(c[1] - ix, c[2] - iy)) <= 200.0]
				if consistent:
					consistent.sort(
						key=lambda item: (1 if item[0].source in ("ai", "hybrid") else 0, item[0].confidence),
						reverse=True,
					)
					return consistent[0]
			candidates.sort(
				key=lambda item: (1 if item[0].source in ("ai", "hybrid") else 0, item[0].confidence),
				reverse=True,
			)
			return candidates[0]

		pred_x = float(np.clip(self.kf.x[0, 0], 0.0, self.homography.real_w))
		pred_y = float(np.clip(self.kf.x[1, 0], 0.0, self.homography.real_h))

		# Group A #2 Fix: Never disable the kinematic gate during occlusion!
		# Grow the search radius with elapsed time, but cap it at a physically plausible maximum (750mm)
		# while the track is active (`self.initialized == True`).
		dt_gate = max(self.dt, min(1.5, elapsed_since_hit_s))
		gate_radius_mm = min(
			750.0,
			max(
				240.0,
				float(self.cfg.max_robot_speed_mm_s) * dt_gate * 1.25 + 65.0 * min(8, self.missed_frames),
			),
		)

		best_candidate = None
		best_score = -1e9
		prox_weight = float(np.clip(self.cfg.match_thresh, 0.50, 0.90))
		conf_weight = 1.0 - prox_weight

		for det, x_mm, y_mm in candidates:
			dist_last = float(np.hypot(x_mm - self.smooth_x_mm, y_mm - self.smooth_y_mm))
			dist_pred = float(np.hypot(x_mm - pred_x, y_mm - pred_y))
			eff_dist_mm = min(dist_last, 0.5 * (dist_last + dist_pred))

			in_base, _ = self.homography.is_in_launch_area(x_mm, y_mm)

			# Group A #2: Strictly enforce gate_radius_mm for all non-base detections while track is active!
			if eff_dist_mm > gate_radius_mm and not in_base:
				continue

			norm_prox = max(0.0, 1.0 - (eff_dist_mm / max(1.0, gate_radius_mm)))
			src_bonus = 0.12 if det.source == "hybrid" else (0.06 if det.source == "ai" else 0.0)
			score = prox_weight * norm_prox + conf_weight * float(det.confidence) + src_bonus

			if score > best_score:
				best_score = score
				best_candidate = (det, x_mm, y_mm)

		return best_candidate

	def _check_hand_interaction(
		self,
		current_robot_bbox: Optional[np.ndarray],
		robot_x_mm: float,
		robot_y_mm: float,
		hand_dets: List[Detection],
	) -> bool:
		if not self.cfg.hand_veto.enabled or not hand_dets:
			return False

		eff_radius = min(130.0, float(self.cfg.hand_veto.proximity_radius_mm))
		eff_iou = max(0.06, float(self.cfg.hand_veto.overlap_iou_threshold))

		for h_det in hand_dets:
			if h_det.confidence < 0.35:
				continue
			cx, cy = h_det.center_px
			if not self.homography.is_px_in_active_zone(cx, cy):
				continue
			if current_robot_bbox is not None:
				iou = compute_iou(current_robot_bbox, h_det.bbox_xyxy)
				if iou >= eff_iou:
					return True
			hx_mm, hy_mm = self.homography.px_to_mm(cx, cy, clip=False)
			dist_mm = float(np.hypot(hx_mm - robot_x_mm, hy_mm - robot_y_mm))
			if dist_mm <= eff_radius:
				return True

		return False

	def _close_open_run(self, end_ts: float):
		"""Cleanly closes any open SortieRun (Group A #5, Group B #6)."""
		if self.runs and self.runs[-1].end_time is None:
			self.runs[-1].end_time = end_ts

	def update(
		self, frame_idx: int, detections: List[Detection], timestamp: Optional[float] = None
	) -> Optional[TrajectoryPoint]:
		now = float(timestamp) if timestamp is not None else time.time()

		if self._last_frame_idx is not None and frame_idx == self._last_frame_idx and timestamp is None:
			return self.all_points[-1] if self.all_points else None

		frame_delta = 1
		if self._last_frame_idx is not None and frame_idx > self._last_frame_idx:
			frame_delta = int(frame_idx - self._last_frame_idx)
		self._last_frame_idx = frame_idx

		if self._last_timestamp is not None and now > self._last_timestamp:
			real_dt = float(np.clip(now - self._last_timestamp, 1e-3, 1.0))
		else:
			real_dt = self.dt * frame_delta
		self._last_timestamp = now

		elapsed_since_hit_s = (
			float(max(real_dt, now - self._last_hit_timestamp)) if self._last_hit_timestamp is not None else real_dt
		)

		robot_dets = [d for d in detections if d.class_name == "robot"]
		hand_dets = [d for d in detections if d.class_name == "hand"]

		if self.initialized:
			self._update_kalman_transition_matrix(real_dt)
			self.kf.predict()

		selected = self._select_best_robot_detection(robot_dets, elapsed_since_hit_s)
		is_predicted_point = False
		stationary_gated = False
		current_frame_bbox: Optional[np.ndarray] = None

		if selected is not None:
			det, meas_x_mm, meas_y_mm = selected
			current_frame_bbox = det.bbox_xyxy
			self.last_bbox_px = det.bbox_xyxy
			self.missed_frames = 0
			self._last_hit_timestamp = now

			if not self.initialized:
				# Group B #5: Verify spatial consistency across initialization frames!
				if self._init_candidate_xy_mm is None:
					self._init_candidate_xy_mm = (meas_x_mm, meas_y_mm)
					self.consecutive_hits = 1
				else:
					init_dist = float(
						np.hypot(meas_x_mm - self._init_candidate_xy_mm[0], meas_y_mm - self._init_candidate_xy_mm[1])
					)
					if init_dist <= 200.0:
						self.consecutive_hits += 1
					else:
						# Different object far away -> restart initialization on new candidate
						self.consecutive_hits = 1
					self._init_candidate_xy_mm = (meas_x_mm, meas_y_mm)

				req_hits = int(max(1, self.cfg.min_hits))
				if det.source == "motion" and req_hits < 2:
					# Require 2 spatially consistent hits for pure motion blobs outside AI confirmation
					req_hits = 2

				if self.consecutive_hits >= req_hits:
					self.kf.x = np.array([[meas_x_mm], [meas_y_mm], [0.0], [0.0]], dtype=float)
					self.smooth_x_mm = meas_x_mm
					self.smooth_y_mm = meas_y_mm
					self.initialized = True
				else:
					return None
			else:
				self.consecutive_hits += 1
				self.kf.update(np.array([[meas_x_mm], [meas_y_mm]], dtype=float))
				raw_x = float(np.clip(self.kf.x[0, 0], 0.0, self.homography.real_w))
				raw_y = float(np.clip(self.kf.x[1, 0], 0.0, self.homography.real_h))
				self.kf.x[0, 0] = raw_x
				self.kf.x[1, 0] = raw_y

				step = float(np.hypot(meas_x_mm - self.smooth_x_mm, meas_y_mm - self.smooth_y_mm))
				if step >= self.cfg.stationary_noise_gate_mm:
					alpha = float(np.clip(self.cfg.ema_alpha, 0.15, 0.85))
					target_x = 0.65 * meas_x_mm + 0.35 * raw_x
					target_y = 0.65 * meas_y_mm + 0.35 * raw_y
					self.smooth_x_mm = alpha * target_x + (1.0 - alpha) * self.smooth_x_mm
					self.smooth_y_mm = alpha * target_y + (1.0 - alpha) * self.smooth_y_mm
				else:
					# Group B #7 Fix: When stationary_noise_gate_mm suppresses movement,
					# also zero out Kalman velocity so position and velocity never diverge!
					stationary_gated = True
					self.kf.x[0, 0] = self.smooth_x_mm
					self.kf.x[1, 0] = self.smooth_y_mm
					self.kf.x[2, 0] = 0.0
					self.kf.x[3, 0] = 0.0
		else:
			self.missed_frames += frame_delta
			self.consecutive_hits = 0
			self._init_candidate_xy_mm = None

			# Group A #5 & Group B #6: If track expires, close any open run cleanly and reset `initialized`!
			if (
				not self.initialized
				or self.missed_frames > self.cfg.max_age_frames
				or elapsed_since_hit_s > self.max_age_seconds
			):
				self._close_open_run(self._last_hit_timestamp or now)
				self.initialized = False
				self.last_bbox_px = None
				self.state = RobotState.OCCLUDED if self.missed_frames > 0 else RobotState.UNINITIALIZED
				return None

			is_predicted_point = True
			damping = 0.80**frame_delta
			self.kf.x[2, 0] *= damping
			self.kf.x[3, 0] *= damping
			pred_x = float(np.clip(self.kf.x[0, 0], 0.0, self.homography.real_w))
			pred_y = float(np.clip(self.kf.x[1, 0], 0.0, self.homography.real_h))
			self.kf.x[0, 0] = pred_x
			self.kf.x[1, 0] = pred_y
			self.smooth_x_mm = 0.5 * pred_x + 0.5 * self.smooth_x_mm
			self.smooth_y_mm = 0.5 * pred_y + 0.5 * self.smooth_y_mm

		active_bbox_for_hand = (
			current_frame_bbox if self.missed_frames == 0 else (self.last_bbox_px if self.missed_frames <= 2 else None)
		)
		hand_touching = self._check_hand_interaction(
			active_bbox_for_hand, self.smooth_x_mm, self.smooth_y_mm, hand_dets
		)
		eff_cooldown = min(8, int(self.cfg.hand_veto.cooldown_frames))
		if hand_touching:
			self.hand_cooldown = eff_cooldown
		elif self.hand_cooldown > 0:
			self.hand_cooldown = max(0, self.hand_cooldown - frame_delta)

		in_base, _ = self.homography.is_in_launch_area(self.smooth_x_mm, self.smooth_y_mm)

		if hand_touching or self.hand_cooldown > 0:
			self.state = RobotState.MANIPULATED_BY_HAND
			if self.runs and self.runs[-1].end_time is None and not in_base:
				self.runs[-1].interrupted_by_hand = True
				self.runs[-1].end_time = now
		elif in_base:
			self.state = RobotState.IN_BASE
			self._close_open_run(now)
		elif self.missed_frames > 12:
			self.state = RobotState.OCCLUDED
		else:
			self.state = RobotState.AUTONOMOUS
			# Group A #5 & Group B #6: Whenever robot is driving autonomously outside base
			# and there is no open run (e.g. after leaving base OR after re-detection outside base),
			# start a new SortieRun!
			if not self.runs or self.runs[-1].end_time is not None:
				self.current_run_id += 1
				self.runs.append(
					SortieRun(
						run_id=self.current_run_id,
						start_time=now,
						table_id=self.table_id,
						max_plausible_speed_mm_s=float(self.cfg.max_robot_speed_mm_s) * 1.4,
					)
				)

		# Group B #7: Keep velocity 100% consistent with stationary gate
		if stationary_gated:
			vx, vy, speed = 0.0, 0.0, 0.0
		else:
			vx = float(self.kf.x[2, 0])
			vy = float(self.kf.x[3, 0])
			speed = float(np.hypot(vx, vy))

		pt = TrajectoryPoint(
			timestamp=now,
			frame_idx=frame_idx,
			x_mm=round(self.smooth_x_mm, 1),
			y_mm=round(self.smooth_y_mm, 1),
			vx_mm_s=round(vx, 1),
			vy_mm_s=round(vy, 1),
			speed_mm_s=round(speed, 1),
			state=self.state,
			run_id=self.current_run_id,
			table_id=self.table_id,
			is_predicted=is_predicted_point,
		)
		self.all_points.append(pt)

		if self.state == RobotState.AUTONOMOUS and self.runs and self.runs[-1].end_time is None:
			self.runs[-1].points.append(pt)

		return pt
