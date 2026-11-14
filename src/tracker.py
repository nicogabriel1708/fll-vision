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


@dataclass
class SortieRun:
	run_id: int
	start_time: float
	end_time: Optional[float] = None
	points: List[TrajectoryPoint] = field(default_factory=list)
	interrupted_by_hand: bool = False

	@property
	def duration_s(self) -> float:
		end = self.end_time if self.end_time is not None else time.time()
		return max(0.0, end - self.start_time)

	@property
	def total_distance_mm(self) -> float:
		if len(self.points) < 2:
			return 0.0
		coords = np.array([[p.x_mm, p.y_mm] for p in self.points])
		diffs = np.diff(coords, axis=0)
		return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	xA = max(box_a[0], box_b[0])
	yA = max(box_a[1], box_b[1])
	xB = min(box_a[2], box_b[2])
	yB = min(box_a[3], box_b[3])

	inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
	if inter_area <= 0.0:
		return 0.0
	area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
	area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
	return float(inter_area / (area_a + area_b - inter_area + 1e-6))


class FLLRobotTracker:
	"""
	Combines a Constant-Velocity Kalman Filter in real-world millimeter coordinates
	with a Hand-Interaction Veto State Machine and Kinematic Gate.
	Distinguishes true autonomous robot driving from camera noise, moving mission
	models, and human hands touching/carrying the robot.
	"""

	def __init__(self, cfg: TrackerConfig, homography: TableHomography, fps: float = 60.0):
		self.cfg = cfg
		self.homography = homography
		self.dt = 1.0 / max(1.0, fps)

		# 4D State Kalman Filter: [x_mm, y_mm, vx_mm_s, vy_mm_s]
		self.kf = KalmanFilter(dim_x=4, dim_z=2)
		self.kf.F = np.array(
			[
				[1, 0, self.dt, 0],
				[0, 1, 0, self.dt],
				[0, 0, 1, 0],
				[0, 0, 0, 1],
			],
			dtype=float,
		)
		self.kf.H = np.array(
			[
				[1, 0, 0, 0],
				[0, 1, 0, 0],
			],
			dtype=float,
		)
		self.kf.R *= 15.0  # Measurement uncertainty in mm
		self.kf.P *= 100.0
		self.kf.Q = np.eye(4) * 2.0
		self.kf.Q[2:, 2:] *= 25.0

		self.initialized = False
		self.state = RobotState.UNINITIALIZED
		self.consecutive_hits = 0
		self.missed_frames = 0
		self.hand_cooldown = 0

		self.smooth_x_mm: float = 0.0
		self.smooth_y_mm: float = 0.0
		self.last_bbox_px: Optional[np.ndarray] = None

		self.current_run_id: int = 0
		self.runs: List[SortieRun] = []
		self.all_points: List[TrajectoryPoint] = []

	def _select_best_robot_detection(self, robot_dets: List[Detection]) -> Optional[Tuple[Detection, float, float]]:
		"""
		Selects the robot detection that is kinematically consistent with our Kalman
		prediction. Rejects sudden impossible jumps caused by false positives on mission models.
		"""
		if not robot_dets:
			return None

		candidates = []
		for det in robot_dets:
			cx, cy = det.center_px
			x_mm, y_mm = self.homography.px_to_mm(cx, cy)
			# Strictly ignore any false detections in the unused half of the table
			if not self.homography.is_mm_in_active_zone(x_mm, y_mm):
				continue
			candidates.append((det, x_mm, y_mm))

		if not self.initialized:
			# Before initialization, pick highest confidence detection
			candidates.sort(key=lambda item: item[0].confidence, reverse=True)
			return candidates[0]

		pred_x, pred_y = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
		max_step_mm = self.cfg.max_robot_speed_mm_s * self.dt * max(1, self.missed_frames + 1)

		best_candidate = None
		best_score = -1e9

		for det, x_mm, y_mm in candidates:
			dist_mm = float(np.hypot(x_mm - pred_x, y_mm - pred_y))
			# Kinematic Gate: if robot is NOT in base and jump exceeds physical top speed, reject
			in_base, _ = self.homography.is_in_launch_area(x_mm, y_mm)
			if dist_mm > max_step_mm and not in_base and self.missed_frames < 15:
				continue
			score = det.confidence * 100.0 - 0.15 * dist_mm
			if score > best_score:
				best_score = score
				best_candidate = (det, x_mm, y_mm)

		return best_candidate

	def _check_hand_interaction(
		self,
		robot_bbox: Optional[np.ndarray],
		robot_x_mm: float,
		robot_y_mm: float,
		hand_dets: List[Detection],
	) -> bool:
		"""
		Returns True if any detected human hand overlaps the robot bounding box
		or is within proximity_radius_mm of the robot in table coordinates.
		"""
		if not self.cfg.hand_veto.enabled or not hand_dets:
			return False

		for h_det in hand_dets:
			if robot_bbox is not None:
				iou = compute_iou(robot_bbox, h_det.bbox_xyxy)
				if iou >= self.cfg.hand_veto.overlap_iou_threshold:
					return True
			hx_px, hy_px = h_det.center_px
			hx_mm, hy_mm = self.homography.px_to_mm(hx_px, hy_px)
			dist_mm = float(np.hypot(hx_mm - robot_x_mm, hy_mm - robot_y_mm))
			if dist_mm <= self.cfg.hand_veto.proximity_radius_mm:
				return True

		return False

	def update(
		self, frame_idx: int, detections: List[Detection], timestamp: Optional[float] = None
	) -> Optional[TrajectoryPoint]:
		now = timestamp if timestamp is not None else time.time()

		robot_dets = [d for d in detections if d.class_name == "robot"]
		hand_dets = [d for d in detections if d.class_name == "hand"]

		if self.initialized:
			self.kf.predict()

		selected = self._select_best_robot_detection(robot_dets)

		if selected is not None:
			det, meas_x_mm, meas_y_mm = selected
			self.last_bbox_px = det.bbox_xyxy
			self.consecutive_hits += 1
			self.missed_frames = 0

			if not self.initialized:
				if self.consecutive_hits >= self.cfg.min_hits:
					self.kf.x = np.array([[meas_x_mm], [meas_y_mm], [0.0], [0.0]], dtype=float)
					self.smooth_x_mm = meas_x_mm
					self.smooth_y_mm = meas_y_mm
					self.initialized = True
				else:
					return None
			else:
				self.kf.update(np.array([[meas_x_mm], [meas_y_mm]], dtype=float))
				raw_x = float(self.kf.x[0, 0])
				raw_y = float(self.kf.x[1, 0])

				# Stationary micro-jitter noise gate
				step = float(np.hypot(raw_x - self.smooth_x_mm, raw_y - self.smooth_y_mm))
				if step >= self.cfg.stationary_noise_gate_mm:
					alpha = self.cfg.ema_alpha
					self.smooth_x_mm = alpha * raw_x + (1.0 - alpha) * self.smooth_x_mm
					self.smooth_y_mm = alpha * raw_y + (1.0 - alpha) * self.smooth_y_mm
		else:
			self.missed_frames += 1
			self.consecutive_hits = 0
			if not self.initialized or self.missed_frames > self.cfg.max_age_frames:
				self.state = RobotState.OCCLUDED if self.initialized else RobotState.UNINITIALIZED
				return None

		# Evaluate Hand Interaction & Launch Area State Machine
		hand_touching = self._check_hand_interaction(self.last_bbox_px, self.smooth_x_mm, self.smooth_y_mm, hand_dets)
		if hand_touching:
			self.hand_cooldown = self.cfg.hand_veto.cooldown_frames
		elif self.hand_cooldown > 0:
			self.hand_cooldown -= 1

		in_base, _ = self.homography.is_in_launch_area(self.smooth_x_mm, self.smooth_y_mm)

		prev_state = self.state
		if hand_touching or self.hand_cooldown > 0:
			self.state = RobotState.MANIPULATED_BY_HAND
			if self.runs and self.runs[-1].end_time is None and not in_base:
				self.runs[-1].interrupted_by_hand = True
				self.runs[-1].end_time = now
		elif self.missed_frames > 0:
			self.state = RobotState.OCCLUDED
		elif in_base:
			self.state = RobotState.IN_BASE
			if self.runs and self.runs[-1].end_time is None:
				self.runs[-1].end_time = now
		else:
			self.state = RobotState.AUTONOMOUS
			# Start a new autonomous sortie run when leaving base without hands
			if prev_state in (RobotState.IN_BASE, RobotState.UNINITIALIZED, RobotState.MANIPULATED_BY_HAND):
				if not self.runs or self.runs[-1].end_time is not None:
					self.current_run_id += 1
					self.runs.append(SortieRun(run_id=self.current_run_id, start_time=now))

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
		)
		self.all_points.append(pt)

		if self.state == RobotState.AUTONOMOUS and self.runs and self.runs[-1].end_time is None:
			self.runs[-1].points.append(pt)

		return pt
