from typing import List, Optional

import cv2
import numpy as np

from src.detector import Detection
from src.homography import TableHomography
from src.tracker import FLLRobotTracker, RobotState, TrajectoryPoint

RUN_COLORS = [
	(255, 190, 0),  # Cyan-Blue
	(0, 230, 118),  # Bright Green
	(255, 82, 82),  # Coral Red
	(224, 64, 251),  # Purple
	(255, 215, 64),  # Amber
	(100, 255, 218),  # Teal
]


class FLLVisualizer:
	"""Renders camera HUD, autonomous trajectories, hand-veto alerts, and 2D Bird's-Eye Heatmap."""

	def __init__(self, homography: TableHomography, enable_heatmap: bool = True, decay: float = 0.998):
		self.homography = homography
		self.enable_heatmap = enable_heatmap
		self.decay = decay
		self.heatmap = np.zeros((homography.minimap_h, homography.minimap_w), dtype=np.float32)

	def render(
		self,
		frame: np.ndarray,
		detections: List[Detection],
		tracker: FLLRobotTracker,
		latest_point: Optional[TrajectoryPoint],
		fps: float,
	) -> tuple[np.ndarray, np.ndarray]:
		annotated = frame.copy()

		# 1. Draw FLL table calibration border
		table_pts = np.array(self.homography.cfg.src_corners_px, dtype=np.int32)
		cv2.polylines(annotated, [table_pts], isClosed=True, color=(180, 180, 180), thickness=2)

		# 2. Draw detections (Robot, Hands, Jigs)
		for det in detections:
			x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
			if det.class_name == "hand":
				color = (0, 60, 255)  # Red warning for human hands
				label = f"HAND ({det.confidence:.2f})"
			elif det.class_name == "robot":
				color = (0, 255, 120) if tracker.state == RobotState.AUTONOMOUS else (0, 165, 255)
				label = f"ROBOT [{tracker.state.value}] ({det.confidence:.2f})"
			else:
				color = (200, 200, 0)
				label = f"{det.class_name.upper()} ({det.confidence:.2f})"

			cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
			cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

		# 3. Build 2D Top-Down Bird's-Eye Minimap
		minimap = self.homography.warp_to_minimap(frame)

		# Update & overlay Gaussian heatmap only during AUTONOMOUS driving (never when carried by hand!)
		if self.enable_heatmap:
			self.heatmap *= self.decay
			if latest_point is not None and latest_point.state == RobotState.AUTONOMOUS:
				mx, my = self.homography.mm_to_minimap(latest_point.x_mm, latest_point.y_mm)
				if 0 <= mx < self.homography.minimap_w and 0 <= my < self.homography.minimap_h:
					cv2.circle(self.heatmap, (mx, my), radius=14, color=1.0, thickness=-1)

			norm_heat = np.clip(self.heatmap * 25.0, 0, 255).astype(np.uint8)
			norm_heat = cv2.GaussianBlur(norm_heat, (21, 21), 0)
			color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
			mask = norm_heat > 8
			if np.any(mask):
				blended = cv2.addWeighted(minimap, 0.45, color_heat, 0.55, 0)
				minimap[mask] = blended[mask]

		# 4. Draw Autonomous Run Trajectories on both Camera View and Minimap
		for run in tracker.runs:
			if len(run.points) < 2:
				continue
			color = RUN_COLORS[(run.run_id - 1) % len(RUN_COLORS)]
			cam_pts = [self.homography.mm_to_px(p.x_mm, p.y_mm) for p in run.points]
			map_pts = [self.homography.mm_to_minimap(p.x_mm, p.y_mm) for p in run.points]

			cv2.polylines(annotated, [np.array(cam_pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
			cv2.polylines(minimap, [np.array(map_pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

		# 5. Draw Current Robot Position Marker on Minimap
		if latest_point is not None:
			mx, my = self.homography.mm_to_minimap(latest_point.x_mm, latest_point.y_mm)
			marker_col = (0, 255, 0) if latest_point.state == RobotState.AUTONOMOUS else (0, 0, 255)
			cv2.circle(minimap, (mx, my), 8, marker_col, -1)
			cv2.circle(minimap, (mx, my), 11, (255, 255, 255), 2)

		# 6. Draw Telemetry Status Bar
		status_text = (
			f"FPS: {fps:.1f} | State: {tracker.state.value} | "
			f"Run #{tracker.current_run_id} | "
			f"Pos: ({latest_point.x_mm if latest_point else 0:.0f}, {latest_point.y_mm if latest_point else 0:.0f}) mm | "
			f"Speed: {latest_point.speed_mm_s if latest_point else 0:.0f} mm/s"
		)
		cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (20, 20, 20), -1)
		cv2.putText(annotated, status_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

		return annotated, minimap
