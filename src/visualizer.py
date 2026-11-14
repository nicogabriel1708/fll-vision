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
	"""
	Renders camera HUD, autonomous trajectories, hand-veto alerts, and non-stretched
	2D Bird's-Eye Heatmaps for Single Table 1 (Left) and Single Table 2 (Right).
	"""

	def __init__(
		self,
		homography: TableHomography,
		enable_heatmap: bool = True,
		decay: float = 0.998,
		homography_table2: Optional[TableHomography] = None,
	):
		self.homography = homography
		self.homography2 = homography_table2
		self.enable_heatmap = enable_heatmap
		self.decay = decay
		self.heatmap = np.zeros((homography.minimap_h, homography.minimap_w), dtype=np.float32)
		self.heatmap2 = (
			np.zeros((homography_table2.minimap_h, homography_table2.minimap_w), dtype=np.float32)
			if homography_table2 is not None
			else None
		)

	def _render_single_table_minimap(
		self,
		frame: np.ndarray,
		homography: TableHomography,
		heatmap_buf: np.ndarray,
		tracker: Optional[FLLRobotTracker],
		latest_point: Optional[TrajectoryPoint],
		title_label: str,
		border_color: tuple[int, int, int],
	) -> tuple[np.ndarray, np.ndarray]:
		minimap = homography.warp_to_minimap(frame)
		mh, mw = minimap.shape[:2]

		# Dynamically adapt heatmap buffer if table calibration aspect ratio changed
		if heatmap_buf.shape[:2] != (mh, mw):
			heatmap_buf = cv2.resize(heatmap_buf, (mw, mh), interpolation=cv2.INTER_LINEAR)

		if self.enable_heatmap:
			heatmap_buf *= self.decay
			if latest_point is not None and latest_point.state == RobotState.AUTONOMOUS:
				mx, my = homography.mm_to_minimap(latest_point.x_mm, latest_point.y_mm)
				if 0 <= mx < mw and 0 <= my < mh:
					cv2.circle(heatmap_buf, (mx, my), radius=14, color=1.0, thickness=-1)

			norm_heat = np.clip(heatmap_buf * 25.0, 0, 255).astype(np.uint8)
			norm_heat = cv2.GaussianBlur(norm_heat, (21, 21), 0)
			color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
			mask = norm_heat > 8
			if np.any(mask):
				blended = cv2.addWeighted(minimap, 0.45, color_heat, 0.55, 0)
				minimap[mask] = blended[mask]

		if tracker is not None:
			for run in tracker.runs:
				if len(run.points) < 2:
					continue
				color = RUN_COLORS[(run.run_id - 1) % len(RUN_COLORS)]
				map_pts = [homography.mm_to_minimap(p.x_mm, p.y_mm) for p in run.points]
				cv2.polylines(minimap, [np.array(map_pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

		if latest_point is not None:
			mx, my = homography.mm_to_minimap(latest_point.x_mm, latest_point.y_mm)
			marker_col = (0, 255, 0) if latest_point.state == RobotState.AUTONOMOUS else (0, 165, 255)
			cv2.circle(minimap, (mx, my), 8, marker_col, -1)
			cv2.circle(minimap, (mx, my), 11, (255, 255, 255), 2)

		# Clean header badge on minimap
		cv2.rectangle(minimap, (0, 0), (mw - 1, mh - 1), border_color, 3)
		cv2.rectangle(minimap, (0, 0), (min(mw, 280), 30), (15, 23, 42), -1)
		cv2.putText(minimap, title_label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2, cv2.LINE_AA)
		return minimap, heatmap_buf

	def render(
		self,
		frame: np.ndarray,
		detections: List[Detection],
		tracker: FLLRobotTracker,
		latest_point: Optional[TrajectoryPoint],
		fps: float,
		detections_table2: Optional[List[Detection]] = None,
		tracker_table2: Optional[FLLRobotTracker] = None,
		latest_point_table2: Optional[TrajectoryPoint] = None,
		frame_table2: Optional[np.ndarray] = None,
		camera_fps: Optional[float] = None,
	) -> tuple[np.ndarray, np.ndarray]:
		annotated = frame.copy()

		# 1. Draw FLL table calibration borders for Table 1 (Red) and Table 2 (Blue)
		t1_pts = np.array(self.homography.cfg.src_corners_px, dtype=np.int32)
		cv2.polylines(annotated, [t1_pts], isClosed=True, color=(36, 28, 237), thickness=2)
		if self.homography2 is not None:
			t2_pts = np.array(self.homography2.cfg.src_corners_px, dtype=np.int32)
			cv2.polylines(annotated, [t2_pts], isClosed=True, color=(179, 102, 0), thickness=2)

		# 2. Draw detections (Robot, Hands, Jigs)
		all_dets = list(detections) + (list(detections_table2) if detections_table2 else [])
		for det in all_dets:
			x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
			if det.class_name == "hand":
				color = (0, 60, 255)
				label = f"HAND ({det.confidence:.2f})"
			elif det.class_name == "robot":
				color = (0, 255, 120)
				src_tag = "M+AI" if det.source == "hybrid" else det.source.upper()
				label = f"ROBOT [{src_tag}] ({det.confidence:.2f})"
			else:
				color = (200, 200, 0)
				label = f"{det.class_name.upper()} ({det.confidence:.2f})"

			cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
			cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

		# 3. Build non-stretched 2D Top-Down Bird's-Eye Minimaps
		minimap1, self.heatmap = self._render_single_table_minimap(
			frame=frame,
			homography=self.homography,
			heatmap_buf=self.heatmap,
			tracker=tracker,
			latest_point=latest_point,
			title_label="Tisch 1 (Links / Red Team)",
			border_color=(68, 68, 239),
		)

		# Draw camera-space trajectories for Table 1
		for run in tracker.runs:
			if len(run.points) < 2:
				continue
			color = RUN_COLORS[(run.run_id - 1) % len(RUN_COLORS)]
			cam_pts = [self.homography.mm_to_px(p.x_mm, p.y_mm) for p in run.points]
			cv2.polylines(annotated, [np.array(cam_pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

		if self.homography2 is not None and tracker_table2 is not None:
			if self.heatmap2 is None:
				self.heatmap2 = np.zeros((self.homography2.minimap_h, self.homography2.minimap_w), dtype=np.float32)
			src_frame2 = frame_table2 if frame_table2 is not None else frame
			minimap2, self.heatmap2 = self._render_single_table_minimap(
				frame=src_frame2,
				homography=self.homography2,
				heatmap_buf=self.heatmap2,
				tracker=tracker_table2,
				latest_point=latest_point_table2,
				title_label="Tisch 2 (Rechts / Blue Team)",
				border_color=(246, 130, 59),
			)
			for run in tracker_table2.runs:
				if len(run.points) < 2:
					continue
				color = RUN_COLORS[(run.run_id - 1) % len(RUN_COLORS)]
				cam_pts = [self.homography2.mm_to_px(p.x_mm, p.y_mm) for p in run.points]
				cv2.polylines(annotated, [np.array(cam_pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

			# Combine Minimap 1 and Minimap 2 side-by-side with a 12px divider (preserving exact aspect ratio!)
			h_target = minimap1.shape[0]
			if minimap2.shape[0] != h_target:
				new_w2 = max(120, int(round(minimap2.shape[1] * (h_target / float(minimap2.shape[0])))))
				minimap2 = cv2.resize(minimap2, (new_w2, h_target), interpolation=cv2.INTER_LINEAR)
			divider = np.full((h_target, 12, 3), 15, dtype=np.uint8)
			minimap = np.hstack([minimap1, divider, minimap2])
		else:
			minimap = minimap1

		# 4. Draw Telemetry Status Bar on annotated camera frame (Group A #3: Cam FPS & Proc FPS)
		cam_fps_val = camera_fps if camera_fps is not None else fps
		t2_state = f" | T2: {tracker_table2.state.value}" if tracker_table2 is not None else ""
		status_text = (
			f"Cam: {cam_fps_val:.1f} FPS (Proc: {fps:.1f}) | "
			f"T1: {tracker.state.value} (Run #{tracker.current_run_id}){t2_state}"
		)
		cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (20, 20, 20), -1)
		cv2.putText(annotated, status_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

		return annotated, minimap
