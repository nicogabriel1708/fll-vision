"""
Multi-Core Parallel Dual Team-Table Auto-Zoom Stream Compositor for FLL-Vision.
Optimized for NVIDIA DGX Spark (20-core Grace ARM64 CPU):
  - Uses a persistent ThreadPoolExecutor to crop and resize both team streams
    (Stream 1 = Red Team Table, Stream 2 = Blue Team Table) in parallel.
  - Both streams share 100% identical unified zoom settings (`zoom_factor`, `idle_zoom_factor`).
  - Neither stream renders a heatmap or HUD overlay (clean team table views).
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config_loader import StreamsConfig
from src.homography import TableHomography
from src.tracker import TrajectoryPoint, RobotState

logger = logging.getLogger("fll-vision.compositor")


class SmoothAutoZoomViewport:
    """Maintains smoothed center (cx, cy) and zoom factor for a single team's table stream."""

    def __init__(self, out_w: int, out_h: int, target_zoom: float = 2.2, idle_zoom: float = 1.0):
        self.out_w = max(64, int(out_w))
        self.out_h = max(64, int(out_h))
        self.aspect = float(self.out_w) / float(self.out_h)
        self.target_zoom = target_zoom
        self.idle_zoom = idle_zoom

        self.initialized = False
        self.curr_cx: float = 0.0
        self.curr_cy: float = 0.0
        self.curr_zoom: float = idle_zoom
        self.frames_since_robot: int = 999

    def step(
        self,
        active_bbox: Tuple[int, int, int, int],
        robot_px: Optional[Tuple[int, int]],
        auto_zoom_enabled: bool,
        pan_alpha: float,
        zoom_alpha: float,
        hold_frames: int,
    ) -> Tuple[float, float, float]:
        ax1, ay1, ax2, ay2 = active_bbox
        act_cx = 0.5 * (ax1 + ax2)
        act_cy = 0.5 * (ay1 + ay2)

        if not self.initialized:
            self.curr_cx = act_cx
            self.curr_cy = act_cy
            self.curr_zoom = self.idle_zoom
            self.initialized = True

        if auto_zoom_enabled and robot_px is not None:
            self.frames_since_robot = 0
            goal_cx, goal_cy = float(robot_px[0]), float(robot_px[1])
            goal_zoom = max(1.0, self.target_zoom)
        else:
            self.frames_since_robot += 1
            if self.frames_since_robot <= hold_frames and auto_zoom_enabled:
                goal_cx, goal_cy = self.curr_cx, self.curr_cy
                goal_zoom = self.curr_zoom
            else:
                goal_cx, goal_cy = act_cx, act_cy
                goal_zoom = max(1.0, self.idle_zoom)

        self.curr_cx = (1.0 - pan_alpha) * self.curr_cx + pan_alpha * goal_cx
        self.curr_cy = (1.0 - pan_alpha) * self.curr_cy + pan_alpha * goal_cy
        self.curr_zoom = (1.0 - zoom_alpha) * self.curr_zoom + zoom_alpha * goal_zoom

        return self.curr_cx, self.curr_cy, self.curr_zoom

    def render_crop(
        self,
        frame: np.ndarray,
        active_bbox: Tuple[int, int, int, int],
        cx: float,
        cy: float,
        zoom: float,
        border_fill_mode: str = "cover_crop",
    ) -> np.ndarray:
        """
        Extracts a smooth, aspect-ratio-matched viewport around (cx, cy) constrained to the
        active table region, completely eliminating black side borders on ~1:1 streams.
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        try:
            fh, fw = frame.shape[:2]
            ax1, ay1, ax2, ay2 = active_bbox
            act_w = float(max(64, ax2 - ax1))
            act_h = float(max(64, ay2 - ay1))

            if (act_w / act_h) >= self.aspect:
                base_h = act_h
                base_w = base_h * self.aspect
            else:
                base_w = act_w
                base_h = base_w / self.aspect

            crop_w = max(64.0, base_w / max(1.0, zoom))
            crop_h = max(64.0, base_h / max(1.0, zoom))

            half_w = 0.5 * crop_w
            half_h = 0.5 * crop_h

            min_cx = max(half_w, float(ax1) + half_w * 0.85)
            max_cx = min(float(fw) - half_w, float(ax2) - half_w * 0.85)
            clamped_cx = (0.5 * (ax1 + ax2)) if (min_cx > max_cx) else float(np.clip(cx, min_cx, max_cx))

            min_cy = max(half_h, float(ay1) + half_h * 0.85)
            max_cy = min(float(fh) - half_h, float(ay2) - half_h * 0.85)
            clamped_cy = (0.5 * (ay1 + ay2)) if (min_cy > max_cy) else float(np.clip(cy, min_cy, max_cy))

            x1 = max(0, min(fw - 2, int(round(clamped_cx - half_w))))
            y1 = max(0, min(fh - 2, int(round(clamped_cy - half_h))))
            x2 = max(x1 + 2, min(fw, int(round(clamped_cx + half_w))))
            y2 = max(y1 + 2, min(fh, int(round(clamped_cy + half_h))))

            cropped = frame[y1:y2, x1:x2]

            if border_fill_mode == "blurred_backdrop":
                bg = cv2.resize(frame[ay1:ay2, ax1:ax2], (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
                bg = cv2.GaussianBlur(bg, (31, 31), 0)
                fg = cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
                return cv2.addWeighted(fg, 0.96, bg, 0.04, 0)

            return cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        except Exception as exc:
            logger.error(f"Viewport render_crop error: {exc}")
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)


class DualStreamCompositor:
    """
    Generates the two clean, synchronized team-table streams in parallel across CPU cores:
      - Stream 1 (Left): Red Team Table with smooth Robot Auto-Zoom (clean video, no heatmap/HUD)
      - Stream 2 (Right): Blue Team Table with identical smooth Robot Auto-Zoom (clean video, no heatmap/HUD)
    """

    def __init__(
        self,
        cfg: StreamsConfig,
        homography_table1: TableHomography,
        homography_table2: Optional[TableHomography] = None,
    ):
        self.cfg = cfg
        self.homography1 = homography_table1
        self.homography2 = homography_table2 if homography_table2 is not None else homography_table1
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stream-crop")

        self.vp1 = SmoothAutoZoomViewport(
            cfg.stream_width,
            cfg.stream_height,
            target_zoom=cfg.auto_zoom.zoom_factor,
            idle_zoom=cfg.auto_zoom.idle_zoom_factor,
        )
        self.vp2 = SmoothAutoZoomViewport(
            cfg.stream_width,
            cfg.stream_height,
            target_zoom=cfg.auto_zoom.zoom_factor,
            idle_zoom=cfg.auto_zoom.idle_zoom_factor,
        )

    def update_zoom_settings(
        self,
        enabled: Optional[bool] = None,
        zoom_factor: Optional[float] = None,
        idle_zoom: Optional[float] = None,
        border_fill_mode: Optional[str] = None,
        zoom_stream1: Optional[float] = None,
        zoom_stream2: Optional[float] = None,
    ):
        with self._lock:
            if enabled is not None:
                self.cfg.auto_zoom.enabled = bool(enabled)
            unified_z = zoom_factor if zoom_factor is not None else zoom_stream1
            if unified_z is not None:
                self.cfg.auto_zoom.zoom_factor = float(np.clip(unified_z, 1.0, 4.5))
                self.vp1.target_zoom = self.cfg.auto_zoom.zoom_factor
                self.vp2.target_zoom = self.cfg.auto_zoom.zoom_factor
            if idle_zoom is not None:
                self.cfg.auto_zoom.idle_zoom_factor = float(np.clip(idle_zoom, 1.0, 2.5))
                self.vp1.idle_zoom = self.cfg.auto_zoom.idle_zoom_factor
                self.vp2.idle_zoom = self.cfg.auto_zoom.idle_zoom_factor
            if border_fill_mode is not None:
                self.cfg.auto_zoom.border_fill_mode = border_fill_mode
            logger.info(
                f"Updated unified Auto-Zoom: enabled={self.cfg.auto_zoom.enabled}, "
                f"zoom_factor={self.cfg.auto_zoom.zoom_factor}x, idle={self.cfg.auto_zoom.idle_zoom_factor}x"
            )

    def _resolve_robot_px(
        self, homography: TableHomography, pt: Optional[TrajectoryPoint]
    ) -> Optional[Tuple[int, int]]:
        if pt is None:
            return None
        if pt.state not in (RobotState.AUTONOMOUS, RobotState.IN_BASE, RobotState.MANIPULATED_BY_HAND):
            return None
        if not homography.is_mm_in_active_zone(pt.x_mm, pt.y_mm):
            return None
        return homography.mm_to_px(pt.x_mm, pt.y_mm)

    def compose_streams(
        self,
        raw_frame_table1: np.ndarray,
        annotated_frame: Optional[np.ndarray] = None,
        minimap_frame: Optional[np.ndarray] = None,
        latest_point_table1: Optional[TrajectoryPoint] = None,
        raw_frame_table2: Optional[np.ndarray] = None,
        latest_point_table2: Optional[TrajectoryPoint] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composes (stream1_red_team, stream2_blue_team) in parallel on two CPU worker threads.
        """
        frame1 = raw_frame_table1
        frame2 = raw_frame_table2 if raw_frame_table2 is not None else raw_frame_table1
        pt1 = latest_point_table1
        pt2 = latest_point_table2 if latest_point_table2 is not None else latest_point_table1

        fh1, fw1 = frame1.shape[:2]
        fh2, fw2 = frame2.shape[:2]
        active_bbox1 = self.homography1.get_active_bbox_px(fw1, fh1)
        active_bbox2 = self.homography2.get_active_bbox_px(fw2, fh2)

        robot_px1 = self._resolve_robot_px(self.homography1, pt1)
        robot_px2 = self._resolve_robot_px(self.homography2, pt2)

        with self._lock:
            az = self.cfg.auto_zoom
            cx1, cy1, z1 = self.vp1.step(
                active_bbox1,
                robot_px1,
                az.enabled,
                az.pan_smoothing_alpha,
                az.zoom_smoothing_alpha,
                az.hold_frames_before_zoom_out,
            )
            cx2, cy2, z2 = self.vp2.step(
                active_bbox2,
                robot_px2,
                az.enabled,
                az.pan_smoothing_alpha,
                az.zoom_smoothing_alpha,
                az.hold_frames_before_zoom_out,
            )
            fill_mode = az.border_fill_mode

        fut1 = self._pool.submit(
            self.vp1.render_crop, frame1, active_bbox1, cx1, cy1, z1, fill_mode
        )
        fut2 = self._pool.submit(
            self.vp2.render_crop, frame2, active_bbox2, cx2, cy2, z2, fill_mode
        )
        return fut1.result(), fut2.result()
