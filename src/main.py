import argparse
import csv
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from src.config_loader import load_config
from src.logger import setup_logging, optimize_dgx_spark_hardware
from src.database import FLLDatabase
from src.homography import TableHomography
from src.detector import FLLDetector
from src.tracker import FLLRobotTracker
from src.visualizer import FLLVisualizer
from src.stream_compositor import DualStreamCompositor
from src.match_state import MatchStateManager
from src.server import TelemetryDashboardServer


class AsyncVideoReader:
    """
    Dedicated background thread for video / camera frame decoding.
    Supports live hot-swapping between USB cameras (/dev/video*), RTSP streams, and files.
    """

    def __init__(self, source: Union[str, int], target_fps: float = 60.0, loop_file: bool = True):
        self._cap_lock = threading.Lock()
        self._lock = threading.Lock()
        self.target_fps = float(target_fps)
        self.frame_interval = 1.0 / max(1.0, self.target_fps)
        self._default_loop = loop_file

        self.source: Union[str, int] = ""
        self.is_camera: bool = False
        self.loop_file: bool = loop_file
        self.cap: Optional[cv2.VideoCapture] = None
        self._configure_source_unlocked(source)

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def _configure_source_unlocked(self, source: Union[str, int]):
        logger = logging.getLogger("fll-vision.video")
        s_str = str(source).strip()
        self.source = int(s_str) if s_str.isdigit() else s_str
        self.is_camera = isinstance(self.source, int) or str(self.source).startswith(
            ("rtsp://", "rtsps://", "http://", "https://", "/dev/video")
        )
        self.loop_file = self._default_loop and not self.is_camera
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.source)

        # If default USB camera (/dev/video0) is configured but no physical USB camera is plugged in yet,
        # fall back smoothly to the local demo/fallback MP4 so the web server & streams stay responsive.
        if (not self.cap.isOpened()) and str(s_str) in ("/dev/video0", "0"):
            for fb_path in ("/workspace/data/videos/match_input.mp4", "data/videos/match_input.mp4"):
                if os.path.exists(fb_path):
                    logger.warning(
                        f"Standard-USB-Kamera '{s_str}' aktuell nicht angeschlossen – "
                        f"nutze Fallback-Video '{fb_path}', bis Kamera verbunden oder in /control gewählt wird."
                    )
                    self.cap.release()
                    self.cap = cv2.VideoCapture(fb_path)
                    self.is_camera = False
                    self.loop_file = True
                    break

    def switch_source(self, new_source: Union[str, int]) -> Tuple[bool, str]:
        """Hot-swaps the active video/camera source at runtime after verifying it opens."""
        logger = logging.getLogger("fll-vision.video")
        s_str = str(new_source).strip()
        parsed = int(s_str) if s_str.isdigit() else s_str
        test_cap = cv2.VideoCapture(parsed)
        if not test_cap.isOpened():
            test_cap.release()
            return False, f"Quelle '{new_source}' konnte nicht geöffnet werden."
        ret, first_frame = test_cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            test_cap.release()
            return False, f"Quelle '{new_source}' liefert kein Bildsignal."

        with self._cap_lock:
            old_cap = self.cap
            self.source = parsed
            self.is_camera = isinstance(self.source, int) or str(self.source).startswith(
                ("rtsp://", "rtsps://", "http://", "https://", "/dev/video")
            )
            self.loop_file = self._default_loop and not self.is_camera
            self.cap = test_cap
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

        with self._lock:
            self._latest_frame = first_frame
            self._frame_id += 1

        logger.info(f"Hot-swapped live video source to: {new_source} ({first_frame.shape[1]}x{first_frame.shape[0]})")
        return True, f"Quelle erfolgreich auf '{new_source}' umgeschaltet."

    def isOpened(self) -> bool:
        with self._cap_lock:
            return self.cap is not None and self.cap.isOpened()

    def start(self) -> "AsyncVideoReader":
        if not self.isOpened():
            return self
        self._thread = threading.Thread(target=self._reader_loop, name="video-decoder", daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        logger = logging.getLogger("fll-vision.video")
        while not self._stopped:
            t0 = time.perf_counter()
            with self._cap_lock:
                if self.cap is None:
                    time.sleep(0.05)
                    continue
                ret, frame = self.cap.read()
                if (not ret or frame is None) and self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_camera:
                    # Brief backoff and retry if live USB/RTSP camera drops a frame
                    time.sleep(0.05)
                    continue
                logger.warning("Video stream ended or frame read failed.")
                self._stopped = True
                break

            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

            if not self.is_camera:
                elapsed = time.perf_counter() - t0
                sleep_t = self.frame_interval - elapsed
                if sleep_t > 0.001:
                    time.sleep(sleep_t)

    def read_latest(self, last_seen_id: int, timeout_s: float = 1.0) -> Tuple[bool, Optional[np.ndarray], int]:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest_frame is not None and self._frame_id > last_seen_id:
                    return True, self._latest_frame.copy(), self._frame_id
                if self._stopped:
                    return False, None, self._frame_id
            time.sleep(0.002)
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy(), self._frame_id
        return False, None, last_seen_id

    def release(self):
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def run_pipeline(config_path: str):
    cfg = load_config(config_path)
    log_level = os.environ.get("LOG_LEVEL", cfg.project.log_level)
    logger = setup_logging(log_level=log_level, log_dir=cfg.video.output_dir)
    hw_info = optimize_dgx_spark_hardware()

    # Initialize SQLite Database (No default passwords! Only create accounts if passwords were provided via setup/env)
    db = FLLDatabase(db_path=cfg.project.database_path)
    if not db.list_users():
        admin_user = os.environ.get("FLL_ADMIN_USER", "admin").strip() or "admin"
        admin_pw = os.environ.get("FLL_ADMIN_PASSWORD", "").strip()
        vol_user = os.environ.get("FLL_VOLUNTEERS_USER", "volunteers").strip() or "volunteers"
        vol_pw = os.environ.get("FLL_VOLUNTEERS_PASSWORD", "").strip()
        eng_user = os.environ.get("FLL_ENGINEERS_USER", "engineers").strip() or "engineers"
        eng_pw = os.environ.get("FLL_ENGINEERS_PASSWORD", "").strip()

        if admin_pw:
            db.create_or_update_user(admin_user, admin_pw, role="admin")
            logger.info(f"Initialized admin account: '{admin_user}'")
            if vol_pw:
                db.create_or_update_user(vol_user, vol_pw, role="volunteers")
                logger.info(f"Initialized volunteers account: '{vol_user}'")
            if eng_pw:
                db.create_or_update_user(eng_user, eng_pw, role="engineers")
                logger.info(f"Initialized engineers account: '{eng_user}'")
        else:
            logger.warning(
                "No users in database and FLL_ADMIN_PASSWORD is not set. "
                "Run './setup.sh' or 'python3 scripts/setup_db.py' to configure your admin password."
            )

    homography = TableHomography(cfg.table)
    compositor = DualStreamCompositor(cfg.streams, homography)
    match_state = MatchStateManager(default_duration_sec=150)

    detector = FLLDetector(cfg.inference)
    tracker = FLLRobotTracker(cfg.tracker, homography, fps=cfg.video.fps)
    visualizer = FLLVisualizer(
        homography,
        enable_heatmap=cfg.server.enable_heatmap,
        decay=cfg.server.heatmap_decay,
    )

    server = None
    if cfg.server.enabled:
        server = TelemetryDashboardServer(
            cfg=cfg,
            db=db,
            homography=homography,
            compositor=compositor,
            match_state=match_state,
            tracker=tracker,
            visualizer=visualizer,
            detector=detector,
            config_path=config_path,
        )
        server.start_background()
        logger.info(
            f"FLL-Vision Server running on http://{cfg.server.host}:{cfg.server.port} "
            f"(Match-Stream: / | Volunteer Control: /control | System Configuration: /configuration | Admin Panel: /admin)"
        )

    reader = AsyncVideoReader(cfg.video.source, target_fps=cfg.video.fps, loop_file=True)
    if not reader.isOpened():
        logger.error(f"Could not open video source: {cfg.video.source}")
        return
    reader.start()
    if server is not None:
        server.set_video_reader(reader)

    out_dir = Path(cfg.video.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    last_minimap = None

    frame_idx = 0
    last_reader_id = 0
    fps_ema = float(cfg.video.fps)
    auto_detected_once = False

    logger.info("Starting FLL-Vision GPU inference & dual-stream compositor loop...")
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame, last_reader_id = reader.read_latest(last_reader_id, timeout_s=1.0)
            if not ret or frame is None:
                logger.info("End of video stream reached.")
                break

            frame_idx += 1

            try:
                if cfg.table.auto_detect_borders and not auto_detected_once:
                    auto_detected_once = True
                    ok, corners = homography.auto_detect_corners(frame, apply_immediately=True)
                    if ok:
                        logger.info(f"Startup Auto-Detection set table corners to: {corners}")

                # 1. Strictly mask out the unused half of the FLL table + outside surroundings
                masked_frame = homography.apply_roi_mask(frame)

                # 2. Run TensorRT YOLO detection (robot, hand, attachment_jig)
                detections = detector.detect(masked_frame)

                # 3. Update Kalman + Hand-Veto State Machine Tracker
                latest_pt = tracker.update(frame_idx, detections)

                # 4. Compute throughput FPS
                dt = max(1e-6, time.perf_counter() - t0)
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

                # 5. Render HUD & 2D Bird's-Eye View Heatmap (for /configuration)
                annotated, minimap = visualizer.render(frame, detections, tracker, latest_pt, fps_ema)
                last_minimap = minimap

                # 6. Compose Clean Dual Auto-Zoom Team Streams in parallel across CPU cores
                stream1_frame, stream2_frame = compositor.compose_streams(
                    raw_frame_table1=frame,
                    annotated_frame=annotated,
                    minimap_frame=minimap,
                    latest_point_table1=latest_pt,
                )

                if cfg.video.save_annotated_video and video_writer is None:
                    h, w = annotated.shape[:2]
                    out_file = str(out_dir / "fll_tracked_match.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(out_file, fourcc, cfg.video.fps, (w, h))

                if video_writer is not None:
                    video_writer.write(annotated)

                if server is not None:
                    telemetry = {
                        "frame": frame_idx,
                        "fps": round(fps_ema, 1),
                        "hardware": hw_info,
                        "state": tracker.state.value,
                        "current_run_id": tracker.current_run_id,
                        "table_corners_px": homography.cfg.src_corners_px,
                        "active_half_mode": homography.cfg.active_half_mode,
                        "position_mm": {
                            "x": latest_pt.x_mm if latest_pt else None,
                            "y": latest_pt.y_mm if latest_pt else None,
                        },
                        "speed_mm_s": latest_pt.speed_mm_s if latest_pt else 0.0,
                        "runs_summary": [
                            {
                                "run_id": r.run_id,
                                "duration_s": round(r.duration_s, 2),
                                "distance_mm": round(r.total_distance_mm, 1),
                                "interrupted_by_hand": r.interrupted_by_hand,
                                "num_points": len(r.points),
                            }
                            for r in tracker.runs
                        ],
                    }
                    server.update_state(
                        raw_frame=frame,
                        annotated_frame=annotated,
                        minimap_frame=minimap,
                        stream1_frame=stream1_frame,
                        stream2_frame=stream2_frame,
                        telemetry=telemetry,
                    )
            except Exception as frame_exc:
                logger.error(f"Error processing frame #{frame_idx}: {frame_exc}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal (KeyboardInterrupt). Stopping gracefully...")
    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()

        try:
            if last_minimap is not None:
                heatmap_png = out_dir / "match_heatmap_minimap.png"
                cv2.imwrite(str(heatmap_png), last_minimap)

            json_path = out_dir / "match_telemetry.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_frames": frame_idx,
                        "table_corners_px": homography.cfg.src_corners_px,
                        "active_half_mode": homography.cfg.active_half_mode,
                        "runs": [
                            {
                                "run_id": r.run_id,
                                "duration_s": round(r.duration_s, 2),
                                "distance_mm": round(r.total_distance_mm, 1),
                                "avg_speed_mm_s": round(r.total_distance_mm / max(0.1, r.duration_s), 1),
                                "interrupted_by_hand": r.interrupted_by_hand,
                                "trajectory_mm": [
                                    {"t": p.timestamp, "x": p.x_mm, "y": p.y_mm, "v": p.speed_mm_s}
                                    for p in r.points
                                ],
                            }
                            for r in tracker.runs
                        ],
                    },
                    f,
                    indent=2,
                )

            csv_path = out_dir / "runs_summary.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id",
                    "duration_s",
                    "distance_mm",
                    "avg_speed_mm_s",
                    "interrupted_by_hand",
                    "recorded_points",
                ])
                for r in tracker.runs:
                    writer.writerow([
                        r.run_id,
                        round(r.duration_s, 2),
                        round(r.total_distance_mm, 1),
                        round(r.total_distance_mm / max(0.1, r.duration_s), 1),
                        r.interrupted_by_hand,
                        len(r.points),
                    ])
            logger.info("Exported final heatmap PNG, telemetry JSON, and runs CSV.")
        except Exception as exp_exc:
            logger.error(f"Error saving final match artifacts: {exp_exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLL-Vision GPU Robot Tracker & Broadcast Server")
    parser.add_argument("--config", type=str, default="/workspace/config/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
