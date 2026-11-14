import argparse
import csv
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from src.config_loader import load_config
from src.database import FLLDatabase
from src.detector import FLLDetector
from src.homography import TableHomography, detect_two_independent_tables
from src.logger import optimize_dgx_spark_hardware, setup_logging
from src.match_state import MatchStateManager
from src.server import TelemetryDashboardServer
from src.stream_compositor import DualStreamCompositor
from src.tracker import FLLRobotTracker
from src.visualizer import FLLVisualizer


class AsyncVideoReader:
	"""
	Dedicated background thread for video / camera frame decoding.
	Features:
	  - Automatic native FPS detection (`use_native_fps=True`) so recorded video files (.mp4)
	    always play back at their exact original 1.0x real-time speed using a wall-clock PTS pacer.
	  - Live hot-swapping between USB cameras (/dev/video*), RTSP streams, and video files.
	  - Tracks real measured `camera_fps` (Group A #3).
	  - Never returns duplicate frames on timeout (#10) and exposes true camera frame index (#11).
	"""

	def __init__(
		self,
		source: Union[str, int],
		target_fps: float = 30.0,
		loop_file: bool = True,
		use_native_fps: bool = True,
	):
		self._cap_lock = threading.Lock()
		self._lock = threading.Lock()
		self.configured_fps = float(max(1.0, target_fps))
		self.use_native_fps = bool(use_native_fps)
		self.effective_fps = self.configured_fps
		self.camera_fps = self.configured_fps
		self.frame_interval = 1.0 / self.effective_fps
		self._default_loop = loop_file

		self.source: Union[str, int] = ""
		self.is_camera: bool = False
		self.loop_file: bool = loop_file
		self.cap: Optional[cv2.VideoCapture] = None
		self._pacer_reset_event = threading.Event()
		self._configure_source_unlocked(source)

		self._latest_frame: Optional[np.ndarray] = None
		self._latest_frame_ts: float = 0.0
		self.last_capture_ts: float = 0.0
		self._loop_wrapped: bool = False
		self._frame_id: int = 0
		self._stopped = False
		self._thread: Optional[threading.Thread] = None

	def consume_loop_wrap(self) -> bool:
		with self._lock:
			wrapped = self._loop_wrapped
			self._loop_wrapped = False
			return wrapped

	def _update_fps_from_capture_unlocked(self):
		logger = logging.getLogger("fll-vision.video")
		if self.cap is not None and self.use_native_fps:
			raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
			if 5.0 <= raw_fps <= 240.0 and not math.isnan(raw_fps):
				self.effective_fps = raw_fps
			else:
				self.effective_fps = self.configured_fps
		else:
			self.effective_fps = self.configured_fps
		self.camera_fps = self.effective_fps
		self.frame_interval = 1.0 / max(1.0, self.effective_fps)
		logger.info(
			f"Video source '{self.source}' configured at {self.effective_fps:.2f} FPS "
			f"(interval={self.frame_interval * 1000.0:.1f} ms, is_camera={self.is_camera})"
		)

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

		if (not self.cap.isOpened()) and str(s_str) in ("/dev/video0", "0"):
			for fb_path in ("/workspace/data/videos/match_input.mp4", "data/videos/match_input.mp4"):
				if os.path.exists(fb_path):
					logger.warning(
						f"Standard-USB-Kamera '{s_str}' aktuell nicht angeschlossen – "
						f"nutze Fallback-Video '{fb_path}', bis Kamera verbunden oder in /control gewählt wird."
					)
					self.cap.release()
					self.cap = cv2.VideoCapture(fb_path)
					self.source = fb_path
					self.is_camera = False
					self.loop_file = True
					break

		self._update_fps_from_capture_unlocked()
		self._pacer_reset_event.set()

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
			self._update_fps_from_capture_unlocked()
			self._pacer_reset_event.set()
			if old_cap is not None:
				try:
					old_cap.release()
				except Exception:
					pass

		with self._lock:
			self._latest_frame = first_frame
			self._latest_frame_ts = time.time()
			self._frame_id += 1

		logger.info(
			f"Hot-swapped live video source to: {new_source} "
			f"({first_frame.shape[1]}x{first_frame.shape[0]} @ {self.effective_fps:.2f} FPS)"
		)
		return True, f"Quelle erfolgreich auf '{new_source}' ({self.effective_fps:.1f} FPS) umgeschaltet."

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
		loop_start_wall = time.perf_counter()
		frames_in_segment = 0
		prev_decode_wall = time.perf_counter()

		while not self._stopped:
			if self._pacer_reset_event.is_set():
				self._pacer_reset_event.clear()
				loop_start_wall = time.perf_counter()
				frames_in_segment = 0

			with self._cap_lock:
				if self.cap is None:
					time.sleep(0.05)
					continue
				ret, frame = self.cap.read()
				wrapped_now = False
				if (not ret or frame is None) and self.loop_file:
					self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
					ret, frame = self.cap.read()
					loop_start_wall = time.perf_counter()
					frames_in_segment = 0
					wrapped_now = True
				is_cam = self.is_camera
				interval = self.frame_interval

			if not ret or frame is None:
				if is_cam:
					time.sleep(0.05)
					continue
				logger.warning("Video stream ended or frame read failed.")
				self._stopped = True
				break

			now_wall = time.perf_counter()
			dt_cam = max(1e-4, now_wall - prev_decode_wall)
			prev_decode_wall = now_wall
			self.camera_fps = 0.92 * self.camera_fps + 0.08 * min(240.0, 1.0 / dt_cam)

			capture_ts = time.time()
			with self._lock:
				self._latest_frame = frame
				self._latest_frame_ts = capture_ts
				if wrapped_now:
					self._loop_wrapped = True
				self._frame_id += 1

			if not is_cam:
				frames_in_segment += 1
				target_wall_time = loop_start_wall + (frames_in_segment * interval)
				now = time.perf_counter()
				sleep_s = target_wall_time - now
				if sleep_s > 0.001:
					time.sleep(sleep_s)
				elif sleep_s < -0.5:
					loop_start_wall = now
					frames_in_segment = 0

	def read_latest(self, last_seen_id: int, timeout_s: float = 1.0) -> Tuple[bool, Optional[np.ndarray], int]:
		"""
		Waits up to `timeout_s` for a genuinely new camera frame (`_frame_id > last_seen_id`).
		Point #10 Fix: Never returns a duplicate frame if timeout expires!
		  - Returns `(True, frame, new_id)` when a new frame is available.
		  - Returns `(True, None, last_seen_id)` if the stream is still open/alive but no new frame arrived within timeout_s.
		  - Returns `(False, None, last_seen_id)` only when the video stream has ended/stopped.
		"""
		deadline = time.perf_counter() + timeout_s
		while time.perf_counter() < deadline:
			with self._lock:
				if self._latest_frame is not None and self._frame_id > last_seen_id:
					self.last_capture_ts = self._latest_frame_ts
					return True, self._latest_frame.copy(), self._frame_id
				if self._stopped:
					return False, None, self._frame_id
			time.sleep(0.002)

		with self._lock:
			if self._latest_frame is not None and self._frame_id > last_seen_id:
				self.last_capture_ts = self._latest_frame_ts
				return True, self._latest_frame.copy(), self._frame_id
			if self._stopped:
				return False, None, self._frame_id
		return True, None, last_seen_id

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

	homography1 = TableHomography(cfg.table)
	homography2 = TableHomography(cfg.table.make_table2_config())
	compositor = DualStreamCompositor(cfg.streams, homography1, homography2)
	match_state = MatchStateManager(default_duration_sec=150)

	reader = AsyncVideoReader(
		cfg.video.source,
		target_fps=cfg.video.fps,
		loop_file=True,
		use_native_fps=cfg.video.use_native_fps,
	)
	if not reader.isOpened():
		logger.error(f"Could not open primary video source: {cfg.video.source}")
		return
	reader.start()

	# Single central camera setup: both Table 1 and Table 2 are extracted from `reader` (#2, #8)
	detector = FLLDetector(cfg.inference)
	tracker1 = FLLRobotTracker(cfg.tracker, homography1, fps=reader.effective_fps, table_id=1)
	tracker2 = FLLRobotTracker(cfg.tracker, homography2, fps=reader.effective_fps, table_id=2)
	visualizer = FLLVisualizer(
		homography1,
		enable_heatmap=cfg.server.enable_heatmap,
		decay=cfg.server.heatmap_decay,
		homography_table2=homography2,
	)

	server = None
	if cfg.server.enabled:
		server = TelemetryDashboardServer(
			cfg=cfg,
			db=db,
			homography=homography1,
			compositor=compositor,
			match_state=match_state,
			tracker=tracker1,
			visualizer=visualizer,
			detector=detector,
			config_path=config_path,
			homography_table2=homography2,
			tracker_table2=tracker2,
		)
		server.set_video_reader(reader)
		server.start_background()
		logger.info(
			f"FLL-Vision Server running on http://{cfg.server.host}:{cfg.server.port} "
			f"(Match-Stream: / | Volunteer Control: /control | System Configuration: /configuration | Admin Panel: /admin)"
		)

	out_dir = Path(cfg.video.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	video_writer = None
	video_writer_failed = False
	video_record_start_wall: Optional[float] = None
	video_written_frames: int = 0
	last_minimap = None

	last_reader_id = 0
	last_reader2_id = 0
	latest_frame2: Optional[np.ndarray] = None
	proc_fps_ema = float(reader.effective_fps)
	prev_wall = time.perf_counter()

	# Point #17 & Group A #1: Retry auto-detection until it succeeds (up to 25 attempts every 15 frames)
	auto_detect_succeeded = False
	auto_detect_attempts = 0

	logger.info("Starting FLL-Vision Dual-Table Hybrid Motion+GPU inference loop...")
	try:
		while True:
			ret, frame, new_reader_id = reader.read_latest(last_reader_id, timeout_s=1.0)
			if not ret:
				logger.info("End of primary video stream reached.")
				break
			# Point #10: Skip iteration if timeout expired without a new camera frame
			if frame is None or new_reader_id == last_reader_id:
				continue

			# Point #11 & #4: Use true camera frame counter & exact frame capture timestamp from reader!
			last_reader_id = new_reader_id
			frame_idx = last_reader_id
			capture_ts = reader.last_capture_ts if reader.last_capture_ts > 0.0 else time.time()

			# Point #7: If a video file wrapped around to frame 0, reset MOG2 history and tracker states!
			if reader.consume_loop_wrap():
				detector.reset_motion_history()
				tracker1.reset(clear_runs=False)
				tracker2.reset(clear_runs=False)

			# Point #6: If camera source was switched at runtime, trigger fresh auto-calibration check
			if server is not None and getattr(server, "needs_recalibration", False):
				server.needs_recalibration = False
				auto_detect_succeeded = False
				auto_detect_attempts = 0

			t_proc_start = time.perf_counter()

			try:
				# Point #17 & Group A #1: Retry auto_detect_borders until a valid table quad is found
				if (
					cfg.table.auto_detect_borders
					and not auto_detect_succeeded
					and auto_detect_attempts < 25
					and (frame_idx == 1 or frame_idx % 15 == 0)
				):
					auto_detect_attempts += 1
					ok, t1_c, t2_c = detect_two_independent_tables(
						frame, homography1.cfg.src_corners_px, homography2.cfg.src_corners_px
					)
					if ok:
						try:
							homography1.update_corners(t1_c)
							homography2.update_corners(t2_c)
							cfg.table.src_corners_px = homography1.cfg.src_corners_px
							cfg.table.table2_corners_px = homography2.cfg.src_corners_px
							auto_detect_succeeded = True
							logger.info(
								f"Auto-detected Table 1 & Table 2 on attempt #{auto_detect_attempts} "
								f"(frame #{frame_idx}): T1={t1_c}, T2={t2_c}"
							)
						except ValueError as val_err:
							logger.warning(f"Auto-detect candidate rejected by geometry validator: {val_err}")

				# 1. Single central camera covering Table 1 and Table 2: extract & detect both tables
				dets1, dets2 = detector.detect_dual_table(frame, homography1, homography2)

				# 2. Update independent Trackers for Table 1 and Table 2 using exact capture timestamp (#4, #11)
				pt1 = tracker1.update(frame_idx, dets1, timestamp=capture_ts)
				pt2 = tracker2.update(frame_idx, dets2, timestamp=capture_ts)

				# Group A #3: Measure both processing FPS and real camera FPS
				now_wall = time.perf_counter()
				dt_proc = max(1e-4, now_wall - prev_wall)
				prev_wall = now_wall
				proc_fps_ema = 0.9 * proc_fps_ema + 0.1 * min(500.0, 1.0 / dt_proc)
				cam_fps = float(reader.camera_fps)

				# 3. Render HUD & non-stretched Dual-Table 2D Bird's-Eye Heatmap
				annotated, minimap = visualizer.render(
					frame=frame,
					detections=dets1,
					tracker=tracker1,
					latest_point=pt1,
					fps=proc_fps_ema,
					detections_table2=dets2,
					tracker_table2=tracker2,
					latest_point_table2=pt2,
					frame_table2=frame,
					camera_fps=cam_fps,
				)
				last_minimap = minimap

				# 4. Compose Clean Dual Auto-Zoom Team Streams in parallel across CPU cores
				stream1_frame, stream2_frame = compositor.compose_streams(
					raw_frame_table1=frame,
					annotated_frame=annotated,
					minimap_frame=minimap,
					latest_point_table1=pt1,
					raw_frame_table2=frame,
					latest_point_table2=pt2,
				)

				# 5. Cumulative Wall-Clock Video Recording (Point #14 Fix: no 4-frame cap, zero time drift!)
				if cfg.video.save_annotated_video and not video_writer_failed:
					writer_fps = max(1.0, float(reader.effective_fps))
					if video_writer is None:
						h, w = annotated.shape[:2]
						out_file = str(out_dir / "fll_tracked_match.mp4")
						fourcc = cv2.VideoWriter_fourcc(*"mp4v")
						video_writer = cv2.VideoWriter(out_file, fourcc, writer_fps, (w, h))
						if not video_writer.isOpened():
							logger.warning("Could not open VideoWriter; disabling background video recording.")
							video_writer = None
							video_writer_failed = True
						else:
							video_record_start_wall = now_wall
							video_written_frames = 0
					if video_writer is not None and video_record_start_wall is not None:
						elapsed_total = max(0.0, now_wall - video_record_start_wall)
						target_total_frames = int(round(elapsed_total * writer_fps)) + 1
						n_writes = max(1, min(300, target_total_frames - video_written_frames))
						for _ in range(n_writes):
							video_writer.write(annotated)
						video_written_frames += n_writes

				if server is not None:
					combined_runs = [
						{
							"table_id": 1,
							"run_id": r.run_id,
							"duration_s": round(r.duration_s, 2),
							"distance_mm": round(r.total_distance_mm, 1),
							"interrupted_by_hand": r.interrupted_by_hand,
							"num_points": len(r.points),
						}
						for r in tracker1.runs
					] + [
						{
							"table_id": 2,
							"run_id": r.run_id,
							"duration_s": round(r.duration_s, 2),
							"distance_mm": round(r.total_distance_mm, 1),
							"interrupted_by_hand": r.interrupted_by_hand,
							"num_points": len(r.points),
						}
						for r in tracker2.runs
					]
					telemetry = {
						"frame": frame_idx,
						"fps": round(cam_fps, 1),
						"camera_fps": round(cam_fps, 1),
						"processing_fps": round(proc_fps_ema, 1),
						"native_fps": round(reader.effective_fps, 1),
						"hardware": hw_info,
						"model": {
							"path": detector.active_model_path,
							"is_custom_fll": detector.is_custom_fll_model,
							"using_trt": detector.using_trt_engine,
							"classes": detector.id_to_name,
							"errors": detector.total_errors,
							"last_error": detector.last_error,
						},
						"state": tracker1.state.value,
						"state_table2": tracker2.state.value,
						"current_run_id": tracker1.current_run_id,
						"current_run_id_table2": tracker2.current_run_id,
						"table_corners_px": homography1.cfg.src_corners_px,
						"table2_corners_px": homography2.cfg.src_corners_px,
						"active_half_mode": homography1.cfg.active_half_mode,
						"table2_active_half_mode": homography2.cfg.active_half_mode,
						"position_mm": {
							"x": pt1.x_mm if pt1 else None,
							"y": pt1.y_mm if pt1 else None,
						},
						"position_table2_mm": {
							"x": pt2.x_mm if pt2 else None,
							"y": pt2.y_mm if pt2 else None,
						},
						"speed_mm_s": pt1.speed_mm_s if pt1 else 0.0,
						"speed_table2_mm_s": pt2.speed_mm_s if pt2 else 0.0,
						"runs_summary": combined_runs,
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

			all_runs = list(tracker1.runs) + list(tracker2.runs)
			json_path = out_dir / "match_telemetry.json"
			with open(json_path, "w", encoding="utf-8") as f:
				json.dump(
					{
						"total_frames": last_reader_id,
						"table_corners_px": homography1.cfg.src_corners_px,
						"table2_corners_px": homography2.cfg.src_corners_px,
						"runs": [
							{
								"table_id": r.table_id,
								"run_id": r.run_id,
								"duration_s": round(r.duration_s, 2),
								"distance_mm": round(r.total_distance_mm, 1),
								"avg_speed_mm_s": round(r.total_distance_mm / max(0.1, r.duration_s), 1),
								"interrupted_by_hand": r.interrupted_by_hand,
								"trajectory_mm": [
									{"t": p.timestamp, "x": p.x_mm, "y": p.y_mm, "v": p.speed_mm_s} for p in r.points
								],
							}
							for r in all_runs
						],
					},
					f,
					indent=2,
				)

			csv_path = out_dir / "runs_summary.csv"
			with open(csv_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow(
					[
						"table_id",
						"run_id",
						"duration_s",
						"distance_mm",
						"avg_speed_mm_s",
						"interrupted_by_hand",
						"recorded_points",
					]
				)
				for r in all_runs:
					writer.writerow(
						[
							r.table_id,
							r.run_id,
							round(r.duration_s, 2),
							round(r.total_distance_mm, 1),
							round(r.total_distance_mm / max(0.1, r.duration_s), 1),
							r.interrupted_by_hand,
							len(r.points),
						]
					)
			logger.info("Exported final heatmap PNG, telemetry JSON, and runs CSV.")
		except Exception as exp_exc:
			logger.error(f"Error saving final match artifacts: {exp_exc}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="FLL-Vision GPU Robot Tracker & Broadcast Server")
	parser.add_argument("--config", type=str, default="/workspace/config/config.yaml")
	args = parser.parse_args()
	run_pipeline(args.config)
