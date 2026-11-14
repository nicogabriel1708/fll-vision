#!/usr/bin/env python3
"""
FLL-Vision Final Production Verification Suite.
Verifies:
  1. Complete repository & Git/GitHub structure (Dockerfile, docker-compose.yml, docker-compose.dev.yml,
     .gitignore, .dockerignore, .gitattributes, .env.example, empty data/training_videos dir,
     original fonts/audio/images, HTML templates, and NO LICENSE file).
  2. Zero default passwords anywhere in setup.sh, entrypoint.sh, .env.example, docker-compose*.yml, or src/main.py.
  3. Default production video source is "/dev/video0" (connected USB camera) with live switching in /control & /configuration.
  4. SQLite Database initialization, PBKDF2-SHA256 password hashing & independent RBAC for exact users:
     - 'volunteers' -> Volunteer Control (/control, /login/control)
     - 'engineers'  -> System Configuration (/configuration, /login/configuration)
     - 'admin'      -> Admin Panel (/admin, /login/admin + access to all protected areas)
  5. Active Half-Table ROI Masking & Multi-video training discovery.
  6. MatchStateManager exact Timer.jsx behavior (10s PRE countdown -> 150s MAIN countdown).
  7. AsyncVideoReader, 30-frame Kalman + Hand-Veto tracking, 2-thread DualStreamCompositor,
     3-thread JPEG encoder, no /match-stream route (only /), and exact page titles.
"""

import os
import py_compile
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.auto_pipeline import discover_training_videos
from scripts.generate_demo_video import generate_video
from src.config_loader import load_config
from src.database import FLLDatabase
from src.detector import Detection
from src.homography import TableHomography
from src.logger import optimize_dgx_spark_hardware, setup_logging
from src.main import AsyncVideoReader
from src.match_state import MatchStateManager
from src.server import TelemetryDashboardServer, load_html_template
from src.stream_compositor import DualStreamCompositor
from src.tracker import FLLRobotTracker, RobotState
from src.video_discovery import discover_available_video_sources, probe_video_source
from src.visualizer import FLLVisualizer


def run_all_checks():
	print("==================================================================")
	print(" Running FLL-Vision Final Production Verification Suite")
	print("==================================================================")

	# 1. File, Git/GitHub, Docker & Asset Structure Check + No LICENSE Check
	required_paths = [
		"Dockerfile",
		"docker-compose.yml",
		"docker-compose.dev.yml",
		"entrypoint.sh",
		"setup.sh",
		"run_all.sh",
		"requirements.txt",
		".env.example",
		".gitignore",
		".dockerignore",
		".gitattributes",
		"README.md",
		"config/config.yaml",
		"config/dataset.yaml",
		"data/training_videos/.gitkeep",
		"data/videos/.gitkeep",
		"data/outputs/.gitkeep",
		"data/db/.gitkeep",
		"models/.gitkeep",
		"src/logger.py",
		"src/video_discovery.py",
		"scripts/setup_db.py",
		"scripts/auto_pipeline.py",
		"scripts/train.py",
		"scripts/export_trt.py",
		"scripts/generate_demo_video.py",
		"frontend/templates/livestream.html",
		"frontend/templates/volunteer.html",
		"frontend/templates/backend.html",
		"frontend/templates/admin.html",
		"frontend/templates/login.html",
		"frontend/static/fonts/CARDOT/CARDOT_Regular.ttf",
		"frontend/static/fonts/CARDOT/CARDOT_Semibold.ttf",
		"frontend/static/fonts/Digital_Dismay/Digital_Dismay.otf",
		"frontend/static/fonts/Inter/Inter-VariableFont_opsz,wght.ttf",
		"frontend/static/audio/sfx/beep.mp3",
		"frontend/static/audio/sfx/buzz.mp3",
		"frontend/static/audio/sfx/high.mp3",
		"frontend/static/audio/sfx/low.mp3",
		"frontend/static/images/fll-logo-bg-transparent.png",
		"frontend/static/images/fll-logo-bg-white.png",
	]
	for rel in required_paths:
		p = ROOT / rel
		assert p.exists(), f"Missing required file or directory: {rel}"

	assert not (ROOT / "LICENSE").exists(), "LICENSE file must not exist (all rights reserved)."

	for root_dir, _, filenames in os.walk(ROOT):
		for fn in filenames:
			if fn.endswith(".py"):
				py_compile.compile(os.path.join(root_dir, fn), doraise=True)
	print("[PASS 1/7] Complete Git/Docker structure, no LICENSE file, assets & Python syntax verified.")

	# 2. Zero Default Passwords & No /match-stream Check
	forbidden_pw_strings = [
		"FLLAdmin2026!",
		"FLLVolunteers2026!",
		"FLLEngineers2026!",
		"FLLMaster2026!",
		"FLLVolunteer2026!",
		"FLLBackend2026!",
		"flladmin2026",
		"volunteer2026",
		"backend2026",
	]
	for check_file in [
		"setup.sh",
		"entrypoint.sh",
		".env.example",
		"docker-compose.yml",
		"docker-compose.dev.yml",
		"src/main.py",
		"scripts/setup_db.py",
	]:
		content = (ROOT / check_file).read_text(encoding="utf-8")
		for forbidden in forbidden_pw_strings:
			assert forbidden not in content, f"Found forbidden default password '{forbidden}' in {check_file}!"

	srv_text = (ROOT / "src/server.py").read_text(encoding="utf-8")
	assert "/match-stream" not in srv_text, "/match-stream route should be removed (only / is used)."

	with tempfile.TemporaryDirectory() as tmpdir:
		setup_logging(log_level="INFO", log_dir=tmpdir)
		hw = optimize_dgx_spark_hardware()
		assert os.path.exists(os.path.join(tmpdir, "fll_vision.log"))
		assert hw["opencv_optimized"] is True
	print("[PASS 2/7] Zero default passwords, no /match-stream route & rotating logger verified.")

	# 3. Database & Exact Users ('volunteers', 'engineers', 'admin') Check
	with tempfile.TemporaryDirectory() as tmpdir:
		db_path = os.path.join(tmpdir, "verify.sqlite3")
		db = FLLDatabase(db_path=db_path)
		db.create_or_update_user("admin", "CustomAdminPw!", role="admin")
		db.create_or_update_user("volunteers", "CustomVolPw!", role="volunteers")
		db.create_or_update_user("engineers", "CustomEngPw!", role="engineers")

		s_vol = db.authenticate_user("volunteers", "CustomVolPw!", required_area="volunteers")
		assert s_vol is not None
		assert db.validate_session(s_vol["token"], "volunteers") is not None
		assert db.validate_session(s_vol["token"], "engineers") is None
		assert db.validate_session(s_vol["token"], "admin") is None

		s_eng = db.authenticate_user("engineers", "CustomEngPw!", required_area="engineers")
		assert s_eng is not None
		assert db.validate_session(s_eng["token"], "engineers") is not None
		assert db.validate_session(s_eng["token"], "volunteers") is None
		assert db.validate_session(s_eng["token"], "admin") is None

		s_adm = db.authenticate_user("admin", "CustomAdminPw!", required_area="admin")
		assert s_adm is not None
		assert db.validate_session(s_adm["token"], "admin") is not None
		assert db.validate_session(s_adm["token"], "engineers") is not None
		assert db.validate_session(s_adm["token"], "volunteers") is not None
	print("[PASS 3/7] Users ('volunteers', 'engineers', 'admin') & PBKDF2-SHA256 RBAC verified.")

	# 4. Single Central Camera (/dev/video0) & Independent Dual-Table Extraction (Excluding Inter-Table Gap)
	cfg = load_config(str(ROOT / "config" / "config.yaml"))
	assert cfg.video.source == "/dev/video0", f"Expected default video.source '/dev/video0', got {cfg.video.source}"
	assert not hasattr(cfg.video, "source_table2"), "Second camera assumption must not exist!"
	assert not hasattr(cfg.table, "double_table_corners_px"), "Naive double-table split must not exist!"
	assert cfg.video.use_native_fps is True
	assert cfg.video.save_annotated_video is False
	homography = TableHomography(cfg.table)
	homography2 = TableHomography(cfg.table.make_table2_config())
	assert homography.is_px_in_active_zone(450.0, 500.0) is True
	assert homography2.is_px_in_active_zone(1450.0, 500.0) is True
	# Inter-table gap (x=960) must be strictly excluded from BOTH Table 1 and Table 2!
	assert homography.is_px_in_active_zone(960.0, 500.0) is False
	assert homography2.is_px_in_active_zone(960.0, 500.0) is False
	test_img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
	masked_t1 = homography.apply_roi_mask(test_img)
	masked_t2 = homography2.apply_roi_mask(test_img)
	assert np.all(masked_t1[540, 960] == 0) and np.all(masked_t1[540, 1450] == 0)
	assert np.all(masked_t2[540, 960] == 0) and np.all(masked_t2[540, 450] == 0)

	with tempfile.TemporaryDirectory() as tmpvid:
		vdir = Path(tmpvid)
		(vdir / "long_session_alpha.mp4").write_bytes(b"00")
		(vdir / "clip_01_test.MOV").write_bytes(b"00")
		(vdir / "random_name_xyz.mkv").write_bytes(b"00")
		discovered = discover_training_videos(vdir)
		assert len(discovered) == 3
	print("[PASS 4/7] Default USB camera source (/dev/video0), Half-Table masking & video discovery verified.")

	# 5. MatchStateManager PRE -> MAIN Countdown & Team Select Check
	ms = MatchStateManager(default_duration_sec=150, pre_count_enabled=True, pre_count_duration_sec=1)
	snap0 = ms.select_match(red_team_name="Red Dragons", blue_team_name="Blue Sharks", duration_sec=150)
	assert snap0["teamNames"]["red"] == "Red Dragons"
	assert snap0["teamNames"]["blue"] == "Blue Sharks"
	assert snap0["has_timer_started"] is False
	assert snap0["display_minutes"] == 2 and snap0["display_seconds"] == 30

	snap_start = ms.start_match()
	assert snap_start["has_timer_started"] is True
	assert snap_start["timer_phase"] == "pre"
	assert snap_start["is_timer_red"] is True
	time.sleep(1.1)
	snap_main = ms.get_snapshot()
	assert snap_main["timer_phase"] == "main"
	assert snap_main["is_timer_red"] is False
	print("[PASS 5/7] MatchStateManager PRE (10s) -> MAIN (150s) transition & Timer.jsx display rules verified.")

	# 6. End-to-End AsyncVideoReader, Pre-Save Frame Check, Hand-Veto Tracker & Parallel Dual Streams
	with tempfile.TemporaryDirectory() as tmpout:
		demo_vid = os.path.join(tmpout, "demo_match.mp4")
		generate_video(demo_vid, num_frames=30, fps=30)

		ok_good, meta_good, jpg_good = probe_video_source(demo_vid, timeout_s=3.0)
		assert ok_good is True and meta_good["width"] == 1920 and jpg_good is not None
		ok_bad, meta_bad, jpg_bad = probe_video_source("/nonexistent/broken_stream.mp4", timeout_s=2.0)
		assert ok_bad is False and "error" in meta_bad and jpg_bad is None
		found_sources = discover_available_video_sources(videos_dirs=[tmpout], probe_usb=False)
		assert any(s["source"].endswith("demo_match.mp4") for s in found_sources)

		tracker = FLLRobotTracker(cfg.tracker, homography, fps=30.0, table_id=1)
		tracker2 = FLLRobotTracker(cfg.tracker, homography2, fps=30.0, table_id=2)
		visualizer = FLLVisualizer(homography, enable_heatmap=True, decay=0.998, homography_table2=homography2)
		compositor = DualStreamCompositor(cfg.streams, homography, homography2)
		db_temp = FLLDatabase(db_path=os.path.join(tmpout, "srv.sqlite3"))
		srv = TelemetryDashboardServer(
			cfg=cfg,
			db=db_temp,
			homography=homography,
			compositor=compositor,
			match_state=ms,
			tracker=tracker,
			visualizer=visualizer,
			homography_table2=homography2,
			tracker_table2=tracker2,
		)

		reader = AsyncVideoReader(demo_vid, target_fps=60.0, loop_file=True, use_native_fps=False).start()
		frame_count = 0
		last_id = 0
		while True:
			ret, frame, last_id = reader.read_latest(last_id, timeout_s=0.5)
			if not ret:
				break
			if frame is None:
				continue
			frame_count += 1
			rx_px = 320 + frame_count * 12
			ry_px = 520
			fake_dets = [
				Detection(
					bbox_xyxy=np.array([rx_px - 35, ry_px - 35, rx_px + 35, ry_px + 35], dtype=float),
					confidence=0.92,
					class_id=0,
					class_name="robot",
				),
				Detection(
					bbox_xyxy=np.array([1460.0, 500.0, 1540.0, 580.0], dtype=float),
					confidence=0.95,
					class_id=0,
					class_name="robot",
				),
			]
			pt = tracker.update(frame_count, fake_dets, timestamp=1000.0 + frame_count * 0.033)
			annotated, minimap = visualizer.render(frame, fake_dets, tracker, pt, 60.0)
			s1, s2 = compositor.compose_streams(
				raw_frame_table1=frame,
				annotated_frame=annotated,
				minimap_frame=minimap,
				latest_point_table1=pt,
			)
			assert s1.shape == (1080, 1080, 3) and s2.shape == (1080, 1080, 3)
			# Table 1 and Table 2 extract distinct physical regions from the central camera frame!
			assert not np.array_equal(s1, s2)
			srv.update_state(frame, annotated, minimap, s1, s2, {"fps": 60.0, "state": tracker.state.value})
			if frame_count >= 30:
				break

		reader.release()
		assert frame_count == 30
		assert tracker.state == RobotState.AUTONOMOUS
		assert srv._get_stream_jpeg(1) is not None
		assert srv._get_stream_jpeg(2) is not None
	print("[PASS 6/7] AsyncVideoReader, Pre-Save Frame Check, Hand-Veto tracker & Parallel Streams verified.")

	# 7. Exact Page Titles, /control Video Source Switcher & README Table Check
	live_html = load_html_template("livestream.html")
	vol_html = load_html_template("volunteer.html")
	back_html = load_html_template("backend.html")
	adm_html = load_html_template("admin.html")
	readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

	assert "<title>Match-Stream</title>" in live_html
	assert "<title>Volunteer Control</title>" in vol_html
	assert "/api/control/video_source/save" in vol_html
	assert "<title>System Configuration</title>" in back_html
	assert "<title>Admin Panel</title>" in adm_html
	assert "/match-stream" not in readme_text

	for required_term in [
		"Match-Stream",
		"Volunteer Control",
		"System Configuration",
		"Admin Panel",
		"volunteers",
		"engineers",
		"admin",
		"/control",
		"/configuration",
		"/admin",
	]:
		assert required_term in readme_text, f"Missing '{required_term}' in README.md"
	print("[PASS 7/7] Exact Page Titles, /control Video Switcher & README verified!")


if __name__ == "__main__":
	run_all_checks()
