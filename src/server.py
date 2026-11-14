"""
Unified Production Web, WebSocket & Dual-Stream Server for FLL-Vision.
Provides:
  1. Public Match-Stream Page (`/`, Read-Only) with immediate WebSocket updates (`/ws/match`)
     and low-latency Binary WebSocket video streams (`/ws/stream/1`, `/ws/stream/2`) + MJPEG fallback.
  2. Volunteer Control Page (`/control`, Mobile-Optimized, protected by 'volunteers' or 'admin' role).
  3. System Configuration Page (`/configuration`, protected by 'engineers' or 'admin' role)
     with menus for Double-Table & Single-Tables (Table 1 Left & Table 2 Right) Calibration,
     Unified Stream Auto-Zoom, Non-Stretched Heatmaps & Run Docs, and Hybrid Tracker Config.
  4. Admin Panel (`/admin`, protected by 'admin' role).
"""

import asyncio
import base64
import csv
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
	import uvicorn
	from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
	from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
	from fastapi.staticfiles import StaticFiles
except ImportError:
	FastAPI = None
	uvicorn = None

from src.config_loader import AppConfig, save_config_updates_to_yaml
from src.database import FLLDatabase
from src.homography import TableHomography, order_quad_points, split_double_table_into_two_tables
from src.match_state import MatchStateManager
from src.stream_compositor import DualStreamCompositor
from src.video_discovery import discover_available_video_sources, probe_video_source

logger = logging.getLogger("fll-vision.server")
SESSION_COOKIE_NAME = "fll_session_token"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "frontend" / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "static"


def load_html_template(filename: str, replacements: Optional[Dict[str, str]] = None) -> str:
	path = TEMPLATES_DIR / filename
	content = path.read_text(encoding="utf-8")
	if replacements:
		for k, v in replacements.items():
			content = content.replace(k, str(v))
	return content


class TelemetryDashboardServer:
	def __init__(
		self,
		cfg: AppConfig,
		db: FLLDatabase,
		homography: TableHomography,
		compositor: DualStreamCompositor,
		match_state: MatchStateManager,
		tracker=None,
		visualizer=None,
		detector=None,
		config_path: str = "/workspace/config/config.yaml",
		homography_table2: Optional[TableHomography] = None,
		tracker_table2=None,
	):
		self.cfg = cfg
		self.db = db
		self.homography = homography
		self.homography2 = homography_table2 if homography_table2 is not None else homography
		self.compositor = compositor
		self.match_state = match_state
		self.tracker = tracker
		self.tracker2 = tracker_table2
		self.visualizer = visualizer
		self.detector = detector
		self.video_reader = None
		self.config_path = config_path

		self.host = cfg.server.host
		self.port = cfg.server.port
		self.jpeg_quality = cfg.server.stream_jpeg_quality

		self._lock = threading.Lock()
		self._encode_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="jpeg-enc")
		self._latest_raw_frame: Optional[np.ndarray] = None
		self._latest_annotated_frame: Optional[np.ndarray] = None
		self._latest_minimap_frame: Optional[np.ndarray] = None
		self._latest_stream1_jpeg: Optional[bytes] = None
		self._latest_stream2_jpeg: Optional[bytes] = None
		self._latest_minimap_jpeg: Optional[bytes] = None
		self._latest_telemetry: dict = {"status": "initializing", "runs_summary": []}
		self._frame_seq: int = 0

		self.app = FastAPI(title="FLL-Vision Production Server") if FastAPI else None
		if self.app is not None:
			if STATIC_DIR.exists():
				self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
			self._register_routes()

	def set_video_reader(self, reader):
		self.video_reader = reader

	@staticmethod
	def _encode_jpeg_worker(frame: Optional[np.ndarray], quality: int) -> Optional[bytes]:
		if frame is None or frame.size == 0:
			return None
		try:
			ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
			return buf.tobytes() if ok else None
		except Exception as exc:
			logger.error(f"JPEG encode error: {exc}")
			return None

	def update_state(
		self,
		raw_frame: np.ndarray,
		annotated_frame: np.ndarray,
		minimap_frame: np.ndarray,
		stream1_frame: np.ndarray,
		stream2_frame: np.ndarray,
		telemetry: dict,
	):
		f1 = self._encode_pool.submit(self._encode_jpeg_worker, stream1_frame, self.jpeg_quality)
		f2 = self._encode_pool.submit(self._encode_jpeg_worker, stream2_frame, self.jpeg_quality)
		fm = self._encode_pool.submit(self._encode_jpeg_worker, minimap_frame, self.jpeg_quality)

		jpg1 = f1.result()
		jpg2 = f2.result()
		jpgm = fm.result()

		with self._lock:
			self._latest_raw_frame = raw_frame
			self._latest_annotated_frame = annotated_frame
			self._latest_minimap_frame = minimap_frame
			if jpg1 is not None:
				self._latest_stream1_jpeg = jpg1
			if jpg2 is not None:
				self._latest_stream2_jpeg = jpg2
			if jpgm is not None:
				self._latest_minimap_jpeg = jpgm
			self._latest_telemetry = telemetry
			self._frame_seq += 1

	def get_combined_match_payload(self) -> dict:
		snap = self.match_state.get_snapshot()
		with self._lock:
			snap["telemetry"] = self._latest_telemetry
		return snap

	def _get_stream_jpeg(self, stream_num: int) -> Optional[bytes]:
		swap = self.match_state.swap_streams
		effective_num = (2 if stream_num == 1 else 1) if swap else stream_num
		with self._lock:
			if effective_num == 1:
				return self._latest_stream1_jpeg
			elif effective_num == 2:
				return self._latest_stream2_jpeg
			else:
				return self._latest_minimap_jpeg

	def _mjpeg_generator(self, stream_num: int):
		last_seq = -1
		while True:
			with self._lock:
				seq = self._frame_seq
			if seq == last_seq:
				time.sleep(0.012)
				continue
			last_seq = seq
			jpg = self._get_stream_jpeg(stream_num)
			if jpg is None:
				time.sleep(0.02)
				continue
			yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

	def _require_auth(self, request: "Request", area: str) -> Optional[dict]:
		token = request.cookies.get(SESSION_COOKIE_NAME)
		return self.db.validate_session(token, required_area=area)

	def _require_video_source_auth(self, request: "Request") -> Optional[dict]:
		token = request.cookies.get(SESSION_COOKIE_NAME)
		return self.db.validate_session(token, required_area="volunteers") or self.db.validate_session(
			token, required_area="engineers"
		)

	@staticmethod
	def _warp_single_table_preview(
		frame: np.ndarray,
		corners_px: List[List[float]],
		x_ratio: List[float],
		base_h: int = 360,
	) -> np.ndarray:
		"""Warps a single table quad into a non-stretched top-down preview image."""
		src_pts = order_quad_points(np.array(corners_px, dtype=np.float32))
		top_w = float(np.linalg.norm(src_pts[1] - src_pts[0]))
		bot_w = float(np.linalg.norm(src_pts[2] - src_pts[3]))
		left_h = float(np.linalg.norm(src_pts[3] - src_pts[0]))
		right_h = float(np.linalg.norm(src_pts[2] - src_pts[1]))
		avg_w = max(64.0, 0.5 * (top_w + bot_w))
		avg_h = max(64.0, 0.5 * (left_h + right_h))
		aspect = float(np.clip(avg_w / avg_h, 0.45, 3.6))

		out_h = int(base_h)
		out_w = max(180, int(round(out_h * aspect)))
		dst_pts = np.array(
			[
				[0.0, 0.0],
				[float(out_w), 0.0],
				[float(out_w), float(out_h)],
				[0.0, float(out_h)],
			],
			dtype=np.float32,
		)
		H = cv2.getPerspectiveTransform(src_pts, dst_pts)
		warped = cv2.warpPerspective(frame, H, (out_w, out_h))

		rx0, rx1 = float(x_ratio[0]), float(x_ratio[1])
		if rx0 > 0.001 or rx1 < 0.999:
			overlay = warped.copy()
			ax1, ax2 = int(rx0 * out_w), int(rx1 * out_w)
			mask = np.zeros((out_h, out_w), dtype=np.uint8)
			mask[:, ax1:ax2] = 255
			warped[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
			cv2.rectangle(warped, (ax1, 0), (ax2 - 1, out_h - 1), (16, 185, 129), 3)
		return warped

	def _register_routes(self):
		# ------------------------------------------------------------------
		# 1. PUBLIC LIVESTREAM PAGE (Read-Only, only /)
		# ------------------------------------------------------------------
		@self.app.get("/", response_class=HTMLResponse)
		async def public_livestream_page():
			return HTMLResponse(load_html_template("livestream.html"))

		# ------------------------------------------------------------------
		# 2. AUTHENTICATION & LOGIN PAGES
		# ------------------------------------------------------------------
		@self.app.get("/login/{area}", response_class=HTMLResponse)
		async def login_page(area: str, error: Optional[str] = None):
			norm_area = {
				"volunteer": "control",
				"volunteers": "control",
				"backend": "configuration",
				"engineers": "configuration",
			}.get(area, area)
			area_meta = {
				"control": (
					"Volunteer Control",
					"User: volunteers (/control)",
					"Melde dich als 'volunteers' (oder 'admin') an, um Teams auszuwählen, Match-Zeiten zu setzen, die Video-Quelle zu wählen und den Livestream-Timer zu steuern.",
				),
				"configuration": (
					"System Configuration",
					"User: engineers (/configuration)",
					"Melde dich als 'engineers' (oder 'admin') an, um Doppeltisch & beide Einzeltische (Tisch 1 Links & Tisch 2 Rechts), Auto-Zoom und Heatmaps zu verwalten.",
				),
				"admin": (
					"Admin Panel",
					"User: admin (/admin)",
					"Melde dich als 'admin' an, um Benutzer ('volunteers', 'engineers', 'admin') und FLL-Teams zu verwalten.",
				),
			}.get(norm_area, ("Login", "Protected Area", "Bitte anmelden."))

			html = load_html_template(
				"login.html",
				{
					"{{AREA_TITLE}}": area_meta[0],
					"{{AREA_BADGE}}": area_meta[1],
					"{{AREA_DESCRIPTION}}": area_meta[2],
					"{{FORM_ACTION}}": f"/login/{area}",
					"{{ERROR_DISPLAY}}": "block" if error else "none",
					"{{ERROR_MESSAGE}}": error or "",
				},
			)
			return HTMLResponse(html)

		@self.app.post("/login/{area}")
		async def login_submit(
			request: Request,
			area: str,
			username: str = Form(...),
			password: str = Form(...),
		):
			req_role = {
				"control": "volunteers",
				"volunteer": "volunteers",
				"volunteers": "volunteers",
				"configuration": "engineers",
				"backend": "engineers",
				"engineers": "engineers",
				"admin": "admin",
			}.get(area, "admin")

			client_ip = request.client.host if request.client else "127.0.0.1"
			allowed, retry_sec = self.db.check_rate_limit(client_ip)
			if not allowed:
				return RedirectResponse(
					url=f"/login/{area}?error=Zu+viele+Versuche.+Bitte+{retry_sec}s+warten.",
					status_code=303,
				)

			session = self.db.authenticate_user(username, password, required_area=req_role, ip_address=client_ip)
			if not session:
				return RedirectResponse(
					url=f"/login/{area}?error=Ungültige+Zugangsdaten+oder+fehlende+Berechtigung+für+diesen+Bereich.",
					status_code=303,
				)

			target_url = {
				"control": "/control",
				"volunteer": "/control",
				"configuration": "/configuration",
				"backend": "/configuration",
				"admin": "/admin",
			}.get(area, "/")

			resp = RedirectResponse(url=target_url, status_code=303)
			resp.set_cookie(
				key=SESSION_COOKIE_NAME,
				value=session["token"],
				httponly=True,
				samesite="lax",
				max_age=43200,
			)
			return resp

		@self.app.post("/logout")
		async def logout_route(request: Request):
			token = request.cookies.get(SESSION_COOKIE_NAME)
			self.db.logout_session(token)
			resp = RedirectResponse(url="/", status_code=303)
			resp.delete_cookie(SESSION_COOKIE_NAME)
			return resp

		# ------------------------------------------------------------------
		# 3. VOLUNTEER CONTROL PAGE (/control)
		# ------------------------------------------------------------------
		@self.app.get("/control", response_class=HTMLResponse)
		@self.app.get("/volunteer", response_class=HTMLResponse)
		async def volunteer_page(request: Request):
			if not self._require_auth(request, "volunteers"):
				return RedirectResponse(url="/login/control", status_code=303)
			return HTMLResponse(load_html_template("volunteer.html"))

		@self.app.post("/api/volunteer/action")
		@self.app.post("/api/control/action")
		async def volunteer_action(request: Request):
			if not self._require_auth(request, "volunteers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			action = body.get("action")

			if action == "select_match":
				self.match_state.select_match(
					round_name=body.get("round_name", "Robot Game"),
					team_left=body.get("team_left"),
					team_right=body.get("team_right"),
					red_team_name=body.get("red_team_name"),
					blue_team_name=body.get("blue_team_name"),
					duration_sec=body.get("duration_sec"),
					pre_count_enabled=body.get("pre_count_enabled"),
					pre_count_duration_sec=body.get("pre_count_duration_sec"),
					switch_to_ready=True,
				)
			elif action == "upload_team_list":
				names = body.get("team_names", [])
				self.match_state.set_uploaded_team_list(names)
			elif action == "start":
				self.match_state.start_match()
			elif action == "pause":
				self.match_state.pause_match()
			elif action == "toggle_start_pause":
				self.match_state.toggle_start_pause()
			elif action == "restart":
				self.match_state.restart_match()
			elif action == "abort":
				self.match_state.abort_match()
			elif action == "set_idle":
				self.match_state.set_idle(bool(body.get("idle", True)))
			elif action == "set_duration":
				self.match_state.set_duration(
					duration_sec=int(body.get("duration_sec", 150)),
					pre_count_enabled=body.get("pre_count_enabled"),
					pre_count_duration_sec=body.get("pre_count_duration_sec"),
				)
			elif action == "adjust_time":
				self.match_state.adjust_time(int(body.get("delta_sec", 5)))
			elif action == "swap_streams":
				self.match_state.toggle_swap_streams()
			elif action == "toggle_audio":
				self.match_state.toggle_audio()
			else:
				raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

			return JSONResponse(self.get_combined_match_payload())

		# ------------------------------------------------------------------
		# 4. SYSTEM CONFIGURATION PAGE (/configuration)
		# ------------------------------------------------------------------
		@self.app.get("/configuration", response_class=HTMLResponse)
		@self.app.get("/backend", response_class=HTMLResponse)
		@self.app.get("/calibrate", response_class=HTMLResponse)
		async def backend_control_page(request: Request):
			if not self._require_auth(request, "engineers"):
				return RedirectResponse(url="/login/configuration", status_code=303)
			return HTMLResponse(load_html_template("backend.html"))

		@self.app.get("/api/control/video_sources")
		@self.app.get("/api/configuration/video_sources")
		async def list_video_sources(request: Request):
			if not self._require_video_source_auth(request):
				raise HTTPException(status_code=401, detail="Unauthorized")
			sources = discover_available_video_sources(probe_usb=True)
			return JSONResponse(
				{
					"current_source": str(self.cfg.video.source),
					"current_source_table2": str(self.cfg.video.source_table2 or ""),
					"available_sources": sources,
				}
			)

		@self.app.post("/api/control/video_source/test")
		@self.app.post("/api/configuration/video_source/test")
		async def test_video_source_endpoint(request: Request):
			if not self._require_video_source_auth(request):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			candidate = str(body.get("source", "")).strip()
			ok, meta, jpg_bytes = probe_video_source(candidate, timeout_s=4.5)
			if not ok:
				return JSONResponse(
					status_code=400,
					content={"ok": False, "error": meta.get("error", "Kein Bildsignal empfangen.")},
				)
			b64_img = base64.b64encode(jpg_bytes).decode("ascii") if jpg_bytes else ""
			return JSONResponse(
				{
					"ok": True,
					"meta": meta,
					"preview_data_url": f"data:image/jpeg;base64,{b64_img}" if b64_img else "",
				}
			)

		@self.app.post("/api/control/video_source/save")
		@self.app.post("/api/configuration/video_source/save")
		async def save_video_source_endpoint(request: Request):
			if not self._require_video_source_auth(request):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			new_source = str(body.get("source", "")).strip()

			ok, meta, jpg_bytes = probe_video_source(new_source, timeout_s=4.5)
			if not ok:
				return JSONResponse(
					status_code=400,
					content={
						"ok": False,
						"error": f"Speichern abgelehnt: {meta.get('error', 'Quelle liefert kein Bild.')}",
					},
				)

			if self.video_reader is not None:
				swapped, msg = self.video_reader.switch_source(new_source)
				if not swapped:
					return JSONResponse(status_code=400, content={"ok": False, "error": msg})
			# Group B #1 Fix: Reset trackers, Kalman filters, motion history, and auto-zoom viewports on camera switch!
			new_fps = float(self.video_reader.effective_fps) if self.video_reader is not None else 30.0
			for trk in (self.tracker, self.tracker2):
				if trk is not None:
					trk.nominal_fps = max(1.0, new_fps)
					trk.reset(clear_runs=False)
			if self.compositor is not None:
				self.compositor.reset_viewports()
			# Reset trackers & viewports when table calibration changes so coordinates never jump!
			for trk in (self.tracker, self.tracker2):
				if trk is not None:
					trk.reset(clear_runs=False)
			if self.compositor is not None:
				self.compositor.reset_viewports()
			if self.detector is not None:
				self.detector.reset_motion_history()

			if jpg_bytes is not None:
				arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
				decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
				if decoded is not None:
					with self._lock:
						self._latest_raw_frame = decoded

			self.cfg.video.source = new_source
			save_config_updates_to_yaml(self.config_path, video_source=new_source)
			logger.info(
				f"Saved and activated new video source: {new_source} ({meta.get('width')}x{meta.get('height')})"
			)
			b64_img = base64.b64encode(jpg_bytes).decode("ascii") if jpg_bytes else ""
			return JSONResponse(
				{
					"ok": True,
					"source": new_source,
					"meta": meta,
					"preview_data_url": f"data:image/jpeg;base64,{b64_img}" if b64_img else "",
				}
			)

		@self.app.get("/api/calibration/state")
		async def get_calibration_state(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			return JSONResponse(
				{
					"video_source": str(self.cfg.video.source),
					"double_table_corners_px": self.cfg.table.double_table_corners_px,
					"corners_px": self.homography.cfg.src_corners_px,
					"table2_corners_px": self.homography2.cfg.src_corners_px,
					"auto_detect_borders": self.homography.cfg.auto_detect_borders,
					"active_half_mode": self.homography.cfg.active_half_mode,
					"active_x_ratio": self.homography.cfg.active_x_ratio,
					"active_y_ratio": self.homography.cfg.active_y_ratio,
					"table2_active_half_mode": self.homography2.cfg.active_half_mode,
					"table2_active_x_ratio": self.homography2.cfg.active_x_ratio,
					"table2_active_y_ratio": self.homography2.cfg.active_y_ratio,
					"real_width_mm": self.homography.real_w,
					"real_height_mm": self.homography.real_h,
					"streams": {
						"auto_zoom": {
							"enabled": self.compositor.cfg.auto_zoom.enabled,
							"zoom_factor": self.compositor.cfg.auto_zoom.zoom_factor,
							"zoom_factor_stream1": self.compositor.cfg.auto_zoom.zoom_factor,
							"zoom_factor_stream2": self.compositor.cfg.auto_zoom.zoom_factor,
							"idle_zoom_factor": self.compositor.cfg.auto_zoom.idle_zoom_factor,
							"border_fill_mode": self.compositor.cfg.auto_zoom.border_fill_mode,
						}
					},
				}
			)

		@self.app.get("/api/calibration/raw_frame")
		async def get_raw_frame(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			with self._lock:
				frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None
			if frame is None:
				frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
				cv2.putText(
					frame, "Warte auf Video-Frame...", (640, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 3
				)
			ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
			return Response(content=buf.tobytes(), media_type="image/jpeg")

		@self.app.post("/api/calibration/auto_detect")
		async def run_auto_detect(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			with self._lock:
				frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None
			if frame is None:
				raise HTTPException(status_code=400, detail="No camera frame available yet")
			success, dbl_corners = self.homography.auto_detect_corners(frame, apply_immediately=False)
			t1_corners, t2_corners = split_double_table_into_two_tables(dbl_corners)
			return JSONResponse(
				{
					"detected": success,
					"double_table_corners_px": dbl_corners,
					"corners_px": t1_corners,
					"table2_corners_px": t2_corners,
				}
			)

		@self.app.post("/api/calibration/split_double_table")
		async def api_split_double_table(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			dbl = body.get("double_table_corners_px", self.cfg.table.double_table_corners_px)
			t1_corners, t2_corners = split_double_table_into_two_tables(dbl)
			return JSONResponse(
				{
					"double_table_corners_px": dbl,
					"corners_px": t1_corners,
					"table2_corners_px": t2_corners,
				}
			)

		@self.app.post("/api/calibration/preview")
		async def preview_warp(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			with self._lock:
				frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None
			if frame is None:
				frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

			corners1 = body.get("corners_px", self.homography.cfg.src_corners_px)
			corners2 = body.get("table2_corners_px", self.homography2.cfg.src_corners_px)
			x_ratio1 = body.get("active_x_ratio", self.homography.cfg.active_x_ratio)
			x_ratio2 = body.get("table2_active_x_ratio", self.homography2.cfg.active_x_ratio)

			w1 = self._warp_single_table_preview(frame, corners1, x_ratio1, base_h=340)
			cv2.rectangle(w1, (0, 0), (w1.shape[1] - 1, w1.shape[0] - 1), (68, 68, 239), 3)
			cv2.putText(w1, "Tisch 1 (Links / Red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (68, 68, 239), 2)

			w2 = self._warp_single_table_preview(frame, corners2, x_ratio2, base_h=340)
			cv2.rectangle(w2, (0, 0), (w2.shape[1] - 1, w2.shape[0] - 1), (246, 130, 59), 3)
			cv2.putText(w2, "Tisch 2 (Rechts / Blue)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (246, 130, 59), 2)

			div = np.full((340, 10, 3), 15, dtype=np.uint8)
			combined = np.hstack([w1, div, w2])

			ok, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
			return Response(content=buf.tobytes(), media_type="image/jpeg")

		@self.app.post("/api/calibration/save")
		async def save_calibration(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()

			dbl_corners = body.get("double_table_corners_px", self.cfg.table.double_table_corners_px)
			ordered_dbl = [
				[round(float(p[0]), 1), round(float(p[1]), 1)]
				for p in order_quad_points(np.array(dbl_corners, dtype=np.float32))
			]
			self.cfg.table.double_table_corners_px = ordered_dbl

			ordered1 = self.homography.update_calibration(
				corners_px=body.get("corners_px", self.homography.cfg.src_corners_px),
				active_half_mode=body.get("active_half_mode"),
				active_x_ratio=body.get("active_x_ratio"),
				active_y_ratio=body.get("active_y_ratio"),
			)

			t2_corners_in = body.get("table2_corners_px", self.homography2.cfg.src_corners_px)
			t2_mode_in = body.get(
				"table2_active_half_mode", body.get("active_half_mode", self.homography2.cfg.active_half_mode)
			)
			t2_x_in = body.get("table2_active_x_ratio", self.homography2.cfg.active_x_ratio)
			t2_y_in = body.get("table2_active_y_ratio", self.homography2.cfg.active_y_ratio)

			ordered2 = self.homography2.update_calibration(
				corners_px=t2_corners_in,
				active_half_mode=t2_mode_in,
				active_x_ratio=t2_x_in,
				active_y_ratio=t2_y_in,
			)
			self.cfg.table.table2_corners_px = ordered2
			self.cfg.table.table2_active_half_mode = self.homography2.cfg.active_half_mode
			self.cfg.table.table2_active_x_ratio = self.homography2.cfg.active_x_ratio
			self.cfg.table.table2_active_y_ratio = self.homography2.cfg.active_y_ratio

			if self.detector is not None:
				self.detector.reset_motion_history()

			save_config_updates_to_yaml(
				self.config_path,
				double_table_corners_px=ordered_dbl,
				corners_px=ordered1,
				table2_corners_px=ordered2,
				auto_detect=body.get("auto_detect_borders", False),
				active_half_mode=self.homography.cfg.active_half_mode,
				active_x_ratio=self.homography.cfg.active_x_ratio,
				active_y_ratio=self.homography.cfg.active_y_ratio,
				table2_active_half_mode=self.homography2.cfg.active_half_mode,
				table2_active_x_ratio=self.homography2.cfg.active_x_ratio,
				table2_active_y_ratio=self.homography2.cfg.active_y_ratio,
			)
			return JSONResponse(
				{
					"status": "saved",
					"double_table_corners_px": ordered_dbl,
					"corners_px": ordered1,
					"table2_corners_px": ordered2,
				}
			)

		@self.app.post("/api/backend/zoom")
		@self.app.post("/api/configuration/zoom")
		async def update_zoom_config(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			unified_zoom = body.get("zoom_factor", body.get("zoom_factor_stream1"))
			self.compositor.update_zoom_settings(
				enabled=body.get("enabled"),
				zoom_factor=unified_zoom,
				idle_zoom=body.get("idle_zoom_factor"),
				border_fill_mode=body.get("border_fill_mode"),
			)
			save_config_updates_to_yaml(
				self.config_path,
				zoom_factor=self.compositor.cfg.auto_zoom.zoom_factor,
				idle_zoom_factor=self.compositor.cfg.auto_zoom.idle_zoom_factor,
				auto_zoom_enabled=self.compositor.cfg.auto_zoom.enabled,
				border_fill_mode=self.compositor.cfg.auto_zoom.border_fill_mode,
			)
			return JSONResponse({"status": "ok", "zoom_factor": self.compositor.cfg.auto_zoom.zoom_factor})

		@self.app.post("/api/backend/tracker_config")
		async def update_tracker_config(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			body = await request.json()
			for trk in (self.tracker, self.tracker2):
				if trk is not None:
					if "proximity_radius_mm" in body:
						trk.cfg.hand_veto.proximity_radius_mm = float(body["proximity_radius_mm"])
					if "max_robot_speed_mm_s" in body:
						trk.cfg.max_robot_speed_mm_s = float(body["max_robot_speed_mm_s"])
					if "stationary_noise_gate_mm" in body:
						trk.cfg.stationary_noise_gate_mm = float(body["stationary_noise_gate_mm"])
			if self.detector is not None and "conf_threshold" in body:
				self.detector.cfg.conf_threshold = float(body["conf_threshold"])
			return JSONResponse({"status": "ok"})

		@self.app.post("/api/backend/reset_runs")
		async def reset_runs_and_heatmap(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			for trk in (self.tracker, self.tracker2):
				if trk is not None:
					trk.runs.clear()
					trk.all_points.clear()
					trk.current_run_id = 0
			if self.visualizer is not None:
				self.visualizer.heatmap.fill(0.0)
				if self.visualizer.heatmap2 is not None:
					self.visualizer.heatmap2.fill(0.0)
			return JSONResponse({"status": "reset"})

		@self.app.get("/api/backend/export/heatmap.png")
		async def export_heatmap_png(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			with self._lock:
				frame = self._latest_minimap_frame.copy() if self._latest_minimap_frame is not None else None
			if frame is None:
				frame = np.zeros((540, 960, 3), dtype=np.uint8)
			ok, buf = cv2.imencode(".png", frame)
			return Response(content=buf.tobytes(), media_type="image/png")

		@self.app.get("/api/backend/export/runs.csv")
		async def export_runs_csv(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			sio = io.StringIO()
			writer = csv.writer(sio)
			writer.writerow(
				[
					"table_id",
					"run_id",
					"duration_s",
					"distance_mm",
					"avg_speed_mm_s",
					"interrupted_by_hand",
					"num_points",
				]
			)
			runs = (list(self.tracker.runs) if self.tracker else []) + (
				list(self.tracker2.runs) if self.tracker2 else []
			)
			for r in runs:
				writer.writerow(
					[
						getattr(r, "table_id", 1),
						r.run_id,
						round(r.duration_s, 2),
						round(r.total_distance_mm, 1),
						round(r.total_distance_mm / max(0.1, r.duration_s), 1),
						r.interrupted_by_hand,
						len(r.points),
					]
				)
			return Response(content=sio.getvalue(), media_type="text/csv")

		@self.app.get("/api/backend/export/telemetry.json")
		async def export_telemetry_json(request: Request):
			if not self._require_auth(request, "engineers"):
				raise HTTPException(status_code=401, detail="Unauthorized")
			runs = (list(self.tracker.runs) if self.tracker else []) + (
				list(self.tracker2.runs) if self.tracker2 else []
			)
			data = {
				"double_table_corners_px": self.cfg.table.double_table_corners_px,
				"table_corners_px": self.homography.cfg.src_corners_px,
				"table2_corners_px": self.homography2.cfg.src_corners_px,
				"runs": [
					{
						"table_id": getattr(r, "table_id", 1),
						"run_id": r.run_id,
						"duration_s": round(r.duration_s, 2),
						"distance_mm": round(r.total_distance_mm, 1),
						"interrupted_by_hand": r.interrupted_by_hand,
						"trajectory_mm": [
							{"t": p.timestamp, "x": p.x_mm, "y": p.y_mm, "v": p.speed_mm_s} for p in r.points
						],
					}
					for r in runs
				],
			}
			return JSONResponse(data)

		# ------------------------------------------------------------------
		# 5. ADMIN PANEL (/admin)
		# ------------------------------------------------------------------
		@self.app.get("/admin", response_class=HTMLResponse)
		async def master_admin_page(request: Request):
			if not self._require_auth(request, "admin"):
				return RedirectResponse(url="/login/admin", status_code=303)
			return HTMLResponse(load_html_template("admin.html"))

		@self.app.get("/api/admin/users")
		async def api_list_users(request: Request):
			if not self._require_auth(request, "admin"):
				raise HTTPException(status_code=401, detail="Master Admin access required")
			return JSONResponse(self.db.list_users())

		@self.app.post("/api/admin/users")
		async def api_create_user(request: Request):
			if not self._require_auth(request, "admin"):
				raise HTTPException(status_code=401, detail="Master Admin access required")
			body = await request.json()
			try:
				user = self.db.create_or_update_user(
					username=body.get("username", ""),
					password=body.get("password", ""),
					role=body.get("role", "volunteers"),
				)
				return JSONResponse(user)
			except ValueError as e:
				raise HTTPException(status_code=400, detail=str(e))

		@self.app.delete("/api/admin/users/{user_id}")
		async def api_delete_user(request: Request, user_id: int):
			if not self._require_auth(request, "admin"):
				raise HTTPException(status_code=401, detail="Master Admin access required")
			try:
				self.db.delete_user(user_id)
				return JSONResponse({"status": "deleted"})
			except ValueError as e:
				raise HTTPException(status_code=400, detail=str(e))

		@self.app.post("/api/admin/teams")
		async def api_add_team(request: Request):
			if not self._require_auth(request, "admin"):
				raise HTTPException(status_code=401, detail="Master Admin access required")
			body = await request.json()
			t = self.db.add_team(
				team_number=body.get("team_number", ""),
				team_name=body.get("team_name", ""),
				organization=body.get("organization", ""),
			)
			return JSONResponse(t)

		# ------------------------------------------------------------------
		# 6. PUBLIC API & STREAMING ENDPOINTS (WebSocket + MJPEG)
		# ------------------------------------------------------------------
		@self.app.get("/api/teams")
		async def api_list_teams():
			return JSONResponse(self.db.list_teams())

		@self.app.get("/api/health")
		async def api_health():
			return JSONResponse({"status": "healthy", "frame_seq": self._frame_seq})

		@self.app.get("/api/telemetry")
		async def get_telemetry():
			with self._lock:
				return JSONResponse(self._latest_telemetry)

		@self.app.get("/video-stream/red-team")
		@self.app.get("/stream/1.mjpeg")
		@self.app.get("/video_feed")
		async def stream1_mjpeg():
			return StreamingResponse(
				self._mjpeg_generator(1),
				media_type="multipart/x-mixed-replace; boundary=frame",
			)

		@self.app.get("/video-stream/blue-team")
		@self.app.get("/stream/2.mjpeg")
		async def stream2_mjpeg():
			return StreamingResponse(
				self._mjpeg_generator(2),
				media_type="multipart/x-mixed-replace; boundary=frame",
			)

		@self.app.get("/minimap_feed")
		async def minimap_mjpeg():
			return StreamingResponse(
				self._mjpeg_generator(3),
				media_type="multipart/x-mixed-replace; boundary=frame",
			)

		@self.app.websocket("/ws/match")
		async def ws_match_endpoint(websocket: WebSocket):
			await websocket.accept()
			try:
				while True:
					payload = self.get_combined_match_payload()
					await websocket.send_text(json.dumps(payload))
					await asyncio.sleep(0.08)
			except WebSocketDisconnect:
				pass
			except Exception:
				pass

		@self.app.websocket("/ws/stream/{stream_id}")
		async def ws_binary_video_stream(websocket: WebSocket, stream_id: int):
			await websocket.accept()
			last_seq = -1
			try:
				while True:
					with self._lock:
						seq = self._frame_seq
					if seq != last_seq:
						last_seq = seq
						jpg = self._get_stream_jpeg(stream_id)
						if jpg is not None:
							await websocket.send_bytes(jpg)
					await asyncio.sleep(0.015)
			except WebSocketDisconnect:
				pass
			except Exception:
				pass

	def start_background(self):
		if uvicorn is None or self.app is None:
			return
		thread = threading.Thread(
			target=uvicorn.run,
			kwargs={"app": self.app, "host": self.host, "port": self.port, "log_level": "warning"},
			daemon=True,
		)
		thread.start()
