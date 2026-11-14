"""
USB Camera Auto-Discovery & Video Source Probing (USB V4L2, RTSP, HTTP, File).
Ensures a candidate camera or RTSP stream actually delivers a valid image frame
before saving or switching the live pipeline.
"""

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("fll-vision.video_discovery")


def parse_cv_source(source: Union[str, int]) -> Union[str, int]:
	"""Converts numeric strings like '0' or '1' to int camera indices for OpenCV."""
	s = str(source).strip()
	if s.isdigit():
		return int(s)
	return s


def probe_video_source(
	source: Union[str, int],
	timeout_s: float = 4.5,
) -> Tuple[bool, Dict[str, Union[str, int, float]], Optional[bytes]]:
	"""
	Opens `source` in a worker thread with a strict timeout, reads a single frame,
	and verifies that the frame is non-empty.
	Returns: (success, metadata_dict, preview_jpeg_bytes)
	"""
	parsed_src = parse_cv_source(source)
	if parsed_src == "" or parsed_src is None:
		return False, {"error": "Keine Video-Quelle angegeben."}, None

	def _open_and_grab():
		# Set low RTSP timeout so unreachable network streams fail fast
		os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"
		cap = cv2.VideoCapture(parsed_src)
		try:
			if not cap.isOpened():
				return False, {"error": f"Quelle '{source}' konnte nicht geöffnet werden."}, None
			ret, frame = cap.read()
			if not ret or frame is None or frame.size == 0:
				return False, {"error": f"Quelle '{source}' geöffnet, liefert aber kein Bildsignal."}, None

			h, w = frame.shape[:2]
			fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
			if fps <= 0.0 or fps > 240.0 or np.isnan(fps):
				fps = 30.0

			ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
			jpg_bytes = buf.tobytes() if ok_jpg else None
			return (
				True,
				{
					"source": str(source).strip(),
					"width": int(w),
					"height": int(h),
					"fps": round(fps, 1),
				},
				jpg_bytes,
			)
		finally:
			cap.release()

	pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-probe")
	try:
		fut = pool.submit(_open_and_grab)
		return fut.result(timeout=timeout_s)
	except FuturesTimeoutError:
		logger.warning(f"Timeout ({timeout_s}s) while probing video source: {source}")
		return False, {"error": f"Zeitüberschreitung ({timeout_s}s) beim Verbinden mit '{source}'."}, None
	except Exception as exc:
		logger.error(f"Error probing video source '{source}': {exc}")
		return False, {"error": f"Fehler beim Prüfen von '{source}': {exc}"}, None
	finally:
		pool.shutdown(wait=False)


def discover_available_video_sources(
	videos_dirs: Optional[List[str]] = None,
	probe_usb: bool = True,
) -> List[Dict[str, Union[str, int, bool]]]:
	"""
	Automatically detects:
	  1. Connected USB / V4L2 cameras (/dev/video* and /sys/class/video4linux/*)
	  2. Available video files in `/workspace/data/videos` and `/workspace/data/training_videos`
	"""
	discovered: List[Dict[str, Union[str, int, bool]]] = []
	seen_sources = set()

	# 1. Scan Linux V4L2 USB Cameras (/sys/class/video4linux/video*)
	v4l_paths = sorted(glob.glob("/sys/class/video4linux/video*"))
	for sys_path in v4l_paths:
		dev_name = os.path.basename(sys_path)  # e.g. 'video0'
		dev_node = f"/dev/{dev_name}"
		if not os.path.exists(dev_node):
			continue

		# Read human-friendly camera name if available
		label_file = os.path.join(sys_path, "name")
		cam_label = dev_name
		if os.path.exists(label_file):
			try:
				cam_label = Path(label_file).read_text(encoding="utf-8").strip()
			except Exception:
				pass

		# Check if this node is a valid capture device (some USB cameras expose metadata nodes)
		if probe_usb:
			ok, meta, _ = probe_video_source(dev_node, timeout_s=1.5)
			if not ok:
				continue
			res_info = f"{meta.get('width')}x{meta.get('height')}"
		else:
			res_info = "V4L2"

		idx_str = dev_name.replace("video", "")
		discovered.append(
			{
				"type": "usb_camera",
				"source": dev_node,
				"index": int(idx_str) if idx_str.isdigit() else -1,
				"label": f"📷 USB-Kamera: {cam_label} ({dev_node} – {res_info})",
			}
		)
		seen_sources.add(dev_node)

	# 2. Scan /dev/video* directly if /sys/class/video4linux was not mounted
	for dev_node in sorted(glob.glob("/dev/video*")):
		if dev_node in seen_sources:
			continue
		if probe_usb:
			ok, meta, _ = probe_video_source(dev_node, timeout_s=1.5)
			if not ok:
				continue
			res_info = f"{meta.get('width')}x{meta.get('height')}"
		else:
			res_info = "USB"
		discovered.append(
			{
				"type": "usb_camera",
				"source": dev_node,
				"label": f"📷 USB-Kamera ({dev_node} – {res_info})",
			}
		)
		seen_sources.add(dev_node)

	# 3. Scan local video files in data/videos and data/training_videos
	if videos_dirs is None:
		videos_dirs = [
			"/workspace/data/videos",
			"/workspace/data/training_videos",
			"data/videos",
			"data/training_videos",
		]

	exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
	for vdir in videos_dirs:
		pdir = Path(vdir)
		if not pdir.exists():
			continue
		for fp in sorted(pdir.rglob("*")):
			if fp.is_file() and fp.suffix.lower() in exts:
				full_str = str(fp.resolve()) if not str(fp).startswith("/workspace") else str(fp)
				if full_str in seen_sources:
					continue
				seen_sources.add(full_str)
				discovered.append(
					{
						"type": "video_file",
						"source": str(fp),
						"label": f"🎞️ Videodatei: {fp.name} ({fp.parent.name}/)",
					}
				)

	return discovered
