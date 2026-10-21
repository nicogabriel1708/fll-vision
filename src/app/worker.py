import time

import cv2

from src.tools.video_source import VideoSource

from .config import LOST_TIMEOUT, SMOOTH_FACTOR, TABLE_ROI, VIEW_HEIGHT, VIEW_WIDTH
from .detector import DetectorHybrid


def crop_to_table(frame):
	h, w = frame.shape[:2]
	y1 = int(h * TABLE_ROI["top"])
	y2 = int(h * TABLE_ROI["bottom"])
	x1 = int(w * TABLE_ROI["left"])
	x2 = int(w * TABLE_ROI["right"])
	# Return image and offset (to correct coordinates)
	return frame[y1:y2, x1:x2], x1, y1


def get_safe_window(cx, cy, frame_w, frame_h, view_w, view_h):
	# Calculate top-left corner of the window, centered on cx, cy
	left = int(cx - (view_w / 2))
	top = int(cy - (view_h / 2))

	# Check boundaries and adjust if necessary
	left = max(0, left)
	top = max(0, top)
	if left + view_w > frame_w:
		left = frame_w - view_w
	if top + view_h > frame_h:
		top = frame_h - view_h

	return left, top


def gpu_worker(source_path, gpu_id, side, queue):
	cv2.cuda.setDevice(gpu_id)
	detector = DetectorHybrid()

	# wait until the source is available (e.g. network camera)
	while True:
		try:
			source = VideoSource(source_path)
			break
		except:  # noqa: E722
			time.sleep(1)

	cam_x, cam_y = None, None
	frames_lost = 0

	while True:
		full_frame = source.read()
		if full_frame is None:
			# Video Loop
			if isinstance(source_path, str):
				source = VideoSource(source_path)
				continue
			time.sleep(0.001)
			continue

		# Cut the table area from the full frame based on TABLE_ROI
		table_img, _, _ = crop_to_table(full_frame)
		t_h, t_w = table_img.shape[:2]

		if cam_x is None:
			cam_x, cam_y = t_w // 2, t_h // 2

		# Detection (only on the table area)
		bbox = detector.detect(table_img)

		target_x, target_y = cam_x, cam_y

		if bbox:
			frames_lost = 0
			bx, by, bw, bh = bbox
			# Goal is robot center
			target_x = bx + bw // 2
			target_y = by + bh // 2

			# Debug box (green)
			cv2.rectangle(table_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
		else:
			frames_lost += 1
			# After timeout, slowly return to the center
			if frames_lost > LOST_TIMEOUT:
				target_x, target_y = t_w // 2, t_h // 2

		# Smooth camera movement
		cam_x = cam_x * (1 - SMOOTH_FACTOR) + target_x * SMOOTH_FACTOR
		cam_y = cam_y * (1 - SMOOTH_FACTOR) + target_y * SMOOTH_FACTOR

		# Select frame (digital pan) - no zoom/stretch
		wx, wy = get_safe_window(cam_x, cam_y, t_w, t_h, VIEW_WIDTH, VIEW_HEIGHT)
		final_view = table_img[wy : wy + VIEW_HEIGHT, wx : wx + VIEW_WIDTH]

		if not queue.full():
			queue.put(final_view)
