import cv2

from .config import MAX_ASPECT_RATIO, MIN_AREA, MIN_ASPECT_RATIO, MIN_SOLIDITY


class DetectorHybrid:
	def __init__(self):
		# GPU Motion Detection
		self.bg_sub = cv2.cuda.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
		# GPU Filter (remove noise)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		self.gpu_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.kernel)

		# CPU Tracker
		self.tracker = None
		self.tracking_active = False

	def is_robot_shape(self, contour, frame_w, frame_h):
		"""Entscheidet: Roboter oder Hand?"""
		x, y, w, h = cv2.boundingRect(contour)

		# Border check: Does it touch the edge? (Typical for arms)
		margin = 5
		if x <= margin or y <= margin or (x + w) >= frame_w - margin or (y + h) >= frame_h - margin:
			return False

		# Form check: Too long/thin?
		aspect = float(w) / h
		if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
			return False

		# Solidity check: Too many gaps/fingers?
		area = cv2.contourArea(contour)
		hull = cv2.convexHull(contour)
		hull_area = cv2.contourArea(hull)
		if hull_area == 0:
			return False

		solidity = float(area) / hull_area

		return solidity >= MIN_SOLIDITY

	def detect(self, frame):
		h_img, w_img = frame.shape[:2]

		# Tracking (if robot is known)
		if self.tracking_active:
			success, box = self.tracker.update(frame)
			if success:
				return tuple(map(int, box))
			else:
				self.tracking_active = False
				self.tracker = None
				# print("Lost -> Searching for robot...")

		# Search (Motion Detection on GPU)
		gpu_frame = cv2.cuda_GpuMat()
		gpu_frame.upload(frame)

		mask_gpu = self.bg_sub.apply(gpu_frame, learningRate=-1, stream=None)
		mask_gpu = self.gpu_filter.apply(mask_gpu, stream=None)
		mask = mask_gpu.download()

		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		best_box = None
		max_area = 0

		for c in contours:
			area = cv2.contourArea(c)
			if area > MIN_AREA and self.is_robot_shape(c, w_img, h_img) and area > max_area:
				max_area = area
				best_box = cv2.boundingRect(c)

		if best_box:
			# Start tracking the detected robot
			self.tracker = cv2.TrackerCSRT_create()
			self.tracker.init(frame, best_box)
			self.tracking_active = True
			# print("Robot found and tracking started.")
			return best_box

		return None
