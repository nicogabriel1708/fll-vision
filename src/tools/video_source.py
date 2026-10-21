import cv2


class VideoSource:
	def __init__(self, source):
		self.source = source
		self.cap = cv2.VideoCapture(source)

		if isinstance(source, int):
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
			self.cap.set(cv2.CAP_PROP_FPS, 30)

		if not self.cap.isOpened():
			raise RuntimeError(f"Source Error: {source}")

	def read(self):
		ret, frame = self.cap.read()
		return frame if ret else None
