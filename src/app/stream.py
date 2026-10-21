import cv2
from fastapi.responses import StreamingResponse


def mjpeg_stream(queue):
	def gen():
		while True:
			frame = queue.get()
			_, jpeg = cv2.imencode(".jpg", frame)
			yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

	return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
