from multiprocessing import Queue

from fastapi import FastAPI

from .stream import mjpeg_stream

app = FastAPI()

left_q = Queue(maxsize=1)
right_q = Queue(maxsize=1)


@app.get("/left")
def left():
	return mjpeg_stream(left_q)


@app.get("/right")
def right():
	return mjpeg_stream(right_q)
