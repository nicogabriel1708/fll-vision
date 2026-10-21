from multiprocessing import Process, freeze_support

import uvicorn

from src.app.config import LEFT_GPU, RIGHT_GPU, VIDEO_SOURCE
from src.app.logging import setup_logging
from src.app.main import app, left_q, right_q
from src.app.worker import gpu_worker

if __name__ == "__main__":
	freeze_support()
	setup_logging()

	p_left = Process(target=gpu_worker, args=(VIDEO_SOURCE, LEFT_GPU, "left", left_q))
	p_left.start()

	p_right = Process(target=gpu_worker, args=(VIDEO_SOURCE, RIGHT_GPU, "right", right_q))
	p_right.start()

	uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
