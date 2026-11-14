"""
Centralized Logging, Error-Handling & NVIDIA DGX Spark Hardware Optimization Module.
Configures:
  1. Structured console + rotating file logging (`data/outputs/fll_vision.log`).
  2. Multi-core ARM64 Grace CPU optimizations (`cv2.setUseOptimized(True)`, `cv2.setNumThreads`).
  3. NVIDIA GB10 Blackwell GPU optimizations (`cudnn.benchmark`, TF32 tensor cores, FP16 warmup).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2


def setup_logging(log_level: str = "INFO", log_dir: str = "/workspace/data/outputs") -> logging.Logger:
	"""Configures root and application loggers with console and rotating file handlers."""
	level = getattr(logging, str(log_level).upper(), logging.INFO)
	root_logger = logging.getLogger()
	root_logger.setLevel(level)

	# Avoid duplicate handlers if called multiple times
	if root_logger.handlers:
		root_logger.handlers.clear()

	fmt = logging.Formatter(
		fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(level)
	console_handler.setFormatter(fmt)
	root_logger.addHandler(console_handler)

	try:
		out_path = Path(log_dir)
		out_path.mkdir(parents=True, exist_ok=True)
		file_handler = RotatingFileHandler(
			out_path / "fll_vision.log",
			maxBytes=10 * 1024 * 1024,  # 10 MB per file
			backupCount=5,
			encoding="utf-8",
		)
		file_handler.setLevel(level)
		file_handler.setFormatter(fmt)
		root_logger.addHandler(file_handler)
	except Exception as exc:
		root_logger.warning(f"Could not attach rotating file logger in {log_dir}: {exc}")

	return logging.getLogger("fll-vision")


def optimize_dgx_spark_hardware() -> dict:
	"""
	Tunes OpenCV, CPU thread pools, and PyTorch/CUDA backends for the NVIDIA DGX Spark
	(20-core Grace ARM64 CPU + GB10 Blackwell GPU with 128GB unified memory).
	"""
	logger = logging.getLogger("fll-vision.hardware")
	cpu_cores = os.cpu_count() or 8
	opencv_threads = min(16, max(4, cpu_cores - 2))

	info = {
		"cpu_cores": cpu_cores,
		"opencv_threads": opencv_threads,
		"opencv_optimized": False,
		"cuda_available": False,
		"gpu_name": "CPU",
	}

	try:
		cv2.setUseOptimized(True)
		cv2.setNumThreads(opencv_threads)
		info["opencv_optimized"] = bool(cv2.useOptimized())
	except Exception as exc:
		logger.warning(f"OpenCV thread optimization warning: {exc}")

	try:
		import torch

		if torch.cuda.is_available():
			info["cuda_available"] = True
			info["gpu_name"] = torch.cuda.get_device_name(0)
			torch.backends.cudnn.benchmark = True
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
			if hasattr(torch, "set_float32_matmul_precision"):
				torch.set_float32_matmul_precision("high")
			logger.info(
				f"DGX Spark GPU acceleration active: {info['gpu_name']} | "
				f"cuDNN benchmark=True | TF32=True | CPU cores={cpu_cores} (OpenCV threads={opencv_threads})"
			)
		else:
			logger.info(f"Running in CPU mode ({cpu_cores} cores, OpenCV threads={opencv_threads}).")
	except ImportError:
		logger.info(
			f"PyTorch not loaded in this process; OpenCV SIMD={info['opencv_optimized']} ({opencv_threads} threads)."
		)
	except Exception as exc:
		logger.warning(f"CUDA optimization check warning: {exc}")

	return info
