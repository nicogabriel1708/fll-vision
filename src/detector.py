import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

try:
	from ultralytics import YOLO
except ImportError:
	YOLO = None

from src.config_loader import InferenceConfig

logger = logging.getLogger("fll-vision.detector")


@dataclass
class Detection:
	bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] in pixels
	confidence: float
	class_id: int
	class_name: str

	@property
	def center_px(self) -> tuple[float, float]:
		x1, y1, x2, y2 = self.bbox_xyxy
		return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


class FLLDetector:
	"""
	Hardware-accelerated YOLO11 + NVIDIA TensorRT Detector for DGX Spark.
	Detects 'robot', 'hand', and 'attachment_jig' in a single forward pass with
	automatic fallback and error resilience.
	"""

	def __init__(self, cfg: InferenceConfig):
		self.cfg = cfg
		self.id_to_name = {v: k for k, v in cfg.classes.items()}
		self.model = None

		if YOLO is None:
			logger.warning("Ultralytics YOLO package not installed; detector running in no-op mock mode.")
			return

		engine_path = Path(cfg.engine_trt)
		pt_path = Path(cfg.weights_pt)

		try:
			if engine_path.exists():
				logger.info(f"Loading compiled NVIDIA TensorRT Engine: {engine_path}")
				self.model = YOLO(str(engine_path), task="detect")
			elif pt_path.exists():
				logger.info(f"Loading PyTorch weights: {pt_path}")
				self.model = YOLO(str(pt_path), task="detect")
				if cfg.auto_export_trt:
					try:
						logger.info("Compiling PyTorch weights into FP16 NVIDIA TensorRT Engine (.engine)...")
						exported = self.model.export(
							format="engine",
							imgsz=cfg.imgsz,
							quantize=cfg.half_precision,
							device=cfg.device,
							simplify=True,
							workspace=4,
						)
						self.model = YOLO(str(exported), task="detect")
						logger.info(f"Successfully compiled and loaded TensorRT engine: {exported}")
					except Exception as trt_exc:
						logger.warning(f"TensorRT auto-export skipped (using PyTorch weights directly): {trt_exc}")
			else:
				logger.warning(f"Neither {engine_path} nor {pt_path} found! Falling back to base yolo11s.pt.")
				self.model = YOLO("yolo11s.pt", task="detect")

			# Warmup GPU kernels on DGX Spark
			dummy = np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8)
			self.model.predict(dummy, imgsz=cfg.imgsz, device=cfg.device, verbose=False)
			logger.info("GPU inference warmup completed successfully.")
		except Exception as exc:
			logger.error(f"Detector initialization error (continuing in safe fallback mode): {exc}")
			self.model = None

	def detect(self, frame: np.ndarray) -> List[Detection]:
		if self.model is None or frame is None or frame.size == 0:
			return []

		try:
			results = self.model.predict(
				source=frame,
				imgsz=self.cfg.imgsz,
				conf=self.cfg.conf_threshold,
				iou=self.cfg.iou_threshold,
				max_det=self.cfg.max_det,
				device=self.cfg.device,
				quantize=self.cfg.half_precision,
				verbose=False,
			)

			detections: List[Detection] = []
			if not results or results[0].boxes is None:
				return detections

			boxes = results[0].boxes
			xyxy = boxes.xyxy.cpu().numpy()
			confs = boxes.conf.cpu().numpy()
			cls_ids = boxes.cls.cpu().numpy().astype(int)

			for box, conf, cid in zip(xyxy, confs, cls_ids):
				cname = self.id_to_name.get(cid, f"class_{cid}")
				detections.append(
					Detection(
						bbox_xyxy=box,
						confidence=float(conf),
						class_id=int(cid),
						class_name=cname,
					)
				)
			return detections
		except Exception as exc:
			logger.error(f"Frame detection error: {exc}")
			return []
