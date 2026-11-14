from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
	import yaml
except ImportError:
	yaml = None

import ast
import logging

logger = logging.getLogger("fll-vision.config")


def _parse_yaml_fallback(text: str) -> dict:
	"""
	Lightweight pure-Python parser for structured YAML files when PyYAML is not installed (#21).
	Ensures `config.yaml` values are actually read from disk instead of silently returning defaults.
	"""
	root: dict = {}
	stack: list[tuple[int, dict | list]] = [(-1, root)]
	lines = text.splitlines()

	for idx, raw_line in enumerate(lines):
		line = raw_line
		if "#" in line:
			in_q = False
			for ci, ch in enumerate(line):
				if ch in ('"', "'"):
					in_q = not in_q
				elif ch == "#" and not in_q:
					line = line[:ci]
					break
		if not line.strip():
			continue

		indent = len(line) - len(line.lstrip(" "))
		stripped = line.strip()

		while len(stack) > 1 and indent <= stack[-1][0]:
			stack.pop()
		parent = stack[-1][1]

		if stripped.startswith("- "):
			val_str = stripped[2:].strip()
			val = _parse_scalar(val_str)
			if isinstance(parent, list):
				parent.append(val)
			continue

		if ":" in stripped:
			key, rest = stripped.split(":", 1)
			key = key.strip()
			rest = rest.strip()
			if rest == "":
				next_is_list = False
				for nxt in lines[idx + 1 :]:
					s_nxt = nxt.split("#", 1)[0].strip()
					if not s_nxt:
						continue
					next_is_list = s_nxt.startswith("- ")
					break
				child: dict | list = [] if next_is_list else {}
				if isinstance(parent, dict):
					parent[key] = child
				stack.append((indent, child))
			else:
				if isinstance(parent, dict):
					parent[key] = _parse_scalar(rest)
	return root


def _parse_scalar(val_str: str):
	low = val_str.lower()
	if low == "true":
		return True
	if low == "false":
		return False
	if low in ("null", "none", "~"):
		return None
	try:
		return ast.literal_eval(val_str)
	except Exception:
		return val_str.strip('"').strip("'")


@dataclass
class ProjectConfig:
	name: str = "fll-vision"
	season: str = "FLL 2026/27 BIOGLOW"
	log_level: str = "INFO"
	database_path: str = "/workspace/data/db/fll_vision.sqlite3"
	secret_key_env: str = "FLL_SECRET_KEY"


@dataclass
class VideoConfig:
	source: Union[str, int] = "/dev/video0"
	training_videos_dir: str = "/workspace/data/training_videos"
	width: int = 1920
	height: int = 1080
	use_native_fps: bool = True
	fps: int = 30
	use_hw_decode: bool = True
	save_annotated_video: bool = False
	output_dir: str = "/workspace/data/outputs"


@dataclass
class TableConfig:
	real_width_mm: float = 2362.0
	real_height_mm: float = 1143.0
	auto_detect_borders: bool = False
	src_corners_px: List[List[float]] = field(
		default_factory=lambda: [[60.0, 140.0], [910.0, 140.0], [910.0, 960.0], [20.0, 960.0]]
	)
	table2_corners_px: List[List[float]] = field(
		default_factory=lambda: [[1010.0, 140.0], [1860.0, 140.0], [1900.0, 960.0], [1010.0, 960.0]]
	)
	apply_table_roi_mask: bool = True
	roi_padding_px: int = 4
	active_half_mode: str = "full"
	active_x_ratio: List[float] = field(default_factory=lambda: [0.0, 1.00])
	active_y_ratio: List[float] = field(default_factory=lambda: [0.0, 1.00])
	table2_active_half_mode: str = "full"
	table2_active_x_ratio: List[float] = field(default_factory=lambda: [0.0, 1.00])
	table2_active_y_ratio: List[float] = field(default_factory=lambda: [0.0, 1.00])
	launch_areas_mm: Dict[str, List[float]] = field(
		default_factory=lambda: {
			"left_home": [0.0, 0.0, 600.0, 1143.0],
			"right_home": [1762.0, 0.0, 2362.0, 1143.0],
		}
	)

	def make_table2_config(self) -> "TableConfig":
		"""Creates an independent TableConfig instance for Table 2 (Right / Blue Team)."""
		return TableConfig(
			real_width_mm=self.real_width_mm,
			real_height_mm=self.real_height_mm,
			auto_detect_borders=False,
			src_corners_px=[list(p) for p in self.table2_corners_px],
			table2_corners_px=[list(p) for p in self.table2_corners_px],
			apply_table_roi_mask=self.apply_table_roi_mask,
			roi_padding_px=self.roi_padding_px,
			active_half_mode=self.table2_active_half_mode,
			active_x_ratio=list(self.table2_active_x_ratio),
			active_y_ratio=list(self.table2_active_y_ratio),
			table2_active_half_mode=self.table2_active_half_mode,
			table2_active_x_ratio=list(self.table2_active_x_ratio),
			table2_active_y_ratio=list(self.table2_active_y_ratio),
			launch_areas_mm={k: list(v) for k, v in self.launch_areas_mm.items()},
		)


@dataclass
class AutoZoomConfig:
	enabled: bool = True
	zoom_factor: float = 2.2
	idle_zoom_factor: float = 1.0
	pan_smoothing_alpha: float = 0.18
	zoom_smoothing_alpha: float = 0.12
	hold_frames_before_zoom_out: int = 90
	border_fill_mode: str = "cover_crop"

	@property
	def zoom_factor_stream1(self) -> float:
		return self.zoom_factor

	@property
	def zoom_factor_stream2(self) -> float:
		return self.zoom_factor


@dataclass
class StreamsConfig:
	stream_width: int = 1080
	stream_height: int = 1080
	jpeg_quality: int = 85
	target_fps: int = 30
	auto_zoom: AutoZoomConfig = field(default_factory=AutoZoomConfig)

	@property
	def stream1_width(self) -> int:
		return self.stream_width

	@property
	def stream1_height(self) -> int:
		return self.stream_height

	@property
	def stream2_width(self) -> int:
		return self.stream_width

	@property
	def stream2_height(self) -> int:
		return self.stream_height


@dataclass
class InferenceConfig:
	weights_pt: str = "/workspace/models/fll_yolo11s.pt"
	engine_trt: str = "/workspace/models/fll_yolo11s.engine"
	auto_export_trt: bool = True
	device: str = "cuda:0"
	imgsz: int = 960
	half_precision: bool = True
	conf_threshold: float = 0.25
	iou_threshold: float = 0.45
	max_det: int = 25
	use_motion_discovery: bool = True
	motion_min_area_px: int = 450
	motion_max_area_ratio: float = 0.18
	classes: Dict[str, int] = field(default_factory=lambda: {"robot": 0, "hand": 1, "attachment_jig": 2})


@dataclass
class HandVetoConfig:
	enabled: bool = True
	overlap_iou_threshold: float = 0.08
	proximity_radius_mm: float = 115.0
	cooldown_frames: int = 6


@dataclass
class TrackerConfig:
	track_thresh: float = 0.25
	match_thresh: float = 0.80
	max_age_frames: int = 90
	min_hits: int = 1
	ema_alpha: float = 0.50
	max_robot_speed_mm_s: float = 1800.0
	stationary_noise_gate_mm: float = 2.5
	hand_veto: HandVetoConfig = field(default_factory=HandVetoConfig)


@dataclass
class ServerConfig:
	enabled: bool = True
	host: str = "0.0.0.0"
	port: int = 8080
	stream_jpeg_quality: int = 85
	enable_heatmap: bool = True
	heatmap_decay: float = 0.998


@dataclass
class AppConfig:
	project: ProjectConfig = field(default_factory=ProjectConfig)
	video: VideoConfig = field(default_factory=VideoConfig)
	table: TableConfig = field(default_factory=TableConfig)
	streams: StreamsConfig = field(default_factory=StreamsConfig)
	inference: InferenceConfig = field(default_factory=InferenceConfig)
	tracker: TrackerConfig = field(default_factory=TrackerConfig)
	server: ServerConfig = field(default_factory=ServerConfig)


def _filter_dataclass_kwargs(cls, data: dict) -> dict:
	valid_keys = set(cls.__dataclass_fields__.keys())
	return {k: v for k, v in data.items() if k in valid_keys}


def load_config(path: str) -> AppConfig:
	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_path}")
	with open(config_path, "r", encoding="utf-8") as f:
		if yaml is not None:
			raw = yaml.safe_load(f) or {}
		else:
			logger.debug("PyYAML not installed; parsing config.yaml with built-in YAML parser (#21).")
			raw = _parse_yaml_fallback(f.read())

	proj = ProjectConfig(**_filter_dataclass_kwargs(ProjectConfig, raw.get("project", {})))
	vid = VideoConfig(**_filter_dataclass_kwargs(VideoConfig, raw.get("video", {})))
	tbl = TableConfig(**_filter_dataclass_kwargs(TableConfig, raw.get("table", {})))

	streams_raw = dict(raw.get("streams", {}))
	az_raw = dict(streams_raw.pop("auto_zoom", {})) if isinstance(streams_raw, dict) else {}
	if "zoom_factor_stream1" in az_raw and "zoom_factor" not in az_raw:
		az_raw["zoom_factor"] = az_raw.pop("zoom_factor_stream1")
	strms = StreamsConfig(
		**_filter_dataclass_kwargs(StreamsConfig, streams_raw),
		auto_zoom=AutoZoomConfig(**_filter_dataclass_kwargs(AutoZoomConfig, az_raw)),
	)

	inf = InferenceConfig(**_filter_dataclass_kwargs(InferenceConfig, raw.get("inference", {})))

	trk_raw = dict(raw.get("tracker", {}))
	hv_raw = dict(trk_raw.pop("hand_veto", {})) if isinstance(trk_raw, dict) else {}
	trk = TrackerConfig(
		**_filter_dataclass_kwargs(TrackerConfig, trk_raw),
		hand_veto=HandVetoConfig(**_filter_dataclass_kwargs(HandVetoConfig, hv_raw)),
	)

	srv = ServerConfig(**_filter_dataclass_kwargs(ServerConfig, raw.get("server", {})))
	return AppConfig(project=proj, video=vid, table=tbl, streams=strms, inference=inf, tracker=trk, server=srv)


def save_config_updates_to_yaml(
	path: str,
	corners_px: Optional[List[List[float]]] = None,
	table2_corners_px: Optional[List[List[float]]] = None,
	auto_detect: Optional[bool] = None,
	active_half_mode: Optional[str] = None,
	active_x_ratio: Optional[List[float]] = None,
	active_y_ratio: Optional[List[float]] = None,
	table2_active_half_mode: Optional[str] = None,
	table2_active_x_ratio: Optional[List[float]] = None,
	table2_active_y_ratio: Optional[List[float]] = None,
	zoom_factor: Optional[float] = None,
	idle_zoom_factor: Optional[float] = None,
	auto_zoom_enabled: Optional[bool] = None,
	border_fill_mode: Optional[str] = None,
	video_source: Union[str, int, None] = None,
	proximity_radius_mm: Optional[float] = None,
	max_robot_speed_mm_s: Optional[float] = None,
	stationary_noise_gate_mm: Optional[float] = None,
	conf_threshold: Optional[float] = None,
):
	"""Persists independent Table 1 & Table 2 corners, active zones, zoom, tracker & video source to config.yaml."""
	config_path = Path(path)
	if not config_path.exists() or yaml is None:
		return
	with open(config_path, "r", encoding="utf-8") as f:
		raw = yaml.safe_load(f) or {}

	raw.setdefault("video", {})
	raw.setdefault("table", {})
	raw.setdefault("streams", {}).setdefault("auto_zoom", {})

	# Remove obsolete legacy keys if present
	raw["video"].pop("source_table2", None)
	raw["table"].pop("double_table_corners_px", None)

	if video_source is not None:
		raw["video"]["source"] = str(video_source).strip()

	if corners_px is not None:
		raw["table"]["src_corners_px"] = [[round(float(x), 1), round(float(y), 1)] for x, y in corners_px]
	if table2_corners_px is not None:
		raw["table"]["table2_corners_px"] = [[round(float(x), 1), round(float(y), 1)] for x, y in table2_corners_px]
	if auto_detect is not None:
		raw["table"]["auto_detect_borders"] = bool(auto_detect)
	if active_half_mode is not None:
		raw["table"]["active_half_mode"] = str(active_half_mode)
	if active_x_ratio is not None:
		raw["table"]["active_x_ratio"] = [round(float(active_x_ratio[0]), 3), round(float(active_x_ratio[1]), 3)]
	if active_y_ratio is not None:
		raw["table"]["active_y_ratio"] = [round(float(active_y_ratio[0]), 3), round(float(active_y_ratio[1]), 3)]

	if table2_active_half_mode is not None:
		raw["table"]["table2_active_half_mode"] = str(table2_active_half_mode)
	if table2_active_x_ratio is not None:
		raw["table"]["table2_active_x_ratio"] = [
			round(float(table2_active_x_ratio[0]), 3),
			round(float(table2_active_x_ratio[1]), 3),
		]
	if table2_active_y_ratio is not None:
		raw["table"]["table2_active_y_ratio"] = [
			round(float(table2_active_y_ratio[0]), 3),
			round(float(table2_active_y_ratio[1]), 3),
		]

	if zoom_factor is not None:
		raw["streams"]["auto_zoom"]["zoom_factor"] = round(float(zoom_factor), 2)
	if idle_zoom_factor is not None:
		raw["streams"]["auto_zoom"]["idle_zoom_factor"] = round(float(idle_zoom_factor), 2)
	if auto_zoom_enabled is not None:
		raw["streams"]["auto_zoom"]["enabled"] = bool(auto_zoom_enabled)
	if border_fill_mode is not None:
		raw["streams"]["auto_zoom"]["border_fill_mode"] = str(border_fill_mode)

	raw.setdefault("tracker", {}).setdefault("hand_veto", {})
	raw.setdefault("inference", {})
	if proximity_radius_mm is not None:
		raw["tracker"]["hand_veto"]["proximity_radius_mm"] = round(float(proximity_radius_mm), 1)
	if max_robot_speed_mm_s is not None:
		raw["tracker"]["max_robot_speed_mm_s"] = round(float(max_robot_speed_mm_s), 1)
	if stationary_noise_gate_mm is not None:
		raw["tracker"]["stationary_noise_gate_mm"] = round(float(stationary_noise_gate_mm), 2)
	if conf_threshold is not None:
		raw["inference"]["conf_threshold"] = round(float(conf_threshold), 3)

	with open(config_path, "w", encoding="utf-8") as f:
		yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
