from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

try:
    import yaml
except ImportError:
    yaml = None


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
    source_table2: Union[str, int] = ""
    training_videos_dir: str = "/workspace/data/training_videos"
    width: int = 1920
    height: int = 1080
    fps: int = 60
    use_hw_decode: bool = True
    save_annotated_video: bool = True
    output_dir: str = "/workspace/data/outputs"


@dataclass
class TableConfig:
    real_width_mm: float = 2362.0
    real_height_mm: float = 1143.0
    auto_detect_borders: bool = True
    src_corners_px: List[List[float]] = field(
        default_factory=lambda: [[210.0, 140.0], [1710.0, 140.0], [1860.0, 960.0], [60.0, 960.0]]
    )
    table2_corners_px: List[List[float]] = field(
        default_factory=lambda: [[210.0, 140.0], [1710.0, 140.0], [1860.0, 960.0], [60.0, 960.0]]
    )
    apply_table_roi_mask: bool = True
    roi_padding_px: int = 15
    active_half_mode: str = "left_half"
    active_x_ratio: List[float] = field(default_factory=lambda: [0.0, 0.50])
    active_y_ratio: List[float] = field(default_factory=lambda: [0.0, 1.00])
    launch_areas_mm: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "left_home": [0.0, 0.0, 650.0, 1143.0],
            "right_home": [1712.0, 0.0, 2362.0, 1143.0],
        }
    )


@dataclass
class AutoZoomConfig:
    enabled: bool = True
    # Single unified zoom_factor applied identically to both team streams
    zoom_factor: float = 2.2
    idle_zoom_factor: float = 1.0
    pan_smoothing_alpha: float = 0.12
    zoom_smoothing_alpha: float = 0.08
    hold_frames_before_zoom_out: int = 45
    border_fill_mode: str = "cover_crop"

    # Backwards-compatibility properties if legacy code accesses zoom_factor_stream1/2
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
    target_fps: int = 60
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
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    max_det: int = 20
    classes: Dict[str, int] = field(
        default_factory=lambda: {"robot": 0, "hand": 1, "attachment_jig": 2}
    )


@dataclass
class HandVetoConfig:
    enabled: bool = True
    overlap_iou_threshold: float = 0.05
    proximity_radius_mm: float = 180.0
    cooldown_frames: int = 15


@dataclass
class TrackerConfig:
    track_thresh: float = 0.45
    match_thresh: float = 0.80
    max_age_frames: int = 45
    min_hits: int = 4
    ema_alpha: float = 0.35
    max_robot_speed_mm_s: float = 1200.0
    stationary_noise_gate_mm: float = 4.0
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


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if yaml is None:
        return AppConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    proj = ProjectConfig(**raw.get("project", {}))
    vid = VideoConfig(**raw.get("video", {}))
    tbl = TableConfig(**raw.get("table", {}))

    streams_raw = dict(raw.get("streams", {}))
    az_raw = dict(streams_raw.pop("auto_zoom", {})) if isinstance(streams_raw, dict) else {}
    # Migrate legacy zoom_factor_stream1/2 keys if present
    if "zoom_factor_stream1" in az_raw and "zoom_factor" not in az_raw:
        az_raw["zoom_factor"] = az_raw.pop("zoom_factor_stream1")
    az_raw.pop("zoom_factor_stream1", None)
    az_raw.pop("zoom_factor_stream2", None)
    streams_raw.pop("stream1_width", None)
    streams_raw.pop("stream1_height", None)
    streams_raw.pop("stream2_width", None)
    streams_raw.pop("stream2_height", None)
    strms = StreamsConfig(**streams_raw, auto_zoom=AutoZoomConfig(**az_raw))

    inf = InferenceConfig(**raw.get("inference", {}))

    trk_raw = dict(raw.get("tracker", {}))
    hv_raw = dict(trk_raw.pop("hand_veto", {})) if isinstance(trk_raw, dict) else {}
    trk = TrackerConfig(**trk_raw, hand_veto=HandVetoConfig(**hv_raw))

    srv = ServerConfig(**raw.get("server", {}))
    return AppConfig(project=proj, video=vid, table=tbl, streams=strms, inference=inf, tracker=trk, server=srv)


def save_config_updates_to_yaml(
    path: str,
    corners_px: List[List[float]] | None = None,
    table2_corners_px: List[List[float]] | None = None,
    auto_detect: bool | None = None,
    active_half_mode: str | None = None,
    active_x_ratio: List[float] | None = None,
    active_y_ratio: List[float] | None = None,
    zoom_factor: float | None = None,
    idle_zoom_factor: float | None = None,
    auto_zoom_enabled: bool | None = None,
    border_fill_mode: str | None = None,
    video_source: Union[str, int, None] = None,
    video_source_table2: Union[str, int, None] = None,
):
    """Persists updated table crop, active half, and unified stream zoom settings to config.yaml."""
    config_path = Path(path)
    if not config_path.exists() or yaml is None:
        return
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw.setdefault("video", {})
    raw.setdefault("table", {})
    raw.setdefault("streams", {}).setdefault("auto_zoom", {})

    if video_source is not None:
        raw["video"]["source"] = str(video_source).strip()
    if video_source_table2 is not None:
        raw["video"]["source_table2"] = str(video_source_table2).strip()

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

    if zoom_factor is not None:
        raw["streams"]["auto_zoom"]["zoom_factor"] = round(float(zoom_factor), 2)
    if idle_zoom_factor is not None:
        raw["streams"]["auto_zoom"]["idle_zoom_factor"] = round(float(idle_zoom_factor), 2)
    if auto_zoom_enabled is not None:
        raw["streams"]["auto_zoom"]["enabled"] = bool(auto_zoom_enabled)
    if border_fill_mode is not None:
        raw["streams"]["auto_zoom"]["border_fill_mode"] = str(border_fill_mode)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
