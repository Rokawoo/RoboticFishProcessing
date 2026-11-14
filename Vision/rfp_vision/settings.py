"""Vision thresholds and camera settings, loaded from Vision/configs/vision.yaml."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "configs" / "vision.yaml"


@dataclass
class HoldSettings:
    """Vision check that a held fish still moves with the gripper (hold_monitor.py)."""

    use_mask: bool = True  # the fish's outline stays where it was in the image
    use_depth: bool = True  # the surface near the camera stays at its distance
    near_m: float = 0.25  # a held fish is closer to the camera than this; the table is farther
    depth_tolerance_m: float = 0.04  # bellows stretch and the fish sags when lifted
    min_depth_share: float = 0.5  # of the held surface that must still be there
    min_region_share: float = 0.03  # of the image the held surface must cover to be usable
    min_iou: float = 0.5  # outline now against the outline at the grasp
    lost_frames: int = 3  # frames in a row without the fish = dropped (~100 ms at 30 fps)


@dataclass
class VisionSettings:
    # Camera (VS-11); hfov only for sources without intrinsics (D405: 87 deg)
    width: int = 1280
    height: int = 720
    fps: int = 30
    hfov_deg: float = 87.0

    # Detection (VS-20, VS-61, VS-24)
    predict_conf: float = 0.25  # the model's own threshold; weaker detections aren't listed
    min_confidence: float = 0.50
    min_area_share: float = 0.03  # of the frame; smaller detections aren't fish
    border_margin_px: float = 8.0

    # Depth (VS-45, VS-66, C.4)
    use_depth: bool = True  # False, or a camera without depth: the fish lies on the table plane
    depth_range_m: tuple[float, float] = (0.10, 0.50)
    depth_window_m: float = 0.025
    min_depth_valid: float = 0.80
    default_thickness_m: float = 0.10  # fish height on the table, used without depth
    outline_height_share: float = 0.5  # the outline is seen at about half the fish's height

    # Heading (fish_heading.py, H-20)
    require_heading: bool = True  # False: a fish with an unknown head end is still valid
    min_heading_confidence: float = 0.6

    # Geometry checks (VS-65: Table SY-1 +- 10%, VS-33)
    check_size: bool = True  # False: sizes aren't checked (videos not taken by the cell's camera)
    length_range_m: tuple[float, float] = (0.54, 0.825)
    width_range_m: tuple[float, float] = (0.0945, 0.154)
    thickness_range_m: tuple[float, float] = (0.081, 0.132)
    cup_footprint_m: tuple[float, float] = (0.14, 0.085)  # along and across the body (HW-31)
    overlap_margin_px: float = 6.0  # outlines this close touch (Table VS-1)

    # Stability over the frames of one observation (VS-52)
    stable_frames: int = 3
    stable_position_m: float = 0.005
    stable_heading_deg: float = 5.0
    match_distance_m: float = 0.05  # the same fish in two fish lists

    hold: HoldSettings = field(default_factory=HoldSettings)


def from_mapping(cls, data: dict, where: str):
    """A settings dataclass from a mapping; unknown names are an error, not ignored."""
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping of settings")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown setting(s): {', '.join(unknown)}")
    values = {}
    for name, value in data.items():
        if dataclasses.is_dataclass(fields[name].default_factory):
            value = from_mapping(fields[name].default_factory, value, f"{where}.{name}")
        elif isinstance(value, list):
            value = tuple(value)
        values[name] = value
    return cls(**values)


def load_settings(path: Path | None = None) -> VisionSettings:
    """Settings from a YAML file (default: Vision/configs/vision.yaml); defaults if it's missing."""
    import yaml

    path = path or DEFAULT_FILE
    if not path.is_file():
        if path != DEFAULT_FILE:
            raise FileNotFoundError(f"vision settings not found: {path}")
        return VisionSettings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return from_mapping(VisionSettings, data, path.name)
