"""Vision thresholds and camera settings, loaded from Vision/configs/vision.yaml."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "configs" / "vision.yaml"


PANEL_FILE = Path(__file__).resolve().parents[2] / "control_panel.yaml"
PANEL_SECTIONS = ("run", "workcell", "vision", "control", "simulation")


def panel_section(panel: Path | None, name: str) -> dict:
    """One section of the control panel file (control_panel.yaml); {} without the file."""
    import yaml

    if panel is None or not panel.is_file():
        return {}
    data = yaml.safe_load(panel.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(data) - set(PANEL_SECTIONS)) if isinstance(data, dict) else ["?"]
    if unknown:
        raise ValueError(f"{panel.name}: unknown section(s): {', '.join(unknown)}")
    section = data.get(name) or {}
    if not isinstance(section, dict):
        raise TypeError(f"{panel.name}: '{name}' must be a mapping of settings")
    return section


def merged(base: dict, over: dict) -> dict:
    """`base` with the values of `over` on top; mappings inside are merged, not replaced."""
    result = dict(base)
    for key, value in over.items():
        both = isinstance(value, dict) and isinstance(result.get(key), dict)
        result[key] = merged(result[key], value) if both else value
    return result


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
class FishType:
    """The fish being picked, as vision measures it: outside these sizes it isn't picked
    (VS-65, VS-33), and the cups go where it balances."""

    length_range_m: tuple[float, float] = (0.54, 0.825)
    width_range_m: tuple[float, float] = (0.0945, 0.154)  # across the body at its middle
    thickness_range_m: tuple[float, float] = (0.081, 0.132)  # its height on the table
    # The head is heavier: the fish balances this far ahead of its outline's middle, as a
    # share of its measured length. The cups go there when the head end is known (0: middle)
    balance_ahead_share: float = 0.11


@dataclass
class Catfish(FishType):
    """Real catfish: Table SY-1 +- 10%. A catfish balances ~0.11 of its length ahead of its
    middle, but the cup array must still seal behind the pectoral fins: on the catfish
    model all six (a 140 mm array then) did up to 0.062 of the length ahead at 0.55 m, 0.094
    at 0.79 m. Tune on real fish: one that tips head-down when lifted wants more."""

    balance_ahead_share: float = 0.065


@dataclass
class StandIn(FishType):
    """The printed stand-in (Simulation/models/catfish.3mf, 0.79 m) as vision measures it in
    the simulation: about 14% long (its raised ends) and taller than real catfish; its true
    center of mass lies 0.126 x the measured length ahead of the measured middle."""

    length_range_m: tuple[float, float] = (0.80, 1.00)
    width_range_m: tuple[float, float] = (0.15, 0.22)
    thickness_range_m: tuple[float, float] = (0.15, 0.21)
    balance_ahead_share: float = 0.126


FISH_TYPES = ("stand_in", "catfish")


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
    # Close looks from 0.62 m see a flat fish's top up to ~0.55 m away (the D405 measures
    # from 0.07 m; past ~0.5 m its noise grows)
    depth_range_m: tuple[float, float] = (0.10, 0.75)
    depth_window_m: float = 0.025
    min_depth_valid: float = 0.80
    default_thickness_m: float = 0.10  # fish height on the table, used without depth
    outline_height_share: float = 0.5  # the outline is seen at about half the fish's height
    # The fish's surface under the cups, from depth: the cups are tilted onto it
    align_to_surface: bool = True  # False: always approach straight down
    max_surface_tilt_deg: float = 15.0  # a steeper surface is approached straight down (AS-21)
    surface_fit_rms_m: float = 0.012  # a rougher fit isn't trusted

    # Heading (fish_heading.py, H-20)
    require_heading: bool = True  # False: a fish with an unknown head end is still valid
    min_heading_confidence: float = 0.6

    # Geometry checks (VS-65: Table SY-1 +- 10%, VS-33)
    check_size: bool = True  # False: sizes aren't checked (videos not taken by the cell's camera)
    fish_type: str = "stand_in"  # the fish being picked (FISH_TYPES)
    stand_in: StandIn = field(default_factory=StandIn)
    catfish: Catfish = field(default_factory=Catfish)
    cup_footprint_m: tuple[float, float] = (0.20, 0.10)  # along and across the body
    cup_diameter_m: float = 0.04  # six of them, 3 x 2, filling the footprint
    cup_stroke_m: float = 0.012  # unevenness under the cups the bellows take up and still seal
    overlap_margin_px: float = 6.0  # outlines this close touch (Table VS-1)

    # Stability over the frames of one observation (VS-52)
    stable_frames: int = 3
    stable_position_m: float = 0.005
    stable_heading_deg: float = 5.0
    match_distance_m: float = 0.05  # the same fish in two fish lists

    hold: HoldSettings = field(default_factory=HoldSettings)

    def __post_init__(self) -> None:
        if self.fish_type not in FISH_TYPES:
            raise ValueError(f"fish_type must be one of {', '.join(FISH_TYPES)}")

    @property
    def fish(self) -> FishType:
        """The fish being picked (fish_type): its size limits and balance point."""
        return getattr(self, self.fish_type)


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


def load_settings(path: Path | None = None, panel: Path | None = PANEL_FILE) -> VisionSettings:
    """Settings from a YAML file (default: Vision/configs/vision.yaml; defaults if it's
    missing) with the control panel's `vision` section on top (`panel=None`: without it)."""
    import yaml

    path = path or DEFAULT_FILE
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif path != DEFAULT_FILE:
        raise FileNotFoundError(f"vision settings not found: {path}")
    return from_mapping(VisionSettings, merged(data, panel_section(panel, "vision")), path.name)
