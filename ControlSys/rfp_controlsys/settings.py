"""The Control System's fixed limits (CS-30), loaded from ControlSys/configs/control.yaml."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "configs" / "control.yaml"


@dataclass
class ControlSettings:
    max_attempts_per_fish: int = 3  # CS-62
    max_failed_cycles: int = 3  # in a row, then pause (CS-63)
    empty_passes_to_end: int = 3  # passes over the observation poses with no fish (SY-22)
    max_reobserve: int = 3  # "reobserve" answers in a row, then the fallback rule (Table CS-1)
    fish_list_max_age_s: float = 1.0  # CS-12, CS-53
    recheck_position_m: float = 0.010  # CS-41
    recheck_heading_deg: float = 10.0
    same_fish_m: float = 0.05  # attempts follow a fish found again within this distance
    settle_s: float = 0.3  # wait after a move before frames are used (VS-13)
    require_hold_confirmation: bool = True  # no vacuum switch and no vision cue: don't lift
    vision_hold_check: bool = True  # watch the held fish with the camera
    save_images: bool = True  # the annotated observation of every cycle (CS-82)


def load_settings(path: Path | None = None) -> ControlSettings:
    """Settings from a YAML file (default: ControlSys/configs/control.yaml); defaults if missing."""
    import yaml

    path = path or DEFAULT_FILE
    if not path.is_file():
        if path != DEFAULT_FILE:
            raise FileNotFoundError(f"control settings not found: {path}")
        return ControlSettings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path.name}: expected a mapping of settings")
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(ControlSettings)})
    if unknown:
        raise ValueError(f"{path.name}: unknown setting(s): {', '.join(unknown)}")
    return ControlSettings(**data)
