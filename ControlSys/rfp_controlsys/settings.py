"""The Control System's fixed limits (CS-30), loaded from ControlSys/configs/control.yaml."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "configs" / "control.yaml"


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


def load_settings(path: Path | None = None, panel: Path | None = PANEL_FILE) -> ControlSettings:
    """Settings from a YAML file (default: ControlSys/configs/control.yaml; defaults if it's
    missing) with the control panel's `control` section on top (`panel=None`: without it)."""
    import yaml

    path = path or DEFAULT_FILE
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif path != DEFAULT_FILE:
        raise FileNotFoundError(f"control settings not found: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"{path.name}: expected a mapping of settings")
    data = merged(data, panel_section(panel, "control"))
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(ControlSettings)})
    if unknown:
        raise ValueError(f"{path.name}: unknown setting(s): {', '.join(unknown)}")
    return ControlSettings(**data)


RUN_DEFAULTS = {"mode": None, "device": None, "model": None, "cycles": None, "step": False,
                "window": True, "any_size": False}  # fmt: skip


def run_defaults(panel: Path | None = PANEL_FILE) -> dict:
    """The control panel's `run` section: the defaults of pick_place.py's command line."""
    section = panel_section(panel, "run")
    unknown = sorted(set(section) - set(RUN_DEFAULTS))
    if unknown:
        raise ValueError(f"control panel: unknown run setting(s): {', '.join(unknown)}")
    return {**RUN_DEFAULTS, **section}
