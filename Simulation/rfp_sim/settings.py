"""Simulation settings, from Simulation/sim.yaml with the control panel's `simulation` section."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILE = SIM_DIR / "sim.yaml"
MESH_CACHE = SIM_DIR / "models" / "fr5"  # FAIRINO's FR5 meshes, downloaded (not in Git)


@dataclass
class SimSettings:
    # The fish
    fish_model: str = "models/catfish.3mf"  # .3mf or .stl, relative to Simulation/
    fish_length: float = 0.79  # m snout to tail fin after scaling: the printed stand-ins
    fish_head: str = "-x"  # the model's axis toward the head
    fish_up: str = "+z"  # the model's axis pointing up as the fish lies on the table
    fish_count: int = 1
    seed: int = 1  # where the fish lie; another number, another layout
    # "view": each fish where an observation pose sees all of it, within
    # max_heading_offset_deg of the image's long side (the easy case). "anywhere": anywhere in
    # the pick area at any heading, as fish really lie
    placement: str = "anywhere"
    max_heading_offset_deg: float = 15.0
    fish_color: tuple[float, float, float] = (0.93, 0.93, 0.9)  # the white printed stand-in

    # The arm and the gripper
    robot_meshes: str = ""  # folder with the FR5's STL files; "": ROS's, else the download
    download_robot_meshes: bool = True  # fetch FAIRINO's FR5 meshes once (models/fr5/)
    speed: float = 1.0  # simulated time per real second; 0: as fast as it renders
    step_s: float = 0.05  # simulated seconds between rendered arm poses
    cup_rows: tuple[float, ...] = (-0.05, 0.0, 0.05)  # cup centers along the tool's x (m)
    cup_columns: tuple[float, ...] = (-0.0225, 0.0225)  # and along its y
    cup_radius: float = 0.02
    cup_length: float = 0.04  # bellows, from the plate to the cup faces
    cup_stroke: float = 0.012  # how far a cup compresses
    cup_stiffness: float = 500.0  # N/m per cup
    seal_cups: int = 5  # cups that must touch for the vacuum to hold the fish
    seal_gap: float = 0.002  # m: a cup face this close to the fish touches it
    drop: str = ""  # "lift" or "carry": lose the first fish then, to test drop detection

    # The camera
    width: int = 1280
    height: int = 720
    hfov_deg: float = 87.0  # D405
    depth_range_m: tuple[float, float] = (0.07, 1.0)  # outside it the D405 gives no depth
    depth_noise_m: float = 0.0005
    depth_edge_holes_px: int = 3  # no depth this close to a depth step (stereo occlusion); 0: off
    detector: str = "model"  # "model": the trained YOLO; "truth": the renderer's own masks

    # The 3D view
    window: tuple[int, int] = (1280, 800)


def load_settings(path: Path | None = None, panel: Path | None = None) -> SimSettings:
    """Settings from sim.yaml (defaults if missing) with the control panel's `simulation`
    section on top; `panel` defaults to the repository's control_panel.yaml."""
    import yaml
    from rfp_controlsys.settings import PANEL_FILE, merged, panel_section

    path = path or DEFAULT_FILE
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif path != DEFAULT_FILE:
        raise FileNotFoundError(f"simulation settings not found: {path}")
    data = merged(data, panel_section(PANEL_FILE if panel is None else panel, "simulation"))
    names = {f.name for f in dataclasses.fields(SimSettings)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{path.name}: unknown setting(s): {', '.join(unknown)}")
    return SimSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def model_path(settings: SimSettings) -> Path:
    path = Path(settings.fish_model)
    return path if path.is_absolute() else SIM_DIR / path
