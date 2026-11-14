"""
The workcell: frames, tool, zones, observation poses, placement slots and every motion
and gripper limit (SF-10, SF-11, AS-11, AS-51, AS-40), loaded from Arm/configs/workcell.yaml
before any motion. The file uses meters and degrees; here angles are radians. No ROS.

All positions are in the arm base frame {B} (MoveIt's base_link).
"""

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rfp_arm.poses import invert, tool_down, transform

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "configs" / "workcell.yaml"


class WorkcellError(ValueError):
    """The workcell file is missing something, or its values contradict each other."""


@dataclass(frozen=True)
class Rect:
    """Rectangle on the table: x and y ranges in {B}."""

    x: tuple[float, float]
    y: tuple[float, float]

    def contains(self, xy, margin: float = 0.0) -> bool:
        return (
            self.x[0] + margin <= xy[0] <= self.x[1] - margin
            and self.y[0] + margin <= xy[1] <= self.y[1] - margin
        )

    def overlaps(self, other: "Rect") -> bool:
        return (
            self.x[0] < other.x[1]
            and other.x[0] < self.x[1]
            and self.y[0] < other.y[1]
            and other.y[0] < self.y[1]
        )

    @property
    def area(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.x, self.y


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in {B}."""

    name: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def contains(self, point, margin: float = 0.0) -> bool:
        """Whether `point` is inside; with a margin > 0, also if it is that close outside."""
        return all(
            low - margin <= value <= high + margin
            for value, (low, high) in zip(point, (self.x, self.y, self.z))
        )

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((low + high) / 2 for low, high in (self.x, self.y, self.z))

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(high - low for low, high in (self.x, self.y, self.z))


@dataclass(frozen=True)
class ObservationPose:
    """Where the arm looks at the table from: taught joint positions, or a camera pose."""

    name: str
    joints: tuple[float, ...] | None = None  # rad, in the robot's joint order
    camera: np.ndarray | None = None  # 4x4 T_base_camera, looking straight down


@dataclass
class MotionLimits:
    tcp_speed_max: float = 0.25  # m/s, every motion (SF-21, AS-42)
    tcp_accel_max: float = 0.5  # m/s2, straight moves (AS-43)
    approach_speed: float = 0.05  # m/s, onto the fish (AS-22)
    carry_speed: float = 0.15  # m/s, straight moves with a fish
    free_speed: float = 0.15  # m/s, straight moves without a fish
    ptp_velocity_scale: float = 0.10  # share of the joint speed limits; lowered if the TCP is fast
    ptp_acceleration_scale: float = 0.10
    max_joint_travel_deg: float = 200.0  # a plan that turns a joint further is refused
    goal_tolerance: float = 0.002  # m: a move ending further from its target failed (AS-12)
    planning_time: float = 5.0  # s
    pilz_max_trans_vel: float = 1.0  # MoveIt's pilz_cartesian_limits.yaml, if it can't be read
    pilz_max_trans_acc: float = 2.25
    carry_ptp_fallback: bool = False  # carry point-to-point when no straight path is found


@dataclass
class GraspSettings:
    """Table AS-1. Distances in meters, times in seconds."""

    pregrasp_distance: float = 0.10  # above the grasp point
    press_distance: float = 0.010  # past the grasp point, compressing the bellows
    dry_run_standoff: float = 0.05  # dry run: stop this far short of the fish (CS-93)
    lift_length_share: float = 0.6  # lift height = this share of the fish's length + margin
    lift_margin: float = 0.05
    lift_height_max: float = 0.50  # above the table, whatever the fish's length
    release_clearance: float = 0.015  # cup faces above the fish's resting height (AS-24)
    retreat_distance: float = 0.10
    seal_timeout: float = 1.0  # no hold this long after the approach: E_NO_SEAL
    seal_dwell: float = 0.5  # without a vacuum switch: wait this long instead
    drop_time: float = 0.1  # holding not reported this long while carrying: E_DROP
    release_timeout: float = 1.0  # vacuum still there this long after release: E_RELEASE
    max_approach_tilt_deg: float = 15.0  # only near-vertical approaches are supported so far


@dataclass
class GripperSettings:
    """How the vacuum is switched (AS-30): by a board on a serial port, or by the FR5
    control box I/O of HW-33 and HW-34."""

    driver: str = "controller"  # "serial": a board on a serial port; "controller": FR5 I/O
    serial_port: str = ""  # /dev/ttyUSB0, COM3, ...
    serial_baud: int = 115200
    serial_on: str = "1"  # the line sent to turn the vacuum on
    serial_off: str = "0"
    serial_boot_s: float = 2.0  # wait after opening the port before the first command
    vacuum_do: int = 0
    blowoff_do: int = 1
    holding_di: int = 0  # vacuum switch output 1: on at the holding threshold (AS-31)
    vacuum_di: int = 1  # vacuum switch output 2: on while vacuum is 2.5 kPa or more (AS-33)
    vacuum_switch: bool = True  # False: no switch wired; the hold is checked by vision only
    blowoff_time: float = 0.2  # s; 0: no blow-off
    state_timeout: float = 0.5  # s without an I/O reading: gripper error (Table AS-4)


@dataclass
class Workcell:
    verified: bool  # a person measured this cell and set every value; the real arm needs it
    group: str
    base_frame: str
    tip_link: str
    joints: tuple[str, ...]
    tcp: np.ndarray  # 4x4 T_tip_tcp: cup face center, z out of the cups, x along the footprint
    camera: np.ndarray  # 4x4 T_tip_camera: the camera's optical frame (hand-eye calibration)
    calibrated: bool
    calibration_version: str
    tool_box: Box | None  # cups, bracket and camera as a box in the tip link frame (collisions)
    table_z: float
    table_clearance: float  # the cups and the camera stay at least this far above the table
    work_zone: Box
    pick_area: Rect
    placement_zone: Rect
    exclusion_zones: list[Box]
    observation_poses: list[ObservationPose]
    placement_heading: float  # rad (AS-52)
    slots: list[tuple[float, float]]
    motion: MotionLimits = field(default_factory=MotionLimits)
    grasp: GraspSettings = field(default_factory=GraspSettings)
    gripper: GripperSettings = field(default_factory=GripperSettings)
    source: Path | None = None

    # -- frames ------------------------------------------------------------
    def tcp_of_tip(self, t_base_tip: np.ndarray) -> np.ndarray:
        return t_base_tip @ self.tcp

    def tip_of_tcp(self, t_base_tcp: np.ndarray) -> np.ndarray:
        return t_base_tcp @ invert(self.tcp)

    def camera_of_tcp(self, t_base_tcp: np.ndarray) -> np.ndarray:
        return t_base_tcp @ invert(self.tcp) @ self.camera

    def tcp_of_camera(self, t_base_camera: np.ndarray) -> np.ndarray:
        return t_base_camera @ invert(self.camera) @ self.tcp

    def observation_tcp(self, pose: ObservationPose) -> np.ndarray | None:
        """TCP pose of an observation pose given as a camera pose (None for taught joints)."""
        return None if pose.camera is None else self.tcp_of_camera(pose.camera)

    # -- limits (SF-10, SF-13) ---------------------------------------------
    def pose_problem(
        self, t_base_tcp: np.ndarray, fish_length: float = 0.0, work_zone: bool = True
    ) -> str | None:
        """
        Why the tool may not be at this TCP pose, or None. The TCP, the camera and (with
        `fish_length`) both ends of the held fish must be outside every exclusion zone and
        inside the work zone, and over the table the TCP and camera must stay above it.
        `work_zone=False` skips the inside-the-work-zone part (a path may start outside).
        """
        tcp = t_base_tcp[:3, 3]
        points = [("TCP", tcp, True), ("camera", self.camera_of_tcp(t_base_tcp)[:3, 3], True)]
        if fish_length > 0:
            along = t_base_tcp[:3, 0] * fish_length / 2  # the fish lies along the TCP's x axis
            points += [("fish end", tcp + along, False), ("fish end", tcp - along, False)]
        zone = self.work_zone
        floor = self.table_z + self.table_clearance
        for name, point, is_tool in points:
            for exclusion in self.exclusion_zones:
                if exclusion.contains(point):
                    return (
                        f"{name} would enter the exclusion zone '{exclusion.name}' at {_mm(point)}"
                    )
            over_table = Rect(zone.x, zone.y).contains(point, -1e-9)
            if work_zone and (not over_table or point[2] > zone.z[1] + 1e-9):
                return f"{name} would leave the work zone at {_mm(point)}"
            if over_table and is_tool and point[2] < floor - 1e-9:
                return (
                    f"{name} would be {(point[2] - self.table_z) * 1000:.1f} mm above the table "
                    f"(minimum {self.table_clearance * 1000:.1f} mm)"
                )
        return None

    # -- self-check ----------------------------------------------------------
    def problems(self) -> list[str]:
        """What is wrong with the workcell's own values (empty = consistent)."""
        found = []
        zone = Rect(self.work_zone.x, self.work_zone.y)
        for name, rect in (("pick area", self.pick_area), ("placement zone", self.placement_zone)):
            corners = [(x, y) for x in rect.x for y in rect.y]
            if not all(zone.contains(c, -1e-9) for c in corners):
                found.append(f"the {name} is not inside the work zone")
        if self.pick_area.overlaps(self.placement_zone):
            found.append("the placement zone overlaps the pick area (AS-51)")
        if len(self.slots) < 2:
            found.append("the placement zone needs at least 2 slots (AS-51)")
        for i, slot in enumerate(self.slots, 1):
            if not self.placement_zone.contains(slot):
                found.append(f"slot {i} is outside the placement zone")
        if not self.observation_poses:
            found.append("no observation pose")
        for pose in self.observation_poses:
            tcp = self.observation_tcp(pose)
            if tcp is not None and (problem := self.pose_problem(tcp)):
                found.append(f"observation pose '{pose.name}': {problem}")
            if pose.joints is not None and len(pose.joints) != len(self.joints):
                found.append(f"observation pose '{pose.name}' needs {len(self.joints)} joints")
        if self.work_zone.z[0] < self.table_z - 1e-9:
            found.append("the work zone starts below the table")
        limits = self.motion
        for name in ("approach_speed", "carry_speed", "free_speed"):
            if not 0 < getattr(limits, name) <= limits.tcp_speed_max:
                found.append(f"motion.{name} must be above 0 and at most tcp_speed_max")
        if not 0 < limits.ptp_velocity_scale <= 1 or not 0 < limits.ptp_acceleration_scale <= 1:
            found.append("motion.ptp_*_scale must be above 0 and at most 1")
        if self.gripper.driver not in ("serial", "controller"):
            found.append("gripper.driver must be 'serial' or 'controller'")
        elif self.gripper.driver == "serial" and not self.gripper.serial_port:
            found.append("gripper.serial_port is empty")
        return found


def _mm(point) -> str:
    return "(" + ", ".join(f"{v * 1000:.0f}" for v in point) + ") mm"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_workcell(path: Path | None = None) -> Workcell:
    """The workcell of a YAML file (default: Arm/configs/workcell.yaml), checked."""
    import yaml

    path = path or DEFAULT_FILE
    if not path.is_file():
        raise WorkcellError(f"workcell file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        workcell = workcell_from_mapping(data)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        detail = f"missing '{e.args[0]}'" if isinstance(e, KeyError) else str(e)
        raise WorkcellError(f"{path.name}: {detail}") from None
    workcell.source = path
    problems = workcell.problems()
    if problems:
        raise WorkcellError(f"{path.name}: " + "; ".join(problems))
    return workcell


def workcell_from_mapping(data: dict) -> Workcell:
    _only(data, "workcell", {"verified", "robot", "tool", "camera", "table", "zones",
                             "observation", "placement", "motion", "grasp", "gripper"})  # fmt: skip
    robot, tool, camera, table = data["robot"], data["tool"], data["camera"], data["table"]
    zones, observation, placement = data["zones"], data["observation"], data["placement"]
    table_z = float(table["z"])
    work = zones["work"]
    body = tool.get("body")

    poses = []
    for i, entry in enumerate(observation["poses"], 1):
        _only(entry, f"observation pose {i}", {"name", "joints_deg", "camera_xy", "yaw_deg"})
        name = str(entry.get("name", f"pose {i}"))
        if "joints_deg" in entry:
            joints = tuple(math.radians(float(v)) for v in entry["joints_deg"])
            poses.append(ObservationPose(name, joints=joints))
        else:
            x, y = (float(v) for v in entry["camera_xy"])
            z = table_z + float(observation["height"])
            yaw = math.radians(float(entry.get("yaw_deg", 0.0)))
            poses.append(ObservationPose(name, camera=tool_down((x, y, z), yaw)))

    return Workcell(
        verified=bool(data.get("verified", False)),
        group=str(robot["group"]),
        base_frame=str(robot["base_frame"]),
        tip_link=str(robot["tip_link"]),
        joints=tuple(str(j) for j in robot["joints"]),
        tcp=_pose(tool["tcp"]),
        camera=_pose(camera["mount"]),
        calibrated=bool(camera.get("calibrated", False)),
        calibration_version=str(camera.get("calibration_version", "")),
        tool_box=None if body is None else Box("tool", *(_range(body[k]) for k in "xyz")),
        table_z=table_z,
        table_clearance=float(table.get("clearance", 0.005)),
        work_zone=Box(
            "work zone",
            _range(work["x"]),
            _range(work["y"]),
            (table_z, table_z + float(work["height"])),
        ),
        pick_area=_rect(zones["pick_area"]),
        placement_zone=_rect(zones["placement"]),
        exclusion_zones=[
            Box(str(z["name"]), _range(z["x"]), _range(z["y"]), _range(z["z"]))
            for z in zones.get("exclusion") or []
        ],
        observation_poses=poses,
        placement_heading=math.radians(float(placement.get("heading_deg", 0.0))),
        slots=[(float(x), float(y)) for x, y in placement["slots"]],
        motion=_settings(MotionLimits, data.get("motion"), "motion"),
        grasp=_settings(GraspSettings, data.get("grasp"), "grasp"),
        gripper=_settings(GripperSettings, data.get("gripper"), "gripper"),
    )


def _only(data, where: str, allowed: set[str]) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown entry {', '.join(unknown)}")


def _range(values) -> tuple[float, float]:
    low, high = (float(v) for v in values)
    if low >= high:
        raise ValueError(f"range [{low}, {high}] must be [min, max]")
    return low, high


def _rect(data: dict) -> Rect:
    return Rect(_range(data["x"]), _range(data["y"]))


def _pose(data: dict) -> np.ndarray:
    _only(data, "pose", {"xyz", "rpy_deg"})
    rpy = [math.radians(float(v)) for v in data.get("rpy_deg", (0, 0, 0))]
    return transform([float(v) for v in data["xyz"]], rpy)


def _settings(cls, data: dict | None, where: str):
    data = data or {}
    _only(data, where, {f.name for f in dataclasses.fields(cls)})
    return cls(**data)


def describe(workcell: Workcell) -> list[str]:
    """A few lines for the operator: what this workcell is."""
    pick, place = workcell.pick_area, workcell.placement_zone
    return [
        f"table at z = {workcell.table_z * 1000:.0f} mm; work zone {_span(workcell.work_zone)}",
        (
            f"pick area x {_pair(pick.x)}, y {_pair(pick.y)} mm; placement zone "
            f"x {_pair(place.x)}, y {_pair(place.y)} mm, {len(workcell.slots)} slots, "
            f"heading {math.degrees(workcell.placement_heading):.0f} deg"
        ),
        (
            f"{len(workcell.observation_poses)} observation pose(s); "
            f"camera calibration: {workcell.calibration_version or 'none'}"
            f"{'' if workcell.calibrated else ' (NOT calibrated)'}"
        ),
    ]


def _pair(values) -> str:
    return f"{values[0] * 1000:.0f}..{values[1] * 1000:.0f}"


def _span(box: Box) -> str:
    return f"x {_pair(box.x)}, y {_pair(box.y)}, z {_pair(box.z)} mm"
