"""
The cell's layout as a person sets it up (pick_place.py --mode setup): the table (work
zone), pick area, placement zone, slots and heading, exclusion zones and observation poses.

Read from the workcell file with the control panel on top, turned about the arm's base in
quarter turns, checked against what the arm reaches and the camera sees (`check_layout`),
and saved where each value is set: the control panel where it overrides the file, else the
file. Saving changes only those lines, so every comment stays, keeps a .bak copy of each
file, and reads the files back: if anything but the layout changed, they are restored.
No ROS.
"""

import copy
import json
import math
import re
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from rfp_arm.kinematics import FR5, Chain
from rfp_arm.poses import tool_down
from rfp_arm.workcell import (
    DEFAULT_FILE,
    HEADING_DEG,
    PANEL_FILE,
    Rect,
    Workcell,
    merged,
    panel_section,
    workcell_from_mapping,
)


class LayoutError(ValueError):
    """The layout can't be saved; nothing was changed."""


@dataclass
class Layout:
    """The layout part of the workcell file, in its units (meters, degrees)."""

    work: dict  # {"x": [low, high], "y": [low, high], "height": h}: the table
    pick_area: dict  # {"x": [low, high], "y": [low, high]}
    placement_zone: dict
    exclusions: list  # [{"name", "x", "y", "z"}]
    poses: list  # [{"name", "camera_xy", "yaw_deg"}] or [{"name", "joints_deg"}]
    heading_deg: float
    slots: list  # [[x, y]]


# Where each part is in the workcell file (and under `workcell:` in the control panel)
PATHS = {
    "work": ("zones", "work"),
    "pick_area": ("zones", "pick_area"),
    "placement_zone": ("zones", "placement"),
    "exclusions": ("zones", "exclusion"),
    "poses": ("observation", "poses"),
    "heading_deg": ("placement", "heading_deg"),
    "slots": ("placement", "slots"),
}
LISTS = ("exclusions", "poses", "slots")


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _rect(data: dict) -> dict:
    return {"x": _floats(data["x"]), "y": _floats(data["y"])}


def layout_of(data: dict) -> Layout:
    """The layout of a workcell mapping (the file with the control panel's section on top)."""
    zones, placement = data["zones"], data["placement"]
    poses = []
    for i, entry in enumerate(data["observation"]["poses"], 1):
        name = str(entry.get("name", f"pose {i}"))
        if "joints_deg" in entry:
            poses.append({"name": name, "joints_deg": _floats(entry["joints_deg"])})
        else:
            poses.append(
                {
                    "name": name,
                    "camera_xy": _floats(entry["camera_xy"]),
                    "yaw_deg": float(entry.get("yaw_deg", 0.0)),
                }
            )
    return Layout(
        work={**_rect(zones["work"]), "height": float(zones["work"]["height"])},
        pick_area=_rect(zones["pick_area"]),
        placement_zone=_rect(zones["placement"]),
        exclusions=[
            {"name": str(zone["name"]), **_rect(zone), "z": _floats(zone["z"])}
            for zone in zones.get("exclusion") or []
        ],
        poses=poses,
        heading_deg=float(placement.get("heading_deg", HEADING_DEG)),
        slots=[_floats(slot) for slot in placement["slots"]],
    )


def with_layout(data: dict, layout: Layout) -> dict:
    """A copy of a workcell mapping with `layout` in it."""
    data = copy.deepcopy(data)
    for part, path in PATHS.items():
        value = copy.deepcopy(getattr(layout, part))
        if part == "exclusions" and not value and "exclusion" not in data["zones"]:
            continue
        _put(data, path, value)
    return data


def _put(data: dict, path: tuple, value) -> None:
    for key in path[:-1]:
        data = data.setdefault(key, {})
    data[path[-1]] = value


def _has(data: dict, path: tuple) -> bool:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return False
        data = data[key]
    return True


def load_layout_data(workcell_file: Path = DEFAULT_FILE, panel_file: Path | None = PANEL_FILE):
    """(The workcell mapping with the control panel's section on top, the layout in it)."""
    import yaml

    data = yaml.safe_load(Path(workcell_file).read_text(encoding="utf-8"))
    data = merged(data, panel_section(panel_file, "workcell"))
    return data, layout_of(data)


# ---------------------------------------------------------------------------
# Turning about the arm's base
# ---------------------------------------------------------------------------
def _wrap_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def turned(layout: Layout, quarter_turns: int) -> Layout:
    """The whole layout turned about the arm's base (its z axis) by quarter turns, counter-
    clockwise seen from above: a table on another side of the arm. Zones stay boxes along
    the axes; taught observation poses turn with the first joint."""
    k = quarter_turns % 4

    def point(xy) -> list[float]:
        x, y = (float(v) for v in xy)
        for _ in range(k):
            x, y = -y, x
        return [round(x, 9), round(y, 9)]

    def rect(data: dict) -> dict:
        corners = [point((x, y)) for x in data["x"] for y in data["y"]]
        xs, ys = zip(*corners)
        return {**copy.deepcopy(data), "x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}

    poses = []
    for pose in layout.poses:
        pose = copy.deepcopy(pose)
        if "camera_xy" in pose:
            pose["camera_xy"] = point(pose["camera_xy"])
            pose["yaw_deg"] = _wrap_deg(pose["yaw_deg"] + 90.0 * k)
        else:
            pose["joints_deg"][0] = _wrap_deg(pose["joints_deg"][0] + 90.0 * k)
        poses.append(pose)
    return Layout(
        work=rect(layout.work),
        pick_area=rect(layout.pick_area),
        placement_zone=rect(layout.placement_zone),
        exclusions=[rect(zone) for zone in layout.exclusions],
        poses=poses,
        heading_deg=_wrap_deg(layout.heading_deg + 90.0 * k),
        slots=[point(slot) for slot in layout.slots],
    )


# ---------------------------------------------------------------------------
# What the arm reaches and the camera sees
# ---------------------------------------------------------------------------
@dataclass
class FishSize:
    """What the checks assume of the fish: the largest that vision passes."""

    length: float  # m, as vision measures it
    width: float  # m, across the body
    thickness: float  # m, its height lying on the table
    # m: how far ahead of its middle, toward the head, the cups may hold it (its balance
    # point): placed, the tool is that far past the slot
    cups_ahead: float = 0.0


@dataclass
class CameraView:
    hfov_deg: float
    width: int  # px
    height: int


@dataclass
class Finding:
    error: bool  # True: the cell can't work like this; False: a warning
    text: str
    where: tuple[float, float] | None = None  # m: the place on the table it is about
    # A hint for the process (the placement heading), not about where things are: printed,
    # not shown in the setup window
    minor: bool = False


YAWS = tuple(math.radians(a) for a in (0, 90, 180, -90))
EMPTY_WRIST = math.radians(90.0)  # as grasp.EMPTY_WRIST: the empty gripper near the middle
SLOT_WRIST_SLACK = 30.0  # deg: a slot's wrist angles this close to +-90 aren't worth a word
# Elbow up, tool straight down; turned to face each target before solving from it
REACH_SEED = np.radians([0.0, -100.0, -100.0, -70.0, 90.0, 0.0])


def lift_height(cell: Workcell, fish_length: float) -> float:
    """The height (z in {B}) a fish this long is lifted to and carried at (plan_grasp)."""
    g = cell.grasp
    return cell.table_z + min(g.lift_length_share * fish_length + g.lift_margin, g.lift_height_max)


def reach_joints(
    cell: Workcell, tcp: np.ndarray, chain: Chain = FR5, wrist_range: bool = True
) -> np.ndarray | None:
    """Joint positions putting the TCP at `tcp` with the arm facing it, elbow up, the wrist
    inside its range (the vacuum tube; `wrist_range=False`: anywhere), the arm clear of
    itself (the cell's collision model); None if there are none."""
    tip = cell.tip_of_tcp(tcp)
    seed = REACH_SEED.copy()
    seed[0] = math.atan2(tip[1, 3], tip[0, 3]) - _seed_bearing(chain)
    q = chain.ik(tip, seed, iterations=60, steps=False)
    if q is None:
        return None
    low, high = cell.wrist_range
    if wrist_range and not low <= q[-1] <= high:
        return None
    if cell.collision is not None and cell.collision.hit(chain, q):
        return None
    return q


def _seed_bearing(chain: Chain) -> float:
    """Where the seed posture's tip is, seen from the base: turning the first joint by the
    difference faces the seed toward a target."""
    tip = chain.fk(REACH_SEED)
    return math.atan2(tip[1, 3], tip[0, 3])


def _reach(cell: Workcell, chain: Chain, xy, z: float, yaws) -> np.ndarray | None:
    """The joints putting the tool straight down at (xy, z), turned to the first of `yaws`
    that the arm reaches; None if it reaches none."""
    for yaw in yaws:
        q = reach_joints(cell, tool_down((xy[0], xy[1], z), yaw), chain)
        if q is not None:
            return q
    return None


def reach_radius(cell: Workcell, z: float, chain: Chain = FR5) -> float:
    """How far from the base's axis (m) the TCP reaches with the tool straight down at height
    `z` in {B} (the FR5's reach is nearly the same in every direction)."""
    yaws = YAWS[:2]  # straight down, the tool's turn only moves the last joint
    low, high = 0.15, 1.5
    if _reach(cell, chain, (low, 0.0), z, yaws) is None:
        return 0.0
    for _ in range(11):  # to about a millimeter
        middle = (low + high) / 2
        reached = _reach(cell, chain, (middle, 0.0), z, yaws) is not None
        low, high = (middle, high) if reached else (low, middle)
    return low


def view_footprint(t_base_camera: np.ndarray, view: CameraView, plane_z: float) -> np.ndarray:
    """(4, 2) the image's corners on the plane z = plane_z, in {B}."""
    f = view.width / 2 / math.tan(math.radians(view.hfov_deg) / 2)
    corners = []
    for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ray = t_base_camera[:3, :3] @ np.array([u * view.width / 2 / f, v * view.height / 2 / f, 1])
        if ray[2] >= -1e-9:
            raise ValueError("the camera doesn't look down at the table")
        s = (plane_z - t_base_camera[2, 3]) / ray[2]
        corners.append(t_base_camera[:2, 3] + s * ray[:2])
    return np.array(corners)


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Which of the (n, 2) points are inside a convex polygon (any winding)."""
    signs = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge, to_point = b - a, points - a
        signs.append(edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0])
    signs = np.array(signs)
    return (signs >= -1e-12).all(axis=0) | (signs <= 1e-12).all(axis=0)


def _slot_wrist(cell, chain, k, slot, z, heading, middle) -> Finding | None:
    """Placed, the wrist ends at one of two angles half a turn apart (the cups either way
    round). Far from +-90 from its middle, a fish that must arrive the far way round to be
    turned at most half a turn needs the empty gripper past 90 at the grasp (plan_grasp):
    say so, and which placement heading would put them at +-90."""

    def wrist(yaw: float) -> float | None:
        q = reach_joints(cell, tool_down((slot[0], slot[1], z), yaw), chain, wrist_range=False)
        return None if q is None else float(q[-1]) - middle

    near = wrist(heading)
    if near is None:
        return None
    off = abs(abs(math.degrees(near)) - 90.0)  # the other angle is as far off
    if off <= SLOT_WRIST_SLACK:
        return None
    nudged = wrist(heading + 0.01)
    if nudged is None:
        return None
    sign = 1.0 if math.remainder(nudged - near, math.tau) > 0 else -1.0
    turns = [sign * (math.radians(target) - near) for target in (90.0, -90.0)]
    better = math.degrees(math.remainder(heading + min(turns, key=abs), math.tau))
    other = math.degrees(math.remainder(near + math.pi, math.tau))
    return Finding(
        False,
        f"slot {k}: placed, the wrist ends at {math.degrees(near):.0f} or {other:.0f} deg (the "
        f"cups either way round): a fish arriving the far way round needs the empty gripper "
        f"up to {90 + off:.0f} deg out at the grasp; a placement heading of {better:.0f} deg "
        "would keep it within 90",
        tuple(slot),
        minor=True,
    )


def _hidden(pick: Rect, views: list[np.ndarray], half_width: float, step: float = 0.01):
    """A place in the pick area (x, y) whose whole surroundings within `half_width`, a fish
    lying there, are outside every view; None if there is none. A fish partly in a view is
    found and gets a closer look."""
    xs = np.arange(pick.x[0], pick.x[1] + 1e-9, step)
    ys = np.arange(pick.y[0], pick.y[1] + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    seen = np.zeros(len(grid), bool)
    for corners in views:
        seen |= _inside_convex(grid, corners)
    unseen = ~seen.reshape(gx.shape)
    radius = max(1, int(half_width / step))
    hidden = unseen.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(unseen)  # outside the pick area counts as unseen
            shifted[:] = True
            x0, x1 = max(0, -dx), min(unseen.shape[0], unseen.shape[0] - dx)
            y0, y1 = max(0, -dy), min(unseen.shape[1], unseen.shape[1] - dy)
            shifted[x0:x1, y0:y1] = unseen[x0 + dx : x1 + dx, y0 + dy : y1 + dy]
            hidden &= shifted
    if not hidden.any():
        return None
    i, j = np.argwhere(hidden)[0]
    return float(xs[i]), float(ys[j])


def observation_camera(cell: Workcell, pose, chain: Chain = FR5) -> np.ndarray:
    """T_base_camera of an observation pose (a camera pose, or taught joints)."""
    if pose.camera is not None:
        return pose.camera
    return cell.camera_of_tcp(cell.tcp_of_tip(chain.fk(pose.joints)))


def _overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def check_layout(
    cell: Workcell, fish: FishSize, view: CameraView | None = None, chain: Chain = FR5
) -> list[Finding]:
    """
    What keeps this layout from working, errors first: the workcell's own problems, poses
    the arm can't reach (observation poses; the pick area at a fish's top and at the lift
    height; each slot at the carry and release heights, the way the fish goes in), the arm
    in an exclusion zone at those poses (its links and the gripper: the collision model),
    fish that would reach out of the table or into an exclusion zone when lifted or placed,
    placed fish in the pick area, slots closer than a fish is wide, and the part of the
    pick area no observation pose sees.
    """
    found = [Finding(True, problem) for problem in cell.problems()]
    table = cell.table_z
    lift = lift_height(cell, fish.length)
    top = table + fish.thickness  # the cups on a fish's top
    release = table + fish.thickness + cell.grasp.release_clearance
    postures = []  # (what, joints) the arm takes, checked against the exclusion zones

    low, high = cell.wrist_range
    middle = (low + high) / 2
    for pose in cell.observation_poses:
        if pose.camera is not None:
            joints = reach_joints(cell, cell.observation_tcp(pose), chain, wrist_range=False)
        elif len(pose.joints) == len(chain.joints):  # taught (else one of the cell's problems)
            joints = np.asarray(pose.joints, float)
        else:
            joints = None
        if joints is not None:
            postures.append((f"observation pose '{pose.name}'", joints))
        if pose.camera is not None:
            if joints is None:
                found.append(
                    Finding(
                        True,
                        f"observation pose '{pose.name}': out of the arm's reach",
                        tuple(pose.camera[:2, 3]),
                    )
                )
            elif abs(joints[-1] - middle) > EMPTY_WRIST + 1e-6:
                found.append(
                    Finding(
                        True,
                        f"observation pose '{pose.name}': the wrist at "
                        f"{math.degrees(joints[-1]):.0f} deg, more than 90 from its middle, "
                        "the camera toward the arm: turn the pose's yaw_deg by 180",
                        tuple(pose.camera[:2, 3]),
                    )
                )
        else:
            outside = [
                joint.name
                for joint, angle in zip(chain.joints, pose.joints)
                if not joint.lower <= angle <= joint.upper
            ]
            if outside:
                found.append(
                    Finding(
                        True, f"observation pose '{pose.name}': {', '.join(outside)} past its limit"
                    )
                )

    pick = cell.pick_area
    points = [(x, y) for x in pick.x for y in pick.y] + [(sum(pick.x) / 2, sum(pick.y) / 2)]
    for xy in points:
        where = f"({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm"
        if cell.close_look_height is not None and all(
            reach_joints(cell, cell.tcp_of_camera(cell.close_look_camera(xy, yaw)), chain) is None
            for yaw in YAWS
        ):
            found.append(
                Finding(
                    False,
                    f"pick area at {where}: a closer look is out of reach there: a fish too "
                    "long for one view isn't seen whole",
                    xy,
                )
            )
        at_top = _reach(cell, chain, xy, top, YAWS)
        at_lift = None if at_top is None else _reach(cell, chain, xy, lift, YAWS)
        postures += [(f"picking at {where}", q) for q in (at_top, at_lift) if q is not None]
        if at_top is None:
            found.append(Finding(True, f"pick area at {where}: out of the arm's reach", xy))
        elif at_lift is None:
            found.append(
                Finding(
                    False,
                    f"pick area at {where}: a fish can't be lifted to {lift - table:.2f} m "
                    "there (out of reach): move the pick area nearer the arm",
                    xy,
                )
            )

    heading = cell.placement_heading
    ways = (heading, heading + math.pi)
    along = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-along[1], along[0]])
    zone = cell.work_zone
    table_rect = Rect(zone.x, zone.y)
    for k, slot in enumerate(cell.slots, 1):
        problem = None
        for ahead in (0.0, fish.cups_ahead) if fish.cups_ahead > 0 else (0.0,):
            tool = np.asarray(slot, float) + ahead * along  # the cups ahead, toward the head
            for z, what in ((lift, "carry"), (release, "release")):
                q = None if problem else _reach(cell, chain, tool, z, ways)
                if q is not None:
                    postures.append((f"slot {k} at the {what} height", q))
                elif problem is None:
                    problem = (
                        f"slot {k}: out of the arm's reach at the {what} height "
                        f"({z - table:.2f} m), the way fish go in"
                        + (f", the cups {ahead * 1000:.0f} mm ahead of the fish's middle"
                           if ahead else "")
                        + ": move it nearer the arm"
                    )  # fmt: skip
        if problem:
            found.append(Finding(True, problem, tuple(slot)))
        elif finding := _slot_wrist(cell, chain, k, slot, lift, heading, middle):
            found.append(finding)
        corners = [
            np.asarray(slot) + a * fish.length / 2 * along + b * fish.width / 2 * across
            for a in (-1, 1)
            for b in (-1, 1)
        ]
        if not all(table_rect.contains(c, -1e-9) for c in corners):
            found.append(
                Finding(
                    True,
                    f"slot {k}: a fish placed there reaches past the table's edge",
                    tuple(slot),
                )
            )
        for exclusion in cell.exclusion_zones:
            if any(exclusion.contains((c[0], c[1], z)) for c in corners for z in (release, lift)):
                found.append(
                    Finding(
                        True,
                        f"slot {k}: a fish placed there reaches into '{exclusion.name}'",
                        tuple(slot),
                    )
                )
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        if _overlap((min(xs), max(xs)), pick.x) and _overlap((min(ys), max(ys)), pick.y):
            found.append(
                Finding(
                    False,
                    f"slot {k}: a fish placed there lies partly in the pick area: "
                    "it would be picked again",
                    tuple(slot),
                )
            )
    for k, a in enumerate(cell.slots, 1):
        for m, b in enumerate(cell.slots[k:], k + 1):
            gap = abs((np.asarray(b) - np.asarray(a)) @ across)
            if gap < fish.width and abs((np.asarray(b) - np.asarray(a)) @ along) < fish.length:
                found.append(
                    Finding(
                        False,
                        f"slots {k} and {m} are {gap * 1000:.0f} mm apart across "
                        f"the fish, which are up to {fish.width * 1000:.0f} mm wide",
                    )
                )

    half = fish.length / 2
    reach_x, reach_y = (pick.x[0] - half, pick.x[1] + half), (pick.y[0] - half, pick.y[1] + half)
    if not (
        zone.x[0] <= reach_x[0]
        and reach_x[1] <= zone.x[1]
        and zone.y[0] <= reach_y[0]
        and reach_y[1] <= zone.y[1]
    ):
        found.append(
            Finding(
                False,
                f"a fish near the pick area's edge reaches {half:.2f} m past it, "
                "past the table's edge: it can't be picked there",
            )
        )
    for exclusion in cell.exclusion_zones:
        if (
            _overlap(reach_x, exclusion.x)
            and _overlap(reach_y, exclusion.y)
            and _overlap((lift - fish.thickness, lift), exclusion.z)
        ):
            found.append(
                Finding(
                    False,
                    f"a fish near the pick area's edge reaches into "
                    f"'{exclusion.name}' when lifted: it can't be picked there",
                )
            )
    # The arm's links and gripper, not only the tool's point: e.g. the elbow over the base
    # when it reaches near, into a zone behind the arm. Paths aren't checked: MoveIt does.
    if cell.collision is not None:
        for exclusion in cell.exclusion_zones:
            for what, joints in postures:
                if (hit := cell.collision.in_box(chain, joints, exclusion)) is not None:
                    body, point = hit
                    found.append(
                        Finding(
                            True,
                            f"{what}: the arm's {body} would be in '{exclusion.name}'",
                            (float(point[0]), float(point[1])),
                        )
                    )
                    break

    if view is not None and cell.observation_poses:
        plane = table + fish.thickness / 2
        try:
            views = [
                view_footprint(observation_camera(cell, p, chain), view, plane)
                for p in cell.observation_poses
            ]
        except ValueError as e:
            found.append(Finding(True, f"observation pose: {e}"))
            views = []
        if views:
            hidden = _hidden(pick, views, fish.width / 2)
            if hidden is not None:
                found.append(
                    Finding(
                        False,
                        "a fish can lie in the pick area where no observation pose sees any "
                        "of it: it isn't found",
                        hidden,
                    )
                )
    return sorted(found, key=lambda f: not f.error)


# ---------------------------------------------------------------------------
# Saving: only the layout's lines change
# ---------------------------------------------------------------------------
def _num(value: float) -> str:
    """Meters to the millimeter, at least two decimals: 0.20, -0.35, 0.375."""
    text = f"{round(float(value), 3):.3f}".rstrip("0")
    whole, fraction = text.split(".")
    if whole == "-0" and not fraction.strip("0"):
        whole = "0"
    return f"{whole}.{fraction:0<2}"


def _deg(value: float) -> str:
    value = round(float(value), 1)
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def _pair(values) -> str:
    return f"[{_num(values[0])}, {_num(values[1])}]"


def _name(text: str) -> str:
    return text if re.fullmatch(r"[A-Za-z][\w .-]*", text) else json.dumps(text)


def rounded(layout: Layout) -> Layout:
    """The layout as it is written: meters to the millimeter, degrees to a tenth."""

    def rect(data: dict) -> dict:
        return {k: ([round(v, 3) for v in data[k]] if k in "xyz" else data[k]) for k in data}

    def pose(data: dict) -> dict:
        if "joints_deg" in data:
            return {"name": data["name"], "joints_deg": [round(v, 1) for v in data["joints_deg"]]}
        return {
            "name": data["name"],
            "camera_xy": [round(v, 3) for v in data["camera_xy"]],
            "yaw_deg": round(data["yaw_deg"], 1),
        }

    work = rect(layout.work)
    work["height"] = round(layout.work["height"], 3)
    return Layout(
        work=work,
        pick_area=rect(layout.pick_area),
        placement_zone=rect(layout.placement_zone),
        exclusions=[rect(zone) for zone in layout.exclusions],
        poses=[pose(p) for p in layout.poses],
        heading_deg=round(layout.heading_deg, 1),
        slots=[[round(v, 3) for v in slot] for slot in layout.slots],
    )


def _text(part: str, value) -> str | list[str]:
    """The YAML of one part: a flow value, or the list's items."""
    if part == "work":
        return (
            f"{{x: {_pair(value['x'])}, y: {_pair(value['y'])}, height: {_num(value['height'])}}}"
        )
    if part in ("pick_area", "placement_zone"):
        return f"{{x: {_pair(value['x'])}, y: {_pair(value['y'])}}}"
    if part == "heading_deg":
        return _deg(value)
    if part == "exclusions":
        return [
            f"{{name: {_name(z['name'])}, x: {_pair(z['x'])}, y: {_pair(z['y'])}, "
            f"z: {_pair(z['z'])}}}"
            for z in value
        ]
    if part == "poses":
        items = []
        for pose in value:
            if "joints_deg" in pose:
                joints = ", ".join(_deg(v) for v in pose["joints_deg"])
                items.append(f"{{name: {_name(pose['name'])}, joints_deg: [{joints}]}}")
            else:
                items.append(
                    f"{{name: {_name(pose['name'])}, camera_xy: {_pair(pose['camera_xy'])}, "
                    f"yaw_deg: {_deg(pose['yaw_deg'])}}}"
                )
        return items
    if part == "slots":
        return [_pair(slot) for slot in value]
    raise KeyError(part)


_KEY_LINE = re.compile(r"([A-Za-z_]\w*):(\s|$)")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _locate(lines: list[str], path: tuple, where: str) -> int:
    """The index of the line holding the last key of a block-mapping path."""
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent(body)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stripped.startswith("-"):
            continue
        match = _KEY_LINE.match(stripped)
        if match is None:
            continue
        stack.append((indent, match.group(1)))
        if tuple(key for _, key in stack) == tuple(path):
            return i
    raise LayoutError(f"{where}: no '{'.'.join(path)}' entry to change")


def _split_comment(text: str) -> tuple[str, str]:
    """(value, '# comment' or '') of the text after a key's colon or a list item's dash."""
    depth, quote = 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "#" and depth == 0 and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip(), text[i:]
    return text.rstrip(), ""


def _with_comment(head: str, comment: str, column: int) -> str:
    if not comment:
        return head
    return head + " " * max(2, column - len(head)) + comment


def _set_value(lines: list[str], path: tuple, value: str, where: str) -> None:
    i = _locate(lines, path, where)
    body = lines[i].rstrip("\r\n")
    end = lines[i][len(body) :]
    key, rest = body.split(":", 1)
    old, comment = _split_comment(rest)
    if not old.strip():
        raise LayoutError(
            f"{where}: '{'.'.join(path)}' isn't written on one line: change it there by hand"
        )
    lines[i] = _with_comment(f"{key}: {value}", comment, body.find(comment)) + end


def _set_items(lines: list[str], path: tuple, items: list[str], where: str) -> None:
    i = _locate(lines, path, where)
    body = lines[i].rstrip("\r\n")
    end = lines[i][len(body) :] or "\n"
    old, _ = _split_comment(body.split(":", 1)[1])
    if old.strip():  # a flow list on the key's own line
        _set_value(lines, path, "[" + ", ".join(items) + "]", where)
        return
    key_indent = _indent(body)
    block = []
    for j in range(i + 1, len(lines)):
        text = lines[j].rstrip("\r\n")
        stripped = text.strip()
        if not stripped:
            break
        indent = _indent(text)
        if indent < key_indent or (indent == key_indent and not stripped.startswith("- ")):
            break
        block.append(j)
    item_rows = [j for j in block if lines[j].strip().startswith("-")]
    others = [lines[j] for j in block if j not in item_rows]
    item_indent = _indent(lines[item_rows[0]]) if item_rows else key_indent + 2
    comments = []
    for j in item_rows:
        text = lines[j].rstrip("\r\n")
        _, note = _split_comment(text.strip()[1:])
        comments.append((note, text.find(note) if note else 0))
    new = []
    for k, item in enumerate(items):
        head = " " * item_indent + "- " + item
        note, column = comments[k] if k < len(comments) else ("", 0)
        new.append(_with_comment(head, note, column) + end)
    lines[i + 1 : i + 1 + len(block)] = others + new


def save_layout(
    layout: Layout,
    workcell_file: Path = DEFAULT_FILE,
    panel_file: Path | None = PANEL_FILE,
) -> list[str]:
    """
    Write `layout` (rounded to the millimeter) where each part is set, and say what was
    written. A layout with problems isn't written. A verified cell whose layout changes is
    set to not verified: the zones must be checked against the real cell again. Raises
    LayoutError (with nothing changed) if the files can't be changed that way.
    """
    import yaml

    workcell_file = Path(workcell_file)
    panel_file = Path(panel_file) if panel_file is not None else None
    files = [workcell_file] + ([panel_file] if panel_file and panel_file.is_file() else [])
    texts = {f: f.read_text(encoding="utf-8") for f in files}
    parsed = {f: yaml.safe_load(texts[f]) or {} for f in files}
    in_panel = {}
    if panel_file in parsed:
        in_panel = parsed[panel_file].get("workcell") or {}
    data = merged(parsed[workcell_file], in_panel)
    before, layout = layout_of(data), rounded(layout)
    problems = workcell_from_mapping(with_layout(data, layout)).problems()
    if problems:
        raise LayoutError("the layout has problems: " + "; ".join(problems))

    lines = {f: texts[f].splitlines(keepends=True) for f in files}
    expected = copy.deepcopy(parsed)
    said = []
    for part, path in PATHS.items():
        value = getattr(layout, part)
        if _same(value, getattr(before, part)):
            continue
        target = panel_file if _has(in_panel, path) else workcell_file
        full = ("workcell", *path) if target == panel_file else path
        text = _text(part, value)
        if part == "exclusions" and not value and not _has(parsed[target], full):
            continue
        if part in LISTS:
            _set_items(lines[target], full, text, target.name)
        else:
            _set_value(lines[target], full, text, target.name)
        _put(expected[target], full, copy.deepcopy(value))
        said.append(f"{'.'.join(path)} in {target.name}")
    if not said:
        return []
    verified = ("workcell", "verified") if _has(in_panel, ("verified",)) else ("verified",)
    where = panel_file if verified[0] == "workcell" else workcell_file
    if data.get("verified"):
        _set_value(lines[where], verified, "false", where.name)
        _put(expected[where], verified, False)
        said.append(
            f"verified: false in {where.name} (check the zones on the real cell, then "
            "set it true again)"
        )

    written = {f: "".join(lines[f]) for f in files}
    for f in files:  # what the files will say must be the old content with the layout in it
        if _plain(yaml.safe_load(written[f]) or {}) != _plain(expected[f]):
            raise LayoutError(
                f"{f.name}: changing only the layout's lines didn't give the "
                "layout: change it by hand"
            )
    for f in files:
        if written[f] != texts[f]:
            shutil.copy2(f, f.with_name(f.name + ".bak"))
            f.write_text(written[f], encoding="utf-8")
    return said


def _plain(value):
    """Numbers as floats, so 90 and 90.0 compare equal."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 9)
    return value


def _same(a, b) -> bool:
    return _plain(a) == _plain(b)


def layout_changed(a: Layout, b: Layout) -> bool:
    return any(not _same(getattr(a, f.name), getattr(b, f.name)) for f in fields(Layout))
