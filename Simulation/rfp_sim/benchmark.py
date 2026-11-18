"""
The benchmark: the real pick-and-place cycle in the virtual cell over many seeded scenes,
scored against the simulator's truth, so a change to vision, planning or the cycle can be
shown to do better (or not) before it reaches the arm.

Per fish: where it ended up (placed in the placement zone, left in the pick area, or
elsewhere), how far its middle lies from its slot and how far its head points from the
placement heading, and where the cups took it relative to its true center of mass. Per
scene: how the run ended, failures by code, looks, wrist travel, simulated time, how high
fish were carried and how far they cleared the table and the fish under them.
"""

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rfp_arm.poses import invert, wrap, yaw_of
from rfp_arm.workcell import Workcell
from rfp_controlsys.console import Console
from rfp_controlsys.cycle import PickPlaceCycle
from rfp_controlsys.selector import RuleSelector
from rfp_controlsys.settings import ControlSettings
from rfp_vision.fish_list import FishLocator
from rfp_vision.geometry import to_frame
from rfp_vision.settings import VisionSettings
from rfp_vision.vision_system import VisionSystem

from rfp_sim.devices import VirtualArm, VirtualGripper
from rfp_sim.render import TruthDetector
from rfp_sim.session import build_session
from rfp_sim.settings import SimSettings
from rfp_sim.world import World

NEAR_FISH = 0.25  # m: a grasp point this close to a fish's middle was aimed at that fish
SAMPLE_POINTS = 3000  # of a fish's mesh, for how far a carried one clears what is under it


class QuietConsole(Console):
    """The operator console with nobody there: nothing printed or shown, every pause ends
    the scene (as an unattended run stops)."""

    def __init__(self):
        super().__init__(window=False, interactive=False)
        self.lines: list[str] = []
        self.detail = ""  # the last state's detail: the grasp step running

    def state(self, state: str, detail: str = "") -> None:
        self.lines.append(f"[{state}] {detail}")
        self.detail = detail

    def info(self, text: str) -> None:
        self.lines.append(text)

    def alert(self, message: str) -> bool:
        self.lines.append(f"paused: {message}")
        return False

    def show(self, image, lines=None) -> None:
        pass


class _Rows:
    """A run log that keeps the cycle rows in memory."""

    def __init__(self):
        self.rows: list[dict] = []

    def cycle(self, row: dict) -> None:
        self.rows.append(dict(row))

    def image(self, name: str, image) -> None:
        pass


class _CountingVision(VisionSystem):
    looks = 0

    def observe(self, t_base_camera, on_frame=None):
        self.looks += 1
        return super().observe(t_base_camera, on_frame)


@dataclass
class Bench:
    """One virtual cell, reused scene after scene."""

    session: object
    vision: _CountingVision
    cell: Workcell
    sim: SimSettings
    control: ControlSettings
    hold_check: bool


def build_bench(
    cell: Workcell,
    vision_settings: VisionSettings,
    control_settings: ControlSettings,
    sim_settings: SimSettings,
    detector=None,
    hold_check: bool = True,
) -> Bench:
    """`detector`: a FishDetector (the trained model), or None for the renderer's truth."""
    sim = sim_settings
    session = build_session(cell, sim, show=False, viewer=False, log=lambda _: None)
    if len(session.world.fish) != sim.fish_count:
        raise ValueError(f"no room for {sim.fish_count} fish of {sim.fish_length} m")
    detector = detector or TruthDetector(session.camera)
    locator = FishLocator(
        vision_settings, cell.table_z, cell.pick_area.area, getattr(detector, "version", "truth"),
        cell.calibration_version,
    )  # fmt: skip
    if hasattr(detector, "warm_up"):
        detector.warm_up(session.camera.read().color)
    vision = _CountingVision(session.camera, detector, locator, vision_settings)
    return Bench(session, vision, cell, sim, control_settings, hold_check)


def run_scene(bench: Bench, seed: int) -> list[dict]:
    """One scene: lay the fish (the same number every scene, another layout), run the cycle
    until it ends, score each fish. Returns one row per fish."""
    world, sim, cell = bench.session.world, bench.sim, bench.cell
    for attempt in range(20):  # a layout with room for every fish (the renderer has an
        world.fish.clear()  # actor per fish, so the count must stay the same)
        place = world.scatter_fish if sim.placement == "anywhere" else world.place_fish
        place(sim.fish_count, seed + 1000 * attempt)
        if len(world.fish) == sim.fish_count:
            break
    else:
        raise ValueError(f"no layout with room for {sim.fish_count} fish (seed {seed})")
    world.vacuum = False
    world._grabbed_at = None
    world._drop = sim.drop
    for fish in world.fish:
        fish.held = None
    world.set_joints(world.start_joints())
    bench.session.camera.reseed(seed)  # the scene's own noise, whatever came before it
    start = {fish.name: fish.pose.copy() for fish in world.fish}

    wrist = {"last": float(world.joints[-1]), "travel": 0.0}
    rows, console = _Rows(), QuietConsole()
    clearances = []

    def on_step() -> None:
        now = float(world.joints[-1])
        wrist["travel"] += abs(now - wrist["last"])
        wrist["last"] = now
        if console.detail == "carry" and (gap := carry_clearance(world)) is not None:
            clearances.append(gap)

    waited = {"s": 0.0}

    def sleep(seconds: float) -> None:
        waited["s"] += seconds

    arm = VirtualArm(world, cell, on_step=on_step)
    bench.vision.looks = 0
    # Without the camera's hold check something else must confirm a hold, or no fish is
    # lifted: then the gripper has a vacuum switch (the world knows what the cups hold)
    gripper = VirtualGripper(world, has_switch=not bench.hold_check)
    cycle = PickPlaceCycle(
        arm, gripper, bench.vision, RuleSelector(), cell, bench.control, console, rows,
        vision_hold=bench.hold_check, sleep=sleep,
    )  # fmt: skip
    limit = sim.fish_count * (bench.control.max_attempts_per_fish + 1)
    started = time.monotonic()
    report = cycle.run(limit)
    scene = {
        "seed": seed,
        "ended": report.ended,
        "cycles": report.cycles,
        "failures": "+".join(f"{code}x{n}" for code, n in sorted(report.failures.items())),
        "failure_details": " | ".join(
            dict.fromkeys(
                f"{row['failed_step']}: {row['detail']}"
                for row in rows.rows
                if row.get("result") == "failed"
            )
        ),
        "looks": bench.vision.looks,
        "wrist_travel_deg": round(math.degrees(wrist["travel"])),
        "wrist_turn_max_deg": round(math.degrees(max(arm.wrist_turns, default=0.0))),
        "sim_time_s": round(sum(seconds for _, seconds in arm.moves) + waited["s"], 1),
        "compute_s": round(time.monotonic() - started, 1),
        "carry_mm": _mean([r["carry_mm"] for r in rows.rows if r.get("carry_mm") is not None]),
        "carry_clearance_mm": round(min(clearances) * 1000) if clearances else None,
    }
    first_grasp = _first_grasps(rows.rows, start)
    return [
        {**scene, **_score_fish(fish, start[fish.name], first_grasp.get(fish.name), cell)}
        for fish in world.fish
    ]


def carry_clearance(world: World) -> float | None:
    """How far (m) the lowest point of the fish on the cups is above the table, or above the
    top of a fish lying under it, whichever is less; None while nothing is held."""
    held = world.holding
    if held is None:
        return None
    points = to_frame(held.pose, _points(held.model))
    gap = float(points[:, 2].min() - world.workcell.table_z)
    for other in world.fish:
        if other is held:
            continue
        outline = other.outline()[::3]
        near = (points[:, :2] >= outline.min(axis=0)).all(axis=1) & (
            points[:, :2] <= outline.max(axis=0)
        ).all(axis=1)
        over = near.copy()
        over[near] = _inside(points[near, :2], outline)
        if over.any():
            top = other.pose[2, 3] + other.model.mesh.bounds[5]
            gap = min(gap, float(points[over, 2].min() - top))
    return gap


_POINTS: dict[int, tuple[object, np.ndarray]] = {}


def _points(model) -> np.ndarray:
    """A sample of the model's mesh points, its lowest among them (taken once per model)."""
    if id(model) not in _POINTS:
        points = np.asarray(model.mesh.points, float)
        sample = points[:: max(1, len(points) // SAMPLE_POINTS)]
        _POINTS[id(model)] = (model, np.vstack([sample, points[points[:, 2].argmin()]]))
    return _POINTS[id(model)][1]


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Which (n, 2) points lie inside the polygon (even-odd rule)."""
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    px, py = points[:, :1], points[:, 1:2]
    crosses = ((y > py) != (yn > py)) & (px < (xn - x) * (py - y) / (yn - y + 1e-12) + x)
    return np.count_nonzero(crosses, axis=1) % 2 == 1


def _mean(values: list) -> float | None:
    return round(float(np.mean(values))) if values else None


def _first_grasps(rows: list[dict], start: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The grasp point of each fish's first attempt, while it still lay where it started."""
    found: dict[str, np.ndarray] = {}
    for row in rows:
        point = np.array([row["grasp_x_mm"], row["grasp_y_mm"]]) / 1000.0
        name, pose = min(start.items(), key=lambda kv: np.linalg.norm(kv[1][:2, 3] - point))
        if name not in found and np.linalg.norm(pose[:2, 3] - point) <= NEAR_FISH:
            found[name] = point
    return found


def _score_fish(fish, start: np.ndarray, grasp: np.ndarray | None, cell: Workcell) -> dict:
    middle = fish.pose[:2, 3]
    if cell.placement_zone.contains(middle):
        outcome = "placed"
    elif cell.pick_area.contains(middle):
        outcome = "left"
    else:
        outcome = "elsewhere"
    row = {"fish": fish.name, "outcome": outcome}
    if outcome == "placed":
        slot = min(cell.slots, key=lambda s: np.hypot(*(middle - s)))
        heading = wrap(yaw_of(fish.pose) - cell.placement_heading)
        row.update(
            slot_error_mm=round(float(np.hypot(*(middle - slot))) * 1000, 1),
            heading_error_deg=round(math.degrees(heading), 1),
            head_right=int(abs(heading) < math.pi / 2),
        )
    if grasp is not None:  # in the fish's own frame: x toward the head
        to_fish = invert(start)
        local = to_fish[:2, :2] @ grasp + to_fish[:2, 3]
        com = fish.model.center_of_mass
        row.update(
            grasp_from_middle_mm=round(float(local[0]) * 1000, 1),
            grasp_from_com_mm=round(float(local[0] - com[0]) * 1000, 1),
            grasp_across_mm=round(float(local[1] - com[1]) * 1000, 1),
        )
    return row


def summarize(rows: list[dict]) -> dict:
    """The numbers a change is judged by."""

    def values(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r.get(key) is not None], float)

    def stats(key: str, absolute: bool = True) -> dict | None:
        v = values(key)
        if len(v) == 0:
            return None
        if not absolute:  # a signed offset: where it lies, and its spread
            low, high = np.percentile(v, [10, 90])
            return {"median": _r(np.median(v)), "p10": _r(low), "p90": _r(high)}
        v = np.abs(v)
        return {"median": _r(np.median(v)), "p90": _r(np.percentile(v, 90)), "max": _r(v.max())}

    scenes = {r["seed"]: r for r in rows}.values()
    fish = len(rows)
    placed = [r for r in rows if r["outcome"] == "placed"]
    cleared = [s["carry_clearance_mm"] for s in scenes if s.get("carry_clearance_mm") is not None]
    failures: Counter = Counter()
    for scene in scenes:
        for item in filter(None, scene["failures"].split("+")):
            code, n = item.rsplit("x", 1)
            failures[code] += int(n)
    return {
        "scenes": len(scenes),
        "fish": fish,
        "placed": len(placed),
        "placed_share": round(len(placed) / fish, 3) if fish else 0.0,
        "head_right": sum(r.get("head_right", 0) for r in placed),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "slot_error_mm": stats("slot_error_mm"),
        "heading_error_deg": stats("heading_error_deg"),
        "grasp_from_com_mm": stats("grasp_from_com_mm"),
        "grasp_from_com_signed_mm": stats("grasp_from_com_mm", absolute=False),
        "failures": dict(failures),
        "ended": dict(Counter(_reason(s["ended"]) for s in scenes)),
        "looks_per_fish": round(sum(s["looks"] for s in scenes) / fish, 2) if fish else 0.0,
        "wrist_travel_deg_per_placed": (
            round(sum(s["wrist_travel_deg"] for s in scenes) / len(placed)) if placed else None
        ),
        "wrist_turn_max_deg": max((s["wrist_turn_max_deg"] for s in scenes), default=0),
        "sim_time_s_per_placed": (
            round(sum(s["sim_time_s"] for s in scenes) / len(placed), 1) if placed else None
        ),
        "carry_mm": _mean([s["carry_mm"] for s in scenes if s.get("carry_mm") is not None]),
        # the least any carried fish cleared the table or a fish under it, in the worst scene
        "carry_clearance_mm": min(cleared) if cleared else None,
    }


def _r(value) -> float:
    return round(float(value), 1)


def _reason(ended: str) -> str:
    """How a scene ended, without the fish numbers (so equal reasons count together)."""
    for prefix in ("stopped at: Fish are on the table but none can be picked",):
        if ended.startswith(prefix):
            return prefix.replace("stopped at: ", "")
    return ended.replace("stopped at: ", "")


def report_lines(summary: dict, baseline: dict | None = None) -> list[str]:
    """The summary as text, with the change from `baseline` where there is one."""

    def num(key: str) -> str:
        value = summary.get(key)
        before = baseline.get(key) if baseline else None
        if isinstance(value, (int, float)) and isinstance(before, (int, float)):
            return f"{value} (was {before})"
        return f"{value}"

    def spread(key: str, unit: str) -> str:
        def words(value: dict | None) -> str:
            if not value:
                return "-"
            if "p10" in value:
                return f"median {value['median']} {unit}, p10-p90 {value['p10']} to {value['p90']}"
            return f"median {value['median']} {unit}, p90 {value['p90']}, max {value['max']}"

        text = words(summary.get(key))
        if baseline is not None:
            text += f"   (was {words(baseline.get(key))})"
        return text

    f = summary
    lines = [
        (
            f"fish placed:        {num('placed')} of {f['fish']} in {f['scenes']} scenes"
            f"  ({num('placed_share')})"
        ),
        f"head end right:     {num('head_right')} of the placed",
        f"outcomes:           {f['outcomes']}",
        f"slot error:         {spread('slot_error_mm', 'mm')}",
        f"heading error:      {spread('heading_error_deg', 'deg')}",
        "cups from the true center of mass, along the body (+: toward the head):",
        f"                    {spread('grasp_from_com_signed_mm', 'mm')}",
        f"failures:           {f['failures'] or 'none'}",
        f"scenes ended:       {f['ended']}",
        f"looks per fish:     {num('looks_per_fish')}",
        (
            f"wrist travel:       {num('wrist_travel_deg_per_placed')} deg per placed fish, "
            f"at most {num('wrist_turn_max_deg')} in one move"
        ),
        f"simulated time:     {num('sim_time_s_per_placed')} s per placed fish (moves and waits)",
        (
            f"carried:            {num('carry_mm')} mm above the table on average, clearing "
            f"the table and fish under it by at least {num('carry_clearance_mm')} mm"
        ),
    ]
    if baseline is not None and baseline.get("outcomes") != f["outcomes"]:
        lines.insert(3, f"                    (was {baseline.get('outcomes')})")
    return lines


def run_benchmark(
    bench: Bench, scenes: int, first_seed: int, progress: Callable[[str], None] = print
) -> list[dict]:
    rows = []
    for i in range(scenes):
        rows += run_scene(bench, first_seed + i)
        placed = sum(r["outcome"] == "placed" for r in rows)
        progress(f"   scene {i + 1}/{scenes}: {placed} of {len(rows)} fish placed so far")
    return rows
