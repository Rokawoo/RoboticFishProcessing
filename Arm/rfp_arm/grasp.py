"""
The grasp sequence of Table AS-1 (AS-20): `plan_grasp` turns a fish and a slot into the TCP
poses of the nine steps, checked against the workcell; `GraspSequence.run` drives an Arm and
a Gripper through them and stops at a failed step with its Table AS-2 code (AS-25). The same
code runs the simulated arm, a dry run and the real thing. The approach follows the fish's
surface (up to 15 deg off vertical, AS-21); on an arm that reports joint torques the last
centimeters are felt. No ROS.
"""

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rfp_arm.gripper import wait_for
from rfp_arm.interfaces import (
    E_COMM,
    E_DROP,
    E_MOTION,
    E_NO_SEAL,
    E_RELEASE,
    Arm,
    ArmError,
    Gripper,
    GripperState,
    HoldCheck,
    MotionAborted,
    Tick,
)
from rfp_arm.kinematics import Chain
from rfp_arm.poses import rot_z, tilt_from_down, tool_down, wrap, yaw_of
from rfp_arm.workcell import Workcell

# Wrist arcs (rad) this close count as equal: then the one nearer the range's middle
WRIST_TIE = math.radians(5.0)
# The empty gripper stays this close to the wrist's middle where the fish allows (the cups
# are symmetric); there the camera faces away from the arm
EMPTY_WRIST = math.radians(90.0)
HELD_TURN = math.radians(180.0)  # a held fish is turned at most half a turn
WRIST_MOVE = math.radians(180.0)  # no single move turns the wrist further
# Plans keep the wrist this far inside its range where a way allows: the tilted approach
# turns it up to 14 deg from where it was predicted
WRIST_MARGIN = math.radians(10.0)
# A carry goes through a pose every CARRY_STEP m and CARRY_WRIST_STEP of wrist turn, each
# turned so the wrist goes evenly from grasp to slot: along one straight move the base swings
# unevenly, which can drag the wrist into its stop half way
CARRY_WRIST_STEP = math.radians(45.0)
CARRY_STEP = 0.15
TURN_SPOT_STEP = 0.05  # m between the spots tried for turning a held fish where it has room
# m: a fish may be set down again this close to where the cups took it (which can be past
# the pick area's edge: the cups hold it ahead of its middle)
PUT_BACK_NEAR = 0.03
# m into the slow descent before the force is zeroed: moving steadily, the joints' friction
# is counted in (at rest it isn't)
ZERO_AFTER = 0.002

STEPS = ("pre-grasp", "vacuum on", "approach", "confirm hold", "lift", "carry", "lower",
         "release", "retreat")  # fmt: skip


@dataclass
class GraspPlan:
    """TCP poses in {B} for one fish and one slot (4x4, tool pointing down)."""

    pregrasp: np.ndarray
    approach_end: np.ndarray
    lift: np.ndarray
    carry: np.ndarray
    lower: np.ndarray
    retreat: np.ndarray
    fish_length: float
    turn: float  # rad the fish is turned between pick and place
    dry_run: bool
    surface: np.ndarray  # the cups just touching the fish where vision saw its surface
    # the last joint's angle (rad) now, at the grasp and at the slot, as the arm predicted
    # them. None if the arm can't tell.
    wrist: tuple[float, float, float] | None = None
    middle_x: float = 0.0  # m along the TCP's x axis from the cups to the fish's middle
    # the wrist turns back to its middle where it is before the pre-grasp: the turn to the
    # grasp is longer than WRIST_MOVE from where it is
    reset_first: bool = False
    # TCP poses the carry goes through on its straight way, each turned so the wrist moves
    # evenly the way the plan chose (a straight move alone turns the tool the short way,
    # which can wind the wrist into its stop half way)
    carry_via: list = field(default_factory=list)

    @property
    def wrist_sweep(self) -> float | None:
        """The arc (rad) the wrist sweeps; it turns twice that over the cycle."""
        return None if self.wrist is None else max(self.wrist) - min(self.wrist)

    def at_depth(self, depth: float) -> np.ndarray:
        """The TCP pose `depth` m past the fish's surface along the approach (< 0: short of it)."""
        pose = self.surface.copy()
        pose[:3, 3] += pose[:3, 2] * depth
        return pose

    def poses(self) -> dict[str, np.ndarray]:
        return {
            "pre-grasp": self.pregrasp,
            "approach": self.approach_end,
            "lift": self.lift,
            "carry": self.carry,
            "lower": self.lower,
            "retreat": self.retreat,
        }


def plan_grasp(
    workcell: Workcell,
    grasp_pose: np.ndarray,
    heading_known: bool,
    fish_length: float,
    thickness: float,
    slot: tuple[float, float],
    tool_yaw: float,
    dry_run: bool = False,
    wrist: Callable[[np.ndarray], float | None] | None = None,
    middle: np.ndarray | None = None,
    clear_z: float | None = None,
) -> GraspPlan:
    """
    The poses of Table AS-1. `grasp_pose`: on the fish, z the approach, x toward the head.
    `tool_yaw`: the TCP's yaw now (the arm comes back here). `wrist` (Arm.wrist_angle): j6
    now (None) and at a pose; without it the wrist turns with the tool's yaw. `middle`: the
    fish's middle, which goes on the slot when the cups grasp it elsewhere. `clear_z`: the
    top of what lies on the way, carried just over (at most lift_height_max); None: the fixed
    lift height. Raises ArmError(E_MOTION) if the fish or a pose is outside the limits.
    """
    g = workcell.grasp
    point = grasp_pose[:3, 3]
    tilt = math.degrees(tilt_from_down(grasp_pose))
    if tilt > g.max_approach_tilt_deg + 0.01:
        raise ArmError(
            E_MOTION,
            f"approach {tilt:.0f} deg off vertical (at most {g.max_approach_tilt_deg:.0f})",
        )
    if not workcell.in_pick_area(point if middle is None else middle):
        raise ArmError(E_MOTION, "the fish is outside the pick area (CS-13)")
    if not workcell.placement_zone.contains(slot):
        raise ArmError(E_MOTION, "the slot is outside the placement zone")

    heading = yaw_of(grasp_pose)
    turn = wrap(workcell.placement_heading - heading)
    turns = [turn] if heading_known else [turn, wrap(turn + math.pi)]  # either end: head

    x, y, z = point
    table = workcell.table_z
    if clear_z is None:
        lift_z = table + min(g.lift_length_share * fish_length + g.lift_margin, g.lift_height_max)
    else:  # its underside lift_margin over what lies on the way
        hang = max(thickness, 0.0) + g.droop_share * fish_length
        lift_z = min(max(clear_z, table) + hang + g.lift_margin, table + g.lift_height_max)
    lift_z = max(lift_z, z + g.lift_margin)
    release_z = table + max(thickness, 0.0) + g.release_clearance

    # Which way round (the cups either way, and a fish whose head end isn't known into the
    # slot either way). For the vacuum tube: a held fish turned at most HELD_TURN, the wrist
    # inside its range, the empty gripper within EMPTY_WRIST of the middle where the fish
    # allows, no move over WRIST_MOVE (a longer one to the grasp turns back to the middle
    # first: reset_first). Of the ways allowed: clear of the range's ends, then the least
    # turning in all; of those within WRIST_TIE, the one nearest the middle.
    low, high = workcell.wrist_range
    center = (low + high) / 2
    move_limit = min(WRIST_MOVE, math.radians(workcell.motion.max_joint_travel_deg) - WRIST_MARGIN)
    now = wrist(None) if wrist else None
    behind = np.zeros(2) if middle is None else np.asarray(middle, float)[:2] - point[:2]

    def at_slot(grasp_yaw: float, place_yaw: float) -> np.ndarray:
        """The TCP's x, y at the slot that puts the fish's middle on the slot: its offset
        from the cups turns with the fish."""
        return np.asarray(slot, float) - rot_z(place_yaw - grasp_yaw)[:2, :2] @ behind

    ways = []
    for flip in (0.0, math.pi):
        for fish_turn in turns:
            grasp_yaw = wrap(heading + flip)
            place_yaw = wrap(grasp_yaw + fish_turn)
            angles = None
            place = at_slot(grasp_yaw, place_yaw)
            if wrist and now is not None:
                wrist_grasp = wrist(tool_down((x, y, lift_z), grasp_yaw))
                wrist_slot = wrist(tool_down((place[0], place[1], lift_z), place_yaw))
                if wrist_grasp is not None and wrist_slot is not None:
                    angles = (now, wrist_grasp, wrist_slot)
            predicted = angles is not None
            if not predicted:  # the arm can't tell: the wrist turns as the tool's yaw does
                start = center + wrap(grasp_yaw - tool_yaw)
                angles = (center, start, start + fish_turn)
            begin, grasp_at, slot_at = angles
            reset_first = predicted and abs(grasp_at - begin) > move_limit
            if reset_first:  # by way of the middle
                to_grasp = abs(begin - center) + abs(grasp_at - center)
            else:
                to_grasp = abs(grasp_at - begin)
            targets = (grasp_at, slot_at)
            ways.append(
                {
                    "inside": all(low <= a <= high for a in targets),
                    "held": abs(slot_at - grasp_at),
                    "empty": abs(grasp_at - center),
                    "travel": to_grasp + abs(slot_at - grasp_at) + abs(slot_at - center),
                    "peak": max(abs(a - center) for a in targets),
                    "worst": max(targets, key=lambda a: abs(a - center)),
                    "reset_first": reset_first,
                    "flip": flip,
                    "turn": fish_turn,
                    "place_yaw": place_yaw,
                    "place": place,
                    "wrist": tuple(float(a) for a in angles) if predicted else None,
                }
            )
    inside = [way for way in ways if way["inside"]]
    if not inside:
        worst = min(ways, key=lambda way: way["peak"])["worst"]
        raise ArmError(
            E_MOTION,
            f"the wrist would turn to {math.degrees(worst):.0f} deg (range "
            f"{math.degrees(low):.0f} to {math.degrees(high):.0f}: the vacuum tube)",
        )
    allowed = [way for way in inside if way["held"] <= HELD_TURN + 1e-6]
    if not allowed:
        held = min(way["held"] for way in inside)
        raise ArmError(
            E_MOTION,
            f"the fish would be turned {math.degrees(held):.0f} deg on the gripper (at most "
            f"{math.degrees(HELD_TURN):.0f} inside the wrist's range)",
        )

    def kind(way: dict) -> tuple[bool, bool]:
        return (
            way["peak"] > (high - low) / 2 - WRIST_MARGIN,
            way["empty"] > EMPTY_WRIST + 1e-6,
        )

    best = min(kind(way) for way in allowed)
    allowed = [way for way in allowed if kind(way) == best]
    if not heading_known:  # the head end vision guessed, unsure, still beats a coin toss:
        guessed = [way for way in allowed if abs(wrap(way["turn"] - turn)) < 1e-9]
        allowed = guessed or allowed  # into the slot as guessed where the wrist allows
    least = min(way["travel"] for way in allowed)
    way = min(
        (way for way in allowed if way["travel"] <= least + WRIST_TIE),
        key=lambda way: way["peak"],
    )
    flip, turn, place_yaw, place = way["flip"], way["turn"], way["place_yaw"], way["place"]
    # The cups meet the fish along the grasp pose's z axis; `flip` turns them about it.
    surface = grasp_pose.copy()
    surface[:3, :3] = grasp_pose[:3, :3] @ rot_z(flip)
    grasp_yaw = yaw_of(surface)
    plan = GraspPlan(
        pregrasp=surface,
        approach_end=surface,
        lift=surface,
        carry=tool_down((place[0], place[1], lift_z), place_yaw),
        lower=tool_down((place[0], place[1], release_z), place_yaw),
        retreat=tool_down((place[0], place[1], release_z + g.retreat_distance), place_yaw),
        fish_length=fish_length,
        turn=turn,
        dry_run=dry_run,
        surface=surface,
        wrist=way["wrist"],
        middle_x=float(behind @ surface[:2, 0]) / max(float(np.hypot(*surface[:2, 0])), 1e-9),
        reset_first=way["reset_first"],
    )
    plan.pregrasp = plan.at_depth(-g.pregrasp_distance)
    plan.approach_end = plan.at_depth(-g.dry_run_standoff if dry_run else g.press_distance)
    end = plan.approach_end[:3, 3]
    plan.lift = tool_down((end[0], end[1], max(lift_z, end[2] + g.lift_margin)), grasp_yaw)
    for name, pose in plan.poses().items():
        held = fish_length if name in ("lift", "carry", "lower") else 0.0
        if problem := workcell.pose_problem(pose, held, True, plan.middle_x):
            raise ArmError(E_MOTION, f"{name} pose: {problem}")
    if wrist and plan.wrist is not None:
        plan.carry_via = _carry_via(workcell, plan, wrist)
    return plan


def _carry_via(workcell: Workcell, plan: GraspPlan, wrist) -> list[np.ndarray]:
    """The poses a carry goes through: on the straight way with the wrist turning evenly
    (`_even_wrist_carry`), or, where the held fish would sweep off the table or into a
    keep-out zone on that way, by way of a spot where it can turn (`_turning_carry`)."""
    via = _even_wrist_carry(workcell, plan, wrist)
    if via is None:
        via = _turning_carry(workcell, plan, wrist) or []
    return via


def _aimed(point, yaw: float, wanted: float, wrist) -> np.ndarray | None:
    """The tool straight down at `point`, turned (from `yaw`) so the wrist is at `wanted`:
    the wrist turns one for one with the tool, one way or the other. None if it can't."""
    for _ in range(4):
        pose = tool_down(point, yaw)
        now = wrist(pose)
        if now is None:
            return None
        if abs(now - wanted) < 1e-4:
            return pose
        nudged = wrist(tool_down(point, yaw + 1e-3))
        if nudged is None or abs(nudged - now) < 1e-6:
            return None
        yaw += (wanted - now) * 1e-3 / (nudged - now)
    return None


def _even_wrist_carry(workcell: Workcell, plan: GraspPlan, wrist) -> list[np.ndarray] | None:
    """Poses on the straight way from the lift to the carry pose, each turned so the wrist
    is a matching share of the way from its angle at the grasp to its angle at the slot
    (`plan.wrist`). [] if the carry is short: one straight move. None if a pose can't be
    found or isn't allowed (the held fish's ends)."""
    _, at_grasp, at_slot = plan.wrist
    start, end = plan.lift[:3, 3], plan.carry[:3, 3]
    stretches = max(
        math.ceil(abs(at_slot - at_grasp) / CARRY_WRIST_STEP),
        math.ceil(float(np.linalg.norm(end - start)) / CARRY_STEP),
    )
    if stretches < 2:
        return []
    yaw_start, yaw_end = yaw_of(plan.lift), yaw_of(plan.carry)
    poses = []
    for k in range(1, stretches):
        share = k / stretches
        point = start + (end - start) * share
        guess = yaw_start + wrap(yaw_end - yaw_start) * share
        pose = _aimed(point, guess, at_grasp + (at_slot - at_grasp) * share, wrist)
        if pose is None or workcell.pose_problem(pose, plan.fish_length, True, plan.middle_x):
            return None
        poses.append(pose)
    return poses


def _turning_carry(workcell: Workcell, plan: GraspPlan, wrist) -> list[np.ndarray] | None:
    """By way of a spot where the held fish can turn: straight there keeping its heading,
    turned there the way the plan chose (the wrist evenly), straight on to the slot. Of
    the spots (TURN_SPOT_STEP apart on the table) where every pose of it, and of the
    straight ways, keeps the fish's ends on the table and out of the keep-out zones and the
    wrist in its range, the one making the least detour. None if there is none."""
    low, high = workcell.wrist_range
    _, at_grasp, at_slot = plan.wrist
    start, end = plan.lift[:3, 3], plan.carry[:3, 3]
    yaw_start, yaw_end = yaw_of(plan.lift), yaw_of(plan.carry)
    zone = workcell.work_zone
    xs = np.arange(zone.x[0], zone.x[1] + 1e-9, TURN_SPOT_STEP)
    ys = np.arange(zone.y[0], zone.y[1] + 1e-9, TURN_SPOT_STEP)
    spots = [np.array([x, y, start[2]]) for x in xs for y in ys]
    direct = float(np.linalg.norm(end - start))

    def detour(spot) -> float:
        return float(np.linalg.norm(spot - start) + np.linalg.norm(end - spot))

    def clear(pose) -> bool:
        return workcell.pose_problem(pose, plan.fish_length, True, plan.middle_x) is None

    for spot in sorted((s for s in spots if detour(s) < 2 * direct + 0.6), key=detour):
        first, last = tool_down(spot, yaw_start), tool_down(spot, yaw_end)
        before, after = wrist(first), wrist(last)
        if before is None or after is None:
            continue
        # the wrist stays in range, and goes on from the grasp and to the slot (a leg's
        # base swing moves it less than half a turn)
        if not (low <= before <= high and low <= after <= high):
            continue
        if abs(before - at_grasp) > math.pi / 2 or abs(after - at_slot) > math.pi / 2:
            continue
        legs = [tool_down(start + (spot - start) * s, yaw_start) for s in (0.25, 0.5, 0.75)]
        legs += [tool_down(spot + (end - spot) * s, yaw_end) for s in (0.25, 0.5, 0.75)]
        if not all(clear(pose) for pose in legs + [first, last]):
            continue
        stretches = max(2, math.ceil(abs(after - before) / CARRY_WRIST_STEP) * 2)
        turning, ok = [], True
        for k in range(1, stretches):
            share = k / stretches
            guess = yaw_start + wrap(yaw_end - yaw_start) * share
            pose = _aimed(spot, guess, before + (after - before) * share, wrist)
            if pose is None or not clear(pose):
                ok = False
                break
            turning.append(pose)
        if ok:
            return [first, *turning, last]
    return None


def _nearest_segment(point: np.ndarray, corners: list[np.ndarray]) -> int:
    """Which segment of the polyline through `corners` lies nearest to `point`."""
    best, nearest = 0, math.inf
    for i, (a, b) in enumerate(itertools.pairwise(corners)):
        along = b - a
        share = float(np.clip((point - a) @ along / max(along @ along, 1e-12), 0.0, 1.0))
        distance = float(np.linalg.norm(point - (a + share * along)))
        if distance < nearest - 1e-9:
            best, nearest = i, distance
    return best


class OperatorStop(Exception):
    """The operator (or the caller's Tick) stopped the sequence. `reason` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def wrist_back_pose(arm: Arm, workcell: Workcell) -> np.ndarray | None:
    """The tool where it is, turned about the vertical so the wrist (j6) is at the middle of
    its range; None if the arm can't tell where the wrist would be."""
    low, high = workcell.wrist_range
    middle = (low + high) / 2
    here, now = arm.tcp_pose(), arm.wrist_angle()
    if now is None:
        return None
    for turn in (middle - now, now - middle):  # which way the tool turns: the arm's build
        pose = here.copy()
        pose[:3, :3] = rot_z(turn) @ here[:3, :3]
        predicted = arm.wrist_angle(pose)
        if predicted is not None and abs(predicted - middle) < math.radians(5):
            return pose
    return None


def plan_problem(plan: GraspPlan, workcell: Workcell, chain: Chain, joints) -> str | None:
    """What keeps the arm from going through a plan's poses from where it is (`joints`, the
    chain's order), the way it moves (each solved from the last, in the same posture): no
    joint solution, or the arm into itself there (workcell.collision). None if nothing does.
    Checked before anything moves, so a pick is never begun that would have to stop half
    way with a fish on the cups; each move's path is still checked as it is planned."""
    q = np.array(joints, float)
    if plan.reset_first:  # the wrist turns back to its middle first
        low, high = workcell.wrist_range
        q[-1] = (low + high) / 2
    steps = [
        ("pre-grasp", plan.pregrasp),
        ("approach", plan.approach_end),
        ("lift", plan.lift),
        *(("carry", via) for via in plan.carry_via),
        ("carry", plan.carry),
        ("lower", plan.lower),
        ("retreat", plan.retreat),
    ]
    for step, pose in steps:
        q = chain.ik(workcell.tip_of_tcp(pose), q)
        if q is None:
            return f"no joint solution at '{step}' in the arm's posture"
        hit = workcell.collision.hit(chain, q) if workcell.collision is not None else None
        if hit:
            return f"{hit} at '{step}'"
    return None


class GraspSequence:
    """
    Runs a GraspPlan. `tick` is called while the arm moves (display, stop key): a returned
    reason stops the arm and raises OperatorStop. `hold_check` is asked, along with the
    vacuum switch, whether the fish is still on the cups while it is lifted and carried.
    """

    def __init__(
        self,
        arm: Arm,
        gripper: Gripper,
        workcell: Workcell,
        hold_check: HoldCheck | None = None,
        tick: Tick | None = None,
        on_step: Callable[[str], None] | None = None,
        require_hold_confirmation: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        controller: Callable[[str], str] | None = None,
    ):
        """`controller`: runs a command line on the arm's controller (FairinoIO.execute),
        for the grasp settings' press_*_commands; raises ArmError if the controller refuses
        it (what its answer means depends on the driver)."""
        self.arm = arm
        self.gripper = gripper
        self.workcell = workcell
        self.hold_check = hold_check
        self.require_hold_confirmation = require_hold_confirmation
        self._tick = tick
        self._on_step = on_step or (lambda step: None)
        self._clock = clock
        self._sleep = sleep
        self.step = ""  # the step running, or the one that failed
        self.holding = False  # a fish is (believed to be) on the cups
        self.contact_depth: float | None = None  # m past the expected surface where it was felt
        self._controller = controller
        self._not_holding_since: float | None = None
        self._release_z = math.inf  # TCP height at which a held fish rests on the table
        self._picked_at: np.ndarray | None = None  # x, y where the cups took the fish
        self._carried = 0  # carry stops reached: a carry planned again goes on from there

    def run(self, plan: GraspPlan) -> None:
        """All nine steps. Raises ArmError (the arm stopped at `self.step`) or OperatorStop."""
        m = self.workcell.motion
        vacuum = not plan.dry_run
        self._release_z = plan.lower[2, 3]
        self._picked_at = plan.surface[:2, 3].copy()
        self._carried = 0
        self._begin("pre-grasp")
        if plan.reset_first:
            self._move(self._reset_wrist)
        self._move(lambda: self.arm.move_ptp(plan.pregrasp, self._moving))
        if vacuum:
            self._begin("vacuum on")  # before the cups reach the fish (CS-44)
            self.gripper.vacuum_on()
        self._begin("approach")
        try:
            self._send(self.workcell.grasp.press_start_commands)
            self._move(lambda: self._approach(plan))
            if vacuum:
                self._begin("confirm hold")
                self._confirm_hold()
                self.holding = True
                self.arm.set_held_fish(plan.fish_length, plan.middle_x)
        except BaseException:
            try:  # whatever was switched on for the press goes off again
                self._send(self.workcell.grasp.press_end_commands)
            except ArmError:
                pass
            raise
        self._send(self.workcell.grasp.press_end_commands)
        self._begin("lift")
        if not self._blended(plan):  # one at a time, a stop at each corner
            self._move(lambda: self._lin(plan.lift, m.carry_speed))
            self._begin("carry")
            self._move(lambda: self._carry(plan.carry, plan.carry_via))
            self._begin("lower")
            self._move(lambda: self._lin(plan.lower, m.carry_speed))
        if vacuum:
            self._begin("release")
            self._release()
        self._begin("retreat")
        self._move(lambda: self._lin(plan.retreat, m.free_speed))
        self.step = ""

    def _blended(self, plan: GraspPlan) -> bool:
        """Lift, carry and lower as one move with rounded corners, where the arm can plan
        that (motion.blend_radius): True once done. False, nothing moved: one at a time.
        While it moves the step follows where the cups are."""
        m = self.workcell.motion
        through = getattr(self.arm, "move_through", None)
        if through is None or m.blend_radius <= 0:
            return False
        poses = [plan.lift, *plan.carry_via, plan.carry, plan.lower]
        corners = [self.arm.tcp_pose()[:3, 3]] + [p[:3, 3] for p in poses]
        segments = ["lift", *["carry"] * (len(plan.carry_via) + 1), "lower"]
        steps = ("lift", "carry", "lower")

        def reach(step: str) -> None:  # each step on the way is begun, in order
            for later in steps[steps.index(self.step) + 1 : steps.index(step) + 1]:
                self._begin(later)

        def moving() -> str | None:
            reach(segments[_nearest_segment(self.arm.tcp_pose()[:3, 3], corners)])
            return self._moving()

        done = []
        self._move(lambda: done.append(through(poses, m.carry_speed, m.tcp_accel_max,
                                               m.blend_radius, moving)))  # fmt: skip
        if done and done[-1]:
            reach("lower")
            return True
        return False

    def _reset_wrist(self) -> None:
        """Turn the wrist back to its middle where the tool is (about the vertical), so the
        turn to the grasp that follows is shorter than WRIST_MOVE."""
        pose = wrist_back_pose(self.arm, self.workcell)
        if pose is None:
            return
        if problem := self.workcell.pose_problem(pose):
            raise ArmError(E_MOTION, f"turning the wrist back: {problem}")
        self.arm.move_ptp(pose, self._moving)

    def let_go(self) -> None:
        """Vacuum off and forget the fish (after a failed grasp or a drop)."""
        self.gripper.release()
        self._dropped()

    def retreat_up(self) -> None:
        """Straight up from here by the retreat distance, as far as the work zone allows."""
        pose = self.arm.tcp_pose()
        top = self.workcell.work_zone.z[1] - 0.01
        pose[2, 3] = min(pose[2, 3] + self.workcell.grasp.retreat_distance, top)
        if pose[2, 3] > self.arm.tcp_pose()[2, 3] + 1e-3:
            self._lin(pose, self.workcell.motion.free_speed)

    # -- steps ---------------------------------------------------------------
    def _begin(self, step: str) -> None:
        self.step = step
        self._on_step(step)

    def _lin(self, pose: np.ndarray, speed: float, until: Callable[[], bool] | None = None) -> bool:
        return self.arm.move_lin(
            pose, speed, self.workcell.motion.tcp_accel_max, self._moving, until
        )

    def _send(self, commands) -> None:
        """Controller commands of the grasp settings; one the controller refuses stops here."""
        for command in commands if self._controller else ():
            try:
                self._controller(command)
            except ArmError as e:
                raise self._error(E_COMM, e.message or str(e)) from None

    def _approach(self, plan: GraspPlan) -> None:
        """Onto the fish. If the arm can feel: to just above where the fish should be, then
        on down in one slow move that stops as soon as the fish pushes back and the cups have
        pressed on; otherwise to the planned depth."""
        g, m = self.workcell.grasp, self.workcell.motion
        self.contact_depth = None
        feel = g.feel_contact and not plan.dry_run and self.arm.contact_force() is not None
        if not feel:
            self._lin(plan.approach_end, m.approach_speed)
            return
        start = -g.feel_start
        self._lin(plan.at_depth(start), m.approach_speed)
        deepest = self._deepest(plan, g.feel_max_depth + g.press_distance, start + ZERO_AFTER)
        surface, axis = plan.surface[:3, 3], plan.surface[:3, 2]
        zeroed = False

        def pressed() -> bool:  # polled as the arm moves down
            nonlocal zeroed
            depth = float((self.arm.tcp_pose()[:3, 3] - surface) @ axis)
            if not zeroed:
                if depth < start + ZERO_AFTER:
                    return False
                self.arm.contact_force(zero=True)
                zeroed = True
                return False
            force = self.arm.contact_force()
            if force is None:  # the controller stopped saying its joint torques
                raise self._error(E_COMM, "the joint torques stopped while feeling for the fish")
            if self.contact_depth is None and force >= g.contact_force:
                self.contact_depth = depth
            if self.contact_depth is None:
                return depth >= g.feel_max_depth - 1e-9
            pressed_on = depth >= self.contact_depth + g.press_distance - 1e-9
            return pressed_on or force >= g.press_force_max

        self._lin(plan.at_depth(deepest), g.feel_speed, until=pressed)
        if self.contact_depth is None:
            reached = min(deepest, g.feel_max_depth)
            raise self._error(
                E_NO_SEAL, f"nothing felt {reached * 1000:.0f} mm past where the fish should be"
            )

    def _deepest(self, plan: GraspPlan, depth: float, shallowest: float) -> float:
        """`depth` past the fish's expected surface, or as far as the cups may go short of it
        (the table's clearance under a thin fish, a zone)."""
        while self.workcell.pose_problem(plan.at_depth(depth)):
            depth -= 0.001
            if depth <= shallowest:
                raise self._error(E_MOTION, "no room to feel for the fish above the table")
        return depth

    def _carry(self, pose: np.ndarray, via: list | None = None) -> None:
        """Straight to the carry pose (through `via`: the wrist turning evenly the way the
        plan chose), else point-to-point (carry_ptp_fallback). Planned again (_move), it goes
        on from the stop it was heading for: back to the first, the tool would turn the short
        way, the way `via` keeps the wrist from."""
        stops = [*(via or []), pose]
        try:
            while self._carried < len(stops):
                self._lin(stops[self._carried], self.workcell.motion.carry_speed)
                self._carried += 1
        except ArmError as e:
            if e.code != E_MOTION or not self.workcell.motion.carry_ptp_fallback:
                raise
            self.arm.move_ptp(pose, self._moving)

    def _move(self, motion: Callable[[], None]) -> None:
        """Run a motion; E_MOTION is planned again once (Table CS-1), a stop is sorted out."""
        try:
            try:
                motion()
            except ArmError as e:
                if e.code != E_MOTION:
                    raise
                motion()
        except MotionAborted as aborted:
            if aborted.reason != E_DROP:
                raise OperatorStop(aborted.reason) from None
            raise self._drop_error() from None
        except ArmError as e:
            e.gripper_state = self.gripper.state()
            if e.tcp_pose is None:
                e.tcp_pose = self.arm.tcp_pose()
            if e.code == E_MOTION and self.holding:
                e.fish_put_back = self._put_back()
                e.still_holding = self.holding
            raise

    def _moving(self) -> str | None:
        """The Tick given to the arm: the caller's, then the hold checks (AS-70)."""
        reason = self._tick() if self._tick else None
        if reason or not self.holding:
            return reason
        if self.gripper.has_switch:
            if self.gripper.state() == GripperState.HOLDING:
                self._not_holding_since = None
            else:
                now = self._clock()
                if self._not_holding_since is None:
                    self._not_holding_since = now
                elif now - self._not_holding_since > self.workcell.grasp.drop_time:
                    return E_DROP
        if self.hold_check is not None and self.hold_check.lost():
            return E_DROP
        return None

    def _confirm_hold(self) -> None:
        g = self.workcell.grasp
        if self.gripper.has_switch:
            if not wait_for(self.gripper, GripperState.HOLDING, g.seal_timeout, self._clock,
                            self._sleep):  # fmt: skip
                raise self._error(E_NO_SEAL, f"no hold {g.seal_timeout:g} s after the approach")
        else:
            self._sleep(g.seal_dwell)
        seen = self.hold_check.start() if self.hold_check is not None else False
        if not (self.gripper.has_switch or seen) and self.require_hold_confirmation:
            raise self._error(
                E_NO_SEAL, "the hold can't be confirmed: no vacuum switch and no vision cue"
            )
        self._not_holding_since = None

    def _release(self) -> None:
        g = self.workcell.grasp
        self.gripper.release()
        if self.gripper.has_switch and not wait_for(
            self.gripper, GripperState.VACUUM_OFF, g.release_timeout, self._clock, self._sleep
        ):
            raise self._error(E_RELEASE, f"vacuum still on {g.release_timeout:g} s after release")
        if not self.gripper.has_switch:
            self._sleep(g.release_dwell)
        self._dropped()

    def _dropped(self) -> None:
        self.holding = False
        self.arm.set_held_fish(0.0)
        if self.hold_check is not None:
            self.hold_check.stop()

    def _drop_error(self) -> ArmError:
        """The fish is gone (AS-72). The vacuum goes off unless the switch still reports a
        hold: then vision and the switch disagree, and letting go could drop a held fish."""
        error = self._error(E_DROP, f"fish lost during '{self.step}'")
        if self.gripper.has_switch and error.gripper_state == GripperState.HOLDING:
            error.still_holding = True
            error.message += "; the vacuum switch still reports a hold: vacuum left on"
        else:
            self.let_go()
        return error

    def _put_back(self) -> bool:
        """After a failed move with a fish on the cups: set it down where the arm is, if
        that is where it lay, in the pick area (as plan_grasp counts it) or in the placement
        zone."""
        try:
            here = self.arm.tcp_pose()
            xy = here[:2, 3]
            cell = self.workcell
            where_it_lay = self._picked_at is not None and (
                float(np.linalg.norm(xy - self._picked_at)) <= PUT_BACK_NEAR
            )
            if not (where_it_lay or cell.in_pick_area(xy) or cell.placement_zone.contains(xy)):
                return False
            down = here.copy()
            down[2, 3] = min(here[2, 3], self._release_z)
            self._lin(down, cell.motion.carry_speed)
            self._release()
            self.retreat_up()
            return True
        except (ArmError, MotionAborted):
            return False

    def _error(self, code: str, message: str) -> ArmError:
        error = ArmError(code, message, self.arm.tcp_pose())
        error.gripper_state = self.gripper.state()
        return error
