"""
The grasp sequence of Table AS-1 (AS-20): `plan_grasp` turns a fish's grasp pose and a
placement slot into the TCP poses of the nine steps and checks them against the workcell;
`GraspSequence.run` drives an Arm and a Gripper through them and stops at the step that
fails, with its Table AS-2 error code (AS-25). The same code runs the simulated arm, a dry
run and the real thing. No ROS.

The approach follows the grasp pose's z axis, which the Vision System lays square onto
the fish's surface (within AS-21's 15 degrees of straight down). On an arm that reports
joint torques the last centimeters are felt: down in small steps until the fish pushes
back, then pressed on.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

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
from rfp_arm.poses import rot_z, tilt_from_down, tool_down, wrap, yaw_of
from rfp_arm.workcell import Workcell

# Wrist arcs (rad) this close count as equal: then the one nearer the range's middle
WRIST_TIE = math.radians(5.0)
# Plans keep the wrist this far (rad) inside its range when a way round allows: the tilted
# approach turns it up to 14 deg from where it was predicted (straight down; 10 deg in 95%
# of grasps), and the tube twists less. In the placeholder cell some way always allows it.
WRIST_MARGIN = math.radians(10.0)

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
    # them; it returns to the first. None if the arm can't tell.
    wrist: tuple[float, float, float] | None = None
    middle_x: float = 0.0  # m along the TCP's x axis from the cups to the fish's middle

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
) -> GraspPlan:
    """
    The poses of Table AS-1. `grasp_pose` is the fish's grasp pose from the fish list (on
    the fish, z = approach direction, x = along the body toward the head); `tool_yaw` is
    the TCP's yaw now, at the observation pose, where the arm returns after the place.
    `wrist` (Arm.wrist_angle) tells the last joint's angle now (None) and at a TCP pose;
    without it the wrist is taken to turn with the tool's yaw from the middle of its range.
    `middle`: the fish's middle in {B} where the cups grasp it elsewhere (toward its heavier
    head): the slot gets the middle. Raises ArmError(E_MOTION) if the fish or a pose is
    outside the limits.
    """
    g = workcell.grasp
    point = grasp_pose[:3, 3]
    tilt = math.degrees(tilt_from_down(grasp_pose))
    if tilt > g.max_approach_tilt_deg + 0.01:
        raise ArmError(
            E_MOTION,
            f"approach {tilt:.0f} deg off vertical (at most {g.max_approach_tilt_deg:.0f})",
        )
    if not workcell.pick_area.contains(point if middle is None else middle):
        raise ArmError(E_MOTION, "the fish is outside the pick area (CS-13)")
    if not workcell.placement_zone.contains(slot):
        raise ArmError(E_MOTION, "the slot is outside the placement zone")

    heading = yaw_of(grasp_pose)
    turn = wrap(workcell.placement_heading - heading)
    turns = [turn] if heading_known else [turn, wrap(turn + math.pi)]  # either end: head

    x, y, z = point
    table = workcell.table_z
    lift_z = table + min(g.lift_length_share * fish_length + g.lift_margin, g.lift_height_max)
    lift_z = max(lift_z, z + g.lift_margin)
    release_z = table + max(thickness, 0.0) + g.release_clearance

    # The cups fit the body both ways round, and a fish whose head end isn't known may go
    # into the slot either way round. Each way the wrist turns from where it is to the
    # grasp, with the fish to the slot, and back to where it is now (the observation pose):
    # twice the arc it sweeps. Take the smallest arc inside the wrist's range (the vacuum
    # tube), clear of its ends if a way allows; of arcs within a few degrees of it, the one
    # that stays nearest the middle.
    low, high = workcell.wrist_range
    center = (low + high) / 2
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
            targets = angles[1:]
            ways.append(
                {
                    "inside": all(low <= a <= high for a in targets),
                    "sweep": max(angles) - min(angles),
                    "peak": max(abs(a - center) for a in targets),
                    "worst": max(targets, key=lambda a: abs(a - center)),
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
    roomy = [way for way in inside if way["peak"] <= (high - low) / 2 - WRIST_MARGIN] or inside
    smallest = min(way["sweep"] for way in roomy)
    way = min(
        (way for way in roomy if way["sweep"] <= smallest + WRIST_TIE),
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
    )
    plan.pregrasp = plan.at_depth(-g.pregrasp_distance)
    plan.approach_end = plan.at_depth(-g.dry_run_standoff if dry_run else g.press_distance)
    end = plan.approach_end[:3, 3]
    plan.lift = tool_down((end[0], end[1], max(lift_z, end[2] + g.lift_margin)), grasp_yaw)
    for name, pose in plan.poses().items():
        held = fish_length if name in ("lift", "carry", "lower") else 0.0
        if problem := workcell.pose_problem(pose, held, True, plan.middle_x):
            raise ArmError(E_MOTION, f"{name} pose: {problem}")
    return plan


class OperatorStop(Exception):
    """The operator (or the caller's Tick) stopped the sequence. `reason` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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
        """`controller`: sends a command line to the arm's controller and returns its
        answer (FairinoIO.send), for the grasp settings' press_*_commands."""
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

    def run(self, plan: GraspPlan) -> None:
        """All nine steps. Raises ArmError (the arm stopped at `self.step`) or OperatorStop."""
        m = self.workcell.motion
        vacuum = not plan.dry_run
        self._release_z = plan.lower[2, 3]
        self._begin("pre-grasp")
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
        self._move(lambda: self._lin(plan.lift, m.carry_speed))
        self._begin("carry")
        self._move(lambda: self._carry(plan.carry))
        self._begin("lower")
        self._move(lambda: self._lin(plan.lower, m.carry_speed))
        if vacuum:
            self._begin("release")
            self._release()
        self._begin("retreat")
        self._move(lambda: self._lin(plan.retreat, m.free_speed))
        self.step = ""

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

    def _lin(self, pose: np.ndarray, speed: float) -> None:
        self.arm.move_lin(pose, speed, self.workcell.motion.tcp_accel_max, self._moving)

    def _send(self, commands) -> None:
        """Controller commands of the grasp settings; an answer other than "0" is an error."""
        for command in commands if self._controller else ():
            answer = self._controller(command)
            if answer != "0":
                raise self._error(E_COMM, f"the controller answered '{answer}' to {command}")

    def _approach(self, plan: GraspPlan) -> None:
        """Onto the fish. If the arm can feel: slowly down from just above where the fish
        should be until it pushes back, then press; otherwise to the planned depth."""
        g, m = self.workcell.grasp, self.workcell.motion
        self.contact_depth = None
        feel = g.feel_contact and not plan.dry_run and self.arm.contact_force() is not None
        if not feel:
            self._lin(plan.approach_end, m.approach_speed)
            return
        depth = -g.feel_start
        self._lin(plan.at_depth(depth), m.approach_speed)
        self._sleep(g.feel_settle)
        self.arm.contact_force(zero=True)
        while True:
            force = self.arm.contact_force()
            if self.contact_depth is None and force >= g.contact_force:
                self.contact_depth = depth
            if self.contact_depth is None:
                limit = g.feel_max_depth
                if depth >= limit - 1e-9:
                    raise self._error(
                        E_NO_SEAL,
                        f"nothing felt {limit * 1000:.0f} mm past where the fish should be",
                    )
            else:
                limit = self.contact_depth + g.press_distance
                if depth >= limit - 1e-9 or force >= g.press_force_max:
                    return
            depth = min(depth + g.feel_step, limit)
            self._lin(plan.at_depth(depth), g.feel_speed)
            self._sleep(g.feel_settle)

    def _carry(self, pose: np.ndarray) -> None:
        try:
            self._lin(pose, self.workcell.motion.carry_speed)
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
        that is on the table inside the pick area or the placement zone."""
        try:
            here = self.arm.tcp_pose()
            xy = here[:2, 3]
            cell = self.workcell
            if not (cell.pick_area.contains(xy) or cell.placement_zone.contains(xy)):
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
