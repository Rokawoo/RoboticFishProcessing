"""
The grasp sequence of Table AS-1 (AS-20): `plan_grasp` turns a fish's grasp pose and a
placement slot into the TCP poses of the nine steps and checks them against the workcell;
`GraspSequence.run` drives an Arm and a Gripper through them and stops at the step that
fails, with its Table AS-2 error code (AS-25). The same code runs the simulated arm, a dry
run and the real thing. No ROS.

Only approaches straight down are planned so far (the cups meet a fish lying on its side
within AS-21's 15 degrees).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rfp_arm.gripper import wait_for
from rfp_arm.interfaces import (
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
from rfp_arm.poses import tilt_from_down, tool_down, wrap, yaw_of
from rfp_arm.workcell import Workcell

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
) -> GraspPlan:
    """
    The poses of Table AS-1. `grasp_pose` is the fish's grasp pose from the fish list (on
    the fish, z = approach direction, x = along the body toward the head); `tool_yaw` is
    the TCP's yaw now, at the observation pose. Raises ArmError(E_MOTION) if the fish or a
    pose is outside the limits.
    """
    g = workcell.grasp
    point = grasp_pose[:3, 3]
    tilt = math.degrees(tilt_from_down(grasp_pose))
    if tilt > g.max_approach_tilt_deg:
        raise ArmError(E_MOTION, f"approach {tilt:.0f} deg off vertical isn't supported yet")
    if not workcell.pick_area.contains(point):
        raise ArmError(E_MOTION, "the grasp point is outside the pick area (CS-13)")
    if not workcell.placement_zone.contains(slot):
        raise ArmError(E_MOTION, "the slot is outside the placement zone")

    heading = yaw_of(grasp_pose)
    turn = wrap(workcell.placement_heading - heading)
    if not heading_known and abs(turn) > math.pi / 2:  # either end may be the head: turn less
        turn = wrap(turn + math.pi)
    # The cups fit the body both ways round. The wrist turns from `tool_yaw` to the grasp
    # and then by `turn` to the slot: take the way that keeps it nearest where it started,
    # away from the limits of j6.
    ways = [wrap(heading + flip - tool_yaw) for flip in (0.0, math.pi)]
    wrist = min(ways, key=lambda offset: max(abs(offset), abs(offset + turn)))
    grasp_yaw = wrap(tool_yaw + wrist)
    place_yaw = wrap(grasp_yaw + turn)

    x, y, z = point
    table = workcell.table_z
    end_z = z + g.dry_run_standoff if dry_run else z - g.press_distance
    lift_z = table + min(g.lift_length_share * fish_length + g.lift_margin, g.lift_height_max)
    lift_z = max(lift_z, z + g.lift_margin)
    release_z = table + max(thickness, 0.0) + g.release_clearance
    plan = GraspPlan(
        pregrasp=tool_down((x, y, z + g.pregrasp_distance), grasp_yaw),
        approach_end=tool_down((x, y, end_z), grasp_yaw),
        lift=tool_down((x, y, lift_z), grasp_yaw),
        carry=tool_down((slot[0], slot[1], lift_z), place_yaw),
        lower=tool_down((slot[0], slot[1], release_z), place_yaw),
        retreat=tool_down((slot[0], slot[1], release_z + g.retreat_distance), place_yaw),
        fish_length=fish_length,
        turn=turn,
        dry_run=dry_run,
    )
    for name, pose in plan.poses().items():
        held = fish_length if name in ("lift", "carry", "lower") else 0.0
        if problem := workcell.pose_problem(pose, held):
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
    ):
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
        self._move(lambda: self._lin(plan.approach_end, m.approach_speed))
        if vacuum:
            self._begin("confirm hold")
            self._confirm_hold()
            self.holding = True
            self.arm.set_held_fish(plan.fish_length)
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
