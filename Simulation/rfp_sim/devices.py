"""
The virtual FR5 and suction gripper, behind the same interfaces as the real ones
(rfp_arm.interfaces.Arm and Gripper), so the cycle runs unchanged. Like the MoveIt arm,
every move is planned first (inverse kinematics along the path), checked by
safety.check_trajectory, then run, a pose every `step_s` of simulated time.
"""

import math
import time
from collections.abc import Callable, Sequence

import numpy as np
from rfp_arm.interfaces import E_MOTION, ArmError, GripperState, MotionAborted, Tick
from rfp_arm.kinematics import wrist_angle_at
from rfp_arm.poses import rotation_angle
from rfp_arm.safety import check_trajectory
from rfp_arm.sim import interpolate
from rfp_arm.workcell import Workcell

from rfp_sim.world import World

JOINT_SPEED = 3.15  # rad/s, the FR5's joint limit (joint_limits.yaml)
ROTATION_SPEED = 1.0  # rad/s of the tool on straight moves


class VirtualArm:
    def __init__(
        self,
        world: World,
        workcell: Workcell,
        on_step: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.world = world
        self.workcell = workcell
        self.chain = world.chain
        self.step_s = world.settings.step_s
        self.speed = world.settings.speed  # simulated seconds per real one; 0: no pacing
        self._on_step = on_step or (lambda: None)
        self._clock = clock
        self._sleep = sleep
        self._fish_length = 0.0
        self._zero = 0.0
        self._stop = False
        self.moves: list[tuple[str, float]] = []  # (kind, seconds) of each move run

    # -- state -------------------------------------------------------------------
    def tcp_pose(self) -> np.ndarray:
        return self.world.tcp()

    def joints(self) -> dict[str, float]:
        return dict(zip(self.chain.joint_names, self.world.joints))

    def wrist_angle(self, tcp_pose: np.ndarray | None = None) -> float | None:
        if tcp_pose is None:
            return float(self.world.joints[-1])
        return wrist_angle_at(self.chain, self.joints(), self.workcell.tip_of_tcp(tcp_pose))

    def contact_force(self, zero: bool = False) -> float | None:
        force = self.world.contact_force()
        if zero:
            self._zero = force
        return force - self._zero

    def set_held_fish(self, length: float) -> None:
        self._fish_length = length

    def stop(self) -> None:
        self._stop = True

    def fault(self) -> str | None:
        return None

    # -- moves ---------------------------------------------------------------------
    def move_joints(self, joints: Sequence[float], tick: Tick | None = None) -> None:
        self._run("joints", *self._joint_path(np.array(joints, float)), tick)

    def move_ptp(self, tcp_pose: np.ndarray, tick: Tick | None = None) -> None:
        target = self.chain.ik(self.workcell.tip_of_tcp(tcp_pose), self.world.joints)
        if target is None:
            raise ArmError(E_MOTION, "point-to-point move: no joint solution", self.tcp_pose())
        self._run("ptp", *self._joint_path(target), tick)

    def move_lin(
        self, tcp_pose: np.ndarray, speed: float, accel: float, tick: Tick | None = None
    ) -> None:
        start = self.tcp_pose()
        speed = min(speed, self.workcell.motion.tcp_speed_max)
        distance = float(np.linalg.norm(tcp_pose[:3, 3] - start[:3, 3]))
        duration = max(distance / speed, rotation_angle(start, tcp_pose) / ROTATION_SPEED)
        steps = max(1, math.ceil(duration / self.step_s))
        positions, q = [self.world.joints.copy()], self.world.joints
        for i in range(1, steps + 1):
            q = self.chain.ik(self.workcell.tip_of_tcp(interpolate(start, tcp_pose, i / steps)), q)
            if q is None:
                raise ArmError(E_MOTION, "straight move: no joint solution", self.tcp_pose())
            positions.append(q)
        self._run("lin", np.array(positions), np.linspace(0, duration, steps + 1), tick)

    def _joint_path(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space path at the point-to-point speed scale, slowed where the TCP would
        pass 90% of its speed limit."""
        start = self.world.joints
        joint_speed = self.workcell.motion.ptp_velocity_scale * JOINT_SPEED
        steps = max(1, math.ceil(np.abs(target - start).max() / joint_speed / self.step_s))
        positions = start + (target - start) * np.linspace(0, 1, steps + 1)[:, None]
        tcp = np.array([self.workcell.tcp_of_tip(self.chain.fk(q))[:3, 3] for q in positions])
        fastest = 0.9 * self.workcell.motion.tcp_speed_max
        step_time = np.maximum(
            np.abs(np.diff(positions, axis=0)).max(axis=1) / joint_speed,
            np.linalg.norm(np.diff(tcp, axis=0), axis=1) / fastest,
        )
        return positions, np.concatenate([[0.0], np.cumsum(step_time)])

    def _run(self, kind: str, positions: np.ndarray, times: np.ndarray, tick: Tick | None) -> None:
        check = check_trajectory(
            self.workcell, self.chain, self.chain.joint_names, positions, times, self._fish_length
        )
        if not check.ok:
            raise ArmError(E_MOTION, f"{kind} move refused: {check.problem}", self.tcp_pose())
        self._stop = False
        started = self._clock()
        for q, t in zip(positions[1:], times[1:]):
            self.world.set_joints(q)
            self._on_step()
            if self.speed > 0:
                self._sleep(max(0.0, started + t / self.speed - self._clock()))
            reason = tick() if tick else None
            if reason or self._stop:
                raise MotionAborted(reason or "stopped")
        self.moves.append((kind, float(times[-1])))


class VirtualGripper:
    """The six cups under one vacuum. Like the real valve board it can't tell whether the
    cups hold a fish (`has_switch` False): the camera has to confirm the hold."""

    def __init__(self, world: World, has_switch: bool = False):
        self.world = world
        self.has_switch = has_switch

    def vacuum_on(self) -> None:
        self.world.set_vacuum(True)

    def release(self) -> None:
        self.world.set_vacuum(False)

    def state(self) -> GripperState:
        if not self.world.vacuum:
            return GripperState.VACUUM_OFF
        held = self.world.holding is not None
        return GripperState.HOLDING if self.has_switch and held else GripperState.NO_SEAL
