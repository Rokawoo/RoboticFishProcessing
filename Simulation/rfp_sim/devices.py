"""
The virtual FR5 and suction gripper, behind the same interfaces as the real ones
(rfp_arm.interfaces.Arm and Gripper), so the cycle runs unchanged. Like the MoveIt arm,
every move is planned first (inverse kinematics along the path), checked by
safety.check_trajectory, then run, a pose every `step_s` of simulated time. Moves are timed
as MoveIt's Pilz planner times them on the arm: from rest to rest, speeding up and slowing
down at the acceleration limits (point-to-point: the joints' limits in
Arm/ros/joint_limits.yaml times the workcell's ptp scales; straight: the move's speed and
motion.tcp_accel_max), then slowed wherever any point of the arm would pass tcp_speed_max.
"""

import itertools
import math
import time
from collections.abc import Callable, Sequence

import numpy as np
from rfp_arm.interfaces import E_MOTION, ArmError, GripperState, MotionAborted, Tick
from rfp_arm.kinematics import wrist_angle_at
from rfp_arm.moveit_arm import BLEND_ROOM
from rfp_arm.poses import interpolate, rotation_angle
from rfp_arm.safety import check_trajectory
from rfp_arm.workcell import Workcell

from rfp_sim.world import World

# The FR5's joint limits MoveIt plans with (Arm/ros/joint_limits.yaml: a test compares)
JOINT_SPEEDS = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])  # rad/s
JOINT_ACCELS = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])  # rad/s2
ROTATION_SPEED = 1.0  # rad/s of the tool on straight moves
ROTATION_ACCEL = 2.0  # rad/s2


def rest_to_rest_time(distance: float, speed: float, accel: float) -> float:
    """Seconds a move over `distance` takes from rest to rest at most `speed` and `accel`."""
    if distance <= 0.0:
        return 0.0
    ramp = speed / accel
    if distance <= speed * ramp:  # too short to reach `speed`: up and straight down again
        return 2.0 * math.sqrt(distance / accel)
    return distance / speed + ramp


def rest_to_rest(distance: float, speed: float, accel: float, step: float):
    """(times, shares): every `step` s of a rest-to-rest move over `distance` at most `speed`
    and `accel` (a trapezoid, or a triangle when too short), how much of the way is done
    (0 to 1). At least one step."""
    total = rest_to_rest_time(distance, speed, accel)
    times = np.linspace(0.0, total, max(1, math.ceil(total / step)) + 1)
    if distance <= 0.0:
        return times, np.linspace(0.0, 1.0, len(times))
    ramp = min(speed / accel, total / 2.0)
    top = accel * ramp  # the speed it reaches
    done = np.where(
        times < ramp,
        0.5 * accel * times**2,
        np.where(
            times <= total - ramp,
            0.5 * accel * ramp**2 + top * (times - ramp),
            distance - 0.5 * accel * (total - times) ** 2,
        ),
    )
    return times, np.clip(done / distance, 0.0, 1.0)


def rounded_path(frames: Sequence[np.ndarray], blend: float, spacing: float = 0.002) -> list:
    """TCP poses `spacing` m apart along straight moves through `frames` (4x4), each corner
    between two moves rounded as Pilz's sequence blends it: within min(blend, BLEND_ROOM x
    the shorter move) of the corner, on a curve from where the path enters that sphere to
    where it leaves it, the tool turning on from the one to the other."""
    points = [np.asarray(f, float)[:3, 3] for f in frames]
    lengths = [float(np.linalg.norm(b - a)) for a, b in itertools.pairwise(points)]
    radius = [0.0] * len(frames)
    for k in range(1, len(frames) - 1):
        radius[k] = min(blend, BLEND_ROOM * min(lengths[k - 1], lengths[k]))

    def along(i: int, distance: float) -> np.ndarray:
        share = distance / lengths[i] if lengths[i] > 1e-9 else 1.0
        return interpolate(frames[i], frames[i + 1], min(max(share, 0.0), 1.0))

    path = [np.asarray(frames[0], float)]
    for i, length in enumerate(lengths):
        start, end = radius[i], length - radius[i + 1]
        count = max(1, math.ceil((end - start) / spacing))
        path += [along(i, d) for d in np.linspace(start, end, count + 1)[1:]]
        if radius[i + 1] > 0:  # round the corner at frames[i + 1]
            enter, leave = along(i, end), along(i + 1, radius[i + 1])
            corner = points[i + 1]
            for share in np.linspace(0.0, 1.0, max(2, math.ceil(2 * radius[i + 1] / spacing)) + 1)[
                1:
            ]:
                pose = interpolate(enter, leave, share)
                pose[:3, 3] = ((1 - share) ** 2 * enter[:3, 3] + 2 * (1 - share) * share * corner
                               + share**2 * leave[:3, 3])  # fmt: skip
                path.append(pose)
    return path


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
        self._fish_middle_x = 0.0
        self._zero = 0.0
        self._stop = False
        self.moves: list[tuple[str, float]] = []  # (kind, seconds) of each move run
        self.wrist_turns: list[float] = []  # rad the last joint turned in each move run

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

    def set_held_fish(self, length: float, middle_x: float = 0.0) -> None:
        self._fish_length, self._fish_middle_x = length, middle_x

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
        turn = rotation_angle(start, tcp_pose)
        # Along the line as Pilz's LIN goes, the turn alongside; whichever takes longer sets
        # the pace
        if rest_to_rest_time(distance, speed, accel) >= rest_to_rest_time(
            turn, ROTATION_SPEED, ROTATION_ACCEL
        ):
            times, shares = rest_to_rest(distance, speed, accel, self.step_s)
        else:
            times, shares = rest_to_rest(turn, ROTATION_SPEED, ROTATION_ACCEL, self.step_s)
        positions, q = [self.world.joints.copy()], self.world.joints
        for share in shares[1:]:
            q = self.chain.ik(self.workcell.tip_of_tcp(interpolate(start, tcp_pose, share)), q)
            if q is None:
                raise ArmError(E_MOTION, "straight move: no joint solution", self.tcp_pose())
            positions.append(q)
        positions = np.array(positions)
        self._run("lin", positions, self._paced(positions, np.diff(times)), tick)

    def _joint_path(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space path timed as Pilz's PTP: every joint from rest to rest together, the
        one that takes longest at its scaled limits setting the pace; slowed where any point
        of the arm (a joint, the TCP, the camera) would pass 90% of the speed limit."""
        start = self.world.joints
        m = self.workcell.motion
        delta = np.asarray(target, float) - start
        speeds = m.ptp_velocity_scale * JOINT_SPEEDS[: len(delta)]
        accels = m.ptp_acceleration_scale * JOINT_ACCELS[: len(delta)]
        lead = int(np.argmax([rest_to_rest_time(abs(d), v, a)
                              for d, v, a in zip(delta, speeds, accels)]))  # fmt: skip
        times, shares = rest_to_rest(abs(delta[lead]), speeds[lead], accels[lead], self.step_s)
        positions = start + shares[:, None] * delta
        return positions, self._paced(positions, np.diff(times))

    def _paced(self, positions: np.ndarray, step_time) -> np.ndarray:
        """Times of the path's points: each step takes `step_time` (one or one per step), or
        longer where a point of the arm (a joint, the TCP, the camera) would pass 90% of the
        speed limit, as the safety check counts it."""
        cell = self.workcell
        points = []
        for q in positions:
            links = self.chain.link_poses(q)
            tcp = cell.tcp_of_tip(links[-1])
            points.append(
                np.vstack([[t[:3, 3] for t in links], tcp[:3, 3], cell.camera_of_tcp(tcp)[:3, 3]])
            )
        fastest = 0.9 * cell.motion.tcp_speed_max
        moved = np.linalg.norm(np.diff(np.array(points), axis=0), axis=2).max(axis=1)
        steps = np.maximum(np.broadcast_to(step_time, moved.shape), moved / fastest)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def move_through(
        self,
        poses: Sequence[np.ndarray],
        speed: float,
        accel: float,
        blend: float,
        tick: Tick | None = None,
    ) -> bool:
        """Straight moves through `poses` as one, each corner rounded by up to `blend` m and
        no stop at it, as the arm runs them (MoveItArm.move_through, Pilz's sequence): the
        whole path planned and checked first, then run, timed from rest to rest over its
        length: True. False, with nothing moved, where it can't be (then one at a time)."""
        if blend <= 0 or len(poses) < 2:
            return False
        frames = [self.tcp_pose(), *(np.asarray(p, float) for p in poses)]
        path = rounded_path(frames, blend)
        steps = np.linalg.norm(np.diff([p[:3, 3] for p in path], axis=0), axis=1)
        lengths = np.concatenate([[0.0], np.cumsum(steps)])
        total = float(lengths[-1])
        turn = sum(rotation_angle(a, b) for a, b in itertools.pairwise(frames))
        speed = min(speed, self.workcell.motion.tcp_speed_max)
        if rest_to_rest_time(total, speed, accel) >= rest_to_rest_time(
            turn, ROTATION_SPEED, ROTATION_ACCEL
        ):
            times, shares = rest_to_rest(total, speed, accel, self.step_s)
        else:
            times, shares = rest_to_rest(turn, ROTATION_SPEED, ROTATION_ACCEL, self.step_s)
        positions, q = [self.world.joints.copy()], self.world.joints
        for share in shares[1:]:
            at = share * total
            j = int(np.clip(np.searchsorted(lengths, at), 1, len(path) - 1))
            part = (at - lengths[j - 1]) / max(lengths[j] - lengths[j - 1], 1e-12)
            pose = interpolate(path[j - 1], path[j], float(np.clip(part, 0.0, 1.0)))
            q = self.chain.ik(self.workcell.tip_of_tcp(pose), q)
            if q is None:
                return False
            positions.append(q)
        positions = np.array(positions)
        times = self._paced(positions, np.diff(times))
        if self._problem(positions, times) is not None:
            return False
        self._play("blend", positions, times, tick)
        return True

    def _problem(self, positions: np.ndarray, times: np.ndarray) -> str | None:
        check = check_trajectory(
            self.workcell, self.chain, self.chain.joint_names, positions, times,
            self._fish_length, None, self._fish_middle_x,
        )  # fmt: skip
        return None if check.ok else check.problem

    def _run(self, kind: str, positions: np.ndarray, times: np.ndarray, tick: Tick | None) -> None:
        if problem := self._problem(positions, times):
            raise ArmError(E_MOTION, f"{kind} move refused: {problem}", self.tcp_pose())
        self._play(kind, positions, times, tick)

    def _play(self, kind: str, positions: np.ndarray, times: np.ndarray, tick: Tick | None) -> None:
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
        self.wrist_turns.append(float(abs(positions[-1][-1] - positions[0][-1])))


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
