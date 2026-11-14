"""
A simulated arm with no robot, no MoveIt and no ROS: it keeps a TCP pose, applies the
workcell's limits and walks to each target in small steps, calling the Tick on the way.
For tests and for running the whole cycle offline on a recorded video.
"""

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from rfp_arm.interfaces import E_MOTION, ArmError, MotionAborted, Tick
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.poses import axis_rotation, from_parts, rotation_angle
from rfp_arm.workcell import Workcell

STEP = 0.02  # m between simulated points


def interpolate(start: np.ndarray, end: np.ndarray, share: float) -> np.ndarray:
    """The pose `share` (0-1) of the way from start to end: straight line, constant turn rate."""
    relative = start[:3, :3].T @ end[:3, :3]
    angle = rotation_angle(np.eye(3), relative)
    rotation = start[:3, :3]
    if angle > 1e-9:
        if math.pi - angle < 1e-6:  # half turn: the axis is the eigenvector of eigenvalue 1
            values, vectors = np.linalg.eig(relative)
            axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        else:
            axis = np.array(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ]
            )
        rotation = rotation @ axis_rotation(axis, angle * share)
    return from_parts(rotation, start[:3, 3] + (end[:3, 3] - start[:3, 3]) * share)


class SimArm:
    def __init__(
        self,
        workcell: Workcell,
        start_tcp: np.ndarray | None = None,
        chain: Chain = FR5,
        realtime: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """`realtime`: moves take as long as on the arm; otherwise they are instant."""
        self.workcell = workcell
        self.chain = chain
        self.realtime = realtime
        self._sleep = sleep
        self._fish_length = 0.0
        self._stop = False
        if start_tcp is None:  # above the table, at the first observation pose's height
            first = workcell.observation_poses[0]
            start_tcp = (
                workcell.observation_tcp(first)
                if first.camera is not None
                else workcell.tcp_of_tip(chain.fk(first.joints))
            )
        self._tcp = np.array(start_tcp, float)
        self.moves: list[tuple[str, np.ndarray]] = []  # (kind, target TCP pose) of each move
        self.fail: Callable[[str, np.ndarray], str | None] | None = None  # test hook: E_MOTION

    def tcp_pose(self) -> np.ndarray:
        return self._tcp.copy()

    def set_held_fish(self, length: float) -> None:
        self._fish_length = length

    def move_joints(self, joints: Sequence[float], tick: Tick | None = None) -> None:
        target = self.workcell.tcp_of_tip(self.chain.fk(joints))
        self._move("joints", target, self.workcell.motion.free_speed, tick, straight=False)

    def move_ptp(self, tcp_pose: np.ndarray, tick: Tick | None = None) -> None:
        self._move("ptp", tcp_pose, self.workcell.motion.free_speed, tick, straight=False)

    def move_lin(
        self, tcp_pose: np.ndarray, speed: float, accel: float, tick: Tick | None = None
    ) -> None:
        self._move("lin", tcp_pose, speed, tick, straight=True)

    def stop(self) -> None:
        self._stop = True

    def fault(self) -> str | None:
        return None

    def _move(self, kind: str, target: np.ndarray, speed: float, tick, straight: bool) -> None:
        target = np.array(target, float)
        self._stop = False
        if speed > self.workcell.motion.tcp_speed_max + 1e-9:
            raise ArmError(E_MOTION, f"{speed * 1000:.0f} mm/s is above the TCP speed limit")
        problem = self.fail(kind, target) if self.fail else None
        problem = problem or self.workcell.pose_problem(target, self._fish_length)
        if problem:
            raise ArmError(E_MOTION, problem, self.tcp_pose())
        start = self._tcp.copy()
        distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        steps = max(1, math.ceil(distance / STEP))
        for i in range(1, steps + 1):
            pose = interpolate(start, target, i / steps)
            # Only a straight move's path is known here; a real PTP path is checked by MoveIt.
            if straight and (problem := self.workcell.pose_problem(pose, self._fish_length)):
                raise ArmError(E_MOTION, problem, self.tcp_pose())
            if self.realtime:
                self._sleep(distance / steps / speed)
            self._tcp = pose
            reason = tick() if tick else None
            if reason or self._stop:
                raise MotionAborted(reason or "stopped")
        self._tcp = target
        self.moves.append((kind, target))
