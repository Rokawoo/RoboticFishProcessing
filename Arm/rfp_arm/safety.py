"""
Fixed-code check of a planned joint trajectory before it runs (CS-30, CS-43, SF-10, SF-21):
the TCP's speed, where the tool, the camera and the held fish go, and how far the joints
turn. MoveIt checks collisions; this is the layer that doesn't depend on its scene. No ROS.
"""

from dataclasses import dataclass

import numpy as np

from rfp_arm.kinematics import Chain
from rfp_arm.workcell import Workcell


@dataclass
class TrajectoryCheck:
    problem: str | None  # None: the trajectory may run
    too_fast: bool  # the only kind of problem a slower plan fixes
    peak_tcp_speed: float  # m/s
    duration: float  # s
    max_joint_travel: float  # rad, largest change of any joint from the start
    end_tcp: np.ndarray | None  # 4x4 TCP pose at the last point

    @property
    def ok(self) -> bool:
        return self.problem is None

    def summary(self) -> str:
        return (
            f"{self.duration:.1f} s, TCP up to {self.peak_tcp_speed * 1000:.0f} mm/s, "
            f"joints turn up to {np.degrees(self.max_joint_travel):.0f} deg"
        )


def check_trajectory(
    workcell: Workcell,
    chain: Chain,
    joint_names: list[str],
    positions: np.ndarray,
    times: np.ndarray,
    fish_length: float = 0.0,
    tcp_speed_max: float | None = None,
) -> TrajectoryCheck:
    """
    Check trajectory points (rows of joint positions in `joint_names` order, at `times`
    seconds from the start). The path may start outside the work zone (e.g. where the arm
    was left), but once inside it has to stay inside, and it has to end inside.
    """
    positions = np.asarray(positions, float)
    times = np.asarray(times, float)
    limit = workcell.motion.tcp_speed_max if tcp_speed_max is None else tcp_speed_max
    empty = TrajectoryCheck("the plan has no points", False, 0.0, 0.0, 0.0, None)
    if len(positions) == 0 or positions.shape[1] != len(joint_names):
        return empty
    missing = [name for name in chain.joint_names if name not in joint_names]
    if missing:
        empty.problem = f"the plan has no joint {', '.join(missing)}"
        return empty

    poses = []
    for row in positions:
        joints = dict(zip(joint_names, row))
        if violation := chain.limit_violation(joints, margin=1e-3):
            empty.problem = f"joint limit: {violation}"
            return empty
        poses.append(workcell.tcp_of_tip(chain.fk(joints)))
    points = np.array([pose[:3, 3] for pose in poses])

    peak = 0.0
    steps = np.diff(times)
    if len(steps) and (steps < 0).any():
        empty.problem = "the plan's times go backward"
        return empty
    moving = steps > 1e-6
    if moving.any():
        peak = float(
            (np.linalg.norm(np.diff(points, axis=0), axis=1)[moving] / steps[moving]).max()
        )
    travel = float(np.abs(positions - positions[0]).max())
    check = TrajectoryCheck(None, False, peak, float(times[-1] - times[0]), travel, poses[-1])

    entered = False
    for i, pose in enumerate(poses):
        problem = workcell.pose_problem(pose, fish_length)
        if problem is None:
            entered = True
            continue
        # Outside the work zone is allowed only before the path first enters it.
        allowed = not entered and i < len(poses) - 1
        if allowed:
            problem = workcell.pose_problem(pose, fish_length, work_zone=False)
        if problem is not None:
            check.problem = f"{problem} ({times[i] - times[0]:.1f} s into the move)"
            return check
    max_travel = np.radians(workcell.motion.max_joint_travel_deg)
    if travel > max_travel:
        check.problem = (
            f"a joint would turn {np.degrees(travel):.0f} deg (limit "
            f"{workcell.motion.max_joint_travel_deg:.0f}): the arm would change its posture"
        )
    elif peak > limit * 1.02:  # 2%: rounding of the sampled points
        check.problem = f"TCP speed {peak * 1000:.0f} mm/s is above {limit * 1000:.0f} mm/s"
        check.too_fast = True
    return check
