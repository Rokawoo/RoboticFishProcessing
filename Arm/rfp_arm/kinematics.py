"""
Forward kinematics of a serial chain, from a URDF or the built-in FR5 chain. Used to check
planned trajectories in fixed code (TCP speed, zones) and by the simulated arm. No ROS.
"""

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rfp_arm.poses import (
    axis_rotation,
    from_parts,
    interpolate,
    rotation_angle,
    transform,
    wrap,
)

# When one solve can't reach a far target, it is reached in steps of at most these
IK_STEP_SWING = math.radians(10.0)  # about the base
IK_STEP_TURN = math.radians(15.0)  # of the tip's orientation
IK_STEP = 0.05  # m
IK_MAX_STEPS = 40
# The tip this close to the first joint's axis is on neither side of it (see Chain.ahead)
SIDE_MIN = 0.03  # m


@dataclass(frozen=True)
class Joint:
    name: str
    origin: np.ndarray  # 4x4: joint frame in the parent link frame, at position 0
    axis: tuple[float, float, float]
    kind: str = "revolute"  # revolute, continuous, prismatic or fixed
    lower: float = -np.inf
    upper: float = np.inf

    def motion(self, position: float) -> np.ndarray:
        if self.kind in ("revolute", "continuous"):
            return from_parts(axis_rotation(self.axis, position), (0.0, 0.0, 0.0))
        if self.kind == "prismatic":
            return from_parts(np.eye(3), np.asarray(self.axis) * position)
        return np.eye(4)


@dataclass(frozen=True)
class Chain:
    """Joints from the base link to the tip link, in order."""

    joints: tuple[Joint, ...]
    base_link: str = "base_link"
    tip_link: str = "wrist3_link"
    links: tuple[str, ...] = ()  # the child link of each joint, base to tip

    @property
    def joint_names(self) -> list[str]:
        """The moving joints, base to tip."""
        return [j.name for j in self.joints if j.kind != "fixed"]

    def fk(self, positions: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        """T_base_tip for joint positions (by name, or in joint_names order)."""
        if not isinstance(positions, Mapping):
            positions = dict(zip(self.joint_names, positions))
        t = np.eye(4)
        for joint in self.joints:
            t = t @ joint.origin
            if joint.kind != "fixed":
                t = t @ joint.motion(float(positions[joint.name]))
        return t

    def link_poses(self, positions: Mapping[str, float] | Sequence[float]) -> list[np.ndarray]:
        """T_base_link of the base link and of each joint's child link, base to tip."""
        if not isinstance(positions, Mapping):
            positions = dict(zip(self.joint_names, positions))
        t = np.eye(4)
        poses = [t]
        for joint in self.joints:
            t = t @ joint.origin
            if joint.kind != "fixed":
                t = t @ joint.motion(float(positions[joint.name]))
            poses.append(t)
        return poses

    def ik(
        self,
        target: np.ndarray,
        seed: Sequence[float],
        iterations: int = 200,
        steps: bool = True,
    ) -> np.ndarray | None:
        """
        Joint positions (joint_names order) putting the tip at `target`, found from `seed` by
        damped least squares (in steps about the base if one solve can't get there, unless
        `steps` is False: quicker where the seed already faces the target), so the arm keeps
        the seed's posture. Each angle is taken the nearest way round to the seed. None if it
        doesn't converge, a joint limit is hit or only the other side of the base reaches it
        (see `ahead`).
        """
        seed = np.array(seed, float)
        q = self._solve(target, seed, iterations)
        if q is not None and not self._same_side(q, seed):
            q = None  # near the base the other side's posture is nearer: swing round instead
        if q is None and steps:  # a long move: one solve overshoots into another posture
            q = self._solve_in_steps(target, seed, iterations)
        if q is None or not self._same_side(q, seed):
            return None
        lower = np.array([j.lower for j in self.joints if j.kind != "fixed"])
        upper = np.array([j.upper for j in self.joints if j.kind != "fixed"])
        q = q - 2 * np.pi * np.round((q - seed) / (2 * np.pi))
        q = np.where(q > upper, q - 2 * np.pi, np.where(q < lower, q + 2 * np.pi, q))
        return q if ((q >= lower) & (q <= upper)).all() else None

    def ahead(self, positions: Sequence[float]) -> float:
        """How far ahead of the first joint's axis (m, along the plane the arm reaches in) the
        tip is, for an arm like the FR5 (first joint vertical, its zero toward +x). A point is
        reached from either side, the base turned up to half a turn between: ahead, or behind
        with the arm folded back over its base, where the tool passes close to the forearm.
        Near the base the two are close in joint angles."""
        tip = self.fk(positions)
        first = float(positions[0])
        return float(tip[0, 3] * math.cos(first) + tip[1, 3] * math.sin(first))

    def _same_side(self, q: np.ndarray, seed: np.ndarray) -> bool:
        a, b = self.ahead(q), self.ahead(seed)
        return a * b >= 0 or min(abs(a), abs(b)) < SIDE_MIN

    def _solve(self, target: np.ndarray, q: np.ndarray, iterations: int) -> np.ndarray | None:
        for _ in range(iterations):
            now = self.fk(q)
            error = _twist(now, target)
            if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-6:
                return q
            jacobian = np.empty((6, len(q)))
            for i in range(len(q)):
                bumped = q.copy()
                bumped[i] += 1e-6
                jacobian[:, i] = _twist(now, self.fk(bumped)) / 1e-6
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
            q = q + np.clip(step, -0.3, 0.3)
        return None

    def _solve_in_steps(
        self, target: np.ndarray, seed: np.ndarray, iterations: int
    ) -> np.ndarray | None:
        """Through tip poses between where the seed puts it and the target, each solved from
        the last: swinging about the base (the first joint) and turning a little at a time."""
        start = self.fk(seed)
        swing = abs(wrap(_bearing(target) - _bearing(start)))
        turn = rotation_angle(start, target)
        distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        steps = math.ceil(max(swing / IK_STEP_SWING, turn / IK_STEP_TURN, distance / IK_STEP))
        steps = min(max(steps, 2), IK_MAX_STEPS)
        q = seed
        for i in range(1, steps + 1):
            q = self._solve(_swung(start, target, i / steps), q, iterations)
            if q is None:
                return None
        return q

    def limit_violation(self, positions: Mapping[str, float], margin: float = 0.0) -> str | None:
        for joint in self.joints:
            if joint.kind in ("revolute", "prismatic"):
                value = float(positions[joint.name])
                if not joint.lower - margin <= value <= joint.upper + margin:
                    return f"{joint.name} = {value:.3f} outside [{joint.lower}, {joint.upper}]"
        return None

    @classmethod
    def from_urdf(cls, urdf: str, base_link: str, tip_link: str) -> "Chain":
        """The chain base_link -> tip_link of a URDF (text). Raises ValueError if there is none."""
        by_child: dict[str, ET.Element] = {}
        for element in ET.fromstring(urdf).iter("joint"):
            child = element.find("child")
            if child is not None and element.find("parent") is not None:
                by_child[child.get("link")] = element
        joints: list[Joint] = []
        links: list[str] = []
        link = tip_link
        while link != base_link:
            element = by_child.get(link)
            if element is None:
                raise ValueError(f"URDF has no chain from '{base_link}' to '{tip_link}'")
            joints.append(_joint(element))
            links.append(link)
            link = element.find("parent").get("link")
        return cls(tuple(reversed(joints)), base_link, tip_link, tuple(reversed(links)))


def _bearing(pose: np.ndarray) -> float:
    return math.atan2(pose[1, 3], pose[0, 3])


def _swung(start: np.ndarray, end: np.ndarray, share: float) -> np.ndarray:
    """The pose `share` (0-1) of the way from start to end, its position swung about the
    base's z axis (bearing, distance from the axis and height each go evenly)."""
    pose = interpolate(start, end, share)
    (x0, y0, z0), (x1, y1, z1) = start[:3, 3], end[:3, 3]
    r0, r1 = math.hypot(x0, y0), math.hypot(x1, y1)
    if min(r0, r1) < 1e-3:  # on the axis there is no bearing: the straight line will do
        return pose
    bearing = _bearing(start) + share * wrap(_bearing(end) - _bearing(start))
    r = r0 + share * (r1 - r0)
    pose[:3, 3] = (r * math.cos(bearing), r * math.sin(bearing), z0 + share * (z1 - z0))
    return pose


def _twist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """6-vector taking pose a to pose b: translation, then rotation vector (base frame)."""
    r = b[:3, :3] @ a[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    rotation = axis / 2 if angle < 1e-6 else axis / (2 * np.sin(angle)) * angle
    return np.concatenate([b[:3, 3] - a[:3, 3], rotation])


def wrist_angle_at(chain: Chain, joints: Mapping[str, float], tip_pose: np.ndarray) -> float | None:
    """
    The angle the chain's last joint would have with the tip at `tip_pose`, for an arm
    like the FR5 (first joint vertical, last joint along the tool) holding the tool
    vertical in its present posture, given by `joints`. None if that doesn't hold.

    With the tool vertical its yaw is the base angle plus or minus the last joint's, and
    the tool axis keeps a fixed sideways offset from the plane the arm reaches in. Both
    constants are read off the present posture, so no inverse kinematics is needed.
    """
    names = chain.joint_names
    first, last = joints[names[0]], joints[names[-1]]
    now = chain.fk(joints)
    if abs(now[2, 2]) < 0.99 or abs(tip_pose[2, 2]) < 0.99 or now[2, 2] * tip_pose[2, 2] < 0:
        return None

    def yaw(pose: np.ndarray) -> float:
        return float(np.arctan2(pose[1, 0], pose[0, 0]))

    def wrapped(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    nudged = chain.fk({**joints, names[-1]: last + 0.01})
    sign = 1.0 if wrapped(yaw(nudged) - yaw(now)) > 0 else -1.0
    constant = yaw(now) - first - sign * last
    offset = -now[0, 3] * np.sin(first) + now[1, 3] * np.cos(first)
    x, y = tip_pose[0, 3], tip_pose[1, 3]
    reach = float(np.hypot(x, y))
    if reach <= abs(offset) + 1e-9:
        return None
    toward, lean = np.arctan2(y, x), np.arcsin(offset / reach)
    # the posture's side of the base (Chain.ik keeps it): ahead, or folded back over it
    ahead = now[0, 3] * np.cos(first) + now[1, 3] * np.sin(first)
    if abs(ahead) >= SIDE_MIN:
        base = toward - lean if ahead > 0 else toward - np.pi + lean
    else:
        base = min((toward - lean, toward - np.pi + lean), key=lambda a: abs(wrapped(a - first)))
    return wrapped(sign * (yaw(tip_pose) - base - constant))


def _numbers(text: str | None, default: Sequence[float]) -> list[float]:
    return [float(v) for v in text.split()] if text else list(default)


def _joint(element: ET.Element) -> Joint:
    origin = element.find("origin")
    xyz = _numbers(origin.get("xyz") if origin is not None else None, (0, 0, 0))
    rpy = _numbers(origin.get("rpy") if origin is not None else None, (0, 0, 0))
    axis = element.find("axis")
    limit = element.find("limit")
    kind = element.get("type", "fixed")
    limited = kind in ("revolute", "prismatic") and limit is not None
    return Joint(
        element.get("name"),
        transform(xyz, rpy),
        tuple(_numbers(axis.get("xyz") if axis is not None else None, (1, 0, 0))),
        kind,
        float(limit.get("lower", -np.inf)) if limited else -np.inf,
        float(limit.get("upper", np.inf)) if limited else np.inf,
    )


def _fr5() -> Chain:
    """The FR5 chain of frcobot_ros2's fairino_description/urdf/fairino5_v6.urdf."""
    half_pi = 1.5708  # as written in the URDF
    rows = [
        ("j1", "shoulder_link", (0, 0, 0), (0, 0, 0), -3.0543, 3.0543),
        ("j2", "upperarm_link", (0, 0, 0.152), (half_pi, 0, 0), -4.6251, 1.4835),
        ("j3", "forearm_link", (-0.425, 0, 0), (0, 0, 0), -2.8274, 2.8274),
        ("j4", "wrist1_link", (-0.39501, 0, 0), (0, 0, 0), -4.6251, 1.4835),
        ("j5", "wrist2_link", (0, 0, 0.1021), (half_pi, 0, 0), -3.0543, 3.0543),
        ("j6", "wrist3_link", (0, 0, 0.102), (-half_pi, 0, 0), -3.0543, 3.0543),
    ]
    joints = tuple(
        Joint(name, transform(xyz, rpy), (0.0, 0.0, 1.0), "revolute", lower, upper)
        for name, _, xyz, rpy, lower, upper in rows
    )
    return Chain(joints, "base_link", "wrist3_link", tuple(row[1] for row in rows))


FR5 = _fr5()
