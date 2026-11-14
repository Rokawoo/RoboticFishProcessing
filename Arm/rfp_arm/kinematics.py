"""
Forward kinematics of a serial chain, from a URDF or the built-in FR5 chain. Used to check
planned trajectories in fixed code (TCP speed, zones) and by the simulated arm. No ROS.
"""

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rfp_arm.poses import axis_rotation, from_parts, transform


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
