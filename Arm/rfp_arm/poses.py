"""
Rigid transforms as 4x4 NumPy matrices, meters and radians. T_a_b is the pose of frame b
in frame a: it maps points given in b to a (p_a = T_a_b @ p_b). No ROS.
"""

import math
from collections.abc import Sequence

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation from roll, pitch, yaw about the fixed X, Y, Z axes (the URDF/ROS convention)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_matrix; at pitch = +-90 deg roll is set to 0."""
    r = np.asarray(rotation)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-9:
        return math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], sy), math.atan2(r[1, 0], r[0, 0])
    return 0.0, math.atan2(-r[2, 0], sy), math.atan2(-r[0, 1], r[1, 1])


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation by `angle` about `axis` (Rodrigues)."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def from_parts(rotation: np.ndarray, position: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = np.asarray(rotation)[:3, :3]
    t[:3, 3] = position
    return t


def transform(xyz: Sequence[float] = (0.0, 0.0, 0.0), rpy: Sequence[float] = (0.0, 0.0, 0.0)):
    return from_parts(rpy_matrix(*rpy), xyz)


def invert(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3].T
    return from_parts(r, -r @ t[:3, 3])


def quaternion(t: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation or transform matrix, w >= 0."""
    r = np.asarray(t)[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(r)))  # the largest diagonal term keeps s away from 0
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def from_quaternion(xyz: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Transform from a position and an (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return from_parts(rotation, xyz)


def wrap(angle: float) -> float:
    """The same angle in (-pi, pi]."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def tool_down(xyz: Sequence[float], yaw: float) -> np.ndarray:
    """Pose whose z axis points straight down (-Z) and whose x axis points along `yaw`."""
    return from_parts(rot_z(yaw) @ rot_x(math.pi), xyz)


def yaw_of(t: np.ndarray) -> float:
    """Direction of the frame's x axis in the XY plane, from +X toward +Y."""
    return math.atan2(t[1, 0], t[0, 0])


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation that takes orientation a to orientation b."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, cos)))


def tilt_from_down(t: np.ndarray) -> float:
    """Angle between the frame's z axis and straight down."""
    return math.acos(min(1.0, max(-1.0, -t[2, 2])))
