"""
What a carried fish passes over (plan_grasp's low carry): its footprint swept along the
straight carry, against the footprints of what lies on the table. Rectangles in {B} seen
from above, no heights: anything under the way and the fish is carried as high as before.
No ROS.
"""

import math
from dataclasses import dataclass

import numpy as np

from rfp_arm.poses import wrap, yaw_of

STEP = 0.01  # m the carried fish's farthest point moves between two places checked


@dataclass(frozen=True)
class Footprint:
    """A rectangle on the table in {B}: something lying there, seen from above."""

    name: str
    center: tuple[float, float]
    yaw: float  # rad: the direction of its length
    length: float
    width: float

    @classmethod
    def around(cls, name: str, points, yaw: float) -> "Footprint":
        """The rectangle along `yaw` holding every one of the (n, 2) points (an outline)."""
        along, across = _axes(yaw)
        points = np.asarray(points, float).reshape(-1, 2)
        a, b = points @ along, points @ across
        center = along * (a.max() + a.min()) / 2 + across * (b.max() + b.min()) / 2
        return cls(name, (float(center[0]), float(center[1])), float(yaw),
                   float(a.max() - a.min()), float(b.max() - b.min()))  # fmt: skip

    @classmethod
    def box(cls, name: str, x: tuple[float, float], y: tuple[float, float]) -> "Footprint":
        return cls(name, ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), 0.0, x[1] - x[0], y[1] - y[0])

    def corners(self) -> np.ndarray:
        along, across = _axes(self.yaw)
        return np.array(
            [
                np.asarray(self.center) + along * a * self.length / 2 + across * b * self.width / 2
                for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))
            ]
        )

    def holds(self, points, grow: float = 0.0) -> np.ndarray:
        """Which of the (n, 2) points lie inside, the rectangle grown by `grow` all round."""
        along, across = _axes(self.yaw)
        offset = np.asarray(points, float).reshape(-1, 2) - self.center
        return (np.abs(offset @ along) <= self.length / 2 + grow) & (
            np.abs(offset @ across) <= self.width / 2 + grow
        )

    def moved(self, start: np.ndarray, end: np.ndarray, name: str | None = None) -> "Footprint":
        """Where it lies once carried on the cups from TCP pose `start` to `end`, the tool
        turning about the vertical."""
        here = _rotate(np.asarray(self.center) - start[:2, 3], -yaw_of(start))
        center = end[:2, 3] + _rotate(here, yaw_of(end))
        turn = yaw_of(end) - yaw_of(start)
        return Footprint(name or self.name, (float(center[0]), float(center[1])),
                         float(wrap(self.yaw + turn)), self.length, self.width)  # fmt: skip


def in_the_way(
    carried: Footprint,
    start: np.ndarray,
    end: np.ndarray,
    things: list[Footprint],
    margin: float,
) -> str | None:
    """
    The name of the first of `things` that the carried fish comes within `margin` of, or
    None. `carried` is where the fish lies with the TCP at pose `start`; it rides along as
    the TCP goes straight to `end`, turning the short way about the vertical (Pilz LIN).
    """
    x0, x1 = start[:2, 3], end[:2, 3]
    yaw0 = yaw_of(start)
    turn = wrap(yaw_of(end) - yaw0)
    reach = float(np.linalg.norm(carried.corners() - x0, axis=1).max())
    moves = float(np.linalg.norm(x1 - x0)) + reach * abs(turn)
    s = np.linspace(0.0, 1.0, max(2, math.ceil(moves / STEP) + 1))
    yaws = yaw0 + s * turn
    here = _rotate(np.asarray(carried.center) - x0, -yaw0)
    centers = x0 + s[:, None] * (x1 - x0) + np.column_stack(
        [here[0] * np.cos(yaws) - here[1] * np.sin(yaws),
         here[0] * np.sin(yaws) + here[1] * np.cos(yaws)]
    )  # fmt: skip
    half = (carried.length / 2 + margin, carried.width / 2 + margin)
    for thing in things:
        if _overlap(centers, carried.yaw - yaw0 + yaws, half, thing).any():
            return thing.name
    return None


def _overlap(centers: np.ndarray, yaws: np.ndarray, half, thing: Footprint) -> np.ndarray:
    """For rectangles at (n, 2) centers and (n,) yaws with half sizes `half`: which overlap
    `thing` (separating axis test)."""
    ua = np.column_stack([np.cos(yaws), np.sin(yaws)])
    va = np.column_stack([-np.sin(yaws), np.cos(yaws)])
    ub, vb = (np.broadcast_to(axis, ua.shape) for axis in _axes(thing.yaw))
    between = np.asarray(thing.center) - centers
    apart = np.zeros(len(centers), bool)
    for axis in (ua, va, ub, vb):
        reach = (
            half[0] * np.abs(np.sum(ua * axis, axis=1))
            + half[1] * np.abs(np.sum(va * axis, axis=1))
            + thing.length / 2 * np.abs(np.sum(ub * axis, axis=1))
            + thing.width / 2 * np.abs(np.sum(vb * axis, axis=1))
        )
        apart |= np.abs(np.sum(between * axis, axis=1)) > reach
    return ~apart


def _axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])


def _rotate(xy, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])
