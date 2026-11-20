"""
The arm's own body as spheres (the FR5's links and the gripper as built, at most `margin`
past them; Arm/configs/collision_model.yaml from Simulation/make_collision_model.py): whether
a pose or a planned path runs the arm into itself, or a pose puts it in a box (keep-out
zones). Checked on every planned path and in the layout's reach, whatever MoveIt's scene
says. NumPy only, no ROS.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rfp_arm.kinematics import Chain

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "configs" / "collision_model.yaml"
GRIPPER = "gripper"  # the gripper's mesh moves with the tip link
BODY_NAMES = {
    "base_link": "base",
    "shoulder_link": "shoulder",
    "upperarm_link": "upper arm",
    "forearm_link": "forearm",
    "wrist1_link": "wrist 1",
    "wrist2_link": "wrist 2",
    "wrist3_link": "wrist 3",
    GRIPPER: "gripper",
}
GROUP = 16  # spheres per group: groups far apart aren't compared sphere by sphere
CHUNK = 4096  # group pairs compared at once (memory)
FAR = 1e6  # m: where the padding of a group lies (it never touches anything)
# Along a path, poses are checked so that no point of the arm moves more than STEP from one
# to the next: a plan's own points are up to a few cm apart (Pilz samples every 0.1 s). A
# joint turning by a (rad) moves no point more than REACH x a: the spheres reach at most
# 1.15 m from any joint's axis (j2's, the arm stretched out).
REACH = 1.2  # m
STEP = 0.01  # m


@dataclass(eq=False)  # compared by identity: one model per file (load_collision_model)
class Body:
    name: str
    link: str  # the link it moves with
    spheres: np.ndarray  # n x 4: x, y, z, radius in the link's frame (m)
    margin: float  # m: how far its spheres reach past it at most
    members: np.ndarray = field(init=False, repr=False)  # groups x GROUP x 4
    bounds: np.ndarray = field(init=False, repr=False)  # groups x 4: around each group

    def __post_init__(self):
        """Neighboring spheres in groups of up to GROUP (split along the longest side), each
        with a sphere around it, padded to GROUP with spheres that touch nothing."""
        groups = _split(np.arange(len(self.spheres)), self.spheres[:, :3])
        self.members = np.full((len(groups), GROUP, 4), [FAR, FAR, FAR, 0.0])
        self.bounds = np.empty((len(groups), 4))
        for g, index in enumerate(groups):
            self.members[g, : len(index)] = self.spheres[index]
            self.bounds[g] = _around(self.spheres[index])


def _split(index: np.ndarray, centers: np.ndarray) -> list[np.ndarray]:
    if len(index) <= GROUP:
        return [index]
    points = centers[index]
    axis = int(np.argmax(np.ptp(points, axis=0)))
    order = index[np.argsort(points[:, axis], kind="stable")]
    half = len(order) // 2
    return _split(order[:half], centers) + _split(order[half:], centers)


def _around(spheres: np.ndarray) -> np.ndarray:
    """A sphere around spheres (x, y, z, radius)."""
    low = (spheres[:, :3] - spheres[:, 3:]).min(axis=0)
    high = (spheres[:, :3] + spheres[:, 3:]).max(axis=0)
    center = (low + high) / 2
    radius = (np.linalg.norm(spheres[:, :3] - center, axis=1) + spheres[:, 3]).max()
    return np.append(center, radius)


@dataclass(eq=False)
class CollisionModel:
    bodies: list[Body]
    pairs: list[tuple[int, int]]  # the bodies checked against each other

    def touching(self, chain: Chain, positions) -> np.ndarray:
        """poses x pairs: which pairs touch in each pose (rows of joint positions in the
        chain's order)."""
        positions = np.atleast_2d(np.asarray(positions, float))
        links = [chain.base_link, *chain.links]
        frames = np.array([chain.link_poses(q) for q in positions])  # poses x links x 4 x 4
        spheres, groups = [], []  # in the base frame: poses x groups (x GROUP) x 3
        for body in self.bodies:
            t = frames[:, links.index(body.link)]
            rotation, origin = t[:, :3, :3], t[:, :3, 3]
            spheres.append(
                np.einsum("pij,gmj->pgmi", rotation, body.members[..., :3]) + origin[:, None, None]
            )
            groups.append(np.einsum("pij,gj->pgi", rotation, body.bounds[:, :3]) + origin[:, None])
        found = np.zeros((len(positions), len(self.pairs)), bool)
        for k, (a, b) in enumerate(self.pairs):
            first, second = self.bodies[a], self.bodies[b]
            reach = first.bounds[:, 3][:, None] + second.bounds[:, 3][None, :]
            gaps = np.linalg.norm(groups[a][:, :, None] - groups[b][:, None], axis=3)
            pose, i, j = np.nonzero(gaps < reach)  # group pairs close enough to look at
            for start in range(0, len(pose), CHUNK):
                p, gi, gj = (v[start : start + CHUNK] for v in (pose, i, j))
                apart = spheres[a][p, gi][:, :, None] - spheres[b][p, gj][:, None]
                limit = first.members[gi, :, 3][:, :, None] + second.members[gj, :, 3][:, None]
                touch = (np.einsum("tmnk,tmnk->tmn", apart, apart) < limit**2).any(axis=(1, 2))
                found[p[touch], k] = True
        return found

    def what(self, pair: tuple[int, int]) -> str:
        a, b = (self.bodies[i].name for i in pair)
        return f"the {b} would hit the {a}"

    def hit(self, chain: Chain, joints: Sequence[float]) -> str | None:
        """What runs into what with the arm at `joints` (the chain's order); None if
        nothing does."""
        found = self.touching(chain, [joints])[0]
        return self.what(self.pairs[int(np.argmax(found))]) if found.any() else None

    def in_box(self, chain: Chain, joints: Sequence[float], box) -> tuple[str, np.ndarray] | None:
        """(a body's name, a point of the box it reaches) where the arm at `joints` would be
        inside `box` (x, y and z ranges in the base frame: a workcell.Box), within the
        margins; None if it stays out."""
        frames = chain.link_poses(np.asarray(joints, float))
        links = [chain.base_link, *chain.links]
        low = np.array([box.x[0], box.y[0], box.z[0]], float)
        high = np.array([box.x[1], box.y[1], box.z[1]], float)
        for body in self.bodies:
            t = frames[links.index(body.link)]
            centers = body.spheres[:, :3] @ t[:3, :3].T + t[:3, 3]
            nearest = np.clip(centers, low, high)  # the box's point nearest each sphere
            gap = nearest - centers
            inside = np.einsum("ij,ij->i", gap, gap) < body.spheres[:, 3] ** 2
            if inside.any():
                return body.name, nearest[int(np.argmax(inside))]
        return None

    def over(self, chain: Chain, joints: Sequence[float], xy, below: float) -> str | None:
        """The body (not on the tool's link) between a camera at height `below` looking
        down and the table at (x, y), the arm at `joints`; None if none is. The camera
        can't see a fish there: what it takes for one there is the arm itself."""
        frames = chain.link_poses(np.asarray(joints, float))
        links = [chain.base_link, *chain.links]
        for body in self.bodies:
            if body.link == chain.tip_link:  # the gripper: the camera's view is clear of it
                continue
            t = frames[links.index(body.link)]
            centers = body.spheres[:, :3] @ t[:3, :3].T + t[:3, 3]
            gap = np.hypot(centers[:, 0] - xy[0], centers[:, 1] - xy[1])
            if ((gap < body.spheres[:, 3]) & (centers[:, 2] < below)).any():
                return body.name
        return None

    def path_problem(self, chain: Chain, positions) -> tuple[int, str] | None:
        """(the index of the given pose by which it happens, what hits what) where a path
        (rows of joint positions in the chain's order, joint-space straight from one to the
        next) runs the arm into itself; None if it doesn't. Checked every STEP of the way,
        not only at the given poses. A pair touching where the path starts (the arm left
        there, within the margins) may go on touching until it comes apart, not touch
        again, and it must come apart."""
        positions = np.atleast_2d(np.asarray(positions, float))
        if not len(positions):
            return None
        turns = np.abs(np.diff(positions, axis=0)).sum(axis=1)
        steps = np.maximum(1, np.ceil(turns * REACH / STEP)).astype(int)
        dense = [positions[:1]]
        for start, end, n in zip(positions[:-1], positions[1:], steps):
            dense.append(start + (end - start) * (np.arange(1, n + 1) / n)[:, None])
        given = np.repeat(np.arange(len(positions)), np.concatenate([[1], steps]))
        found = self.touching(chain, np.vstack(dense))
        for k in np.flatnonzero(found[0]):
            apart = np.flatnonzero(~found[:, k])
            if not len(apart):
                return 0, self.what(self.pairs[k]) + " all the way"
            found[: apart[0], k] = False
        hits = np.flatnonzero(found.any(axis=1))
        if not len(hits):
            return None
        i = int(hits[0])
        return int(given[i]), self.what(self.pairs[int(np.argmax(found[i]))])


@functools.lru_cache(maxsize=4)
def load_collision_model(path: Path = DEFAULT_MODEL) -> CollisionModel:
    """The model in a file made by Simulation/make_collision_model.py (read once)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bodies = [
        Body(
            str(name),
            str(entry["link"]),
            np.array(entry["spheres"], float).reshape(-1, 4),
            float(entry["margin"]),
        )
        for name, entry in data["bodies"].items()
    ]
    index = {body.name: i for i, body in enumerate(bodies)}
    ignore = {frozenset((index[a], index[b])) for a, b in data.get("ignore") or []}
    pairs = [
        (i, j)
        for i in range(len(bodies))
        for j in range(i + 1, len(bodies))
        if frozenset((i, j)) not in ignore
    ]
    return CollisionModel(bodies, pairs)
