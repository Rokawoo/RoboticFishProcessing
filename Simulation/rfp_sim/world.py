"""
The virtual workcell's state: the arm's joints, the fish on the table and on the cups, and
the little physics the cycle needs. Each suction cup is a ray along the tool axis: a cup
touches when the fish's surface is within its stroke, the fish pushes back like a spring as
the cups compress, and the vacuum holds the fish only once enough cups touch (a firm,
square contact). A held fish moves with the cups once they lift (pressing on, the table
holds it up); let go, it lies down where it is.
"""

from collections.abc import Callable

import numpy as np
import vtk
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.poses import from_parts, invert, rot_z, yaw_of
from rfp_arm.workcell import Workcell
from rfp_vision.geometry import Intrinsics, to_frame

from rfp_sim.assets import FishModel
from rfp_sim.settings import SimSettings

STARTS = np.radians(  # postures tried for the first observation pose (elbow up, tool down)
    [[20, -100, -100, -70, 90, -60], [160, -80, 100, -110, -90, 0], [0, -90, -90, -90, 90, 0],
     [180, -90, 90, -90, -90, 0], [-90, -90, -90, -90, 90, 0], [90, -90, 90, -90, -90, 0]]
)  # fmt: skip
BACK_OFF = 0.05  # m behind a cup face where its ray starts
REACH = 0.05  # m in front of it where the ray ends


class Fish:
    def __init__(self, name: str, model: FishModel, pose: np.ndarray):
        self.name = name
        self.model = model
        self.pose = pose  # 4x4 in {B}
        self.held: np.ndarray | None = None  # T_tcp_fish while on the cups
        self._locator = _locator(model)

    def hit(self, start: np.ndarray, end: np.ndarray) -> float | None:
        """Distance from `start` (in {B}) to where the segment first meets the fish."""
        to_fish = invert(self.pose)
        a, b = (to_frame(to_fish, p)[0] for p in (start, end))
        points = vtk.vtkPoints()
        if (
            not self._locator.IntersectWithLine(a, b, points, None)
            or points.GetNumberOfPoints() == 0
        ):
            return None
        hits = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        return float(np.linalg.norm(hits - a, axis=1).min())

    def outline(self) -> np.ndarray:
        """Top-view outline in {B} (n, 2), as it lies."""
        xy = self.model.outline
        return xy @ self.pose[:2, :2].T + self.pose[:2, 3]


class World:
    def __init__(
        self, workcell: Workcell, settings: SimSettings, model: FishModel, chain: Chain = FR5
    ):
        self.workcell = workcell
        self.settings = settings
        self.model = model
        self.chain = chain
        self.fish: list[Fish] = []
        self.vacuum = False
        self.listeners: list[Callable[[], None]] = []  # called after every change
        self._drop = settings.drop
        self._grabbed_at: np.ndarray | None = None
        self.joints = self.start_joints()

    # -- arm -------------------------------------------------------------------
    def start_joints(self) -> np.ndarray:
        """The arm at the first observation pose, wrist (last joint) as near 0 as it goes."""
        first = self.workcell.observation_poses[0]
        if first.joints is not None:
            return np.array(first.joints, float)
        tip = self.workcell.tip_of_tcp(self.workcell.observation_tcp(first))
        found = [q for q in (self.chain.ik(tip, seed) for seed in STARTS) if q is not None]
        if not found:
            raise ValueError("the arm can't reach the first observation pose")
        return min(found, key=lambda q: abs(q[-1]))

    def tip(self) -> np.ndarray:
        return self.chain.fk(self.joints)

    def tcp(self) -> np.ndarray:
        return self.workcell.tcp_of_tip(self.tip())

    def camera_pose(self) -> np.ndarray:
        return self.workcell.camera_of_tcp(self.tcp())

    def set_joints(self, joints) -> None:
        self.joints = np.array(joints, float)
        tcp = self.tcp()
        held = self.holding
        if held is not None:
            follows = tcp @ held.held
            if follows[2, 3] < self.workcell.table_z - 1e-9:  # pressed on: the table holds it up
                held.held = invert(tcp) @ held.pose
            else:  # lifting or moving: it comes along
                held.pose = follows
            if self._should_drop(tcp):
                self._drop = ""
                self.let_go()
        elif self.vacuum:
            self.try_grab()
        self.changed()

    def changed(self) -> None:
        for listener in self.listeners:
            listener()

    # -- cups and fish -----------------------------------------------------------
    def cups(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(center of the cup face, tool axis) of each cup, in {B}."""
        tcp = self.tcp()
        axis = tcp[:3, 2]
        s = self.settings
        return [
            (tcp[:3, :3] @ (x, y, 0.0) + tcp[:3, 3], axis)
            for x in s.cup_rows
            for y in s.cup_columns
        ]

    def cup_gaps(self) -> list[tuple[float | None, Fish | None]]:
        """Per cup: (m from its face to the fish ahead, < 0 when pressed in; that fish)."""
        result = []
        for center, axis in self.cups():
            start, end = center - axis * BACK_OFF, center + axis * REACH
            best = (None, None)
            for fish in self.fish:
                distance = fish.hit(start, end)
                if distance is not None and (best[0] is None or distance - BACK_OFF < best[0]):
                    best = (distance - BACK_OFF, fish)
            result.append(best)
        return result

    def contact_force(self) -> float:
        """N the fish pushes back on the cups: springs while they compress, stiff beyond."""
        s = self.settings
        force = 0.0
        for gap, _ in self.cup_gaps():
            if gap is not None and gap < 0:
                pressed = -gap
                force += s.cup_stiffness * min(pressed, s.cup_stroke)
                force += 20 * s.cup_stiffness * max(0.0, pressed - s.cup_stroke)
        return force

    @property
    def holding(self) -> Fish | None:
        return next((f for f in self.fish if f.held is not None), None)

    def set_vacuum(self, on: bool) -> None:
        self.vacuum = on
        if on:
            self.try_grab()
        else:
            self.let_go()
        self.changed()

    def try_grab(self) -> Fish | None:
        """With the vacuum on: hold the fish that enough cups touch."""
        touching: dict[int, int] = {}
        for gap, fish in self.cup_gaps():
            if gap is not None and gap <= self.settings.seal_gap:
                touching[id(fish)] = touching.get(id(fish), 0) + 1
        for fish in self.fish:
            if touching.get(id(fish), 0) >= self.settings.seal_cups:
                tcp = self.tcp()
                fish.held = invert(tcp) @ fish.pose
                self._grabbed_at = tcp[:3, 3].copy()
                return fish
        return None

    def let_go(self) -> None:
        fish = self.holding
        if fish is not None:
            fish.held = None
            self.settle(fish)

    def settle(self, fish: Fish) -> None:
        """The fish lies down where it is, flat on the table, keeping its heading."""
        x, y = fish.pose[:2, 3]
        fish.pose = from_parts(rot_z(yaw_of(fish.pose)), (x, y, self.workcell.table_z))

    def _should_drop(self, tcp: np.ndarray) -> bool:
        if not self._drop or self._grabbed_at is None:
            return False
        moved = tcp[:3, 3] - self._grabbed_at
        if self._drop == "lift":
            return moved[2] > 0.08
        return float(np.hypot(moved[0], moved[1])) > 0.15  # "carry"

    # -- laying fish out ---------------------------------------------------------------
    def place_fish(self, count: int, seed: int) -> list[str]:
        """Lay `count` fish on the table where an observation pose sees all of each, apart.
        Returns a warning for each fish that found no place."""
        s = self.settings
        rng = np.random.default_rng(seed)
        intrinsics = Intrinsics.from_fov(s.width, s.height, s.hfov_deg)
        cameras = [self._observation_camera(p) for p in self.workcell.observation_poses]
        warnings = []
        for i in range(count):
            for _ in range(2000):
                camera = cameras[rng.integers(len(cameras))]
                yaw = yaw_of(camera) + rng.choice([0.0, np.pi])
                yaw += np.radians(rng.uniform(-s.max_heading_offset_deg, s.max_heading_offset_deg))
                along = np.array([np.cos(yaw), np.sin(yaw)])
                across = np.array([-along[1], along[0]])
                center = (
                    camera[:2, 3]
                    + along * rng.uniform(-0.05, 0.05)
                    + across * rng.uniform(-0.08, 0.08)
                )
                pose = from_parts(rot_z(yaw), (center[0], center[1], self.workcell.table_z))
                fish = Fish(f"fish {i + 1}", self.model, pose)
                if self._fits(fish, camera, intrinsics):
                    self.fish.append(fish)
                    break
            else:
                warnings.append(f"no free place in view for fish {i + 1}")
        self.changed()
        return warnings

    def _observation_camera(self, pose) -> np.ndarray:
        if pose.camera is not None:
            return pose.camera
        return self.workcell.camera_of_tcp(self.workcell.tcp_of_tip(self.chain.fk(pose.joints)))

    def _fits(self, fish: Fish, camera: np.ndarray, intrinsics: Intrinsics) -> bool:
        if not self.workcell.pick_area.contains(fish.pose[:2, 3], 0.02):
            return False
        outline = fish.outline()
        # Seen from the camera: all of the fish (a sample of its mesh) 25 px inside the frame
        points = np.asarray(fish.model.mesh.points)[:: max(1, fish.model.mesh.n_points // 4000)]
        in_camera = to_frame(invert(camera) @ fish.pose, points)
        pixels = intrinsics.project(in_camera)
        margin = 25
        inside = (
            (in_camera[:, 2] > 0).all()
            and (pixels[:, 0] >= margin).all()
            and (pixels[:, 1] >= margin).all()
            and (pixels[:, 0] <= intrinsics.width - 1 - margin).all()
            and (pixels[:, 1] <= intrinsics.height - 1 - margin).all()
        )
        if not inside:
            return False
        for other in self.fish:  # 3 cm apart at least
            gaps = np.linalg.norm(outline[:, None, :] - other.outline()[None, :, :], axis=2)
            if (
                gaps.min() < 0.03
                or _inside(outline, other.outline())
                or _inside(other.outline(), outline)
            ):
                return False
        return True


_LOCATORS: dict[int, tuple[FishModel, vtk.vtkOBBTree]] = {}


def _locator(model: FishModel) -> vtk.vtkOBBTree:
    """The model's ray-casting tree, built once (a second for a 100k-triangle fish)."""
    if id(model) not in _LOCATORS:
        tree = vtk.vtkOBBTree()
        tree.SetDataSet(model.mesh)
        tree.BuildLocator()
        _LOCATORS[id(model)] = (model, tree)  # the model is kept so its id stays unique
    return _LOCATORS[id(model)][1]


def _inside(points: np.ndarray, polygon: np.ndarray) -> bool:
    """Whether any of `points` lies inside `polygon` (even-odd rule)."""
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    for px, py in points[:: max(1, len(points) // 50)]:
        crosses = ((y > py) != (yn > py)) & (px < (xn - x) * (py - y) / (yn - y + 1e-12) + x)
        if np.count_nonzero(crosses) % 2:
            return True
    return False
