"""
Drawing the virtual workcell with VTK (pyvista): the virtual D405 renders color and depth
from the camera's pose on the arm, as frames the real vision code reads, and the viewer
shows the whole cell in a 3D window (and can record it).
"""

import functools
import math
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pyvista as pv
from rfp_vision.camera import Frame
from rfp_vision.detector import Detection
from rfp_vision.fish_list import FishList
from rfp_vision.geometry import Intrinsics, pixels_on_plane, to_frame
from rfp_vision.overlay import cup_circles

from rfp_sim.assets import GripperModel, load_gripper
from rfp_sim.settings import gripper_path
from rfp_sim.world import World

TEXTURE_PX_PER_MM = 1.5
HOLE_PITCH_MM = 25.0  # the table is an optical breadboard: M6 holes on a 25 mm grid
HOLE_RADIUS_MM = 3.0
LINK_COLOR = (0.92, 0.92, 0.94)
GRIPPER_COLOR = (0.62, 0.63, 0.66)
TUBE_COLOR = (0.18, 0.4, 0.8)
OPENING_PX = 15  # at 1280 px wide (about 9 mm at the table): barbels go, fins stay
CAMERA_BODY_M = 0.06  # the camera's own view leaves out the gripper this close behind the lens
TUBE_START_Z = 0.17  # tip link z where the vacuum tube leaves the gripper's plate
TABLE_THICKNESS = 0.04
FLOOR_DROP = 0.75  # m from the table top to the floor
TUBE_RADIUS = 0.058  # m from the wrist's axis where the vacuum tube winds round it
WRIST_Z = 0.06  # where it leaves the wrist, in the last link's frame
FOREARM_CLIPS = [(-0.36, 0.0, 0.085), (-0.12, 0.0, 0.085)]  # tube clips, forearm link frame
CLOUD_STEP = 6  # depth pixels: one point in this many each way
VALID = (0.45, 0.85, 0.25)  # what vision found: can be picked
FAILED = (1.0, 0.55, 0.0)  # and can't


def breadboard_texture(size_m: tuple[float, float]) -> pv.Texture:
    """A perforated gray metal plate, like the real table."""
    w, h = (round(s * 1000 * TEXTURE_PX_PER_MM) for s in size_m)
    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(128, 6, (h, w)), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    pitch = HOLE_PITCH_MM * TEXTURE_PX_PER_MM
    radius = max(1, round(HOLE_RADIUS_MM * TEXTURE_PX_PER_MM))
    for y in np.arange(pitch / 2, h, pitch):
        for x in np.arange(pitch / 2, w, pitch):
            cv2.circle(image, (round(x), round(y)), radius, 35, -1, cv2.LINE_AA)
    return pv.Texture(np.dstack([image] * 3))


def _tube(a, b, radius: float) -> pv.PolyData:
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    return pv.Cylinder(center=(a + b) / 2, direction=b - a, radius=radius, height=length)


@functools.lru_cache(maxsize=4)
def cached_gripper(path: Path) -> GripperModel:
    return load_gripper(path)


def without_camera_body(mesh: pv.PolyData, t_tip_camera: np.ndarray) -> pv.PolyData:
    """The gripper less what lies around and behind the lens (the camera's own housing),
    which a camera placed inside it would see instead of the table."""
    local = (mesh.cell_centers().points - t_tip_camera[:3, 3]) @ t_tip_camera[:3, :3]
    around = (np.linalg.norm(local, axis=1) < CAMERA_BODY_M) & (local[:, 2] < 0.01)
    return mesh.extract_cells(~around).extract_surface(algorithm="dataset_surface")


def add_arm(plotter: pv.Plotter, chain, meshes: dict) -> list:
    """One actor per link (base first), in the link's frame: its mesh, or a plain tube to the
    next joint. Place them with chain.link_poses (actor.user_matrix)."""
    actors = []
    for k, name in enumerate(["base_link", *chain.links]):
        if name in meshes:
            mesh = meshes[name]
        else:
            mesh = pv.Sphere(radius=0.05)
            if k < len(chain.joints) and np.linalg.norm(chain.joints[k].origin[:3, 3]) > 1e-6:
                mesh = mesh.merge(_tube((0, 0, 0), chain.joints[k].origin[:3, 3], 0.04))
            if k == 0:
                mesh = mesh.merge(pv.Cylinder(center=(0, 0, 0.03), direction=(0, 0, 1),
                                              radius=0.08, height=0.06))  # fmt: skip
        actors.append(plotter.add_mesh(mesh, color=LINK_COLOR, smooth_shading=True))
    return actors


class Scene:
    """The cell's actors in one plotter; update() moves them to the world's state."""

    def __init__(
        self,
        plotter: pv.Plotter,
        world: World,
        robot_meshes: dict,
        texture: bool = True,
        camera_body: bool = True,
        tube: bool = False,
        room: bool = False,
    ):
        """`camera_body`: draw the D405's housing (not from the camera itself: it is inside).
        `tube`: draw the vacuum tube, coiling round the wrist as it turns. `room`: the
        table's slab and legs and a floor. Tube and room are for the eye only: the camera's
        scene leaves them out rather than invent things in its view."""
        self.plotter = plotter
        self.camera_body = camera_body
        self.world = world
        cell = world.workcell
        zone = cell.work_zone
        size = (zone.x[1] - zone.x[0], zone.y[1] - zone.y[0])
        center = ((zone.x[0] + zone.x[1]) / 2, (zone.y[0] + zone.y[1]) / 2, cell.table_z)
        table = pv.Plane(center=center, direction=(0, 0, 1), i_size=size[0], j_size=size[1])
        if texture:
            self.table = plotter.add_mesh(table, texture=breadboard_texture(size), ambient=0.25)
        else:
            self.table = plotter.add_mesh(table, color=(0.5, 0.5, 0.52))
        if room:
            self._add_room()
        s = world.settings
        self.fish = [
            plotter.add_mesh(f.model.mesh, color=s.fish_color, smooth_shading=True, specular=0.2)
            for f in world.fish
        ]
        self.links = self._add_arm(robot_meshes)
        self.gripper = self._add_gripper()
        self.tube = None
        if tube:
            self.tube = pv.PolyData()
            plotter.add_mesh(self.tube, color=TUBE_COLOR, smooth_shading=True, specular=0.4)
        self.update()

    def _add_room(self) -> None:
        cell = self.world.workcell
        (x0, x1), (y0, y1), top = cell.work_zone.x, cell.work_zone.y, cell.table_z
        floor = top - FLOOR_DROP
        add = self.plotter.add_mesh
        add(pv.Box((x0, x1, y0, y1, top - TABLE_THICKNESS, top - 0.0005)), color=(0.42, 0.43, 0.45))
        legs = [
            pv.Box((x, x + 0.06, y, y + 0.06, floor, top - TABLE_THICKNESS))
            for x in (x0 + 0.04, x1 - 0.10)
            for y in (y0 + 0.04, y1 - 0.10)
        ]
        add(pv.merge(legs), color=(0.3, 0.31, 0.33))
        add(pv.Plane(center=((x0 + x1) / 2, (y0 + y1) / 2, floor), i_size=6, j_size=6),
            color=(0.22, 0.23, 0.25))  # fmt: skip

    def _add_arm(self, meshes: dict) -> list:
        return add_arm(self.plotter, self.world.chain, meshes)

    def _add_gripper(self):
        """The gripper's CAD model (cups, plate, camera, bracket) in the tip link frame."""
        mesh = cached_gripper(gripper_path(self.world.settings)).mesh
        if not self.camera_body:
            mesh = without_camera_body(mesh, self.world.workcell.camera)
        return self.plotter.add_mesh(mesh, color=GRIPPER_COLOR, smooth_shading=True, specular=0.2)

    def tube_centerline(self, poses: list[np.ndarray]) -> np.ndarray:
        """The vacuum tube: from the plate it winds round the wrist by as much as the wrist
        (j6) has turned from where the tube hangs free, then runs to the forearm."""
        world = self.world
        tip, forearm = poses[-1], poses[3]
        twist = float(world.joints[-1]) - world.relaxed_wrist
        f = np.linspace(0.0, 1.0, 24)
        z = TUBE_START_Z + (WRIST_Z - TUBE_START_Z) * f
        angle = math.pi / 2 - twist * f  # leaves the plate on the side away from the camera
        around = np.column_stack([TUBE_RADIUS * np.cos(angle), TUBE_RADIUS * np.sin(angle), z])
        helix = around @ tip[:3, :3].T + tip[:3, 3]
        back = helix[-1] - tip[:3, 2] * 0.07  # off the wrist, back toward the arm
        clips = [forearm[:3, :3] @ p + forearm[:3, 3] for p in FOREARM_CLIPS]
        rest = pv.Spline(np.array([helix[-1], back, *clips]), 40).points
        return np.vstack([helix, rest[1:]])

    def update(self) -> None:
        for actor, fish in zip(self.fish, self.world.fish):
            actor.user_matrix = fish.pose
            actor.visibility = fish.taken_from is None
        poses = self.world.chain.link_poses(self.world.joints)
        for actor, pose in zip(self.links, poses):
            actor.user_matrix = pose
        self.gripper.user_matrix = poses[-1]
        if self.tube is not None:
            line = pv.lines_from_points(self.tube_centerline(poses))
            self.tube.copy_from(line.tube(radius=0.006, n_sides=10))

    def set_visible(self, table: bool, arm: bool, fish: list[bool]) -> None:
        self.table.visibility = table
        for actor in [*self.links, self.gripper]:
            actor.visibility = arm
        for actor, shown, lying in zip(self.fish, fish, self.world.fish):
            actor.visibility = shown and lying.taken_from is None


class VirtualCamera:
    """The D405 on the gripper, rendering 1280 x 720 color and depth (rfp_vision Camera)."""

    has_depth = True

    def __init__(self, world: World, robot_meshes: dict, labels: bool = False):
        self.world = world
        self.name = "virtual D405"
        self.intrinsics = k = world.lens
        self.labels = labels
        self.last_color: np.ndarray | None = None
        self.last_masks: list[np.ndarray] = []  # one per fish in view (labels only)
        size = max(3, round(OPENING_PX * k.width / 1280) | 1)
        self._opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        self.reseed(7)
        self._count = 0
        self.plotter = pv.Plotter(off_screen=True, window_size=(k.width, k.height))
        self.plotter.set_background("black")
        self.scene = Scene(self.plotter, world, robot_meshes, camera_body=False)
        self.plotter.show(auto_close=False)
        vertical = 2 * math.atan(self.intrinsics.cy / self.intrinsics.fy)
        self.plotter.camera.view_angle = math.degrees(vertical)

    def reseed(self, seed: int) -> None:
        """Restart the depth noise: the same seed, the same noise from here on."""
        self._rng = np.random.default_rng(seed)

    def _aim(self) -> None:
        pose = self.world.camera_pose()  # optical frame: z out of the lens, x right, y down
        camera = self.plotter.camera
        camera.position = tuple(pose[:3, 3])
        camera.focal_point = tuple(pose[:3, 3] + pose[:3, 2])
        camera.up = tuple(-pose[:3, 1])
        camera.clipping_range = (0.01, 4.0)

    def read(self, timeout_s: float = 2.0) -> Frame:
        s = self.world.settings
        self.scene.update()
        self._aim()
        self.plotter.render()
        color = np.ascontiguousarray(self.plotter.screenshot(return_img=True)[:, :, ::-1])
        depth = -self.plotter.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False)
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        if s.depth_edge_holes_px > 0:  # one camera of the pair can't see past an edge
            steps = np.hypot(cv2.Sobel(depth, cv2.CV_32F, 1, 0), cv2.Sobel(depth, cv2.CV_32F, 0, 1))
            size = 2 * s.depth_edge_holes_px + 1
            holes = cv2.dilate((steps > 0.04).astype(np.uint8), np.ones((size, size), np.uint8))
            depth[holes > 0] = 0.0  # > 0.04: a step of more than about 1 cm between pixels
        if s.depth_noise_m > 0:
            depth += self._rng.normal(0, s.depth_noise_m, depth.shape).astype(np.float32)
        low, high = s.depth_range_m
        depth[(depth < low) | (depth > high)] = 0.0  # no measurement, as on the D405
        if self.labels:
            self.last_masks = self._masks()
        self._count += 1
        self.last_color = color
        return Frame(color, depth, self.intrinsics, time.monotonic(), time.time(), self._count)

    def _masks(self) -> list[np.ndarray]:
        """Each fish's silhouette in the image, rendered alone and unlit (the ground truth)."""
        fish = self.world.fish
        masks = []
        for actor in self.scene.fish:
            actor.prop.lighting = False
        for i in range(len(fish)):
            self.scene.set_visible(False, False, [j == i for j in range(len(fish))])
            self.plotter.render()
            image = self.plotter.screenshot(return_img=True)
            mask = (image[:, :, 0] > 127).astype(np.uint8)
            # Thin parts (barbels, fin rays) go, as in a segmentation model's outline
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._opening)
            if mask.any():
                masks.append(mask)
        for actor in self.scene.fish:
            actor.prop.lighting = True
        self.scene.set_visible(True, True, [True] * len(fish))
        return masks

    def close(self) -> None:
        self.plotter.close()


class TruthDetector:
    """A detector that knows where the fish are: the virtual camera's silhouettes. For
    testing everything but the trained model."""

    version = "ground truth (simulation)"
    names: ClassVar[dict[int, str]] = {0: "catfish"}

    def __init__(self, camera: VirtualCamera, min_area_share: float = 0.03):
        camera.labels = True
        self.camera = camera
        self.min_area_share = min_area_share

    def warm_up(self, image) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[Detection]:
        found = []
        for mask in self.camera.last_masks:
            if mask.mean() < self.min_area_share:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = max(contours, key=cv2.contourArea)
            polygon = cv2.approxPolyDP(outline, 1.0, True).reshape(-1, 2).astype(float)
            x, y, w, h = cv2.boundingRect(outline)
            found.append(Detection(0, 0.99, (x, y, x + w, y + h), polygon))
        return sorted(
            found, key=lambda d: cv2.contourArea(d.polygon.astype(np.float32)), reverse=True
        )


VIEW_COLOR = (1.0, 0.75, 0.2)
VIEW_REACH = 1.5  # m along the optical axis at most, where a ray misses the table (tilted)
_VIEW_LINES = [2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 0, 4, 5, 1, 2, 3, 4, 1]
_VIEW_FACES = [3, 0, 1, 2, 3, 0, 2, 3, 3, 0, 3, 4, 3, 0, 4, 1, 4, 1, 2, 3, 4]


def view_pyramid(camera: np.ndarray, intrinsics: Intrinsics, table_z: float) -> np.ndarray:
    """The camera's view as a pyramid: its lens (row 0) and where the image's four corner
    rays meet the table (rows 1-4; VIEW_REACH out along the optical axis for a ray that
    doesn't), in the frame `camera` (T_base_camera) is given in."""
    i = intrinsics
    corners = [(-i.cx, -i.cy), (i.width - i.cx, -i.cy), (i.width - i.cx, i.height - i.cy),
               (-i.cx, i.height - i.cy)]  # fmt: skip
    lens = camera[:3, 3]
    points = [lens]
    for u, v in corners:
        ray = camera[:3, :3] @ np.array([u / i.fx, v / i.fy, 1.0])
        reach = VIEW_REACH
        if ray[2] < -1e-6:  # down toward the table
            reach = min(reach, (table_z - lens[2]) / ray[2])
        points.append(lens + ray * max(reach, 0.0))
    return np.array(points)


def add_view_pyramid(plotter) -> tuple:
    """A pyramid (edges, faces) to show a camera's view in; give its points with
    view_pyramid()."""
    edges = pv.PolyData(np.zeros((5, 3)), lines=_VIEW_LINES)
    sides = pv.PolyData(np.zeros((5, 3)), faces=_VIEW_FACES)
    plotter.add_mesh(edges, color=VIEW_COLOR, line_width=2)
    plotter.add_mesh(sides, color=VIEW_COLOR, opacity=0.12, show_edges=False, lighting=False)
    return edges, sides


class VisionLayer:
    """What one look of vision found, in a 3D window: each fish's outline where vision puts it
    (green: can be picked, orange: not), the six cups where the grasp would put them, its
    heading where the head end is known, and the camera's depth as points colored by height.
    The same in the twin beside a run and in the 3D simulation, where it lies over the true
    fish: what vision measured, against what is there."""

    def __init__(self, plotter, table_z: float):
        self.plotter = plotter
        self.table_z = table_z
        self.actors: list = []
        self.cloud = None

    def show(self, fish_list: FishList, frame: Frame) -> None:
        for actor in self.actors:
            self.plotter.remove_actor(actor)
        self.actors = []
        add = self.plotter.add_mesh
        camera = fish_list.camera_pose
        for fish in fish_list.fish:
            table = fish.table_z if fish.table_z is not None else self.table_z
            height = table + max(fish.thickness, 0.0) * 0.5
            outline = pixels_on_plane(frame.intrinsics, camera, fish.mask_polygon, height)
            if not np.isfinite(outline).all():
                continue
            color = VALID if fish.vision_ok else FAILED
            ring = pv.lines_from_points(np.vstack([outline, outline[:1]]))
            self.actors.append(add(ring, color=color, line_width=4))
            if fish.grasp_pose is None:
                continue
            for rim in cup_circles(fish.grasp_pose):
                circle = pv.lines_from_points(np.vstack([rim, rim[:1]]))
                self.actors.append(add(circle, color=color, line_width=2))
            if fish.heading is not None and fish.heading_known:
                start = fish.grasp_pose[:3, 3] + (0.0, 0.0, 0.01)
                direction = (math.cos(fish.heading), math.sin(fish.heading), 0.0)
                arrow = pv.Arrow(
                    start=start, direction=direction, scale=0.15, tip_radius=0.12, shaft_radius=0.04
                )
                self.actors.append(add(arrow, color=color))
        self._show_cloud(frame, camera)

    def _show_cloud(self, frame: Frame, camera: np.ndarray) -> None:
        if self.cloud is not None:
            self.plotter.remove_actor(self.cloud)
            self.cloud = None
        if frame.depth is None:
            return
        depth = frame.depth[::CLOUD_STEP, ::CLOUD_STEP]
        rows, columns = np.nonzero((depth > 0.05) & (depth < 1.5))
        if len(rows) == 0:
            return
        pixels = np.column_stack([columns * CLOUD_STEP, rows * CLOUD_STEP])
        points = to_frame(camera, frame.intrinsics.rays(pixels) * depth[rows, columns][:, None])
        cloud = pv.PolyData(points)
        cloud["height (m)"] = points[:, 2] - self.table_z
        self.cloud = self.plotter.add_mesh(
            cloud, scalars="height (m)", point_size=3, cmap="viridis", clim=(0.0, 0.25),
            show_scalar_bar=False,
        )  # fmt: skip


class Viewer:
    """The cell in a 3D window (mouse: turn, zoom), with the camera's view cone, the zones
    and an inset of the camera's image; optionally recorded to a video."""

    def __init__(self, world: World, robot_meshes: dict, camera: VirtualCamera | None = None,
                 show: bool = True, video: str | None = None):  # fmt: skip
        self.world = world
        self.camera = camera
        self.show = show
        self.status = ""
        self.on_close = None  # called once when the user closes the window
        size = world.settings.window
        self.plotter = pv.Plotter(
            off_screen=not show, window_size=size, title="Fish cell simulation"
        )
        self.plotter.set_background((0.16, 0.17, 0.2), top=(0.32, 0.34, 0.4))
        self.scene = Scene(self.plotter, world, robot_meshes, tube=True, room=True)
        self._add_zones()
        self.cone = add_view_pyramid(self.plotter)
        self.vision = VisionLayer(self.plotter, world.workcell.table_z)
        self.text = self.plotter.add_text("", position="upper_left", font_size=11, color="white")
        cell = world.workcell
        middle = np.array(
            [
                (cell.pick_area.x[0] + cell.placement_zone.x[1]) / 2,
                (cell.pick_area.y[0] + cell.placement_zone.y[1]) / 2,
                cell.table_z,
            ]
        )
        self.plotter.camera_position = [
            tuple(middle + (1.5, -1.3, 1.1)),
            tuple(middle + (0, 0, 0.15)),
            (0, 0, 1),
        ]
        self._exited = False
        if show:
            self.plotter.show(interactive_update=True, auto_close=False)
            # Closing the window (or q) asks VTK to exit; caught here, the window stays until
            # close(), and `closed` tells the program.
            self.plotter.iren.add_observer("ExitEvent", self._on_exit)
        else:
            self.plotter.show(auto_close=False)
        self._writer = None
        self._video = video
        self._last = 0.0
        self.update()

    def _add_zones(self) -> None:
        cell = self.world.workcell
        z = cell.table_z + 0.002
        for rect, color in (
            (cell.pick_area, (0.3, 0.9, 0.4)),
            (cell.placement_zone, (0.3, 0.6, 1.0)),
        ):
            (x0, x1), (y0, y1) = rect.x, rect.y
            corners = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]])
            self.plotter.add_mesh(pv.lines_from_points(corners), color=color, line_width=3)
        for x, y in cell.slots:
            slot = pv.Disc(center=(x, y, z), inner=0.0, outer=0.015, normal=(0, 0, 1))
            self.plotter.add_mesh(slot, color=(0.3, 0.6, 1.0))

    @property
    def closed(self) -> bool:
        return self.show and (self._exited or self.plotter.render_window is None)

    def _on_exit(self, *_) -> None:
        if not self._exited:
            self._exited = True
            if self.on_close is not None:
                self.on_close()

    def set_status(self, text: str) -> None:
        self.status = text

    def show_fish(self, fish_list: FishList, frame: Frame) -> None:
        """What vision found in a look, over the cell as it is (VisionLayer)."""
        if not self.closed:
            self.vision.show(fish_list, frame)

    def update(self) -> None:
        if self.show and not self.closed:
            # Events first: the window's close button destroys the window (WM_CLOSE on
            # Windows) and nothing may be drawn into it after that.
            self.plotter.iren.process_events()
        if self.closed:
            return
        self.scene.update()
        pose = self.world.camera_pose()
        intrinsics = self.camera.intrinsics if self.camera else self.world.lens
        points = view_pyramid(pose, intrinsics, self.world.workcell.table_z)
        for part in self.cone:
            part.points = points
        held = self.world.holding
        low, high = (math.degrees(a) for a in self.world.workcell.wrist_range)
        wrist = math.degrees(self.world.joints[-1])
        twist = wrist - math.degrees(self.world.relaxed_wrist)
        tube = (
            f"wrist (j6) {wrist:+.0f} deg, tube twisted {twist:+.0f} (range {low:.0f}..{high:.0f})"
        )
        vacuum = f"vacuum {'ON' if self.world.vacuum else 'off'}"
        lines = [self.status, vacuum + (f", holding {held.name}" if held else ""), tube]
        self.text.set_text("upper_left", "\n".join(line for line in lines if line))
        self.plotter.render()
        if self._video:
            self._record()

    def _record(self) -> None:
        now = time.monotonic()
        if now - self._last < 1 / 15:  # 15 frames a second of real time is plenty
            return
        self._last = now
        image = np.ascontiguousarray(self.plotter.screenshot(return_img=True)[:, :, ::-1])
        if self.camera is not None and self.camera.last_color is not None:
            h, w = image.shape[:2]
            ch, cw = self.camera.last_color.shape[:2]
            inset = cv2.resize(self.camera.last_color, (w // 3, round(w / 3 * ch / cw)))
            ih, iw = inset.shape[:2]
            image[10 : 10 + ih, w - iw - 10 : w - 10] = inset
            cv2.rectangle(image, (w - iw - 11, 9), (w - 9, 10 + ih), (40, 190, 255), 2)
        if self._writer is None:
            h, w = image.shape[:2]
            self._writer = cv2.VideoWriter(self._video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        self._writer.write(image)

    def wait(self, seconds: float, speed: float) -> None:
        """Let `seconds` of simulated time pass, keeping the window alive."""
        end = time.monotonic() + (seconds / speed if speed > 0 else 0.0)
        while True:
            self.update()
            if time.monotonic() >= end:
                return
            time.sleep(min(0.03, max(0.0, end - time.monotonic())))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if not self.closed:
            self.plotter.close()
