"""
Drawing the virtual workcell with VTK (pyvista): the virtual D405 renders color and depth
from the camera's pose on the arm, as frames the real vision code reads, and the viewer
shows the whole cell in a 3D window (and can record it).
"""

import math
import time
from typing import ClassVar

import cv2
import numpy as np
import pyvista as pv
from rfp_arm.poses import from_parts
from rfp_vision.camera import Frame
from rfp_vision.detector import Detection
from rfp_vision.geometry import Intrinsics

from rfp_sim.world import World

TEXTURE_PX_PER_MM = 1.5
HOLE_PITCH_MM = 25.0  # the table is an optical breadboard: M6 holes on a 25 mm grid
HOLE_RADIUS_MM = 3.0
LINK_COLOR = (0.92, 0.92, 0.94)
CUP_COLOR = (0.12, 0.12, 0.12)
METAL_COLOR = (0.7, 0.72, 0.75)
CAMERA_COLOR = (0.2, 0.2, 0.22)
OPENING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


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


class Scene:
    """The cell's actors in one plotter; update() moves them to the world's state."""

    def __init__(
        self,
        plotter: pv.Plotter,
        world: World,
        robot_meshes: dict,
        texture: bool = True,
        camera_body: bool = True,
    ):
        """`camera_body`: draw the D405's housing (not from the camera itself: it is inside)."""
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
        s = world.settings
        self.fish = [
            plotter.add_mesh(f.model.mesh, color=s.fish_color, smooth_shading=True, specular=0.2)
            for f in world.fish
        ]
        self.links = self._add_arm(robot_meshes)
        self.gripper = self._add_gripper()
        self.gripper[-1].visibility = camera_body
        self.update()

    def _add_arm(self, meshes: dict) -> list:
        """One actor per link, in the link's frame: its mesh, or a plain tube to the next joint."""
        chain = self.world.chain
        names = ["base_link", *chain.links]
        actors = []
        for k, name in enumerate(names):
            if name in meshes:
                mesh = meshes[name]
            else:
                mesh = pv.Sphere(radius=0.05)
                if k < len(chain.joints) and np.linalg.norm(chain.joints[k].origin[:3, 3]) > 1e-6:
                    mesh = mesh.merge(_tube((0, 0, 0), chain.joints[k].origin[:3, 3], 0.04))
                if k == 0:
                    mesh = mesh.merge(pv.Cylinder(center=(0, 0, 0.03), direction=(0, 0, 1),
                                                  radius=0.08, height=0.06))  # fmt: skip
            actors.append(self.plotter.add_mesh(mesh, color=LINK_COLOR, smooth_shading=True))
        return actors

    def _add_gripper(self) -> list:
        """Flange stem, plate, six cups and the camera, in the tip link's frame."""
        cell, s = self.world.workcell, self.world.settings
        tcp = cell.tcp
        axis = tcp[:3, 2]
        plate_center = tcp[:3, 3] - axis * (s.cup_length + 0.008)
        cups = [
            _tube(
                tcp[:3, :3] @ (x, y, -s.cup_length) + tcp[:3, 3],
                tcp[:3, :3] @ (x, y, 0.0) + tcp[:3, 3],
                s.cup_radius,
            )
            for x in s.cup_rows
            for y in s.cup_columns
        ]
        span_x = max(s.cup_rows) - min(s.cup_rows) + 2 * s.cup_radius + 0.02
        span_y = max(s.cup_columns) - min(s.cup_columns) + 2 * s.cup_radius + 0.02
        plate = pv.Box((-span_x / 2, span_x / 2, -span_y / 2, span_y / 2, -0.008, 0.008))
        plate = plate.transform(from_parts(tcp[:3, :3], plate_center), inplace=False)
        flange = np.array([0.0, 0.0, 0.10])  # the flange face, 100 mm along the tip link's z
        stem = _tube(flange, plate_center - axis * 0.008, 0.03)
        camera = pv.Box((-0.021, 0.021, -0.021, 0.021, -0.0115, 0.0115))
        camera = camera.transform(cell.camera, inplace=False)
        bracket = _tube(
            cell.camera[:3, 3], flange + (cell.camera[:3, 3] - flange) * [0, 0, 1], 0.008
        )
        add = self.plotter.add_mesh
        return [
            add(pv.merge(cups), color=CUP_COLOR),
            add(plate.merge(stem).merge(bracket), color=METAL_COLOR),
            add(camera, color=CAMERA_COLOR),
        ]

    def update(self) -> None:
        for actor, fish in zip(self.fish, self.world.fish):
            actor.user_matrix = fish.pose
        poses = self.world.chain.link_poses(self.world.joints)
        for actor, pose in zip(self.links, poses):
            actor.user_matrix = pose
        tip = poses[-1]
        for actor in self.gripper:
            actor.user_matrix = tip

    def set_visible(self, table: bool, arm: bool, fish: list[bool]) -> None:
        self.table.visibility = table
        for actor in self.links + self.gripper[:-1]:
            actor.visibility = arm
        self.gripper[-1].visibility = arm and self.camera_body
        for actor, shown in zip(self.fish, fish):
            actor.visibility = shown


class VirtualCamera:
    """The D405 on the gripper, rendering 1280 x 720 color and depth (rfp_vision Camera)."""

    has_depth = True

    def __init__(self, world: World, robot_meshes: dict, labels: bool = False):
        s = world.settings
        self.world = world
        self.name = "virtual D405"
        self.intrinsics = Intrinsics.from_fov(s.width, s.height, s.hfov_deg)
        self.labels = labels
        self.last_color: np.ndarray | None = None
        self.last_masks: list[np.ndarray] = []  # one per fish in view (labels only)
        self._rng = np.random.default_rng(7)
        self._count = 0
        self.plotter = pv.Plotter(off_screen=True, window_size=(s.width, s.height))
        self.plotter.set_background("black")
        self.scene = Scene(self.plotter, world, robot_meshes, camera_body=False)
        self.plotter.show(auto_close=False)
        vertical = 2 * math.atan(self.intrinsics.cy / self.intrinsics.fy)
        self.plotter.camera.view_angle = math.degrees(vertical)

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
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPENING)
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


class Viewer:
    """The cell in a 3D window (mouse: turn, zoom), with the camera's view cone, the zones
    and an inset of the camera's image; optionally recorded to a video."""

    def __init__(self, world: World, robot_meshes: dict, camera: VirtualCamera | None = None,
                 show: bool = True, video: str | None = None):  # fmt: skip
        self.world = world
        self.camera = camera
        self.show = show
        self.status = ""
        size = world.settings.window
        self.plotter = pv.Plotter(
            off_screen=not show, window_size=size, title="Fish cell simulation"
        )
        self.plotter.set_background((0.16, 0.17, 0.2), top=(0.32, 0.34, 0.4))
        self.scene = Scene(self.plotter, world, robot_meshes)
        self._add_zones()
        self.cone = pv.PolyData(
            np.zeros((5, 3)), lines=[2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 0, 4, 5, 1, 2, 3, 4, 1]
        )
        self.plotter.add_mesh(self.cone, color=(1.0, 0.75, 0.2), line_width=2)
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
        self.plotter.show(interactive_update=True, auto_close=False) if show else self.plotter.show(
            auto_close=False
        )
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
        return self.show and self.plotter.render_window is None

    def set_status(self, text: str) -> None:
        self.status = text

    def update(self) -> None:
        if self.closed:
            return
        self.scene.update()
        pose = self.world.camera_pose()
        intrinsics = self.camera.intrinsics if self.camera else Intrinsics.from_fov(1280, 720, 87.0)
        reach = 0.45
        corners = [(-intrinsics.cx, -intrinsics.cy), (intrinsics.cx, -intrinsics.cy),
                   (intrinsics.cx, intrinsics.cy), (-intrinsics.cx, intrinsics.cy)]  # fmt: skip
        points = [pose[:3, 3]] + [
            pose[:3, :3] @ (u / intrinsics.fx * reach, v / intrinsics.fy * reach, reach)
            + pose[:3, 3]
            for u, v in corners
        ]
        self.cone.points = np.array(points)
        held = self.world.holding
        lines = [self.status, f"vacuum {'ON' if self.world.vacuum else 'off'}"
                 + (f", holding {held.name}" if held else "")]  # fmt: skip
        self.text.set_text("upper_left", "\n".join(line for line in lines if line))
        if self.show:
            self.plotter.update()
        else:
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
            inset = cv2.resize(self.camera.last_color, (w // 3, round(w / 3 * 720 / 1280)))
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
        self.plotter.close()
