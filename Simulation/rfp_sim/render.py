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
TUBE_COLOR = (0.18, 0.4, 0.8)
OPENING_PX = 15  # at 1280 px wide (about 9 mm at the table): barbels go, fins stay
FLANGE = np.array([0.0, 0.0, 0.10])  # the FR5's flange face in its last link's frame
PLATE_THICKNESS = 0.016
TABLE_THICKNESS = 0.04
FLOOR_DROP = 0.75  # m from the table top to the floor
TUBE_RADIUS = 0.058  # m from the wrist's axis where the vacuum tube winds round it
WRIST_Z = 0.06  # where it leaves the wrist, in the last link's frame
FOREARM_CLIPS = [(-0.36, 0.0, 0.085), (-0.12, 0.0, 0.085)]  # tube clips, forearm link frame


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


def bellows_cup(radius: float, length: float) -> pv.PolyData:
    """A 1.5-fold bellows suction cup along +z, from its fitting (z = 0) to its face."""
    up = (0.0, 0.0, 1.0)
    neck = pv.Cylinder(center=(0, 0, 0.15 * length), direction=up, radius=0.45 * radius,
                       height=0.3 * length)  # fmt: skip
    fold = pv.ParametricTorus(ringradius=0.7 * radius, crosssectionradius=0.3 * radius)
    fold = fold.translate((0, 0, 0.42 * length), inplace=False)
    body = pv.Cylinder(center=(0, 0, 0.6 * length), direction=up, radius=0.62 * radius,
                       height=0.35 * length)  # fmt: skip
    lip = pv.Cone(center=(0, 0, 0.88 * length), direction=(0, 0, -1), height=0.24 * length,
                  radius=radius, resolution=24, capping=True)  # fmt: skip
    return pv.merge([neck, fold.triangulate(), body, lip])


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
        self.gripper[-1].visibility = camera_body
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
        """Flange adapter, stem, plate, six bellows cups and the camera on its bracket, in
        the tip link's frame (the FR5's flange face is 100 mm along its z)."""
        cell, s = self.world.workcell, self.world.settings
        tcp = cell.tcp
        axis = tcp[:3, 2]
        plate_center = tcp[:3, 3] - axis * (s.cup_length + PLATE_THICKNESS / 2)
        cup = bellows_cup(s.cup_radius, s.cup_length)
        cups = [
            cup.transform(tcp @ from_parts(np.eye(3), (x, y, -s.cup_length)), inplace=False)
            for x in s.cup_rows
            for y in s.cup_columns
        ]
        span_x = max(s.cup_rows) - min(s.cup_rows) + 2 * s.cup_radius + 0.02
        span_y = max(s.cup_columns) - min(s.cup_columns) + 2 * s.cup_radius + 0.02
        half = PLATE_THICKNESS / 2
        plate = pv.Box((-span_x / 2, span_x / 2, -span_y / 2, span_y / 2, -half, half))
        plate = plate.transform(from_parts(tcp[:3, :3], plate_center), inplace=False)
        adapter = _tube(FLANGE, FLANGE + (0, 0, 0.012), 0.0315)  # ISO 9409-1-50 size disk
        stem = _tube(FLANGE + (0, 0, 0.012), plate_center - axis * half, 0.022)
        camera = pv.Box((-0.021, 0.021, -0.021, 0.021, -0.0115, 0.0115))  # the D405: 42x42x23
        camera = camera.transform(cell.camera, inplace=False)
        on_stem = np.array([0.0, 0.0, cell.camera[2, 3]])
        bracket = _tube(cell.camera[:3, 3], on_stem, 0.008)
        add = self.plotter.add_mesh
        return [
            add(pv.merge(cups), color=CUP_COLOR, smooth_shading=True),
            add(plate.merge(stem).merge(bracket).merge(adapter), color=METAL_COLOR),
            add(camera, color=CAMERA_COLOR),
        ]

    def tube_centerline(self, poses: list[np.ndarray]) -> np.ndarray:
        """The vacuum tube: from the plate it winds round the wrist by as much as the wrist
        (j6) has turned from where the tube hangs free, then runs to the forearm."""
        world, cell, s = self.world, self.world.workcell, self.world.settings
        tip, forearm = poses[-1], poses[3]
        plate_top = cell.tcp[:3, :3] @ (0, 0, -(s.cup_length + PLATE_THICKNESS)) + cell.tcp[:3, 3]
        twist = float(world.joints[-1]) - world.relaxed_wrist
        f = np.linspace(0.0, 1.0, 24)
        z = plate_top[2] + (WRIST_Z - plate_top[2]) * f
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
        poses = self.world.chain.link_poses(self.world.joints)
        for actor, pose in zip(self.links, poses):
            actor.user_matrix = pose
        tip = poses[-1]
        for actor in self.gripper:
            actor.user_matrix = tip
        if self.tube is not None:
            line = pv.lines_from_points(self.tube_centerline(poses))
            self.tube.copy_from(line.tube(radius=0.006, n_sides=10))

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
        size = max(3, round(OPENING_PX * s.width / 1280) | 1)
        self._opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
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

    def update(self) -> None:
        if self.show and not self.closed:
            # Events first: the window's close button destroys the window (WM_CLOSE on
            # Windows) and nothing may be drawn into it after that.
            self.plotter.iren.process_events()
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
        if not self.closed:
            self.plotter.close()
