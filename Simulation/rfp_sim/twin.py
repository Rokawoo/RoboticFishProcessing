"""
The arm's twin in a 3D window, beside a run on the real arm (or MoveIt's mock hardware): the
FR5 and the gripper as built, moving with the arm's joint states; the cell's table, zones,
slots and keep-out zones; and what vision makes of each look: the fish's outlines where they
lie (green: can be picked, orange: not), the six cups where the grasp would put them along
the fish's heading, and the camera's depth as a point cloud (colored by height). Nothing here
moves the arm: it only reads. VTK runs in the calling thread (the cycle's), so `update()` is
called from its loop (the console's show hook) and draws at most every DRAW_S.
"""

import math
import time

import numpy as np
import pyvista as pv
from rfp_arm.interfaces import ArmError
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.workcell import Workcell
from rfp_vision.camera import Frame
from rfp_vision.fish_list import FishList
from rfp_vision.geometry import Intrinsics, pixels_on_plane, to_frame
from rfp_vision.overlay import cup_circles

from rfp_sim.render import GRIPPER_COLOR, add_arm, add_view_pyramid, cached_gripper, view_pyramid
from rfp_sim.settings import SimSettings, gripper_path

DRAW_S = 1 / 15  # at most this often, s
CLOUD_STEP = 6  # depth pixels: one point in this many each way
VALID = (0.45, 0.85, 0.25)
FAILED = (1.0, 0.55, 0.0)


class Twin:
    def __init__(
        self,
        workcell: Workcell,
        robot_meshes: dict,
        chain: Chain = FR5,
        settings: SimSettings | None = None,
        show: bool = True,
        title: str = "FR5 twin",
    ):
        self.workcell = workcell
        self.chain = chain
        self.show = show
        self.joints: np.ndarray | None = None
        self.source = None  # an Arm with joints(): followed in update()
        self.status = ""
        self._drawn = 0.0
        self._exited = False
        self._fish_actors: list = []
        self._cloud_actor = None
        self.plotter = pv.Plotter(off_screen=not show, window_size=(1100, 750), title=title)
        self.plotter.set_background((0.16, 0.17, 0.2), top=(0.32, 0.34, 0.4))
        self._add_cell()
        self.links = add_arm(self.plotter, chain, robot_meshes)
        gripper = cached_gripper(gripper_path(settings or SimSettings())).mesh
        self.gripper = self.plotter.add_mesh(gripper, color=GRIPPER_COLOR, smooth_shading=True)
        s = settings or SimSettings()
        self.intrinsics = Intrinsics.from_fov(s.width, s.height, s.hfov_deg)  # the D405's
        self.cone = add_view_pyramid(self.plotter)  # what the camera sees, down to the table
        self.text = self.plotter.add_text("", position="upper_left", font_size=11, color="white")
        pick = workcell.pick_area
        middle = np.array([np.mean(pick.x), np.mean(pick.y), workcell.table_z])
        self.plotter.camera_position = [
            tuple(middle + (1.3, -1.3, 1.1)),
            tuple(middle + (0, 0, 0.15)),
            (0, 0, 1),
        ]
        if show:
            self.plotter.show(interactive_update=True, auto_close=False)
            self.plotter.iren.add_observer("ExitEvent", self._on_exit)
        else:
            self.plotter.show(auto_close=False)

    # -- the scene ----------------------------------------------------------------
    def _add_cell(self) -> None:
        cell, add = self.workcell, self.plotter.add_mesh
        zone = cell.work_zone
        z = cell.table_z
        table = pv.Box((zone.x[0], zone.x[1], zone.y[0], zone.y[1], z - 0.04, z - 0.0005))
        add(table, color=(0.5, 0.5, 0.52))
        for rect, color in (
            (cell.pick_area, (0.3, 0.9, 0.4)),
            (cell.placement_zone, (0.3, 0.6, 1.0)),
        ):
            (x0, x1), (y0, y1) = rect.x, rect.y
            corners = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]])
            add(pv.lines_from_points(corners + (0, 0, 0.002)), color=color, line_width=3)
        for x, y in cell.slots:
            add(pv.Disc(center=(x, y, z + 0.002), inner=0.0, outer=0.015), color=(0.3, 0.6, 1.0))
        for box in cell.exclusion_zones:
            bounds = (*box.x, *box.y, *box.z)
            add(pv.Box(bounds), color=(0.9, 0.2, 0.2), opacity=0.15)

    @property
    def closed(self) -> bool:
        return self.show and (self._exited or self.plotter.render_window is None)

    def _on_exit(self, *_) -> None:
        self._exited = True

    def follow(self, arm) -> None:
        """Take the joints from this arm (anything with joints() by name) in every update."""
        self.source = arm

    def set_joints(self, joints) -> None:
        if isinstance(joints, dict):
            joints = [joints[name] for name in self.chain.joint_names]
        self.joints = np.asarray(joints, float)

    def set_status(self, text: str) -> None:
        self.status = text

    def show_fish(self, fish_list: FishList, frame: Frame) -> None:
        """What one look of vision found: outlines, cups, headings, and the depth as points."""
        for actor in self._fish_actors:
            self.plotter.remove_actor(actor)
        self._fish_actors = []
        add = self.plotter.add_mesh
        camera = fish_list.camera_pose
        for fish in fish_list.fish:
            table = fish.table_z if fish.table_z is not None else self.workcell.table_z
            height = table + max(fish.thickness, 0.0) * 0.5
            outline = pixels_on_plane(frame.intrinsics, camera, fish.mask_polygon, height)
            if not np.isfinite(outline).all():
                continue
            color = VALID if fish.vision_ok else FAILED
            ring = pv.lines_from_points(np.vstack([outline, outline[:1]]))
            self._fish_actors.append(add(ring, color=color, line_width=4))
            if fish.grasp_pose is None:
                continue
            for rim in cup_circles(fish.grasp_pose):
                circle = pv.lines_from_points(np.vstack([rim, rim[:1]]))
                self._fish_actors.append(add(circle, color=color, line_width=2))
            if fish.heading is not None and fish.heading_known:
                start = fish.grasp_pose[:3, 3] + (0.0, 0.0, 0.01)
                direction = (math.cos(fish.heading), math.sin(fish.heading), 0.0)
                arrow = pv.Arrow(
                    start=start, direction=direction, scale=0.15, tip_radius=0.12, shaft_radius=0.04
                )
                self._fish_actors.append(add(arrow, color=color))
        self._show_cloud(frame, camera)

    def _show_cloud(self, frame: Frame, camera: np.ndarray) -> None:
        if self._cloud_actor is not None:
            self.plotter.remove_actor(self._cloud_actor)
            self._cloud_actor = None
        if frame.depth is None:
            return
        depth = frame.depth[::CLOUD_STEP, ::CLOUD_STEP]
        rows, columns = np.nonzero((depth > 0.05) & (depth < 1.5))
        if len(rows) == 0:
            return
        pixels = np.column_stack([columns * CLOUD_STEP, rows * CLOUD_STEP])
        points = to_frame(camera, frame.intrinsics.rays(pixels) * depth[rows, columns][:, None])
        cloud = pv.PolyData(points)
        cloud["height (m)"] = points[:, 2] - self.workcell.table_z
        self._cloud_actor = self.plotter.add_mesh(
            cloud, scalars="height (m)", point_size=3, cmap="viridis", clim=(0.0, 0.25),
            show_scalar_bar=False,
        )  # fmt: skip

    def update(self) -> None:
        """Follow the arm and draw, at most every DRAW_S."""
        if self.show and not self.closed:
            self.plotter.iren.process_events()
        if self.closed:
            return
        now = time.monotonic()
        if now - self._drawn < DRAW_S:
            return
        self._drawn = now
        if self.source is not None:
            try:
                self.set_joints(self.source.joints())
            except ArmError:  # the joint states stopped arriving: the last ones stay shown
                self.status = "no joint states from the arm"
        if self.joints is not None:
            poses = self.chain.link_poses(self.joints)
            for actor, pose in zip(self.links, poses):
                actor.user_matrix = pose
            self.gripper.user_matrix = poses[-1]
            self._aim_cone(poses[-1])
            wrist = math.degrees(self.joints[-1])
            low, high = (math.degrees(a) for a in self.workcell.wrist_range)
            lines = [self.status, f"wrist (j6) {wrist:+.0f} deg (range {low:.0f}..{high:.0f})"]
        else:
            lines = [self.status, "waiting for the arm's joint states"]
        self.text.set_text("upper_left", "\n".join(line for line in lines if line))
        self.plotter.render()

    def _aim_cone(self, tip: np.ndarray) -> None:
        points = view_pyramid(tip @ self.workcell.camera, self.intrinsics, self.workcell.table_z)
        for part in self.cone:
            part.points = points

    def close(self) -> None:
        if not self.closed:
            self.plotter.close()


def open_twin(workcell: Workcell, settings: SimSettings, show: bool = True) -> Twin:
    """A twin with the FR5's meshes (FAIRINO's, as the simulation finds them)."""
    from rfp_sim.assets import find_robot_meshes, robot_link_meshes
    from rfp_sim.settings import MESH_CACHE, SIM_DIR

    folder, _ = find_robot_meshes(
        settings.robot_meshes, SIM_DIR, MESH_CACHE, settings.download_robot_meshes
    )
    meshes = robot_link_meshes(folder, ["base_link", *FR5.links])
    return Twin(workcell, meshes, settings=settings, show=show)
