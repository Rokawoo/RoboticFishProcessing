"""
The arm's twin in a 3D window beside a run on the robot PC: the FR5 and gripper following
the arm's joint states, the cell, and what vision finds (render.VisionLayer). It only reads.
VTK runs in the cycle's thread: `update()` is called from its loop and draws at most every
DRAW_S.
"""

import math
import time

import numpy as np
import pyvista as pv
from rfp_arm.interfaces import ArmError
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.workcell import Box, Workcell
from rfp_vision.camera import Frame
from rfp_vision.fish_list import FishList
from rfp_vision.geometry import Intrinsics
from rfp_vision.settings import VisionSettings

from rfp_sim.render import (
    GRIPPER_COLOR,
    VisionLayer,
    add_arm,
    add_view_pyramid,
    box_mesh,
    cached_gripper,
    outline_mesh,
    view_pyramid,
)
from rfp_sim.settings import SimSettings, gripper_path

DRAW_S = 1 / 15  # at most this often, s


class Twin:
    def __init__(
        self,
        workcell: Workcell,
        robot_meshes: dict,
        chain: Chain = FR5,
        settings: SimSettings | None = None,
        show: bool = True,
        title: str = "FR5 twin",
        lens: Intrinsics | None = None,
    ):
        self.workcell = workcell
        self.chain = chain
        self.show = show
        self.joints: np.ndarray | None = None
        self.source = None  # an Arm with joints(): followed in update()
        self.status = ""
        self._drawn = 0.0
        self._exited = False
        self.plotter = pv.Plotter(off_screen=not show, window_size=(1100, 750), title=title)
        self.plotter.set_background((0.16, 0.17, 0.2), top=(0.32, 0.34, 0.4))
        self._add_cell()
        self.links = add_arm(self.plotter, chain, robot_meshes)
        gripper = cached_gripper(gripper_path(settings or SimSettings())).mesh
        self.gripper = self.plotter.add_mesh(gripper, color=GRIPPER_COLOR, smooth_shading=True)
        # The camera's view: as the settings describe it until a frame says what it is
        self.intrinsics = lens if lens is not None else VisionSettings().lens()
        self.cone = add_view_pyramid(self.plotter)  # what the camera sees, down to the table
        self.vision = VisionLayer(self.plotter, workcell.table_z)
        self.text = self.plotter.add_text("", position="upper_left", font_size=11, color="white")
        middle = np.array([*workcell.pick_area.center, workcell.table_z])
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
        table = Box("table", zone.x, zone.y, (z - 0.04, z - 0.0005), zone.yaw)
        add(box_mesh(table), color=(0.5, 0.5, 0.52))
        for rect, color in (
            (cell.pick_area, (0.3, 0.9, 0.4)),
            (cell.placement_zone, (0.3, 0.6, 1.0)),
        ):
            add(outline_mesh(rect, z + 0.002), color=color, line_width=3)
        for x, y in cell.slots:
            add(pv.Disc(center=(x, y, z + 0.002), inner=0.0, outer=0.015), color=(0.3, 0.6, 1.0))
        for box in cell.exclusion_zones:
            add(box_mesh(box), color=(0.9, 0.2, 0.2), opacity=0.15)

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
        """What one look of vision found (VisionLayer); the view from now on the camera's own."""
        self.intrinsics = frame.intrinsics
        self.vision.show(fish_list, frame)

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


def open_twin(
    workcell: Workcell, settings: SimSettings, lens: Intrinsics | None = None, show: bool = True
) -> Twin:
    """A twin with the FR5's meshes (FAIRINO's, as the simulation finds them)."""
    from rfp_sim.assets import find_robot_meshes, robot_link_meshes
    from rfp_sim.settings import MESH_CACHE, SIM_DIR

    folder, _ = find_robot_meshes(
        settings.robot_meshes, SIM_DIR, MESH_CACHE, settings.download_robot_meshes
    )
    meshes = robot_link_meshes(folder, ["base_link", *FR5.links])
    return Twin(workcell, meshes, settings=settings, lens=lens, show=show)
