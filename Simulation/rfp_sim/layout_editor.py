"""
Lay out the cell by hand in a 3D window (pick_place.py --mode setup): the table, pick area,
placement zone and slots, exclusion zones and observation poses, around the FR5 and the
gripper as built. Nothing moves.

Drag a box's handles to size it (its middle handle moves it), a ball to move a slot or an
observation pose; the buttons turn the whole layout a quarter turn about the arm, turn the
placement heading, add or remove a slot, undo, save. After every change the layout is checked
(rfp_arm.layout.check_layout): red is what can't work (the arm can't reach it, a placed fish
off the table), yellow what may not (a fish that can't be lifted there, part of the pick area
no camera view sees). Save writes the workcell file, and the control panel where it overrides
the file, changing only the layout's lines (a .bak copy of each is kept).
"""

import copy
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from rfp_arm.kinematics import FR5
from rfp_arm.layout import (
    CameraView,
    FishSize,
    Layout,
    LayoutError,
    check_layout,
    layout_changed,
    lift_height,
    load_layout_data,
    observation_camera,
    reach_joints,
    reach_radius,
    save_layout,
    turned,
    view_footprint,
    with_layout,
)
from rfp_arm.workcell import DEFAULT_FILE, PANEL_FILE, Workcell, workcell_from_mapping

from rfp_sim.assets import find_robot_meshes, robot_link_meshes
from rfp_sim.render import GRIPPER_COLOR, add_arm, cached_gripper
from rfp_sim.settings import MESH_CACHE, SIM_DIR, gripper_path
from rfp_sim.settings import load_settings as load_sim_settings

PICK_COLOR = (0.3, 0.9, 0.4)
PLACE_COLOR = (0.3, 0.6, 1.0)
TABLE_COLOR = (0.75, 0.75, 0.78)
EXCLUSION_COLOR = (1.0, 0.35, 0.3)
POSE_COLOR = (1.0, 0.75, 0.2)
ERROR_COLOR = (1.0, 0.3, 0.3)
WARNING_COLOR = (1.0, 0.85, 0.3)
FLAT = 0.01  # m: the pick area and placement zone are drawn as slabs this thick
BUTTON = 26  # px


class LayoutEditor:
    """The cell's layout in a 3D window, edited by hand and checked as it changes."""

    def __init__(
        self,
        workcell_file: Path = DEFAULT_FILE,
        panel_file: Path | None = PANEL_FILE,
        off_screen: bool = False,
        log=print,
    ):
        from rfp_vision.settings import load_settings as load_vision_settings

        self.workcell_file, self.panel_file, self.log = Path(workcell_file), panel_file, log
        self.data, self.saved = load_layout_data(self.workcell_file, panel_file)
        self.layout = copy.deepcopy(self.saved)
        vision = load_vision_settings(panel=panel_file)
        size = vision.fish
        longest = size.length_range_m[1]
        self.fish = FishSize(
            longest,
            size.width_range_m[1],
            size.thickness_range_m[1],
            size.balance_ahead_share * longest,  # placed, the tool is that far past the slot
        )
        self.view = CameraView(vision.hfov_deg, vision.width, vision.height)
        sim = load_sim_settings(panel=panel_file)
        folder, _ = find_robot_meshes(
            sim.robot_meshes, SIM_DIR, MESH_CACHE, sim.download_robot_meshes
        )
        self.meshes = robot_link_meshes(folder, ["base_link", *FR5.links])
        self.gripper = cached_gripper(gripper_path(sim))
        self.confirm_save = False
        self.building = False
        self.message = ""
        self.findings = []

        self.plotter = pv.Plotter(
            off_screen=off_screen,
            window_size=sim.window,
            title="Fish cell layout (pick_place.py --mode setup)",
        )
        self.plotter.set_background((0.16, 0.17, 0.2), top=(0.32, 0.34, 0.4))
        self.arm = add_arm(self.plotter, FR5, self.meshes)
        self.tool = self.plotter.add_mesh(
            self.gripper.mesh, color=GRIPPER_COLOR, smooth_shading=True
        )
        self._add_reach()
        self._add_buttons()
        self.rebuild()
        self.plotter.camera_position = [(-1.1, -1.9, 2.0), (0.35, 0.1, 0.0), (0, 0, 1)]

    # -- the cell as it is now -------------------------------------------------------
    @property
    def cell(self) -> Workcell:
        return workcell_from_mapping(with_layout(self.data, self.layout))

    @property
    def table_z(self) -> float:
        return float(self.data["table"]["z"])

    def _add_reach(self) -> None:
        """Rings on the table: how far the cups reach straight down at a fish's top, and at
        the height fish are lifted and carried at (the same in every direction)."""
        cell = self.cell
        lift = lift_height(cell, self.fish.length)
        rings = (
            (
                reach_radius(cell, self.table_z + self.fish.thickness),
                (0.6, 0.6, 0.65),
                "reach at a fish's top",
            ),
            (
                reach_radius(cell, lift),
                (0.95, 0.55, 0.25),
                f"reach when carrying ({lift - self.table_z:.2f} m up)",
            ),
        )
        z = self.table_z + 0.003
        for radius, color, label in rings:
            angles = np.linspace(0, 2 * math.pi, 181)
            ring = np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z)]
            )
            self.plotter.add_mesh(pv.lines_from_points(ring), color=color, line_width=2)
            self.plotter.add_point_labels(
                [(radius * 0.7071, -radius * 0.7071, z)],
                [label],
                font_size=11,
                text_color=color,
                shape_opacity=0.0,
                show_points=False,
                always_visible=True,
            )
        self.plotter.add_arrows(
            np.zeros((2, 3)), np.array([[0.25, 0, 0], [0, 0.25, 0]]), color="white"
        )
        self.plotter.add_point_labels(
            [(0.28, 0, z), (0, 0.28, z)],
            ["+X", "+Y"],
            font_size=12,
            text_color="white",
            shape_opacity=0.0,
            show_points=False,
            always_visible=True,
        )

    def _add_buttons(self) -> None:
        buttons = [
            ("Turn the layout 90 deg left (counter-clockwise)", lambda: self.turn(1)),
            ("Turn the layout 90 deg right", lambda: self.turn(-1)),
            ("Placement heading +90 deg", lambda: self.turn_heading(90)),
            ("Placement heading -90 deg", lambda: self.turn_heading(-90)),
            ("Placement heading +15 deg", lambda: self.turn_heading(15)),
            ("Placement heading -15 deg", lambda: self.turn_heading(-15)),
            ("Add a slot", self.add_slot),
            ("Remove the last slot", self.remove_slot),
            ("Undo all changes (as saved)", self.undo),
            ("Save", self.save),
        ]
        top = self.plotter.window_size[1] - 40
        for i, (label, action) in enumerate(buttons):
            y = top - i * (BUTTON + 8)
            color = (0.3, 0.75, 0.4) if label == "Save" else (0.85, 0.85, 0.9)
            self.plotter.add_checkbox_button_widget(
                lambda _state, action=action: self._do(action),
                value=False,
                position=(10, y),
                size=BUTTON,
                border_size=2,
                color_on=color,
                color_off=color,
            )
            self.plotter.add_text(label, position=(BUTTON + 18, y + 5), font_size=9, color="white")

    def _do(self, action) -> None:
        action()
        if action != self.save:
            self.confirm_save = False

    # -- the layout's parts ----------------------------------------------------------
    def rebuild(self) -> None:
        """Every widget from the layout (after a change of what there is to drag)."""
        self.building = True  # widgets call back once as they are made: nothing moved
        self.plotter.clear_box_widgets()
        self.plotter.clear_sphere_widgets()
        z0 = self.table_z
        lay = self.layout
        self._box("work", lay.work, (z0 - FLAT, z0), TABLE_COLOR)  # its height stays as set
        self._box("pick_area", lay.pick_area, (z0, z0 + FLAT), PICK_COLOR)
        self._box("placement_zone", lay.placement_zone, (z0, z0 + FLAT), PLACE_COLOR)
        for i, zone in enumerate(lay.exclusions):
            self._box(("exclusions", i), zone, tuple(zone["z"]), EXCLUSION_COLOR)
        for i, slot in enumerate(lay.slots):
            self.plotter.add_sphere_widget(
                lambda center, i=i: self._moved_slot(i, center),
                center=(slot[0], slot[1], z0 + 0.02),
                radius=0.025,
                color=PLACE_COLOR,
                interaction_event="end",
            )
        height = float(self.data["observation"]["height"])
        for i, pose in enumerate(lay.poses):
            if "camera_xy" in pose:
                x, y = pose["camera_xy"]
                self.plotter.add_sphere_widget(
                    lambda center, i=i: self._moved_pose(i, center),
                    center=(x, y, z0 + height),
                    radius=0.03,
                    color=POSE_COLOR,
                    interaction_event="end",
                )
        self.building = False
        self.refresh()

    def _box(self, part, rect: dict, z: tuple, color) -> None:
        bounds = (rect["x"][0], rect["x"][1], rect["y"][0], rect["y"][1], z[0], z[1])
        self.plotter.add_box_widget(
            lambda box, part=part: self._moved_box(part, box.bounds),
            bounds=bounds,
            factor=1.0,
            rotation_enabled=False,
            color=color,
            interaction_event="end",
        )

    def _moved_box(self, part, bounds) -> None:
        if self.building:
            return
        x = sorted((round(bounds[0], 3), round(bounds[1], 3)))
        y = sorted((round(bounds[2], 3), round(bounds[3], 3)))
        if isinstance(part, tuple):  # an exclusion zone: x, y and z
            zone = self.layout.exclusions[part[1]]
            zone.update(x=x, y=y, z=sorted((round(bounds[4], 3), round(bounds[5], 3))))
        else:  # the table and the flat zones: x and y only
            getattr(self.layout, part).update(x=x, y=y)
        self.confirm_save = False
        self.refresh()

    def _moved_slot(self, i: int, center) -> None:
        if self.building:
            return
        self.layout.slots[i] = [round(center[0], 3), round(center[1], 3)]
        self.confirm_save = False
        self.refresh()

    def _moved_pose(self, i: int, center) -> None:
        if self.building:
            return
        self.layout.poses[i]["camera_xy"] = [round(center[0], 3), round(center[1], 3)]
        self.confirm_save = False
        self.refresh()

    # -- buttons ---------------------------------------------------------------------
    def turn(self, quarter_turns: int) -> None:
        self.layout = turned(self.layout, quarter_turns)
        self.message = f"turned the layout {90 * quarter_turns:+d} deg about the arm"
        self.rebuild()

    def turn_heading(self, degrees: float) -> None:
        heading = (self.layout.heading_deg + degrees + 180.0) % 360.0 - 180.0
        self.layout.heading_deg = 180.0 if heading == -180.0 else heading
        self.message = f"placement heading {self.layout.heading_deg:.0f} deg"
        self.refresh()

    def add_slot(self) -> None:
        """A slot a fish's width past the last one, across the heading (inside the zone)."""
        zone = self.layout.placement_zone
        if self.layout.slots:
            heading = math.radians(self.layout.heading_deg)
            across = np.array([-math.sin(heading), math.cos(heading)])
            if len(self.layout.slots) > 1:
                step = np.subtract(self.layout.slots[-1], self.layout.slots[-2])
            else:
                step = across * (self.fish.width + 0.03)
            new = np.add(self.layout.slots[-1], step)
        else:
            new = np.array([sum(zone["x"]) / 2, sum(zone["y"]) / 2])
        self.layout.slots.append([round(float(new[0]), 3), round(float(new[1]), 3)])
        self.message = f"slot {len(self.layout.slots)} added: drag it where it goes"
        self.rebuild()

    def remove_slot(self) -> None:
        if len(self.layout.slots) > 1:
            self.layout.slots.pop()
            self.message = f"{len(self.layout.slots)} slot(s) left"
            self.rebuild()

    def undo(self) -> None:
        self.layout = copy.deepcopy(self.saved)
        self.message = "back to the layout as saved"
        self.rebuild()

    def save(self) -> list[str]:
        """Write the layout. With errors left, the first click only asks for a second one."""
        errors = [f for f in self.findings if f.error]
        if errors and not self.confirm_save:
            self.confirm_save = True
            self.message = f"{len(errors)} error(s) left (red): click Save again to save anyway"
            self.refresh()
            return []
        self.confirm_save = False
        try:
            written = save_layout(self.layout, self.workcell_file, self.panel_file)
        except LayoutError as e:
            self.message = f"not saved: {e}"
            self.log(f"❌ Not saved: {e}")
            self.refresh()
            return []
        self.data, self.saved = load_layout_data(self.workcell_file, self.panel_file)
        self.layout = copy.deepcopy(self.saved)
        if written:
            self.message = "saved (.bak copies kept): " + "; ".join(written)
            self.log("✅ Saved: " + "\n          ".join(written))
            self.log("Next: python main.py --mode sim3d   (the cycle in this layout)")
        else:
            self.message = "nothing changed: nothing to save"
        self.rebuild()
        return written

    # -- what the layout means ---------------------------------------------------------
    def refresh(self) -> None:
        """Check the layout and draw everything that follows from it."""
        cell = self.cell
        self.findings = check_layout(cell, self.fish, self.view)
        z0 = self.table_z + 0.004
        add = self.plotter.add_mesh
        work = cell.work_zone
        table = pv.Plane(
            center=((work.x[0] + work.x[1]) / 2, (work.y[0] + work.y[1]) / 2, z0 - 0.003),
            i_size=work.x[1] - work.x[0],
            j_size=work.y[1] - work.y[0],
        )
        add(table, color=(0.42, 0.43, 0.45), name="table", opacity=0.9)
        add(_grid(work, z0 - 0.002), color=(0.55, 0.56, 0.6), line_width=1, name="grid")
        for name, rect, color in (
            ("pick", cell.pick_area, PICK_COLOR),
            ("place", cell.placement_zone, PLACE_COLOR),
        ):
            add(_outline(rect.x, rect.y, z0), color=color, line_width=3, name=name)
        half = self.fish.length / 2
        pick = cell.pick_area
        reach = ((pick.x[0] - half, pick.x[1] + half), (pick.y[0] - half, pick.y[1] + half))
        add(_outline(*reach, z0), color=PICK_COLOR, line_width=1, opacity=0.5, name="fish reach")
        add(self._placed_fish(cell, z0), color=PLACE_COLOR, line_width=2, name="placed fish")
        add(self._views(cell, z0), color=POSE_COLOR, line_width=2, name="views")
        markers = [
            (f.where, ERROR_COLOR if f.error else WARNING_COLOR)
            for f in self.findings
            if f.where is not None
        ]
        for k in range(len(markers), 12):
            self.plotter.remove_actor(f"finding {k}")
        for k, (where, color) in enumerate(markers[:12]):
            disc = pv.Disc(center=(where[0], where[1], z0 + 0.002), inner=0.03, outer=0.045)
            add(disc, color=color, name=f"finding {k}")
        self._labels(cell, z0)
        self._pose_arm(cell)
        self._text(cell)
        self.plotter.render()

    def _placed_fish(self, cell: Workcell, z: float) -> pv.PolyData:
        """Each slot's fish, the size the checks assume, head along the placement heading."""
        heading = cell.placement_heading
        along = np.array([math.cos(heading), math.sin(heading)])
        across = np.array([-along[1], along[0]])
        lines = []
        for slot in cell.slots:
            c = np.asarray(slot)
            l2, w2 = self.fish.length / 2, self.fish.width / 2
            corners = [
                c - l2 * along - w2 * across,
                c + (l2 - w2) * along - w2 * across,
                c + l2 * along,
                c + (l2 - w2) * along + w2 * across,
                c - l2 * along + w2 * across,
            ]
            points = [(*p, z) for p in corners] + [(*corners[0], z)]
            lines.append(pv.lines_from_points(np.array(points)))
        return pv.merge(lines) if lines else pv.PolyData()

    def _views(self, cell: Workcell, z: float) -> pv.PolyData:
        """What each observation pose sees of the table (at half a fish's height)."""
        lines = []
        plane = self.table_z + self.fish.thickness / 2
        for pose in cell.observation_poses:
            try:
                corners = view_footprint(observation_camera(cell, pose), self.view, plane)
            except ValueError:
                continue
            points = [(*p, z) for p in corners] + [(*corners[0], z)]
            lines.append(pv.lines_from_points(np.array(points)))
        return pv.merge(lines) if lines else pv.PolyData()

    def _labels(self, cell: Workcell, z: float) -> None:
        points, labels = [], []
        pick, place = cell.pick_area, cell.placement_zone
        points.append((pick.x[0], pick.y[0], z))
        labels.append("pick area")
        points.append((place.x[0], place.y[0], z))
        labels.append("placement zone")
        for k, slot in enumerate(cell.slots, 1):
            points.append((slot[0], slot[1], z + 0.03))
            labels.append(f"slot {k}")
        for pose in cell.observation_poses:
            camera = observation_camera(cell, pose)
            points.append(tuple(camera[:3, 3]))
            labels.append(f"view '{pose.name}'")
        for zone in cell.exclusion_zones:
            points.append((zone.x[0], zone.y[0], zone.z[1]))
            labels.append(f"keep out: {zone.name}")
        self.plotter.add_point_labels(
            points,
            labels,
            font_size=11,
            text_color="white",
            shape_opacity=0.35,
            show_points=False,
            always_visible=True,
            name="labels",
        )

    def _pose_arm(self, cell: Workcell) -> None:
        """The arm at the first observation pose (if it reaches it), as sim3d starts."""
        if not cell.observation_poses:
            return
        first = cell.observation_poses[0]
        if first.joints is not None:
            joints = np.array(first.joints)
        else:
            joints = reach_joints(cell, cell.observation_tcp(first))
            if joints is None:
                return
        poses = FR5.link_poses(joints)
        for actor, pose in zip(self.arm, poses):
            actor.user_matrix = pose
        self.tool.user_matrix = poses[-1]

    def _text(self, cell: Workcell) -> None:
        errors = [f.text for f in self.findings if f.error]
        warnings = [f.text for f in self.findings if not f.error]
        status = (
            "no problems found"
            if not self.findings
            else f"{len(errors)} error(s), {len(warnings)} warning(s)"
        )
        changed = " (changed, not saved)" if layout_changed(self.layout, self.saved) else ""
        head = f"Layout{changed}: {status}" + (f"\n{self.message}" if self.message else "")
        self.plotter.add_text(
            head, position="upper_right", font_size=11, color="white", name="status"
        )
        self.plotter.add_text(
            "\n".join("error: " + t for t in errors),
            position=(10, 210),
            font_size=9,
            color=ERROR_COLOR,
            name="errors",
        )
        self.plotter.add_text(
            "\n".join("check: " + t for t in warnings),
            position=(10, 120),
            font_size=9,
            color=WARNING_COLOR,
            name="warnings",
        )
        mm = _mm_pair
        lines = [
            f"table x {mm(cell.work_zone.x)}, y {mm(cell.work_zone.y)} mm",
            f"pick area x {mm(cell.pick_area.x)}, y {mm(cell.pick_area.y)} mm",
            (
                f"placement zone x {mm(cell.placement_zone.x)}, y {mm(cell.placement_zone.y)} mm, "
                f"heading {math.degrees(cell.placement_heading):.0f} deg"
            ),
            "slots " + ", ".join(f"({x * 1000:.0f}, {y * 1000:.0f})" for x, y in cell.slots),
            "views "
            + ", ".join(
                f"'{p['name']}' ({p['camera_xy'][0] * 1000:.0f}, {p['camera_xy'][1] * 1000:.0f})"
                if "camera_xy" in p
                else f"'{p['name']}' (taught joints)"
                for p in self.layout.poses
            ),
            (
                "Drag box handles to size (middle: move), balls to move. Fish drawn: the "
                "largest vision passes."
            ),
        ]
        self.plotter.add_text(
            "\n".join(lines),
            position="lower_left",
            font_size=9,
            color=(0.85, 0.85, 0.9),
            name="values",
        )

    def show(self) -> Layout:
        self.plotter.show()
        if layout_changed(self.layout, self.saved):
            self.log("⚠️  Closed with changes not saved.")
        return self.layout


def _outline(x, y, z: float) -> pv.PolyData:
    corners = [(x[0], y[0], z), (x[1], y[0], z), (x[1], y[1], z), (x[0], y[1], z), (x[0], y[0], z)]
    return pv.lines_from_points(np.array(corners))


def _grid(zone, z: float, step: float = 0.1) -> pv.PolyData:
    """Lines every `step` m over the table, at multiples of it."""
    lines = []
    for x in np.arange(math.ceil(zone.x[0] / step) * step, zone.x[1] + 1e-9, step):
        lines.append(pv.Line((x, zone.y[0], z), (x, zone.y[1], z)))
    for y in np.arange(math.ceil(zone.y[0] / step) * step, zone.y[1] + 1e-9, step):
        lines.append(pv.Line((zone.x[0], y, z), (zone.x[1], y, z)))
    return pv.merge(lines)


def _mm_pair(values) -> str:
    return f"{values[0] * 1000:.0f}..{values[1] * 1000:.0f}"


def edit_layout(workcell_file: Path = DEFAULT_FILE, panel_file: Path | None = PANEL_FILE) -> None:
    """Open the editor; returns when its window is closed."""
    editor = LayoutEditor(workcell_file, panel_file)
    for finding in editor.findings:
        editor.log(("❌ " if finding.error else "⚠️  ") + finding.text)
    editor.show()
