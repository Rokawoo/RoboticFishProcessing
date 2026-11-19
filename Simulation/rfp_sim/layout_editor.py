"""
Lay out the cell by hand in a 3D window (pick_place.py --mode setup), around the FR5 and the
gripper as built. Nothing moves.

- Drag a corner ball to size the table (gray), the pick area (green) or the placement zone
  (blue). Drag a slot (a big blue ball, its fish drawn head first) to where a placed fish
  goes. Drag a keep-out zone's ball (red) to move it: one along a whole table edge, like the
  operator's side, goes along the edge nearest the ball.
- Buttons: turn the whole cell a quarter turn about the arm (the table on another side of
  it), add or remove a slot, undo, save.

The rest follows: the camera's views (yellow) follow the pick area, the placement zone grows
to hold its slots, a keep-out zone along a table edge stays along it as the table is sized,
and everything turns with the cell. After every change the layout is checked
(rfp_arm.layout.check_layout): red is what can't work (the arm can't reach it, a placed fish
off the table), yellow what may not. Save writes the workcell file, and the control panel
where it overrides the file, changing only the layout's lines (a .bak copy of each is kept).
The placement heading, the keep-out zones' sizes and the observation poses are set in the
files.
"""

import copy
import math
import textwrap
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

TABLE_COLOR = (0.62, 0.63, 0.66)
PICK_COLOR = (0.3, 0.9, 0.4)
PLACE_COLOR = (0.3, 0.6, 1.0)
SLOT_COLOR = (0.35, 0.55, 1.0)
EXCLUSION_COLOR = (1.0, 0.35, 0.3)
VIEW_COLOR = (1.0, 0.75, 0.2)
ERROR_COLOR = (1.0, 0.3, 0.3)
WARNING_COLOR = (1.0, 0.85, 0.3)
TEXT_COLOR = (0.85, 0.85, 0.9)
# The parts sized by their corners: (Layout field, name, color)
ZONES = (
    ("work", "table", TABLE_COLOR),
    ("pick_area", "pick area", PICK_COLOR),
    ("placement_zone", "placement zone", PLACE_COLOR),
)
ARM_OPACITY = 0.45
SNAP = 0.005  # m: dragged corners and slots land on this grid
MIN_SIDE = 0.05  # m: a zone stays at least this wide
SLOT_MARGIN = 0.02  # m: the placement zone reaches this far past its outermost slots
SLOT_GAP = 0.02  # m: between the fish an added slot is put next to
CORNER_RADIUS = 0.02  # m
SLOT_RADIUS = 0.03
FAR = 3.0  # m from the arm: a ball dragged past the horizon stops here
BUTTON = 26  # px
LINE = 18  # px: a line of text at font size 9
CHAR = 8.4  # px: a character's width there
HELP = (
    "Drag a corner ball to size the table (gray), pick area (green) or placement zone (blue); "
    "drag a slot (big blue ball) to where a placed fish goes, head along its arrow; drag a red "
    "ball to move a keep-out zone (the operator's side goes along the table edge nearest it). "
    "Yellow: what the camera sees from its observation poses (they follow the pick area).\n"
    "Mouse: drag the empty space to look around, the wheel to zoom, shift + drag to pan."
)


# ---------------------------------------------------------------------------
# What dragging a ball does (no window: tested on its own)
# ---------------------------------------------------------------------------
def _snap(value: float) -> float:
    return round(round(value / SNAP) * SNAP, 3)


def _corner_moved(rect: dict, i: int, j: int, x: float, y: float) -> dict:
    """`rect` with its corner (x[i], y[j]) at (x, y); the other corner stays (the dragged
    sides stop MIN_SIDE short of it, so a corner stays the one it is)."""
    new = copy.deepcopy(rect)  # keeps the table's height
    xs, ys = list(rect["x"]), list(rect["y"])
    xs[i] = min(x, xs[1] - MIN_SIDE) if i == 0 else max(x, xs[0] + MIN_SIDE)
    ys[j] = min(y, ys[1] - MIN_SIDE) if j == 0 else max(y, ys[0] + MIN_SIDE)
    new["x"], new["y"] = [round(v, 3) for v in xs], [round(v, 3) for v in ys]
    return new


def _mapped(xy, old: dict, new: dict) -> list[float]:
    """A point of `old` at the same place relative to `new` (a camera view follows the pick
    area as it is sized)."""
    return [
        round(b[0] + (v - a[0]) * (b[1] - b[0]) / (a[1] - a[0]), 3)
        for v, a, b in zip(xy, (old["x"], old["y"]), (new["x"], new["y"]))
    ]


def _clamped(xy, rect: dict) -> list[float]:
    out = []
    for v, (low, high) in zip(xy, (rect["x"], rect["y"])):
        margin = min(SLOT_MARGIN, (high - low) / 2)
        out.append(round(min(max(v, low + margin), high - margin), 3))
    return out


def _holding(zone: dict, slots: list) -> dict:
    """The placement zone grown to hold every slot, SLOT_MARGIN inside."""
    if not slots:
        return zone
    xs, ys = [s[0] for s in slots], [s[1] for s in slots]
    return {
        **zone,
        "x": [round(min(zone["x"][0], min(xs) - SLOT_MARGIN), 3),
              round(max(zone["x"][1], max(xs) + SLOT_MARGIN), 3)],
        "y": [round(min(zone["y"][0], min(ys) - SLOT_MARGIN), 3),
              round(max(zone["y"][1], max(ys) + SLOT_MARGIN), 3)],
    }  # fmt: skip


def _middle(rect: dict) -> tuple[float, float]:
    return (rect["x"][0] + rect["x"][1]) / 2, (rect["y"][0] + rect["y"][1]) / 2


def _edge(zone: dict, table: dict) -> tuple[int, int] | None:
    """The table edge a keep-out zone lies along, (axis, side) (axis 0: x, 1: y; side 0: the
    low edge, 1: the high one): the zone spans the table's whole length along the edge and
    reaches it or lies past it, not the opposite edge. None for any other box."""
    for axis in (0, 1):
        across, along = "xy"[axis], "xy"[1 - axis]
        if zone[along][0] <= table[along][0] + 1e-6 and zone[along][1] >= table[along][1] - 1e-6:
            low = zone[across][0] <= table[across][0] + 1e-6
            high = zone[across][1] >= table[across][1] - 1e-6
            if low != high:
                return axis, int(high)
    return None


def _nearest_edge(table: dict, xy) -> tuple[int, int]:
    """The table edge (axis, side) nearest a point on the table; off it, the edge the point
    lies furthest past."""
    past = {}
    for axis, name in enumerate("xy"):
        low, high = table[name]
        past[axis, 0], past[axis, 1] = low - xy[axis], xy[axis] - high
    return max(past, key=past.get)


def _along(zone: dict, table: dict, edge: tuple, to_table: dict, to_edge: tuple) -> dict:
    """A keep-out zone along `edge` of `table`, put along `to_edge` of `to_table`: as far into
    the table and out past the edge, and as far past the table's ends along it. It stops
    MIN_SIDE short of the opposite edge, so it stays along this one, unless that would make
    it thinner than MIN_SIDE (a table that small)."""

    def frame(rect: dict, edge: tuple):  # the axes along and across the edge, the edge, inward
        axis, side = edge
        across = "xy"[axis]
        return "xy"[1 - axis], across, rect[across][side], -1.0 if side else 1.0

    along, across, level, inward = frame(table, edge)
    depth = sorted((v - level) * inward for v in zone[across])  # m into the table from the edge
    ends = (table[along][0] - zone[along][0], zone[along][1] - table[along][1])
    along, across, level, inward = frame(to_table, to_edge)
    far = to_table[across][1] - to_table[across][0] - MIN_SIDE
    depth[1] = max(min(depth[1], far), depth[0] + min(depth[1] - depth[0], MIN_SIDE))
    return {
        **zone,
        along: [round(to_table[along][0] - ends[0], 3), round(to_table[along][1] + ends[1], 3)],
        across: sorted(round(level + inward * d, 3) for d in depth),
    }


def _keep_out_moved(zone: dict, table: dict, xy) -> dict:
    """A keep-out zone dragged by its ball to `xy`: one along a table edge (where the operator
    stands) goes along the edge nearest `xy`, as deep; any other box moves as it is, its
    middle to `xy`."""
    edge = _edge(zone, table)
    if edge is not None:
        return _along(zone, table, edge, table, _nearest_edge(table, xy))
    middle = _middle(zone)
    return {
        **zone,
        **{a: [round(v + xy[i] - middle[i], 3) for v in zone[a]] for i, a in enumerate("xy")},
    }


def moved(layout: Layout, handle: tuple, xy) -> Layout:
    """`layout` with one ball dragged to `xy` (m, on the table). A handle is ("slot", k),
    ("keep_out", k) or (zone, i, j): the zone's corner (x[i], y[j]), a zone one of ZONES'
    fields."""
    new = copy.deepcopy(layout)
    x, y = _snap(xy[0]), _snap(xy[1])
    if handle[0] == "slot":
        new.slots[handle[1]] = [x, y]
        new.placement_zone = _holding(new.placement_zone, new.slots)
        return new
    if handle[0] == "keep_out":
        k = handle[1]
        new.exclusions[k] = _keep_out_moved(layout.exclusions[k], layout.work, (x, y))
        return new
    part, i, j = handle
    old = getattr(layout, part)
    rect = _corner_moved(old, i, j, x, y)
    setattr(new, part, rect)
    if part == "work":  # a keep-out zone along a table edge stays along it
        for k, zone in enumerate(layout.exclusions):
            if (edge := _edge(zone, old)) is not None:
                new.exclusions[k] = _along(zone, old, edge, rect, edge)
    elif part == "pick_area":
        for pose in new.poses:
            if "camera_xy" in pose:
                pose["camera_xy"] = _mapped(pose["camera_xy"], old, rect)
    elif part == "placement_zone":
        new.slots = [_clamped(slot, rect) for slot in new.slots]
    return new


def _clear_of(low: np.ndarray, high: np.ndarray, rect: dict) -> np.ndarray:
    """Which boxes (low, high corners: n x 2) don't overlap `rect`."""
    return (
        (high[:, 0] <= rect["x"][0])
        | (low[:, 0] >= rect["x"][1])
        | (high[:, 1] <= rect["y"][0])
        | (low[:, 1] >= rect["y"][1])
    )


def free_spot(layout: Layout, fish: FishSize, reach: float | None, step: float = 0.01):
    """Where another fish can be placed, ((x, y), clear of the other placed fish) or None:
    the whole fish on the table, clear of the pick area and the keep-out zones, within
    `reach` (m from the arm's axis, the cups' offset ahead of its middle included), the
    placement zone grown to hold it clear of the pick area; of those, the one overlapping
    the other placed fish least (not at all, if there is room), nearest the last slot."""
    heading = math.radians(layout.heading_deg)
    along = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-along[1], along[0]])
    (x0, x1), (y0, y1) = layout.work["x"], layout.work["y"]
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    spots = np.column_stack([gx.ravel(), gy.ravel()])
    corners = np.stack(
        [
            spots + a * fish.length / 2 * along + b * fish.width / 2 * across
            for a in (-1, 1)
            for b in (-1, 1)
        ],
        axis=1,
    )
    low, high = corners.min(axis=1), corners.max(axis=1)
    ok = (low[:, 0] >= x0) & (high[:, 0] <= x1) & (low[:, 1] >= y0) & (high[:, 1] <= y1)
    ok &= _clear_of(low, high, layout.pick_area)
    for zone in layout.exclusions:
        ok &= _clear_of(low, high, zone)
    zone = layout.placement_zone  # grown to hold the spot, it stays clear of the pick area
    grown_low = np.minimum((zone["x"][0], zone["y"][0]), spots - SLOT_MARGIN)
    grown_high = np.maximum((zone["x"][1], zone["y"][1]), spots + SLOT_MARGIN)
    ok &= _clear_of(grown_low, grown_high, layout.pick_area)
    if reach is not None:
        for ahead in (0.0, fish.cups_ahead):
            ok &= np.hypot(*(spots + ahead * along).T) <= reach
    if not ok.any():
        return None
    spots = spots[ok]
    clearance = np.full(len(spots), np.inf)  # m past the nearest other fish (< 0: overlaps)
    for slot in layout.slots:
        d = spots - np.asarray(slot)
        apart = np.maximum(
            np.abs(d @ along) - fish.length - SLOT_GAP, np.abs(d @ across) - fish.width - SLOT_GAP
        )
        clearance = np.minimum(clearance, apart)
    overlap = np.minimum(clearance, 0.0)
    last = np.asarray(layout.slots[-1]) if layout.slots else spots.mean(axis=0)
    distance = np.hypot(*(spots - last).T)
    best = np.lexsort((distance, -np.round(overlap, 3)))[0]
    spot = (round(float(spots[best, 0]), 3), round(float(spots[best, 1]), 3))
    return spot, bool(overlap[best] == 0.0)


def with_slot(layout: Layout, fish: FishSize, reach: float | None = None) -> tuple[Layout, bool]:
    """(`layout` with one more slot, whether it is clear of the other placed fish): on the
    best spot `free_spot` finds (if none, a step past the last slot), the placement zone
    grown to hold it."""
    new = copy.deepcopy(layout)
    found = free_spot(new, fish, reach)
    if found is not None:
        spot, clear = found
    elif len(new.slots) > 1:
        spot, clear = np.add(new.slots[-1], np.subtract(new.slots[-1], new.slots[-2])), False
    else:
        spot, clear = np.add(new.slots[-1], (0.0, fish.width + SLOT_GAP)), False
    new.slots.append([_snap(float(spot[0])), _snap(float(spot[1]))])
    new.placement_zone = _holding(new.placement_zone, new.slots)
    return new, clear


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
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
        self.message = ""
        self.findings = []
        self.handles = []  # (vtkSphereWidget, handle)
        self.grabbed = None  # (handle, the layout when the drag began)
        self.offset = np.zeros(2)  # m: from the mouse to the middle of the ball dragged

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
        for actor in (*self.arm, self.tool):  # see-through: the zones under it stay in view
            actor.prop.opacity = ARM_OPACITY
            actor.pickable = False
        self._add_reach()
        self._add_buttons()
        self.rebuild()
        self.look_at_table()

    def look_at_table(self) -> None:
        """From the front (-Y), steeply down, the whole table in view."""
        (x0, x1), (y0, y1) = self.layout.work["x"], self.layout.work["y"]
        middle = ((x0 + x1) / 2, (y0 + y1) / 2, self.table_z)
        self.plotter.camera_position = [
            (middle[0], middle[1] - 1.3, self.table_z + 2.4),
            middle,
            (0, 0, 1),
        ]
        self.plotter.reset_camera(bounds=(x0, x1, y0, y1, self.table_z, self.table_z + 0.3))
        camera = self.plotter.camera
        camera.zoom(1.1)
        shift = np.array([0.0, -0.12 * (y1 - y0), 0.0])  # the table up, clear of the text
        camera.position = tuple(np.add(camera.position, shift))
        camera.focal_point = tuple(np.add(camera.focal_point, shift))

    # -- the cell as it is now -------------------------------------------------------
    @property
    def cell(self) -> Workcell:
        return workcell_from_mapping(with_layout(self.data, self.layout))

    @property
    def table_z(self) -> float:
        return float(self.data["table"]["z"])

    @property
    def handle_z(self) -> float:
        return self.table_z + 0.02

    def _add_reach(self) -> None:
        """Rings on the table: how far the cups reach straight down at a fish's top (the pick
        area inside), and at the height fish are carried at (the slots inside)."""
        cell = self.cell
        lift = lift_height(cell, self.fish.length)
        self.carry_reach = reach_radius(cell, lift)
        rings = (
            (reach_radius(cell, self.table_z + self.fish.thickness), (0.6, 0.6, 0.65), "picking"),
            (self.carry_reach, (0.95, 0.55, 0.25), "carrying: slots inside"),
        )
        z = self.table_z + 0.003
        for radius, color, label in rings:
            angles = np.linspace(0, 2 * math.pi, 181)
            ring = np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z)]
            )
            self.plotter.add_mesh(
                pv.lines_from_points(ring), color=color, line_width=2, pickable=False
            )
            self.plotter.add_point_labels(
                [(radius * 0.7071, -radius * 0.7071, z)],
                [f"arm's reach, {label}"],
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
            ("Turn the cell 90 deg left about the arm", lambda: self.turn(1)),
            ("Turn the cell 90 deg right", lambda: self.turn(-1)),
            ("Add a slot (a place for a fish)", self.add_slot),
            ("Remove the last slot", self.remove_slot),
            ("Undo all changes", self.undo),
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

    # -- the balls -------------------------------------------------------------------
    def _handle_places(self):
        """(handle, (x, y), radius, color) of every ball, from the layout."""
        for part, _, color in ZONES:
            rect = getattr(self.layout, part)
            for i in (0, 1):
                for j in (0, 1):
                    yield (part, i, j), (rect["x"][i], rect["y"][j]), CORNER_RADIUS, color
        for k, slot in enumerate(self.layout.slots):
            yield ("slot", k), tuple(slot), SLOT_RADIUS, SLOT_COLOR
        for k, zone in enumerate(self.layout.exclusions):
            yield ("keep_out", k), _middle(zone), SLOT_RADIUS, EXCLUSION_COLOR

    def rebuild(self) -> None:
        """A ball for every corner, slot and keep-out zone (after a change of what there is
        to drag)."""
        self.plotter.clear_sphere_widgets()
        self.handles = []
        for handle, (x, y), radius, color in self._handle_places():
            widget = self.plotter.add_sphere_widget(
                lambda _center, _widget, handle=handle: self._on_ball(handle),
                center=(x, y, self.handle_z),
                radius=radius,
                color=color,
                interaction_event="always",
                pass_widget=True,
                test_callback=False,
            )
            widget.AddObserver("StartInteractionEvent", lambda *_, h=handle: self._grab(h))
            widget.AddObserver("EndInteractionEvent", lambda *_: self._dropped())
            self.handles.append((widget, handle))
        self.refresh()

    def _place_handles(self, held=None) -> None:
        """Every ball where the layout puts it, on the table; `held`: (handle, xy) of a ball
        that stays where the mouse puts it."""
        places = {handle: xy for handle, xy, _, _ in self._handle_places()}
        if held is not None:
            places[held[0]] = held[1]
        for widget, handle in self.handles:
            if handle in places:
                widget.SetCenter(places[handle][0], places[handle][1], self.handle_z)

    def _under_mouse(self) -> np.ndarray | None:
        """The point of the balls' plane under the mouse (x, y). VTK drags a ball parallel
        to the screen at about the focal point's depth, not the ball's, so it trails the
        mouse: the balls go where the mouse is on the table instead."""
        renderer = self.plotter.renderer
        x, y = self.plotter.iren.interactor.GetEventPosition()
        ends = []
        for depth in (0.0, 1.0):  # the near and far planes: the line of sight
            renderer.SetDisplayPoint(x, y, depth)
            renderer.DisplayToWorld()
            point = renderer.GetWorldPoint()
            ends.append(np.array(point[:3]) / point[3])
        near, far = ends
        if abs(far[2] - near[2]) < 1e-9:
            return None
        t = (self.handle_z - near[2]) / (far[2] - near[2])
        if t < 0:  # looking up: the plane is behind
            return None
        hit = (near + t * (far - near))[:2]
        distance = float(np.hypot(*hit))
        return hit * FAR / distance if distance > FAR else hit

    def _grab(self, handle) -> None:
        """A ball taken: where on it the mouse holds it, kept while it is dragged."""
        under = self._under_mouse()
        place = {h: xy for h, xy, _, _ in self._handle_places()}.get(handle)
        self.offset = np.zeros(2) if under is None or place is None else np.subtract(place, under)

    def _on_ball(self, handle) -> None:
        under = self._under_mouse()
        if under is not None:
            self._dragging(handle, under + self.offset)

    def _dragging(self, handle, xy) -> None:
        """While a ball is dragged: the layout, the balls and what is drawn from it follow;
        the checks wait for the drop."""
        if self.grabbed is None or self.grabbed[0] != handle:
            self.grabbed = (handle, copy.deepcopy(self.layout))
        self.layout = moved(self.grabbed[1], handle, xy)
        # a keep-out zone's ball stays under the mouse while its zone goes from edge to edge
        self._place_handles((handle, tuple(xy)) if handle[0] == "keep_out" else None)
        self._draw()

    def _dropped(self) -> None:
        if self.grabbed is None:
            return
        handle = self.grabbed[0]
        self.grabbed = None
        self.confirm_save = False
        if handle[0] == "slot":
            x, y = self.layout.slots[handle[1]]
            self.message = f"slot {handle[1] + 1} at ({x * 1000:.0f}, {y * 1000:.0f}) mm"
        elif handle[0] == "keep_out":
            zone = self.layout.exclusions[handle[1]]
            self.message = (
                f"keep out '{zone['name']}': x {_mm_pair(zone['x'])}, y {_mm_pair(zone['y'])} mm"
            )
        else:
            name = next(name for part, name, _ in ZONES if part == handle[0])
            rect = getattr(self.layout, handle[0])
            self.message = f"{name}: x {_mm_pair(rect['x'])}, y {_mm_pair(rect['y'])} mm"
        self._place_handles()
        self.refresh()

    def drag(self, handle: tuple, xy) -> None:
        """Drag a ball to `xy` (m, on the table) and let go: what the mouse does."""
        self._dragging(handle, xy)
        self._dropped()

    # -- buttons ---------------------------------------------------------------------
    def turn(self, quarter_turns: int) -> None:
        self.layout = turned(self.layout, quarter_turns)
        self.message = f"turned the cell {90 * quarter_turns:+d} deg about the arm"
        self.rebuild()
        self.look_at_table()

    def add_slot(self) -> None:
        self.layout, clear = with_slot(self.layout, self.fish, self.carry_reach)
        k = len(self.layout.slots)
        self.message = (
            f"slot {k} added where a fish fits: drag it where it goes"
            if clear
            else f"slot {k} added, but no free spot on the table fits a whole fish: make room "
            "(a bigger table or placement zone, slots moved)"
        )
        self.rebuild()

    def remove_slot(self) -> None:
        if len(self.layout.slots) > 2:  # the cell needs at least two
            self.layout.slots.pop()
            self.message = f"{len(self.layout.slots)} slots left"
        else:
            self.message = "two slots at least: drag one instead"
        self.rebuild()

    def undo(self) -> None:
        self.layout = copy.deepcopy(self.saved)
        self.message = "back to the layout as saved"
        self.rebuild()

    def save(self) -> list[str]:
        """Write the layout. With problems left, the first click only asks for a second one."""
        errors = [f for f in self.findings if f.error]
        if errors and not self.confirm_save:
            self.confirm_save = True
            self.message = f"{len(errors)} problem(s) left (red): click Save again to save anyway"
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
            self.message = "saved (.bak copies kept)"
            self.log("✅ Saved: " + "\n          ".join(written))
            for finding in self.findings:
                self.log(("❌ " if finding.error else "⚠️  ") + finding.text)
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
        self._draw(cell)
        shown = [f for f in self.findings if not f.minor]
        z = self.table_z + 0.006
        markers = [
            (f.where, ERROR_COLOR if f.error else WARNING_COLOR)
            for f in shown
            if f.where is not None
        ]
        for k in range(len(markers), 12):
            self.plotter.remove_actor(f"finding {k}", render=False)
        for k, (where, color) in enumerate(markers[:12]):
            disc = pv.Disc(center=(where[0], where[1], z), inner=0.035, outer=0.05)
            self.plotter.add_mesh(
                disc, color=color, name=f"finding {k}", pickable=False, render=False
            )
        self._pose_arm(cell)
        self._text(shown)
        self.plotter.render()

    def _draw(self, cell: Workcell | None = None) -> None:
        """Everything drawn from the layout: quick, so it follows a ball as it is dragged."""
        cell = cell or self.cell
        z = self.table_z + 0.004
        lay = self.layout

        def add(mesh, name, **style):
            if mesh.n_points == 0:  # e.g. no grid line on a tiny table: nothing to draw
                self.plotter.remove_actor(name, render=False)
            else:
                self.plotter.add_mesh(mesh, name=name, pickable=False, render=False, **style)

        work = lay.work
        add(_plane(work, z - 0.003), "table", color=(0.42, 0.43, 0.45), opacity=0.9)
        add(_grid(work, z - 0.002), "grid", color=(0.55, 0.56, 0.6), line_width=1)
        add(_outline(work, z - 0.001), "table edge", color=TABLE_COLOR, line_width=2)
        for part, color in (("pick_area", PICK_COLOR), ("placement_zone", PLACE_COLOR)):
            rect = getattr(lay, part)
            add(_plane(rect, z - 0.001), f"{part} fill", color=color, opacity=0.18)
            add(_outline(rect, z), part, color=color, line_width=3)
        for k in range(len(lay.exclusions), 8):
            self.plotter.remove_actor(f"keep out {k}", render=False)
            self.plotter.remove_actor(f"keep out {k} fill", render=False)
        for k, zone in enumerate(lay.exclusions[:8]):
            add(_plane(zone, z - 0.0015), f"keep out {k} fill", color=EXCLUSION_COLOR, opacity=0.2)
            add(_outline(zone, z), f"keep out {k}", color=EXCLUSION_COLOR, line_width=2)
        add(self._placed_fish(z), "placed fish", color=SLOT_COLOR, line_width=2)
        add(self._views(cell, z), "views", color=VIEW_COLOR, line_width=1)
        self._labels(z)

    def _placed_fish(self, z: float) -> pv.PolyData:
        """Each slot's fish, the size the checks assume, head along the placement heading."""
        heading = math.radians(self.layout.heading_deg)
        along = np.array([math.cos(heading), math.sin(heading)])
        across = np.array([-along[1], along[0]])
        lines = []
        for slot in self.layout.slots:
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

    def _labels(self, z: float) -> None:
        lay = self.layout
        points, labels = [], []
        for part, name, _ in ZONES:
            rect = getattr(lay, part)
            points.append((rect["x"][0] + 0.02, rect["y"][0] + 0.02, z))
            labels.append(name)
        for k, slot in enumerate(lay.slots, 1):
            points.append((slot[0], slot[1], z + 0.04))
            labels.append(f"slot {k}")
        for zone in lay.exclusions:  # at its ball, as a slot's
            points.append((*_middle(zone), z + 0.04))
            labels.append(f"keep out: {zone['name']}")
        self.plotter.add_point_labels(
            points,
            labels,
            font_size=11,
            text_color="white",
            shape_opacity=0.35,
            show_points=False,
            always_visible=True,
            name="labels",
            render=False,
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

    def _text(self, shown: list) -> None:
        errors = [f.text for f in shown if f.error]
        notes = [f.text for f in shown if not f.error]
        if errors:
            status, color = f"{len(errors)} problem(s): it can't work like this", ERROR_COLOR
        elif notes:
            status, color = f"Layout OK, {len(notes)} note(s)", WARNING_COLOR
        else:
            status, color = "Layout OK", PICK_COLOR
        saved = "changed, not saved" if layout_changed(self.layout, self.saved) else "as saved"
        head = f"{status}\n{saved}" + (f"\n{self.message}" if self.message else "")
        self.plotter.add_text(
            head, position="upper_right", font_size=11, color=color, name="status", render=False
        )
        width = max(40, int((self.plotter.window_size[0] - 30) / CHAR))
        help_lines = [line for part in HELP.split("\n") for line in textwrap.wrap(part, width)]
        y = 8
        self.plotter.add_text(
            "\n".join(help_lines),
            position=(10, y),
            font_size=9,
            color=TEXT_COLOR,
            name="help",
            render=False,
        )
        y += LINE * len(help_lines) + 10
        room = max(2, (self.plotter.window_size[1] // 2 - y) // LINE)  # the lower half at most
        lines = []
        for texts, mark in ((errors, "problem: "), (notes, "note: ")):
            lines.append([line for text in texts for line in textwrap.wrap(mark + text, width)])
        error_lines, note_lines = lines
        if len(error_lines) + len(note_lines) > room:
            note_lines = note_lines[: max(0, room - len(error_lines) - 1)]
            note_lines.append("... more in the terminal when you save")
        for name, block, color in (
            ("notes", note_lines, WARNING_COLOR),
            ("errors", error_lines[:room], ERROR_COLOR),
        ):
            self.plotter.add_text(
                "\n".join(block), position=(10, y), font_size=9, color=color, name=name,
                render=False,
            )  # fmt: skip
            y += LINE * len(block)

    def show(self) -> Layout:
        self.plotter.show()
        if layout_changed(self.layout, self.saved):
            self.log("⚠️  Closed with changes not saved.")
        return self.layout


def _plane(rect: dict, z: float) -> pv.PolyData:
    (x0, x1), (y0, y1) = rect["x"], rect["y"]
    return pv.Plane(center=((x0 + x1) / 2, (y0 + y1) / 2, z), i_size=x1 - x0, j_size=y1 - y0)


def _outline(rect: dict, z: float) -> pv.PolyData:
    (x0, x1), (y0, y1) = rect["x"], rect["y"]
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]
    return pv.lines_from_points(np.array(corners))


def _grid(rect: dict, z: float, step: float = 0.1) -> pv.PolyData:
    """Lines every `step` m over the table, at multiples of it."""
    (x0, x1), (y0, y1) = rect["x"], rect["y"]
    lines = []
    for x in np.arange(math.ceil(x0 / step) * step, x1 + 1e-9, step):
        lines.append(pv.Line((x, y0, z), (x, y1, z)))
    for y in np.arange(math.ceil(y0 / step) * step, y1 + 1e-9, step):
        lines.append(pv.Line((x0, y, z), (x1, y, z)))
    return pv.merge(lines) if lines else pv.PolyData()


def _mm_pair(values) -> str:
    return f"{values[0] * 1000:.0f}..{values[1] * 1000:.0f}"


def edit_layout(workcell_file: Path = DEFAULT_FILE, panel_file: Path | None = PANEL_FILE) -> None:
    """Open the editor; returns when its window is closed."""
    editor = LayoutEditor(workcell_file, panel_file)
    for finding in editor.findings:
        editor.log(("❌ " if finding.error else "⚠️  ") + finding.text)
    editor.show()
