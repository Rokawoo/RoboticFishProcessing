"""Tests for rfp_arm.layout: the cell's layout read, turned about the arm, checked against
what the arm reaches and the camera sees, and saved (pick_place.py --mode setup's logic; its
3D window is tested in Simulation/tests)."""

import copy
import math
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from arm_helpers import frozen_layout, workcell_data
from rfp_arm.layout import (
    CameraView,
    FishSize,
    LayoutError,
    check_layout,
    layout_changed,
    lift_height,
    load_layout_data,
    reach_radius,
    rounded,
    save_layout,
    turned,
    view_footprint,
    with_layout,
)
from rfp_arm.poses import tool_down
from rfp_arm.workcell import DEFAULT_FILE, PANEL_FILE, load_workcell, workcell_from_mapping

FISH = FishSize(1.0, 0.25, 0.21, 0.126)  # the largest stand-in vision passes
VIEW = CameraView(87.0, 1280, 720)


@pytest.fixture
def files(tmp_path):
    workcell, panel = tmp_path / "workcell.yaml", tmp_path / "control_panel.yaml"
    shutil.copy(DEFAULT_FILE, workcell)
    shutil.copy(PANEL_FILE, panel)
    return workcell, panel


def check(layout, data) -> list:
    return check_layout(workcell_from_mapping(with_layout(data, layout)), FISH, VIEW)


def frozen() -> tuple[dict, object]:
    """(The workcell mapping, its layout): the tests' own layout, not the shipped one."""
    return workcell_data(), frozen_layout()


def test_the_shipped_layout_works():
    data, layout = load_layout_data()  # the cell: workcell.yaml with the panel on top
    found = check(layout, data)
    assert [f.text for f in found if f.error] == []
    # the arm reaches 0.675 m out with the tool down where fish are carried (0.50 m up)
    cell = workcell_from_mapping(data)
    assert lift_height(cell, FISH.length) == pytest.approx(cell.table_z + 0.50)
    assert reach_radius(cell, cell.table_z + 0.50) == pytest.approx(0.675, abs=0.005)


def test_the_check_finds_what_keeps_the_cell_from_working():
    data, layout = frozen()
    far = copy.deepcopy(layout)
    far.slots[1] = [0.38, 0.60]  # the old placeholder slot: past the reach when carrying
    texts = [f.text for f in check(far, data) if f.error]
    assert any("slot 2: out of the arm's reach at the carry height" in t for t in texts)
    # placed, the tool is the cups' offset past the slot (they hold the fish ahead of its
    # middle): without it this slot looked reachable, and every carry to it failed
    ahead = copy.deepcopy(layout)
    ahead.slots[1] = [0.28, 0.57]
    assert any("126 mm ahead" in f.text for f in check(ahead, data) if f.error)
    middle_only = FishSize(FISH.length, FISH.width, FISH.thickness)
    cell = workcell_from_mapping(with_layout(data, ahead))
    assert not [f for f in check_layout(cell, middle_only, VIEW) if f.error]
    behind = copy.deepcopy(layout)
    behind.poses[0]["yaw_deg"] += 180  # the camera toward the arm: the wrist ~180 out
    assert any("more than 90 from its middle" in f.text for f in check(behind, data) if f.error)
    # placed at heading 0, the wrist ends ~35 deg from the middle or ~145 the other way round:
    # fish arriving that way need the empty gripper past 90; the check says which heading
    # would put both at +-90 (the slots and their zone would follow it)
    slot = [f for f in check(layout, data) if f.text.startswith("slot 1: placed, the wrist")]
    assert slot and not slot[0].error and "placement heading of" in slot[0].text
    assert slot[0].minor  # a hint for the process: printed, not shown in the setup window
    off = copy.deepcopy(layout)
    off.slots[1] = [0.0, 0.65]  # a fish placed there reaches past the table's -X edge
    assert any("past the table's edge" in f.text for f in check(off, data) if f.error)
    blind = copy.deepcopy(layout)
    for pose in blind.poses:  # both views over the pick area's far side only
        pose["camera_xy"] = [0.35, -0.40]
    hidden = [f for f in check(blind, data) if "no observation pose sees" in f.text]
    assert hidden and not hidden[0].error and hidden[0].where is not None
    reaching = copy.deepcopy(layout)
    reaching.exclusions[0]["x"] = [0.95, 1.30]  # a fish at the far edge reaches into it
    assert any("into 'operator side' when lifted" in f.text for f in check(reaching, data))
    # the arm's links, not only the tool: reaching near, the elbow comes back over the base,
    # into a zone just behind the arm that the tool never enters
    back = copy.deepcopy(layout)
    back.exclusions[0]["x"] = [-0.40, -0.10]  # the operator's side moved behind the arm
    arm = [f for f in check(back, data) if "the arm's upper arm would be in" in f.text]
    assert arm and arm[0].error and arm[0].text.startswith("observation pose")
    assert -0.40 <= arm[0].where[0] <= -0.10  # marked where it enters the zone
    assert not any("the arm's" in f.text for f in check(layout, data))  # the shipped one: clear


def test_turning_the_layout_about_the_arm():
    data, layout = frozen()
    quarter = turned(layout, 1)
    x, y = layout.slots[0]
    assert quarter.slots[0] == pytest.approx([-y, x])  # counter-clockwise seen from above
    assert quarter.pick_area["x"] == pytest.approx(sorted(-v for v in layout.pick_area["y"]))
    assert quarter.pick_area["y"] == pytest.approx(layout.pick_area["x"])
    assert quarter.heading_deg == pytest.approx(layout.heading_deg + 90)
    assert quarter.poses[0]["yaw_deg"] == pytest.approx(layout.poses[0]["yaw_deg"] + 90)
    assert quarter.work["height"] == layout.work["height"]
    assert not layout_changed(turned(quarter, 3), layout)  # four quarter turns: back
    # turned a quarter either way, the arm reaches it all (its first joint turns with it);
    # half round, the pick area's middle is behind the arm, where that joint (+-175 deg) stops
    for k in (1, 3):
        assert not [f.text for f in check(turned(layout, k), data) if f.error]
    behind = [f for f in check(turned(layout, 2), data) if f.error]
    assert behind and "out of the arm's reach" in behind[0].text
    taught = copy.deepcopy(layout)
    taught.poses[0] = {"name": "taught", "joints_deg": [10.0, -90, -90, -90, 90, 0]}
    assert turned(taught, -1).poses[0]["joints_deg"][0] == pytest.approx(-80.0)


def test_a_view_is_the_image_on_the_table():
    camera = tool_down((0.4, 0.0, 0.45), 0.0)  # z down; image x along +X
    corners = view_footprint(camera, VIEW, plane_z=0.0)
    width, height = np.ptp(corners[:, 0]), np.ptp(corners[:, 1])
    assert width == pytest.approx(2 * 0.45 * math.tan(math.radians(43.5)))
    assert height == pytest.approx(width * 720 / 1280)
    assert corners.mean(axis=0) == pytest.approx([0.4, 0.0])


def test_saving_changes_only_the_layout_lines(files):
    workcell, panel = files
    before, before_panel = workcell.read_text(), panel.read_text()
    _, layout = load_layout_data(workcell, panel)
    assert save_layout(layout, workcell, panel) == []  # nothing changed: nothing written
    assert not workcell.with_name("workcell.yaml.bak").exists()
    quarter = turned(layout, 1)
    written = save_layout(quarter, workcell, panel)
    # the panel overrides the heading: it is changed there, the rest in the workcell file
    assert "placement.heading_deg in control_panel.yaml" in written
    assert "placement.slots in workcell.yaml" in written
    _, back = load_layout_data(workcell, panel)
    assert not layout_changed(back, rounded(quarter))
    load_workcell(workcell, panel)  # still a valid workcell
    old, new = before.splitlines(), workcell.read_text().splitlines()
    assert len(old) == len(new)
    for a, b in zip(old, new):  # every comment stays
        assert ("#" in a) == ("#" in b) and a.partition("#")[2] == b.partition("#")[2]
    assert workcell.with_name("workcell.yaml.bak").read_text() == before  # the old file kept
    assert (
        sum(a != b for a, b in zip(before_panel.splitlines(), panel.read_text().splitlines())) == 1
    )


def test_a_verified_cell_whose_layout_changes_must_be_checked_again(files):
    workcell, panel = files
    panel.write_text(panel.read_text().replace("verified: false", "verified: true", 1))
    data, layout = load_layout_data(workcell, panel)
    assert data["verified"] is True  # set by the panel
    layout.slots[0][0] += 0.01
    written = save_layout(layout, workcell, panel)
    assert any(line.startswith("verified: false in control_panel.yaml") for line in written)
    assert load_layout_data(workcell, panel)[0]["verified"] is False


def test_a_layout_that_cant_be_written_as_asked_is_not_written(files):
    workcell, panel = files
    before = workcell.read_text()
    _, layout = load_layout_data(workcell, panel)
    layout.slots[0] = [0.0, 0.0]  # outside the placement zone: a problem of the workcell
    with pytest.raises(LayoutError, match="slot 1 is outside"):
        save_layout(layout, workcell, panel)
    area = layout.pick_area
    line = next(row for row in before.splitlines() if row.startswith("  pick_area: {"))
    by_hand = f"  pick_area:\n    x: {area['x']}\n    y: {area['y']}"
    workcell.write_text(before.replace(line, by_hand))
    _, layout = load_layout_data(workcell, panel)
    layout.pick_area["x"] = [v - 0.04 for v in area["x"]]  # several lines: changed by hand there
    with pytest.raises(LayoutError, match="by hand"):
        save_layout(layout, workcell, panel)
    assert not workcell.with_name("workcell.yaml.bak").exists()
