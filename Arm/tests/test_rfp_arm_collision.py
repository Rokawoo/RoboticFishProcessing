"""Tests for rfp_arm.collision: the arm's own body (FR5 links and gripper) as spheres, checked
in a pose, along a planned path (safety.check_trajectory) and for the layout's reach. That the
spheres cover the meshes is checked in Simulation/tests (it needs them)."""

import dataclasses
import itertools

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from arm_helpers import make_workcell, workcell_data
from rfp_arm.collision import BODY_NAMES, CollisionModel
from rfp_arm.kinematics import FR5
from rfp_arm.layout import REACH_SEED, reach_joints
from rfp_arm.poses import tool_down
from rfp_arm.safety import check_trajectory
from rfp_arm.workcell import workcell_from_mapping

CELL = make_workcell()


def joints_at(x: float, y: float, z: float) -> np.ndarray:
    """The tool straight down with its TCP at (x, y, z), elbow up."""
    q = FR5.ik(CELL.tip_of_tcp(tool_down((x, y, z), 0.0)), REACH_SEED, iterations=200)
    assert q is not None
    return q


def test_every_body_is_checked_against_all_but_its_neighbors():
    model = CELL.collision
    assert [body.name for body in model.bodies] == list(BODY_NAMES.values())
    pairs = {(model.bodies[a].name, model.bodies[b].name) for a, b in model.pairs}
    gripper = {(body, "gripper") for body in ("base", "shoulder", "upper arm", "forearm")}
    assert gripper | {("forearm", "wrist 2"), ("forearm", "wrist 3")} <= pairs
    # neighbors meet at their joint: their meshes touch there in every pose
    names = [body.name for body in model.bodies]
    assert not set(itertools.pairwise(names)) & pairs
    # the base and the upper arm never come within 17 mm (Simulation/make_collision_model.py)
    assert ("base", "upper arm") not in pairs
    assert max(body.margin for body in model.bodies) == pytest.approx(0.01)
    # 19 mm apart wherever the tool points down: 7 mm spheres on both, so they don't seem to touch
    margins = {body.name: body.margin for body in model.bodies}
    assert margins["forearm"] + margins["wrist 2"] < 0.019


def test_the_gripper_brought_against_the_base_hits_it():
    clear, against = joints_at(0.30, 0.0, 0.05), joints_at(0.15, 0.0, 0.05)
    assert CELL.collision.hit(FR5, clear) is None
    assert CELL.collision.hit(FR5, against) == "the gripper would hit the base"
    for pose in CELL.observation_poses:  # the shipped cell's views are clear
        assert reach_joints(CELL, CELL.observation_tcp(pose)) is not None
    # a pose the arm reaches only against itself is out of reach for the layout's checks
    target = tool_down((0.15, 0.0, 0.05), 0.0)
    assert reach_joints(CELL, target) is None
    assert reach_joints(dataclasses.replace(CELL, collision=None), target) is not None


def test_what_of_the_arm_is_in_a_box():
    from rfp_arm.workcell import Box

    q = joints_at(0.40, 0.0, 0.30)
    cups = Box("cups", (0.38, 0.42), (-0.02, 0.02), (0.28, 0.32))  # around the TCP
    assert CELL.collision.in_box(FR5, q, cups)[0] == "gripper"
    base = Box("base", (-0.05, 0.05), (-0.05, 0.05), (0.0, 0.05))
    body, point = CELL.collision.in_box(FR5, q, base)
    ranges = (base.x, base.y, base.z)
    assert body == "base" and all(low <= v <= high for v, (low, high) in zip(point, ranges))
    far = Box("far", (1.00, 1.30), (-0.85, 0.85), (0.0, 1.5))  # the shipped operator's side
    assert CELL.collision.in_box(FR5, q, far) is None
    # a box of a turned cell: its ranges in the cell's frame, turned 90 deg left from {B}
    turned = Box("cups", (-0.02, 0.02), (-0.42, -0.38), (0.28, 0.32), np.radians(90))
    body, point = CELL.collision.in_box(FR5, q, turned)
    assert body == "gripper" and np.allclose(point[:2], (0.40, 0.0), atol=0.03)  # in {B}
    other_way = dataclasses.replace(turned, yaw=np.radians(-90))  # on the far side of the base
    assert CELL.collision.in_box(FR5, q, other_way) is None


def test_where_the_arm_hides_the_table_from_the_camera():
    # Looking near the base, the detector took the shoulder for a fish cut off at the edge
    pose = CELL.observation_poses[0]
    tcp = CELL.observation_tcp(pose)
    q, camera_z = reach_joints(CELL, tcp), CELL.camera_of_tcp(tcp)[2, 3]
    model = CELL.collision
    assert model.over(FR5, q, (0.0, 0.0), camera_z) == "base"
    assert model.over(FR5, q, (0.0, -0.15), camera_z) == "upper arm"
    assert model.over(FR5, q, (0.30, 0.0), camera_z) is None  # the table, in view
    assert model.over(FR5, q, (0.0, -0.15), 0.10) is None  # above the camera: not in the way
    # the gripper is beside the camera, out of its view: under it the table is seen
    assert model.over(FR5, q, tcp[:2, 3], camera_z) is None


def test_a_path_into_the_arm_is_refused_and_one_out_of_it_may_run():
    clear, against = joints_at(0.30, 0.0, 0.05), joints_at(0.15, 0.0, 0.05)
    path = clear + np.linspace(0, 1, 40)[:, None] * (against - clear)
    times = np.linspace(0, 4, 40)
    names = FR5.joint_names
    into = check_trajectory(CELL, FR5, names, path, times)
    # on its way down past the shoulder the gripper meets the upper arm first
    assert into.problem.startswith("the gripper would hit the upper arm (")
    assert "s into the move" in into.problem
    # in the plan's own joint order
    shuffled = check_trajectory(CELL, FR5, names[::-1], path[:, ::-1], times)
    assert shuffled.problem == into.problem
    # an arm left against itself (within the margin) may move out, not back
    assert check_trajectory(CELL, FR5, names, path[::-1], times).ok
    assert CELL.collision.path_problem(FR5, path[::-1]) is None
    back = np.vstack([path[::-1], path])
    assert CELL.collision.path_problem(FR5, back)[0] > len(path)
    stuck = CELL.collision.path_problem(FR5, [against, against])
    assert stuck is not None and stuck[1].endswith("all the way")
    # tool.collision_model "" turns the check off
    data = workcell_data()
    data["tool"]["collision_model"] = ""
    off = workcell_from_mapping(data)
    assert off.collision is None and check_trajectory(off, FR5, names, path, times).ok


def test_a_path_is_checked_between_its_points_and_pair_by_pair():
    against = joints_at(0.15, 0.0, 0.05)  # the gripper against the base
    a, b = against.copy(), against.copy()
    a[4] -= np.radians(65)
    b[4] += np.radians(80)  # the wrist (j5) swung either way: clear
    assert CELL.collision.hit(FR5, a) is None and CELL.collision.hit(FR5, b) is None
    # between the plan's two points the gripper swings into the upper arm
    assert CELL.collision.path_problem(FR5, [a, b]) == (1, "the gripper would hit the upper arm")
    # a pair touching where the path starts may come apart, and no other may touch meanwhile
    model = CollisionModel(CELL.collision.bodies, CELL.collision.pairs[:2])
    first, second = (model.what(pair) for pair in model.pairs)
    still = np.zeros((4, 6))  # not moving: the four poses alone are checked
    for rows, expected in (
        ([[1, 0], [1, 0], [0, 0], [0, 0]], None),
        ([[1, 0], [0, 0], [1, 0], [0, 0]], (2, first)),
        ([[1, 0], [1, 1], [0, 1], [0, 0]], (1, second)),
        ([[1, 0], [1, 0], [1, 0], [1, 0]], (0, first + " all the way")),
    ):
        model.touching = lambda chain, positions, rows=rows: np.array(rows, bool)
        assert model.path_problem(FR5, still) == expected, rows


def test_a_long_path_is_checked_quickly():
    import time

    start = joints_at(0.30, -0.2, 0.30)
    path = start + np.linspace(0, 1, 300)[:, None] * np.radians([40, 5, -5, 5, 0, 90])
    began = time.perf_counter()
    assert not CELL.collision.touching(FR5, path).any()
    assert time.perf_counter() - began < 1.0  # about 0.05 s on the development PC
    # a long swing, checked every cm of the way (about 700 poses)
    swing = start + np.linspace(0, 1, 60)[:, None] * np.radians([120, 10, -10, 10, 0, 170])
    began = time.perf_counter()
    assert CELL.collision.path_problem(FR5, swing) is None
    assert time.perf_counter() - began < 2.0  # about 0.13 s
