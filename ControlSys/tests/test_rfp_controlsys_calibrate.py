"""Tests for measuring the camera's mount (rfp_controlsys/calibrate.py): the solver on
sightings made up from a known mount, and writing the result into the workcell file. The
whole measurement on the virtual arm and camera is in Simulation/tests."""

import math
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Arm" / "tests"))
from arm_helpers import make_workcell  # the tests' own layout
from rfp_arm.poses import from_parts, invert, rot_z, rotation_angle, tool_down
from rfp_arm.workcell import DEFAULT_FILE, load_workcell
from rfp_controlsys.calibrate import poses_around, save_mount, solve

CELL = make_workcell()


def sightings(mount: np.ndarray, target, noise: float = 0.0, seed: int = 0):
    """(tip poses, the target in the camera frame from each) around a pose 0.4 m above it."""
    rng = np.random.default_rng(seed)
    start = CELL.tcp_of_camera(tool_down((target[0], target[1] - 0.02, target[2] + 0.40), 0.3))
    tips, points = [], []
    for tcp in [start, *poses_around(start)]:
        tip = CELL.tip_of_tcp(tcp)
        point = invert(tip @ mount) @ np.append(target, 1.0)
        tips.append(tip)
        points.append(point[:3] + rng.normal(0.0, noise, 3))
    return tips, points


def test_the_mount_is_found_from_how_a_still_object_shifts():
    target = np.array([0.40, -0.10, 0.08])
    turned = CELL.camera @ from_parts(rot_z(math.pi), (0.0, 0.0, 0.0))  # mounted half round
    turned[:3, 3] += (0.004, -0.006, 0.0)  # and a few mm from the drawing across the tool
    tips, points = sightings(turned, target, noise=0.0005)
    result = solve(tips, points, CELL.camera)
    assert math.degrees(rotation_angle(result.mount, turned)) < 1.0
    assert np.linalg.norm(result.mount[:3, 3] - turned[:3, 3]) < 0.004
    assert result.mount[2, 3] == CELL.camera[2, 3]  # along the tool: the drawing's
    assert np.allclose(result.target, target, atol=0.004) and result.rms_m < 0.002
    assert result.turned_deg == pytest.approx(180.0, abs=1.0) and not result.mirrored
    text = "\n".join(result.lines(CELL.camera))
    assert "half a turn round" in text and "the other way than the fish" in text
    # the mount the workcell has: nothing to change
    same = solve(*sightings(CELL.camera, target), CELL.camera)
    assert same.turned_deg < 0.1 and np.allclose(same.mount, CELL.camera, atol=1e-6)


def test_a_mirrored_image_is_said_to_be_one():
    target = np.array([0.40, -0.10, 0.08])
    tips, points = sightings(CELL.camera, target)
    flipped = [p * (-1.0, 1.0, 1.0) for p in points]  # the image's x mirrored
    result = solve(tips, flipped, CELL.camera)
    assert result.mirrored and "mirrored" in "\n".join(result.lines(CELL.camera))


def test_the_measured_mount_is_written_into_the_workcell_file(tmp_path):
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(DEFAULT_FILE, workcell)
    before = workcell.read_text(encoding="utf-8").splitlines()
    mount = CELL.camera @ from_parts(rot_z(math.pi), (0.0, 0.0, 0.0))
    mount[:3, 3] += (0.004, -0.006, 0.0)
    written = save_mount(workcell, mount, "measured-test")
    assert written == [
        "camera.mount in workcell.yaml",
        "camera.calibration_version in workcell.yaml",
    ]
    cell = load_workcell(workcell, panel=None)
    assert (
        np.allclose(cell.camera, mount, atol=1e-4) and cell.calibration_version == "measured-test"
    )
    after = workcell.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 2  # those two lines, their comments kept
    for a, b in changed:
        assert a.partition("#")[2] == b.partition("#")[2]
    assert (tmp_path / "workcell.yaml.bak").read_text(encoding="utf-8").splitlines() == before
