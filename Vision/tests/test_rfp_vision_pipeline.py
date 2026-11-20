"""Tests for the runtime vision code: geometry, heading, fish list, hold monitor, camera.
Synthetic fish and frames; no camera, model or GPU."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from rfp_vision import fish_heading as fh
from rfp_vision.camera import Frame, OpenCVCamera, open_camera
from rfp_vision.detector import Detection, detections_of, find_weights, list_runs
from rfp_vision.fish_list import FishLocator, match_fish, merge_stable
from rfp_vision.geometry import (
    Intrinsics,
    deproject,
    depth_at,
    fit_plane,
    pixels_on_plane,
    polygon_mask,
    pose_along,
    pose_down,
    to_frame,
)
from rfp_vision.hold_monitor import HoldMonitor
from rfp_vision.overlay import add_status, cup_circles, draw_axes, draw_fish_list
from rfp_vision.settings import Catfish, HoldSettings, StandIn, VisionSettings, load_settings
from rfp_vision.vision_system import VisionSystem

# Width of the stand-in fish along its length, tail to head, as a share of the length
# (measured on its labeled outlines), narrowed to a real catfish's proportions.
PROFILE = 0.75 * np.array(
    [0.04, 0.13, 0.15, 0.13, 0.11, 0.08, 0.08, 0.13, 0.16, 0.18, 0.22, 0.24, 0.25, 0.24, 0.22,
     0.22, 0.23, 0.24, 0.24, 0.25, 0.26, 0.27, 0.27, 0.28, 0.27, 0.27, 0.26, 0.25, 0.24, 0.23,
     0.21, 0.19, 0.18, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.05]
)  # fmt: skip
K = Intrinsics.from_fov(1280, 720, 87.0)
TABLE_Z = 0.0
# Looking straight down from 450 mm: the image shows 0.43 m of the table along X and
# 0.76 m along Y at half a fish's height, so a 0.6 m fish fits only lying along Y.
CAMERA = pose_down((0.40, -0.10, 0.45), math.pi / 2)
PICK_AREA = ((0.20, 0.60), (-0.35, 0.15))
ALONG_Y = math.radians(85)


def fish_outline(x: float, y: float, heading: float, length: float = 0.60):
    """(n, 2) outline of a catfish whose outline's centroid is at (x, y), head toward `heading`."""
    stations = (np.arange(len(PROFILE)) + 0.5) / len(PROFILE) * length
    half = PROFILE * length / 2
    local = np.vstack([np.column_stack([stations, half]), np.column_stack([stations, -half])[::-1]])
    _, centroid, _ = fh.polygon_moments(local)
    c, s = math.cos(heading), math.sin(heading)
    return (local - centroid) @ np.array([[c, s], [-s, c]]) + (x, y)


def render(
    fish: list[tuple],
    thickness: float = 0.10,
    camera=CAMERA,
    depth: bool = True,
    slope: float = 0.0,
    table: float = TABLE_Z,
):
    """A Frame and its detections for fish given as (x, y, heading, confidence) in {B}.
    `slope`: how much the fish's surface rises per meter toward its head. `table`: where the
    table really is (the locator is told TABLE_Z)."""
    t_camera_base = np.linalg.inv(camera)
    color = np.full((K.height, K.width, 3), 90, np.uint8)
    image_depth = np.full((K.height, K.width), camera[2, 3] - table, np.float32)
    detections = []
    for x, y, heading, confidence in fish:
        outline = fish_outline(x, y, heading)
        at_half_height = np.column_stack([outline, np.full(len(outline), table + thickness / 2)])
        pixels = K.project(to_frame(t_camera_base, at_half_height))
        mask = polygon_mask(pixels, color.shape).astype(bool)
        color[mask] = 230
        rows, columns = np.nonzero(mask)
        top = table + thickness
        on_top = pixels_on_plane(K, camera, np.column_stack([columns, rows]), top)[:, :2]
        toward_head = (on_top - (x, y)) @ (math.cos(heading), math.sin(heading))
        image_depth[rows, columns] = camera[2, 3] - top - slope * toward_head
        box = (*pixels.min(axis=0), *pixels.max(axis=0))
        detections.append(Detection(0, confidence, tuple(float(v) for v in box), pixels))
    frame = Frame(color, image_depth if depth else None, K, 0.0, 0.0, 1)
    return frame, detections


def locator(pick_area=PICK_AREA, **changes) -> FishLocator:
    """The synthetic fish here are real catfish sizes."""
    settings = VisionSettings(**{"fish_type": "catfish", **changes})
    return FishLocator(settings, TABLE_Z, pick_area, "model", "calibration")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_pixels_map_to_the_base_frame_and_back():
    assert K.fx == pytest.approx(640 / math.tan(math.radians(43.5)))
    point = np.array([0.52, -0.21, 0.10])
    in_camera = to_frame(np.linalg.inv(CAMERA), point)[0]
    pixel = K.project(in_camera)[0]
    assert np.allclose(to_frame(CAMERA, deproject(K, pixel, in_camera[2]))[0], point, atol=1e-9)
    assert np.allclose(pixels_on_plane(K, CAMERA, pixel, 0.10)[0], point, atol=1e-9)
    # the image's x axis points along the camera's yaw (+Y of {B} here)
    right = pixels_on_plane(K, CAMERA, np.array([[K.cx + 100, K.cy]]), 0.0)[0]
    assert right[1] > CAMERA[1, 3] and right[0] == pytest.approx(CAMERA[0, 3])
    up = pose_down((0, 0, 0.45), 0.0)
    up[:3, :3] = np.eye(3)  # looking up: the rays never reach the table
    assert np.isnan(pixels_on_plane(K, up, np.array([[640, 360]]), 0.0)).all()


def test_depth_is_the_median_of_the_valid_pixels_around_a_point():
    depth = np.full((100, 100), 0.40, np.float32)
    depth[40:60, 40:50] = 0.0  # a hole: half of the window
    depth[50, 55] = 3.0  # out of range
    value, valid = depth_at(depth, (50, 50), 8, (0.10, 0.50))
    assert value == pytest.approx(0.40) and 0.4 < valid < 0.6
    assert depth_at(np.zeros((100, 100), np.float32), (50, 50), 8, (0.10, 0.50)) == (None, 0.0)
    assert depth_at(depth, (500, 500), 8, (0.10, 0.50)) == (None, 0.0)  # outside the image


# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degrees", range(-170, 181, 35))
def test_heading_of_a_catfish_outline_at_any_angle(degrees):
    heading = math.radians(degrees)
    outline = fish_outline(300.0, 200.0, heading, length=400.0)  # pixels
    centroid, axis = fh.body_axis(outline)
    assert np.allclose(centroid, (300, 200), atol=1e-6)
    vote, confidence, cues = fh.head_vote(outline, centroid, axis, (True, True))
    direction = axis if vote > 0 else -axis
    error = math.atan2(direction[1], direction[0]) - heading
    assert abs(math.atan2(math.sin(error), math.cos(error))) < math.radians(2)
    assert confidence > 0.8 and all(abs(v) > 0.3 for v in cues.values())
    estimate = fh.measure_heading(
        fh.FishObservation(outline, 0.9), (640, 480), fh.HeadingSettings()
    )
    assert estimate.source == "measured" and estimate.angle_image == pytest.approx(
        math.atan2(direction[1], direction[0])
    )
    head_side = (estimate.head_px - centroid) @ direction
    assert head_side > 0 > (estimate.tail_px - centroid) @ direction


def test_heading_is_unknown_for_shapes_without_a_head_end_and_for_cut_fish():
    angles = np.linspace(0, 2 * math.pi, 80, endpoint=False)
    ellipse = np.column_stack([300 + 150 * np.cos(angles), 200 + 40 * np.sin(angles)])
    centroid, axis = fh.body_axis(ellipse)
    assert abs(axis[0]) == pytest.approx(1.0)
    assert fh.head_vote(ellipse, centroid, axis, (True, True))[1] < 0.2
    settings = fh.HeadingSettings()
    unknown = fh.measure_heading(fh.FishObservation(ellipse, 0.9), (640, 480), settings)
    assert unknown.source == "unknown" and unknown.angle_image is None
    assert "ambiguous_shape" in unknown.flags

    cut = fish_outline(520.0, 200.0, 0.0, length=400.0)  # the head leaves a 640 px wide frame
    cut[:, 0] = np.minimum(cut[:, 0], 639)
    _, axis = fh.body_axis(cut)
    assert fh.cut_ends(cut, axis, (640, 480), 3.0) == (True, False)
    estimate = fh.measure_heading(fh.FishObservation(cut, 0.9), (640, 480), settings)
    assert estimate.source == "unknown" and "end_cut" in estimate.flags


def test_polygon_moments_of_a_rectangle_in_either_vertex_order():
    rectangle = np.array([[1, 1], [5, 1], [5, 3], [1, 3]], float)
    for points in (rectangle, rectangle[::-1]):
        area, centroid, covariance = fh.polygon_moments(points)
        assert area == pytest.approx(8.0) and np.allclose(centroid, (3, 2))
        assert np.allclose(covariance, [[16 / 12, 0], [0, 4 / 12]])
    with pytest.raises(ValueError):
        fh.polygon_moments(np.array([[0, 0], [1, 1], [2, 2]], float))  # no area


# ---------------------------------------------------------------------------
# Fish list
# ---------------------------------------------------------------------------
def test_fish_are_located_in_the_base_frame_from_depth():
    truth = [(0.30, -0.12, math.radians(95), 0.93), (0.50, -0.08, math.radians(-85), 0.88)]
    frame, detections = render(truth)
    fish_list = locator().locate(frame, detections, CAMERA)
    assert fish_list.vision_ok and fish_list.failure_code == "none" and len(fish_list.valid) == 2
    for (x, y, heading, confidence), fish in zip(truth, fish_list.fish):  # sorted by image x
        assert fish.failure_code == "none" and fish.confidence == confidence
        assert np.allclose(fish.middle, (x, y, 0.10), atol=0.003)
        # the cups go where it balances: its share of the length ahead, toward the head
        share = VisionSettings().catfish.balance_ahead_share
        ahead = share * fish.total_length * np.array([math.cos(heading), math.sin(heading)])
        assert np.allclose(fish.grasp_point[:2], (x, y) + ahead, atol=0.004)
        assert fish.heading_known
        assert abs(math.remainder(fish.heading - heading, math.tau)) < math.radians(2)
        assert fish.total_length == pytest.approx(0.585, abs=0.005)  # the outline's 40 stations
        assert fish.thickness == pytest.approx(0.10, abs=0.002) and fish.depth_measured
        assert np.allclose(fish.grasp_pose[:3, 2], (0, 0, -1))  # approach straight down
        assert np.allclose(
            fish.grasp_pose[:2, 0], (math.cos(heading), math.sin(heading)), atol=0.05
        )
        assert not fish.overlap and fish.bbox[4:] == (
            pytest.approx(0.0, abs=1e-4),  # the table measured around it
            pytest.approx(0.10, abs=0.002),
        )
        assert fish.table_measured and fish.table_z == pytest.approx(TABLE_Z, abs=1e-4)
    record = fish_list.as_record()
    assert record["fish_count_valid"] == 2 and record["model_version"] == "model"
    first = record["fish"][0]
    assert first["target_pose"][:3] == pytest.approx([300, -120, 100], abs=3)  # mm, the middle
    assert first["heading"] == pytest.approx(95, abs=2) and first["total_length"] > 580
    assert fish_list.by_id(2) is fish_list.fish[1] and fish_list.by_id(9) is None


def test_without_depth_fish_lie_on_the_table_plane():
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)], depth=False)
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.vision_ok and not fish.depth_measured and fish.thickness == 0.10
    assert np.allclose(fish.middle, (0.40, -0.10, 0.10), atol=0.003)
    with_depth, _ = render([(0.40, -0.10, ALONG_Y, 0.9)])
    unused = locator(use_depth=False).locate(with_depth, detections, CAMERA).fish[0]
    assert unused.vision_ok and not unused.depth_measured


def test_each_failed_check_gives_its_code_and_clears_the_poses():
    def code(fish_xyh, confidence=0.9, change=lambda frame, detections: None, **settings):
        frame, detections = render([(*fish_xyh, confidence)])
        change(frame, detections)
        fish_list = locator(**settings).locate(frame, detections, CAMERA)
        fish = fish_list.fish[0]
        assert fish_list.failure_code == ("none" if fish.vision_ok else "no_usable_fish")
        if not fish.vision_ok:  # VS-69
            assert fish.grasp_pose is None and fish.target_pose is None and fish.heading is None
        return fish.failure_code

    middle = (0.40, -0.10, ALONG_Y)
    assert code(middle) == "none"
    assert code(middle, confidence=0.4) == "low_confidence"
    assert code((0.40, -0.10, 0.0)) == "out_of_frame"  # lying along X: longer than the view
    outside = ((0.20, 0.35), (-0.35, 0.15))  # a pick area the fish isn't in
    assert code(middle, pick_area=outside) == "geometry_out_of_range"

    def hole(frame, detections):
        frame.depth[300:420, 580:700] = 0.0  # no depth under the grasp point

    assert code(middle, change=hole) == "invalid_depth"

    def blob(frame, detections):
        angles = np.linspace(0, 2 * math.pi, 60, endpoint=False)
        ellipse = [640 + 330 * np.cos(angles), 360 + 140 * np.sin(angles)]  # cups fit on it
        detections[0].polygon = np.column_stack(ellipse)

    assert code(middle, change=blob, require_heading=True) == "wrong_heading"
    assert code(middle, change=blob, check_size=False) == "none"  # by default it's picked

    def sliver(frame, detections):
        detections[0].polygon = np.array([[600, 350], [700, 352], [700, 356], [600, 354]], float)

    # too narrow for the cups; with the size check off that is still caught
    assert code(middle, change=sliver, require_heading=False, check_size=False) == (
        "grasp_point_off_body"
    )

    def lumpy(frame, detections):  # 3 cm steps every ~6 cm: no spot where all six cups seal
        rows, columns = np.indices(frame.depth.shape)
        on_fish = frame.depth < frame.depth.max() - 0.01
        frame.depth[on_fish] -= 0.03 * (((rows // 90) + (columns // 90)) % 2)[on_fish]

    assert code(middle, change=lumpy, strict_seal_check=True) == "grasp_point_off_body"
    assert code(middle, catfish=Catfish(length_range_m=(0.2, 0.5))) == "geometry_out_of_range"
    big = StandIn(length_range_m=(0.80, 1.00))  # only the big prints
    assert code(middle, fish_type="stand_in", stand_in=big) == "geometry_out_of_range"
    # VS-68: of several failed checks, the first in the table's order is the code
    frame, detections = render([(0.40, -0.10, 0.0, 0.3)])
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.failure_code == "low_confidence" and "out_of_frame" in fish.failures


def test_the_cups_go_only_where_all_six_can_seal():
    grid = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 81), np.linspace(-0.06, 0.06, 49)), -1)
    xy = grid.reshape(-1, 2)
    axis, at = np.array([1.0, 0.0]), np.zeros(2)

    def reach(z, **settings):
        return locator(**settings)._cup_contact(np.column_stack([xy, z]), at, axis)

    sealed, (normal, height) = reach(0.10 + 0 * xy[:, 0])  # flat
    assert sealed and np.allclose(normal, (0, 0, 1)) and height == pytest.approx(0.10)
    sealed, (normal, height) = reach(0.10 + 0.2 * xy[:, 0])  # sloping: the cups lean onto it
    assert sealed and math.degrees(math.acos(normal[2])) == pytest.approx(11.3, abs=0.1)
    assert not reach(0.10 + 0.2 * xy[:, 0], align_to_surface=False)[0]  # straight down: 40 mm
    long, wide = (v / 2 - 0.02 for v in VisionSettings().cup_footprint_m)  # a corner cup
    corner = np.hypot(xy[:, 0] - long, xy[:, 1] - wide) < 0.02
    # the body higher under it: the cups lean toward it and the bellows take up the rest
    assert reach(0.10 + 0.015 * corner)[0]
    assert not reach(0.10 + 0.04 * corner)[0]  # past their stroke
    fin = corner & (np.abs(xy[:, 1] - wide) < 0.004)  # a thin fin standing up under it
    strict = {"strict_seal_check": True}
    assert not reach(0.10 + 0.015 * fin, **strict)[0]  # the cup's lip would rest on it
    assert reach(0.10 + 0.015 * fin, **strict, cup_lip_conform_m=1.0)[0]  # a lip that conforms
    assert reach(0.10 + 0.015 * fin)[0]  # by default: tried, the hold check tells
    assert not reach(0.10 + 0.04 * corner, strict_seal_check=False)[0]  # the stroke, always
    few = np.column_stack([xy, 0.1 + 0 * xy[:, 0]])[:100]  # no depth under the cups
    assert locator()._cup_contact(few, at, axis) == (None, None)  # it can't tell: the outline does
    some = np.column_stack([xy, 0.1 + 0.015 * fin])[np.abs(xy[:, 1]) > 0.0]  # a hole between rows
    seen = locator(min_cups_measured=6, **strict)._cup_contact(some, at, axis)
    assert seen[0] is False  # the fin, seen


def test_the_cups_seal_on_a_narrow_round_fish_but_not_on_a_fin():
    # A real fish 110 mm wide and round: the outer cups sit on its sloping shoulders, which
    # their lips follow. Against the six cups' common plane the slope alone took 5 of the
    # 6 mm a lip was allowed, and the D405's noise the rest; against each cup's own plane
    # it is under 1 mm
    grid = np.stack(np.meshgrid(np.linspace(-0.12, 0.12, 97), np.linspace(-0.055, 0.055, 89)), -1)
    xy = grid.reshape(-1, 2)
    body = 0.10 + 0.04 * np.sqrt(np.clip(1 - (xy[:, 1] / 0.055) ** 2, 0, None))
    body += np.random.default_rng(3).normal(0.0, 0.002, len(body))  # depth noise
    axis, at = np.array([1.0, 0.0]), np.zeros(2)
    strict = locator(strict_seal_check=True)
    sealed, plane = strict._cup_contact(np.column_stack([xy, body]), at, axis)
    assert sealed and plane is not None
    outline = np.column_stack([0.30 * np.cos(np.linspace(0, 6.28, 200)),
                               0.055 * np.sin(np.linspace(0, 6.28, 200))])  # fmt: skip
    assert locator()._footprint_fits(outline, at, axis)  # and they are all on it
    long, wide = (v / 2 - 0.02 for v in VisionSettings().cup_footprint_m)
    fin = (np.hypot(xy[:, 0] - long, xy[:, 1] - wide) < 0.02) & (np.abs(xy[:, 1] - wide) < 0.004)
    assert not strict._cup_contact(np.column_stack([xy, body + 0.015 * fin]), at, axis)[0]


def test_a_table_lower_than_the_workcell_says_doesnt_shrink_the_fish():
    # The real cell's table 80 mm below table.z: the camera is that much higher over the fish
    # than the arm thinks. Measured from table.z, every fish came out ~8 % short and 80 mm thin.
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)], table=-0.08)
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.vision_ok and fish.table_measured
    assert fish.table_z == pytest.approx(-0.08, abs=0.002)
    assert fish.thickness == pytest.approx(0.10, abs=0.003)
    assert fish.total_length == pytest.approx(0.585, abs=0.006)
    assert np.allclose(fish.middle, (0.40, -0.10, 0.02), atol=0.004)
    blind = locator(measure_table=False).locate(frame, detections, CAMERA).fish[0]
    assert blind.failure_code == "geometry_out_of_range" and not blind.table_measured
    assert "length 0.5" in blind.detail and "catfish: 0.54-0.82" in blind.detail
    assert "height 0.02 m" in blind.detail  # measured from table.z
    far = locator(table_search_m=0.05).locate(frame, detections, CAMERA).fish[0]
    assert not far.table_measured  # farther from table.z than it may be: not the table


def test_the_cups_are_checked_as_six_circles_not_their_bounding_box():
    angles = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    slim = np.column_stack([0.30 * np.cos(angles), 0.052 * np.sin(angles)])  # 104 mm wide
    axis, at = np.array([1.0, 0.0]), np.zeros(2)
    # the footprint's corners (100, 50) mm lie outside it, but every cup is well on it
    corner = (0.10 / 0.30) ** 2 + (0.05 / 0.052) ** 2
    assert corner > 1.0 and locator()._footprint_fits(slim, at, axis)
    assert not locator(cup_overhang_m=-0.01)._footprint_fits(slim, at, axis)  # 10 mm inside
    narrow = slim * (1.0, 0.8)  # 83 mm wide: the cups hang off it
    assert not locator()._footprint_fits(narrow, at, axis)


def test_a_failed_fish_says_what_was_measured():
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)])
    small = locator(catfish=Catfish(length_range_m=(0.2, 0.5))).locate(frame, detections, CAMERA)
    assert small.fish[0].detail == "length 0.58 m (catfish: 0.20-0.50)"
    detections[0].polygon = np.array([[600, 350], [700, 352], [700, 356], [600, 354]], float)
    sliver = locator(require_heading=False, check_size=False).locate(frame, detections, CAMERA)
    assert sliver.fish[0].failure_code == "grasp_point_off_body"
    assert sliver.fish[0].detail.startswith("the six cups don't fit on it")
    cut = locator().locate(*render([(0.40, -0.10, 0.0, 0.9)]), CAMERA).fish[0]
    assert cut.detail == "cut off at the image's edge"
    assert render([])[0] is not None and locator().locate(*render([]), CAMERA).fish == []


def test_the_cups_stop_behind_a_fin_on_the_way_to_the_balance_point():
    x, y, heading = 0.40, -0.10, ALONG_Y
    toward_head = np.array([math.cos(heading), math.sin(heading)])

    def ahead_of_middle(fin_at=None) -> float:
        frame, detections = render([(x, y, heading, 0.9)])
        if fin_at is not None:  # a fin 10 mm thick and 30 mm tall across the body
            rows, columns = np.nonzero(frame.depth < frame.depth.max() - 0.01)
            on_top = pixels_on_plane(K, CAMERA, np.column_stack([columns, rows]), 0.10)
            along = (on_top[:, :2] - (x, y)) @ toward_head
            fin = np.abs(along - fin_at) < 0.005
            frame.depth[rows[fin], columns[fin]] -= 0.03
        fish = locator().locate(frame, detections, CAMERA).fish[0]
        assert fish.vision_ok, fish.failures
        return float((fish.grasp_point[:2] - fish.middle[:2]) @ toward_head)

    balance = Catfish().balance_ahead_share * 0.6
    assert ahead_of_middle() == pytest.approx(balance, abs=0.01)  # where it balances
    # 110 mm ahead the fin would be under the front cups (80 mm ahead of the grasp point,
    # 40 mm across): they move back toward the middle until it is clear of them
    assert ahead_of_middle(fin_at=0.11) < 0.015


def test_a_rejected_fish_keeps_where_it_was_seen_for_a_closer_look():
    frame, detections = render([(0.40, -0.10, 0.0, 0.9)])  # along X: longer than the view
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert "out_of_frame" in fish.failures and fish.grasp_pose is None and fish.middle is None
    x, y, axis = fish.seen_at  # only part of it is seen: x is off toward the seen part
    assert abs(x - 0.40) < 0.10 and abs(y - (-0.10)) < 0.01
    assert abs(math.remainder(axis, math.pi)) < math.radians(3)
    with pytest.raises(ValueError, match="fish_type"):
        VisionSettings(fish_type="salmon")


def test_empty_frames_and_touching_fish():
    frame, _ = render([])
    empty = locator().locate(frame, [], CAMERA)
    assert not empty.vision_ok and empty.failure_code == "no_fish" and empty.fish == []
    frame, detections = render([(0.35, -0.10, ALONG_Y, 0.9), (0.45, -0.10, ALONG_Y, 0.9)])
    touching = locator().locate(frame, detections, CAMERA)
    assert [fish.overlap for fish in touching.fish] == [True, True]
    frame, detections = render([(0.27, -0.10, ALONG_Y, 0.9), (0.53, -0.10, ALONG_Y, 0.9)])
    apart = locator().locate(frame, detections, CAMERA)
    assert [fish.overlap for fish in apart.fish] == [False, False]


def test_a_weak_partial_detection_does_not_make_a_fish_look_crowded():
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)])
    head = detections[0].polygon[: len(detections[0].polygon) // 4]  # part of the same fish
    for confidence, crowded in ((0.3, False), (0.8, True)):
        extra = Detection(0, confidence, detections[0].box, head)
        fish = locator().locate(frame, [detections[0], extra], CAMERA).fish
        whole = max(fish, key=lambda f: f.confidence if f.confidence == 0.9 else -1)
        assert whole.confidence == 0.9 and whole.overlap is crowded


def test_a_fish_must_be_at_the_same_place_in_every_frame_of_an_observation():
    settings = VisionSettings()

    def lists(*positions):
        made = []
        for x, heading in positions:
            frame, detections = render([(x, -0.10, heading, 0.9)])
            made.append(locator().locate(frame, detections, CAMERA))
        return made

    still = merge_stable(lists((0.40, 1.5), (0.401, 1.51), (0.40, 1.5)), settings)
    assert still.fish[0].vision_ok
    moved = merge_stable(lists((0.40, 1.5), (0.42, 1.5), (0.42, 1.5)), settings)
    assert moved.fish[0].failure_code == "unstable" and moved.fish[0].grasp_pose is None
    turned = merge_stable(lists((0.40, 1.5), (0.40, 1.5), (0.40, 1.7)), settings)
    assert turned.fish[0].failure_code == "unstable"
    new = lists((0.40, 1.5))[0]
    assert match_fish(new.fish[0], still.fish, 0.01) is still.fish[0]
    assert match_fish(new.fish[0], moved.fish, 0.01) is None  # failed fish have no position


class FakeCamera:
    name = "fake"
    has_depth = True

    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def read(self, timeout_s: float = 2.0):
        frame, detections = self.frames.pop(0)
        self.detections = detections
        return frame

    def close(self):
        self.closed = True


def test_vision_system_observes_over_several_frames():
    scene = [render([(0.40, -0.10, ALONG_Y, 0.9)]) for _ in range(3)]
    camera = FakeCamera(scene)
    detector = SimpleNamespace(detect=lambda image: camera.detections)
    vision = VisionSystem(camera, detector, locator(), VisionSettings())
    seen = []
    observation = vision.observe(CAMERA, on_frame=seen.append)
    assert len(seen) == 3 and camera.frames == [] and observation.frame is seen[-1]
    assert observation.fish_list.valid[0].fish_id == 1 and len(observation.detections) == 1
    image = draw_fish_list(observation.frame.color, observation.fish_list, K, chosen_id=1)
    assert image.shape == observation.frame.color.shape and (image != observation.frame.color).any()
    assert add_status(image, ["state", "more"]).shape[0] > image.shape[0]
    vision.close()
    assert camera.closed


# ---------------------------------------------------------------------------
# Hold monitor
# ---------------------------------------------------------------------------
def held_frame(distance: float | None, frame_id: int = 1, gripper: bool = True) -> Frame:
    """What the camera on the gripper sees: the table 0.45 m away, a held fish closer."""
    depth = np.full((720, 1280), 0.45, np.float32)
    if gripper:
        depth[:, :100] = 0.08  # a cup at the edge of the view, always there
    if distance is not None:
        depth[200:520, 300:1000] = distance
    return Frame(np.zeros((720, 1280, 3), np.uint8), depth, K, 0.0, 0.0, frame_id)


def test_depth_cue_tells_a_held_fish_from_a_dropped_one():
    monitor = HoldMonitor(HoldSettings())
    monitor.set_background(held_frame(None))
    assert monitor.start(held_frame(None)) == [] and not monitor.active  # nothing but the gripper
    assert monitor.start(held_frame(0.12)) == ["depth"] and monitor.active
    swinging = monitor.update(held_frame(0.14))  # bellows stretch: still the fish
    assert swinging.held and not swinging.dropped and swinging.depth_share > 0.9
    for i in range(3):
        status = monitor.update(held_frame(None, i))  # the fish stayed on the table
        assert status.held is False and status.dropped == (i == 2)
    assert "not seen" in str(status) and "close surface" in str(status)
    assert monitor.update(held_frame(0.12)).dropped is False  # seen again: the count restarts
    monitor.stop()
    assert not monitor.active and monitor.cues == []


def test_under_a_thin_fish_the_table_is_not_taken_for_the_held_surface():
    # Cups on a 0.12 m thin fish: the camera is 0.08 m from its top and 0.20 m from the
    # table, nearer than near_m. Taken for held surface, the table went away on the lift and
    # the fish, still on the cups, counted as dropped (mid-carry, vacuum off)
    def frame(table: float, fish: float = 0.08):
        depth = np.full((720, 1280), table, np.float32)
        depth[:, :100] = 0.07  # a cup at the edge of the view
        depth[250:470, 400:900] = fish
        return Frame(np.zeros((720, 1280, 3), np.uint8), depth, K, 0.0, 0.0, 1)

    blind = HoldMonitor(HoldSettings())
    blind.set_background(frame(0.45, fish=0.45))
    blind.start(frame(0.20))
    assert blind.update(frame(0.45)).depth_share < 0.5  # lifted: the table left the region
    monitor = HoldMonitor(HoldSettings())
    monitor.set_background(frame(0.45, fish=0.45))
    assert monitor.start(frame(0.20), table_distance=0.20) == ["depth"]
    lifted = monitor.update(frame(0.45))
    assert lifted.held and lifted.depth_share > 0.9
    monitor.stop()
    assert monitor.start(frame(0.45, fish=0.45), table_distance=0.45) == []  # nothing held


def test_the_gripper_in_view_does_not_count_as_a_fish():
    monitor = HoldMonitor(HoldSettings())  # no background taken
    assert monitor.start(held_frame(0.12)) == ["depth"]
    assert monitor.update(held_frame(None)).depth_share < 0.3  # only the cup is left


class OutlineDetector:
    """Detects one rectangle per call, from a list."""

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, image):
        box = self.boxes.pop(0)
        if box is None:
            return []
        x0, y0, x1, y1 = box
        polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        return [Detection(0, 0.9, box, polygon)]


def test_mask_cue_follows_the_outline_and_either_cue_keeps_the_fish():
    there, shifted, small = (300, 200, 1000, 520), (330, 215, 1030, 535), (560, 320, 700, 400)
    colour_only = Frame(np.zeros((720, 1280, 3), np.uint8), None, K, 0.0, 0.0, 1)
    monitor = HoldMonitor(
        HoldSettings(lost_frames=2), OutlineDetector([there, shifted, small, None])
    )
    assert monitor.start(colour_only) == ["mask"]
    assert monitor.update(colour_only).mask_iou > 0.8  # sagged a little: held
    lost = monitor.update(colour_only)  # the fish far below: its outline shrank
    assert lost.held is False and not lost.dropped and lost.mask_iou < 0.2
    assert monitor.update(colour_only).dropped  # no detection at all

    both = HoldMonitor(HoldSettings(), OutlineDetector([there, None, None]))
    assert both.start(held_frame(0.12)) == ["mask", "depth"]
    status = both.update(held_frame(0.12))  # the surface is there: the model isn't even asked
    assert status.held and status.mask_iou is None and status.depth_share > 0.9
    assert len(both._detector.boxes) == 2
    assert both.update(held_frame(None)).held is False

    assert HoldMonitor(HoldSettings(use_mask=False), OutlineDetector([])).start(colour_only) == []
    unknown = HoldMonitor(HoldSettings()).update(colour_only)
    assert unknown.held is None and not unknown.dropped and "unknown" in str(unknown)


# ---------------------------------------------------------------------------
# Detector, settings, camera
# ---------------------------------------------------------------------------
def test_detections_come_largest_first_without_slivers():
    big = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], np.float32)
    small = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    boxes = SimpleNamespace(
        xyxy=np.array([[0, 0, 10, 10], [0, 0, 200, 100]], np.float32),
        cls=np.array([0.0, 0.0]),
        conf=np.array([0.6, 0.9]),
        __len__=lambda: 2,
    )

    class Boxes(SimpleNamespace):
        def __len__(self):
            return 2

    result = SimpleNamespace(boxes=Boxes(**vars(boxes)), masks=SimpleNamespace(xy=[small, big]))
    found = detections_of(result)
    assert [d.conf for d in found] == [0.9, 0.6] and found[0].box == (0.0, 0.0, 200.0, 100.0)
    assert [d.conf for d in detections_of(result, min_area_px=500)] == [0.9]
    assert detections_of(SimpleNamespace(boxes=None, masks=None)) == []


def test_model_runs_are_found(tmp_path):
    run = tmp_path / "fish_seg_2026"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"")
    (tmp_path / "no_weights").mkdir()
    assert list_runs(tmp_path) == [run] and find_weights(run) == run / "weights" / "last.pt"
    assert find_weights(run / "weights" / "last.pt") == run / "weights" / "last.pt"
    with pytest.raises(FileNotFoundError):
        find_weights(tmp_path / "no_weights")


def test_settings_from_a_file_reject_unknown_names(tmp_path):
    assert load_settings(panel=None) == VisionSettings()
    custom = tmp_path / "vision.yaml"
    custom.write_text("min_confidence: 0.7\nhold: {lost_frames: 5}\ndepth_range_m: [0.1, 0.6]\n")
    settings = load_settings(custom, panel=None)
    assert settings.min_confidence == 0.7 and settings.hold.lost_frames == 5
    assert settings.depth_range_m == (0.1, 0.6) and settings.hold.min_iou == 0.5
    custom.write_text("min_confidense: 0.7\n")
    with pytest.raises(ValueError, match="min_confidense"):
        load_settings(custom, panel=None)
    custom.write_text("hold: {lost_frame: 5}\n")
    with pytest.raises(ValueError, match="lost_frame"):
        load_settings(custom, panel=None)
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")


def test_video_files_play_in_a_loop_as_a_color_camera(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 200, (160, 120))
    for value in (10, 120, 240):
        writer.write(np.full((120, 160, 3), value, np.uint8))
    writer.release()
    camera = open_camera(str(video), hfov_deg=90.0)
    assert isinstance(camera, OpenCVCamera) and not camera.has_depth
    frames = [camera.read() for _ in range(5)]
    assert [f.frame_id for f in frames] == [1, 2, 3, 4, 5] and frames[0].depth is None
    assert frames[0].size == (160, 120) and frames[0].intrinsics.fx == pytest.approx(80.0)
    assert abs(int(frames[3].color[0, 0, 0]) - int(frames[0].color[0, 0, 0])) < 15  # started over
    camera.close()
    from rfp_vision.camera import CameraError

    with pytest.raises(CameraError):
        open_camera(str(Path(tmp_path) / "missing.mp4"))


# ---------------------------------------------------------------------------
# The fish's surface under the cups
# ---------------------------------------------------------------------------
def test_plane_fit_and_poses_laid_on_a_surface():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.05, 0.05, (200, 2))
    points = np.column_stack([xy, 0.3 + 0.2 * xy[:, 0] + rng.normal(0, 0.001, 200)])
    normal, middle, rms = fit_plane(points)
    assert np.allclose(normal, np.array([-0.2, 0, 1]) / math.hypot(0.2, 1), atol=0.01)
    assert rms == pytest.approx(0.001, abs=0.0005) and middle[2] == pytest.approx(0.3, abs=0.005)
    pose = pose_along((1, 2, 3), math.radians(90), -normal)
    assert np.allclose(pose[:3, 2], -normal) and np.allclose(pose[:3, 3], (1, 2, 3))
    assert np.allclose(pose[:3, 0], (0, 1, 0), atol=1e-3)  # along Y: not tilted by a slope along X
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(pose_along((0, 0, 0), 0.3, (0, 0, -1)), pose_down((0, 0, 0), 0.3))


def test_surface_slope_tilts_the_grasp_and_helps_tell_head_from_tail():
    middle = (0.40, -0.10, ALONG_Y, 0.9)
    toward_head = np.array([math.cos(ALONG_Y), math.sin(ALONG_Y)])
    tilt = math.atan(0.10)  # a surface rising 10 mm per 100 mm toward the head

    frame, detections = render([middle], slope=0.10)
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.vision_ok and fish.surface_tilt == pytest.approx(tilt, abs=0.01)
    assert fish.middle[2] == pytest.approx(0.10, abs=0.003)
    ahead = (fish.grasp_point[:2] - fish.middle[:2]) @ toward_head  # the cups toward the head,
    assert ahead > 0.03 and fish.thickness == pytest.approx(0.10 + 0.10 * ahead, abs=0.003)  # up
    approach = fish.grasp_pose[:3, 2]  # square onto the surface: leaning toward the head
    assert approach[2] == pytest.approx(-math.cos(tilt), abs=0.01)
    assert approach[:2] @ toward_head == pytest.approx(math.sin(tilt), abs=0.01)
    assert fish.grasp_pose[2, 0] == pytest.approx(math.sin(tilt), abs=0.01)  # x climbs to the head
    assert fish.heading_known and fish.heading_confidence > 0.9  # shape and slope agree

    level = locator(align_to_surface=False).locate(frame, detections, CAMERA).fish[0]
    assert np.allclose(level.grasp_pose[:3, 2], (0, 0, -1)) and level.surface_tilt > 0.05
    frame, detections = render([middle], slope=0.50)  # steeper than the cups may lean:
    steep = locator().locate(frame, detections, CAMERA).fish[0]  # pressed straight down, the
    assert steep.failure_code == "grasp_point_off_body"  # cup rows' heights differ by 80 mm
    # measured where the cups go; at the middle that is the slope itself (toward the narrower
    # head the cups' footprint also takes in the body's rounded sides)
    at_middle = locator(catfish=Catfish(balance_ahead_share=0.0))
    tilt = at_middle.locate(frame, detections, CAMERA).fish[0].surface_tilt
    assert math.degrees(tilt) == pytest.approx(26.6, abs=1.0)

    # sloping down toward what the outline says is the head: the two disagree, so no guess
    frame, detections = render([middle], slope=-0.10)
    confused = locator(require_heading=True).locate(frame, detections, CAMERA).fish[0]
    assert confused.failure_code == "wrong_heading" and confused.heading_confidence < 0.6
    unsure = locator().locate(frame, detections, CAMERA).fish[0]  # by default: picked anyway
    assert unsure.vision_ok and not unsure.heading_known
    flat, detections = render([middle])  # no slope: the outline decides alone
    assert locator().locate(flat, detections, CAMERA).fish[0].heading_confidence > 0.8

    centroid, axis = fh.body_axis(fish_outline(0.0, 0.0, 0.0))
    outline = fish_outline(0.0, 0.0, 0.0)
    assert fh.head_vote(outline, centroid, axis, (True, True), 0.03)[2]["height"] == 1.0
    assert fh.head_vote(outline, centroid, axis, (True, True), -0.01)[2]["height"] == -0.5
    assert "height" not in fh.head_vote(outline, centroid, axis, (True, True))[2]


def test_the_overlay_shows_the_cups_along_the_fish_and_the_arm_s_axes():
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)])
    fish = locator().locate(frame, detections, CAMERA)
    image = draw_fish_list(frame.color, fish, K)
    rims = cup_circles(fish.fish[0].grasp_pose)
    assert rims.shape == (6, 16, 3)
    bare = copy.deepcopy(fish)
    bare.fish[0].grasp_pose = None  # the same drawing without the grasp
    drawn = np.abs(image.astype(int) - draw_fish_list(frame.color, bare, K)).sum(axis=2) > 0
    for rim in rims:  # each cup's rim drawn where the grasp puts it
        pixels = K.project(to_frame(np.linalg.inv(CAMERA), rim))
        u, v = np.round(pixels[0]).astype(int)  # a point of the rim
        assert drawn[v - 2 : v + 3, u - 2 : u + 3].any()
    across = rims[:, :, :2].mean(axis=1) - fish.fish[0].grasp_point[:2]
    along = np.array([math.cos(ALONG_Y), math.sin(ALONG_Y)])
    assert np.abs(across @ along).max() == pytest.approx(0.08, abs=1e-6)  # 3 rows along it
    axes = draw_axes(frame.color.copy(), frame, CAMERA, TABLE_Z)
    red = np.all(axes == (40, 40, 230), axis=2)
    assert red.any() and not np.array_equal(axes, frame.color)
