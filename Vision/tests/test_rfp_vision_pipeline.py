"""Tests for the runtime vision code: geometry, heading, fish list, hold monitor, camera.
Synthetic fish and frames; no camera, model or GPU."""

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
    pixels_on_plane,
    polygon_mask,
    pose_down,
    to_frame,
)
from rfp_vision.hold_monitor import HoldMonitor
from rfp_vision.overlay import add_status, draw_fish_list
from rfp_vision.settings import HoldSettings, VisionSettings, load_settings
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


def fish_outline(x: float, y: float, heading: float, length: float = 0.60) -> np.ndarray:
    """(n, 2) outline of a catfish whose outline's centroid is at (x, y), head toward `heading`."""
    stations = (np.arange(len(PROFILE)) + 0.5) / len(PROFILE) * length
    half = PROFILE * length / 2
    local = np.vstack([np.column_stack([stations, half]), np.column_stack([stations, -half])[::-1]])
    _, centroid, _ = fh.polygon_moments(local)
    c, s = math.cos(heading), math.sin(heading)
    return (local - centroid) @ np.array([[c, s], [-s, c]]) + (x, y)


def render(fish: list[tuple], thickness: float = 0.10, camera=CAMERA, depth: bool = True):
    """A Frame and its detections for fish given as (x, y, heading, confidence) in {B}."""
    t_camera_base = np.linalg.inv(camera)
    color = np.full((K.height, K.width, 3), 90, np.uint8)
    image_depth = np.full((K.height, K.width), camera[2, 3] - TABLE_Z, np.float32)
    detections = []
    for x, y, heading, confidence in fish:
        outline = fish_outline(x, y, heading)
        at_half_height = np.column_stack([outline, np.full(len(outline), TABLE_Z + thickness / 2)])
        pixels = K.project(to_frame(t_camera_base, at_half_height))
        mask = polygon_mask(pixels, color.shape).astype(bool)
        color[mask] = 230
        image_depth[mask] = camera[2, 3] - TABLE_Z - thickness
        box = (*pixels.min(axis=0), *pixels.max(axis=0))
        detections.append(Detection(0, confidence, tuple(float(v) for v in box), pixels))
    frame = Frame(color, image_depth if depth else None, K, 0.0, 0.0, 1)
    return frame, detections


def locator(pick_area=PICK_AREA, **changes) -> FishLocator:
    return FishLocator(VisionSettings(**changes), TABLE_Z, pick_area, "model", "calibration")


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
        assert np.allclose(fish.grasp_point, (x, y, 0.10), atol=0.003)
        assert fish.heading_known
        assert abs(math.remainder(fish.heading - heading, math.tau)) < math.radians(2)
        assert fish.total_length == pytest.approx(0.585, abs=0.005)  # the outline's 40 stations
        assert fish.thickness == pytest.approx(0.10, abs=0.002) and fish.depth_measured
        assert np.allclose(fish.grasp_pose[:3, 2], (0, 0, -1))  # approach straight down
        assert np.allclose(
            fish.grasp_pose[:2, 0], (math.cos(heading), math.sin(heading)), atol=0.05
        )
        assert not fish.overlap and fish.bbox[4:] == (0.0, pytest.approx(0.10, abs=0.002))
    record = fish_list.as_record()
    assert record["fish_count_valid"] == 2 and record["model_version"] == "model"
    first = record["fish"][0]
    assert first["grasp_pose"][:3] == pytest.approx([300, -120, 100], abs=3)  # mm
    assert first["heading"] == pytest.approx(95, abs=2) and first["total_length"] > 580
    assert fish_list.by_id(2) is fish_list.fish[1] and fish_list.by_id(9) is None


def test_without_depth_fish_lie_on_the_table_plane():
    frame, detections = render([(0.40, -0.10, ALONG_Y, 0.9)], depth=False)
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.vision_ok and not fish.depth_measured and fish.thickness == 0.10
    assert np.allclose(fish.grasp_point, (0.40, -0.10, 0.10), atol=0.003)
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
        ellipse = [640 + 300 * np.cos(angles), 360 + 100 * np.sin(angles)]
        detections[0].polygon = np.column_stack(ellipse)

    assert code(middle, change=blob) == "wrong_heading"
    assert code(middle, change=blob, require_heading=False, check_size=False) == "none"

    def sliver(frame, detections):
        detections[0].polygon = np.array([[600, 350], [700, 352], [700, 356], [600, 354]], float)

    # too narrow for the cups; with the size check off that is still caught
    assert code(middle, change=sliver, require_heading=False, check_size=False) == (
        "grasp_point_off_body"
    )
    assert code(middle, length_range_m=(0.2, 0.5)) == "geometry_out_of_range"
    # VS-68: of several failed checks, the first in the table's order is the code
    frame, detections = render([(0.40, -0.10, 0.0, 0.3)])
    fish = locator().locate(frame, detections, CAMERA).fish[0]
    assert fish.failure_code == "low_confidence" and "out_of_frame" in fish.failures


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
    status = both.update(held_frame(0.12))  # the model misses a frame, the surface is there
    assert status.held and status.mask_iou == 0.0 and status.depth_share > 0.9
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


def test_settings_file_matches_the_defaults_and_rejects_unknown_names(tmp_path):
    assert load_settings() == VisionSettings()  # Vision/configs/vision.yaml
    custom = tmp_path / "vision.yaml"
    custom.write_text("min_confidence: 0.7\nhold: {lost_frames: 5}\ndepth_range_m: [0.1, 0.6]\n")
    settings = load_settings(custom)
    assert settings.min_confidence == 0.7 and settings.hold.lost_frames == 5
    assert settings.depth_range_m == (0.1, 0.6) and settings.hold.min_iou == 0.5
    custom.write_text("min_confidense: 0.7\n")
    with pytest.raises(ValueError, match="min_confidense"):
        load_settings(custom)
    custom.write_text("hold: {lost_frame: 5}\n")
    with pytest.raises(ValueError, match="lost_frame"):
        load_settings(custom)
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
