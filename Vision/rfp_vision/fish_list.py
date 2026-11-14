"""
The fish list (VS-40 to VS-60, IF-20): detections in one frame turned into fish with a
position, heading and size in the arm base frame {B}, each checked and flagged.

Meters and radians here; `as_record()` gives the IF-1/IF-2 fields in mm and degrees for logs.
Without a pose model the grasp point is the centroid of the outline (on the stand-in fish
49% of the length from the snout; VS-33 asks for 40-50%), and the approach is straight down.

Not implemented yet: `covered` (VS-25), keypoints (VS-30, VS-62), surface normals (VS-44).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from rfp_vision import fish_heading
from rfp_vision.camera import Frame
from rfp_vision.detector import Detection
from rfp_vision.geometry import (
    deproject,
    depth_at,
    inside_polygon,
    pixels_on_plane,
    pose_down,
    to_frame,
    touches_border,
)
from rfp_vision.settings import VisionSettings

# Table VS-2, in the order the checks apply (VS-68: the first failed check is reported).
FAILURE_ORDER = (
    "low_confidence",
    "out_of_frame",
    "covered",
    "invalid_keypoints",
    "wrong_heading",
    "grasp_point_off_body",
    "geometry_out_of_range",
    "invalid_depth",
    "unstable",
)

Area = tuple[tuple[float, float], tuple[float, float]]  # ((x min, x max), (y min, y max)) in {B}


@dataclass
class Fish:
    """One fish of a fish list (Table IF-2)."""

    fish_id: int
    vision_ok: bool
    failure_code: str  # "none" or a code of Table VS-2
    confidence: float
    grasp_pose: np.ndarray | None  # 4x4 in {B}: on the fish, z = approach, x = along the body
    target_pose: np.ndarray | None  # 4x4 in {B}: centroid of the outline, on the fish
    heading: float | None  # rad, tail -> head, from +X toward +Y (IF-12)
    total_length: float  # m; 0 if unknown
    width_at_grasp: float
    thickness: float
    bbox: tuple[float, float, float, float, float, float] | None  # x, x, y, y, z, z (min, max)
    overlap: bool
    mask_polygon: np.ndarray  # (n, 2) px
    heading_known: bool = False  # False: `heading` is the body axis, either end may be the head
    heading_confidence: float = 0.0
    depth_measured: bool = False  # False: placed on the table plane with the default thickness
    failures: tuple[str, ...] = ()  # every failed check, in table order

    @property
    def grasp_point(self) -> np.ndarray | None:
        return None if self.grasp_pose is None else self.grasp_pose[:3, 3]

    def as_record(self) -> dict:
        """The IF-2 fields: mm, degrees, poses as [X, Y, Z, roll, pitch, yaw]."""

        def pose(t):
            if t is None:
                return None
            roll = math.atan2(t[2, 1], t[2, 2])
            pitch = math.atan2(-t[2, 0], math.hypot(t[0, 0], t[1, 0]))
            yaw = math.atan2(t[1, 0], t[0, 0])
            angles = [round(math.degrees(a), 2) for a in (roll, pitch, yaw)]
            return [round(float(v) * 1000, 1) for v in t[:3, 3]] + angles

        return {
            "fish_id": self.fish_id,
            "vision_ok": self.vision_ok,
            "failure_code": self.failure_code,
            "confidence": round(self.confidence, 2),
            "grasp_pose": pose(self.grasp_pose),
            "target_pose": pose(self.target_pose),
            "heading": None if self.heading is None else round(math.degrees(self.heading), 1),
            "total_length": round(self.total_length * 1000, 1),
            "width_at_grasp": round(self.width_at_grasp * 1000, 1),
            "thickness": round(self.thickness * 1000, 1),
            "bbox": None if self.bbox is None else [round(v * 1000, 1) for v in self.bbox],
            "overlap": self.overlap,
        }


@dataclass
class FishList:
    """Every fish detected in one observation, valid and failed (Table IF-1)."""

    frame_id: int
    stamp: float  # time.monotonic() of the last frame used
    wall_time: float
    camera_pose: np.ndarray  # 4x4 T_base_camera at capture (VS-13)
    fish: list[Fish] = field(default_factory=list)
    model_version: str = ""
    calibration_version: str = ""

    @property
    def valid(self) -> list[Fish]:
        return [f for f in self.fish if f.vision_ok]

    @property
    def vision_ok(self) -> bool:
        return bool(self.valid)

    @property
    def failure_code(self) -> str:
        """none, no_fish or no_usable_fish (VS-67)."""
        if not self.fish:
            return "no_fish"
        return "none" if self.valid else "no_usable_fish"

    def by_id(self, fish_id: int) -> Fish | None:
        return next((f for f in self.fish if f.fish_id == fish_id), None)

    def as_record(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.wall_time, 3),
            "vision_ok": self.vision_ok,
            "failure_code": self.failure_code,
            "fish_count_valid": len(self.valid),
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "fish": [f.as_record() for f in self.fish],
        }


class FishLocator:
    """Turns the detections of a frame into a FishList, given where the camera was."""

    def __init__(
        self,
        settings: VisionSettings,
        table_z: float,
        pick_area: Area | None = None,
        model_version: str = "",
        calibration_version: str = "",
    ):
        self.settings = settings
        self.table_z = table_z
        self.pick_area = pick_area
        self.model_version = model_version
        self.calibration_version = calibration_version

    def locate(
        self, frame: Frame, detections: list[Detection], t_base_camera: np.ndarray
    ) -> FishList:
        order = sorted(detections, key=lambda d: (d.polygon[:, 0].mean(), d.polygon[:, 1].mean()))
        overlaps = _overlaps([d.polygon for d in order], self.settings.overlap_margin_px)
        fish = [
            self._fish(i + 1, detection, overlaps[i], frame, t_base_camera)
            for i, detection in enumerate(order)
        ]
        return FishList(
            frame.frame_id,
            frame.stamp,
            frame.wall_time,
            t_base_camera,
            fish,
            self.model_version,
            self.calibration_version,
        )

    def _fish(self, fish_id, detection, overlap, frame, t_base_camera) -> Fish:
        s = self.settings
        polygon = np.asarray(detection.polygon, float)
        failed: set[str] = set()
        if detection.conf < s.min_confidence:
            failed.add("low_confidence")
        if touches_border(polygon, frame.size, s.border_margin_px):
            failed.add("out_of_frame")

        fish = Fish(fish_id, False, "none", detection.conf, None, None, None, 0.0, 0.0, 0.0, None,
                    overlap, polygon)  # fmt: skip
        try:
            _, centroid_px, _ = fish_heading.polygon_moments(polygon)
        except ValueError:
            return _finish(fish, failed | {"geometry_out_of_range"})

        # How high the fish is: measured depth at the middle of its outline, else the default.
        fish.thickness = s.default_thickness_m
        if s.use_depth and frame.depth is not None:
            top = self._surface_from_depth(frame, centroid_px, t_base_camera)
            if top is None:
                failed.add("invalid_depth")
            else:
                fish.depth_measured = True
                fish.thickness = float(top[2] - self.table_z)

        # The outline in {B}. The camera sees the silhouette of the fish's widest section,
        # about half way up, so that is the plane its pixels are put on; using the top
        # surface instead would shift an off-center fish toward the camera by centimeters.
        outline_z = self.table_z + max(fish.thickness, 0.0) * s.outline_height_share
        outline = pixels_on_plane(frame.intrinsics, t_base_camera, polygon, outline_z)
        if not np.isfinite(outline).all():
            return _finish(fish, failed | {"geometry_out_of_range"})
        outline_xy = outline[:, :2]
        center, axis = fish_heading.body_axis(outline_xy)
        surface = np.array([center[0], center[1], self.table_z + fish.thickness])  # grasp point
        vote, confidence, _ = fish_heading.head_vote(outline_xy, center, axis, (True, True))
        fish.heading_confidence = confidence
        fish.heading_known = confidence >= s.min_heading_confidence and not (
            "out_of_frame" in failed
        )
        direction = axis if vote >= 0 else -axis
        heading = math.atan2(direction[1], direction[0])
        if not fish.heading_known and s.require_heading:
            failed.add("wrong_heading")

        along = fish_heading.along_axis(outline_xy, center, axis)[:, 0]
        fish.total_length = float(along.max() - along.min())
        fish.width_at_grasp = float(fish_heading.width_profile(outline_xy, center, axis, [0.0])[0])
        fish.bbox = (
            float(outline_xy[:, 0].min()), float(outline_xy[:, 0].max()),
            float(outline_xy[:, 1].min()), float(outline_xy[:, 1].max()),
            float(self.table_z), float(surface[2]),
        )  # fmt: skip

        if not self._footprint_fits(outline_xy, surface[:2], axis):
            failed.add("grasp_point_off_body")
        size_ok = not s.check_size or (
            _within(fish.total_length, s.length_range_m)
            and _within(fish.width_at_grasp, s.width_range_m)
            and _within(fish.thickness, s.thickness_range_m)
        )
        if not (size_ok and _in_area(surface[:2], self.pick_area)):
            failed.add("geometry_out_of_range")

        fish.heading = heading
        fish.grasp_pose = pose_down(surface, heading)
        fish.target_pose = fish.grasp_pose.copy()
        return _finish(fish, failed)

    def _surface_from_depth(self, frame, pixel, t_base_camera) -> np.ndarray | None:
        """The fish's surface point in {B} under `pixel`, or None if the depth there is bad."""
        s = self.settings
        nominal = max(t_base_camera[2, 3] - self.table_z - s.default_thickness_m, 0.05)
        rough, _ = depth_at(frame.depth, pixel, 4, s.depth_range_m)
        radius = s.depth_window_m * frame.intrinsics.fx / (rough or nominal)
        depth, valid = depth_at(frame.depth, pixel, radius, s.depth_range_m)
        if depth is None or valid < s.min_depth_valid:
            return None
        return to_frame(t_base_camera, deproject(frame.intrinsics, pixel, depth))[0]

    def _footprint_fits(self, outline_xy, point_xy, axis) -> bool:
        """Whether the cup footprint, centered on the grasp point along the body, is on the fish."""
        half_long, half_wide = (v / 2 for v in self.settings.cup_footprint_m)
        normal = np.array([-axis[1], axis[0]])
        corners = [
            point_xy + a * half_long * axis + b * half_wide * normal
            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
        ]
        # pointPolygonTest works in float32: millimeters keep the precision
        return bool(inside_polygon(outline_xy * 1000.0, np.array(corners) * 1000.0).all())


def _finish(fish: Fish, failed: set[str]) -> Fish:
    """Set the result fields from the failed checks (VS-60, VS-68, VS-69)."""
    fish.failures = tuple(code for code in FAILURE_ORDER if code in failed)
    fish.vision_ok = not fish.failures
    fish.failure_code = fish.failures[0] if fish.failures else "none"
    if not fish.vision_ok:
        fish.grasp_pose = fish.target_pose = fish.heading = None
    return fish


def _within(value: float, limits: tuple[float, float]) -> bool:
    return limits[0] <= value <= limits[1]


def _in_area(xy, area: Area | None) -> bool:
    if area is None:
        return True
    (x0, x1), (y0, y1) = area
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def _overlaps(polygons: list[np.ndarray], margin_px: float) -> list[bool]:
    """For each outline, whether it touches or overlaps another (Table VS-1)."""
    flags = [False] * len(polygons)
    boxes = [(p.min(axis=0) - margin_px, p.max(axis=0) + margin_px) for p in polygons]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            (a0, a1), (b0, b1) = boxes[i], boxes[j]
            if (a0 > b1).any() or (b0 > a1).any():
                continue
            if (
                inside_polygon(polygons[i], polygons[j], -margin_px).any()
                or inside_polygon(polygons[j], polygons[i], -margin_px).any()
            ):
                flags[i] = flags[j] = True
    return flags


def match_fish(fish: Fish, others: list[Fish], max_distance: float) -> Fish | None:
    """The fish of `others` whose grasp point is nearest to `fish`'s, within `max_distance`."""
    if fish.grasp_point is None:
        return None
    best, best_distance = None, max_distance
    for other in others:
        if other.grasp_point is None:
            continue
        distance = float(np.linalg.norm(other.grasp_point[:2] - fish.grasp_point[:2]))
        if distance <= best_distance:
            best, best_distance = other, distance
    return best


def heading_difference(a: Fish, b: Fish) -> float:
    """Angle between two fish's headings; between their body axes if a head end is unknown."""
    difference = abs(math.atan2(math.sin(a.heading - b.heading), math.cos(a.heading - b.heading)))
    if not (a.heading_known and b.heading_known):
        difference = min(difference, math.pi - difference)
    return difference


def merge_stable(lists: list[FishList], settings: VisionSettings) -> FishList:
    """
    The last of several consecutive fish lists, with each valid fish flagged `unstable`
    unless it is valid in every list at the same place and heading (VS-52).
    """
    result = lists[-1]
    for fish in result.valid:
        for earlier in lists[:-1]:
            same = match_fish(fish, earlier.valid, settings.match_distance_m)
            if (
                same is None
                or np.linalg.norm(same.grasp_point - fish.grasp_point) > settings.stable_position_m
                or heading_difference(same, fish) > math.radians(settings.stable_heading_deg)
            ):
                _finish(fish, set(fish.failures) | {"unstable"})
                break
    return result
