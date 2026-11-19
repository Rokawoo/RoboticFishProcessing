"""
The fish list (VS-40 to VS-60, IF-20): detections in one frame turned into fish with a
position, heading and size in the arm base frame {B}, each checked and flagged.

Meters and radians here; `as_record()` gives the IF-1/IF-2 fields in mm and degrees for logs.
Without a pose model the middle is the centroid of the outline (on the stand-in fish 49% of
the length from the snout; VS-33 asks for 40-50%); the cups go ahead of it toward the heavier
head, as far as they fit, and without depth the approach is straight down.

With depth, a plane through the body under the six cups gives the approach direction (VS-44)
and tells where all six seal; the surface's slope along the body is a head/tail cue; and the
table around each fish is measured, so its height and size don't depend on the workcell's
table.z or how high the camera really is (the outline's plane is between the two).

Not implemented yet: `covered` (VS-25), keypoints (VS-30, VS-62).
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from rfp_vision import fish_heading
from rfp_vision.camera import Frame
from rfp_vision.detector import Detection
from rfp_vision.geometry import (
    deproject,
    depth_at,
    fit_plane,
    inside_polygon,
    pixels_on_plane,
    polygon_mask,
    pose_along,
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
SURFACE_GRID = 160  # depth samples across a fish's image box, for its surface
TABLE_GRID = 200  # the same, for the table around it
# Percentile of the heights around a fish taken as the table's: the lowest surface there,
# above the breadboard's holes and below other fish and the gripper
TABLE_PERCENTILE = 30
MIN_TABLE_POINTS = 50


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
    surface_tilt: float = 0.0  # rad between the surface under the cups and level (0: unknown)
    failures: tuple[str, ...] = ()  # every failed check, in table order
    # (x, y, body axis yaw) of the outline on the table in {B}, kept when vision rejects the
    # fish (e.g. cut off at the image's edge), so the arm can take a closer look at it
    seen_at: tuple[float, float, float] | None = None
    detail: str = ""  # what the first failed check measured, for the operator
    table_z: float | None = None  # the table around it in {B}: measured, else the workcell's
    table_measured: bool = False
    # Of the outline in {B}, along the body axis of `seen_at` from its middle (m): where its
    # ends are as seen, and whether each (-axis end, +axis end) runs out of the image
    seen_span: tuple[float, float] | None = None
    cut_ends: tuple[bool, bool] = (False, False)

    def estimate(self, expected_length: float) -> tuple[np.ndarray, float, float] | None:
        """(Its middle's x, y in {B}, body axis yaw, length) as best known from this view:
        cut off at one end, its middle lies half the expected length (or what is seen, if
        longer) from the end that shows. None if vision couldn't place it."""
        if self.seen_at is None:
            if self.middle is not None and self.heading is not None:
                return np.array(self.middle[:2], float), float(self.heading), self.total_length
            if self.bbox is None:
                return None
            x0, x1, y0, y1 = self.bbox[:4]
            axis = 0.0 if x1 - x0 >= y1 - y0 else math.pi / 2
            middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            return middle, axis, max(self.total_length, x1 - x0, y1 - y0)
        x, y, yaw = self.seen_at
        middle = np.array([x, y], float)
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        length = self.total_length
        minus_cut, plus_cut = self.cut_ends
        if self.seen_span is not None and minus_cut != plus_cut:
            low, high = self.seen_span
            length = max(length, expected_length)
            if plus_cut:  # the -axis end shows: the fish runs on toward +axis
                middle = middle + axis * (low + length / 2)
            else:
                middle = middle + axis * (high - length / 2)
        elif minus_cut and plus_cut:
            length = max(length, expected_length)
        return middle, float(yaw), float(length)

    @property
    def grasp_point(self) -> np.ndarray | None:
        """Where the cups go: where the fish balances, toward its head (or its middle)."""
        return None if self.grasp_pose is None else self.grasp_pose[:3, 3]

    @property
    def middle(self) -> np.ndarray | None:
        """The middle of its outline, on it: what the pick area and the slots are about."""
        return None if self.target_pose is None else self.target_pose[:3, 3]

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
        # A weak detection (often a second outline of part of the same fish) mustn't make a
        # fish look crowded: only detections that are trusted count as neighbors.
        trusted = [d.conf >= self.settings.min_confidence for d in order]
        overlaps = _overlaps([d.polygon for d in order], self.settings.overlap_margin_px, trusted)
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
        details = {"out_of_frame": "cut off at the image's edge"}
        try:
            _, centroid_px, _ = fish_heading.polygon_moments(polygon)
        except ValueError:
            details["geometry_out_of_range"] = "its outline has no area"
            return _finish(fish, failed | {"geometry_out_of_range"}, details)

        # The table around it, from depth: the fish's height and its outline's plane are
        # measured from there, so a wrong workcell table.z (or camera height) doesn't scale it.
        table_z = self.table_z
        depth = s.use_depth and frame.depth is not None
        if depth and s.measure_table:
            around = self._table_around(frame, polygon, t_base_camera)
            if around is not None and abs(around - self.table_z) <= s.table_search_m:
                table_z, fish.table_measured = around, True
        fish.table_z = table_z

        # How high the fish is: measured depth at the middle of its outline, else the default.
        fish.thickness = s.default_thickness_m
        if depth:
            top = self._surface_from_depth(frame, centroid_px, t_base_camera)
            if top is None:
                failed.add("invalid_depth")
                details["invalid_depth"] = "no depth at the middle of its outline"
            else:
                fish.depth_measured = True
                fish.thickness = float(top[2] - table_z)

        # The outline in {B}. The camera sees the silhouette of the fish's widest section,
        # about half way up, so that is the plane its pixels are put on; using the top
        # surface instead would shift an off-center fish toward the camera by centimeters.
        outline_z = table_z + max(fish.thickness, 0.0) * s.outline_height_share
        outline = pixels_on_plane(frame.intrinsics, t_base_camera, polygon, outline_z)
        if not np.isfinite(outline).all():
            details["geometry_out_of_range"] = "its outline doesn't reach the table"
            return _finish(fish, failed | {"geometry_out_of_range"}, details)
        outline_xy = outline[:, :2]
        center, axis = fish_heading.body_axis(outline_xy)
        fish.seen_at = (float(center[0]), float(center[1]), math.atan2(axis[1], axis[0]))
        along = fish_heading.along_axis(outline_xy, center, axis)[:, 0]
        fish.total_length = float(along.max() - along.min())
        fish.seen_span = (float(along.min()), float(along.max()))
        if "out_of_frame" in failed:  # which end runs out of the image (H-10, in {B})
            edge = fish_heading.at_border(polygon, frame.size, s.border_margin_px)
            share = (along[edge] - along.min()) / max(fish.total_length, 1e-9)
            fish.cut_ends = (bool((share <= 0.25).any()), bool((share >= 0.75).any()))
        points = rise = middle_plane = None
        if fish.depth_measured:
            points = self._surface_points(frame, polygon, t_base_camera)
            rise, middle_plane = self._surface_shape(points, center, axis, fish.total_length)
        if middle_plane is not None:
            fish.thickness = float(middle_plane[1] - table_z)
        middle_height = fish.thickness  # at the outline's middle, for the size check
        usable = (True, True)
        vote, confidence, _ = fish_heading.head_vote(outline_xy, center, axis, usable, rise)
        fish.heading_confidence = confidence
        fish.heading_known = confidence >= s.min_heading_confidence and not (
            "out_of_frame" in failed
        )
        direction = axis if vote >= 0 else -axis
        heading = math.atan2(direction[1], direction[0])
        if not fish.heading_known and s.require_heading:
            failed.add("wrong_heading")
            details["wrong_heading"] = f"which end is the head is unclear ({confidence:.2f})"

        # The cups go where the fish balances: ahead of its middle, toward the heavier head,
        # as far as they still fit on the body and all six seal; else the middle, else up to
        # grasp_behind_m behind it. Head end unknown: the middle, then either way. Nowhere:
        # grasp_point_off_body. The cups lean onto the body's plane under them.
        grasp_xy, plane, seals, fits = center, middle_plane, False, 0
        for distance in self._grasp_offsets(fish.total_length, fish.heading_known):
            point = center + direction * distance
            if not self._footprint_fits(outline_xy, point, axis):
                continue
            fits += 1
            sealed, under_cups = self._cup_contact(points, point, axis)
            if sealed is not False:  # None: too little depth under the cups; the outline says
                grasp_xy, seals = point, True
                plane = under_cups if under_cups is not None else plane
                break
        approach = (0.0, 0.0, -1.0)
        if plane is not None:  # the surface's height and slope right at the grasp point
            normal, height = plane
            fish.thickness = float(height - table_z)
            fish.surface_tilt = float(math.acos(min(1.0, normal[2])))
            if s.align_to_surface and fish.surface_tilt <= math.radians(s.max_surface_tilt_deg):
                approach = -normal
        surface = np.array([grasp_xy[0], grasp_xy[1], table_z + fish.thickness])

        offset = float((grasp_xy - center) @ axis)
        fish.width_at_grasp = float(
            fish_heading.width_profile(outline_xy, center, axis, [offset])[0]
        )
        width_middle = float(fish_heading.width_profile(outline_xy, center, axis, [0.0])[0])
        fish.bbox = (
            float(outline_xy[:, 0].min()), float(outline_xy[:, 0].max()),
            float(outline_xy[:, 1].min()), float(outline_xy[:, 1].max()),
            float(table_z), float(table_z + max(fish.thickness, middle_height)),
        )  # fmt: skip

        if not seals:  # the cups would hang off the body, or not all of them reach it
            failed.add("grasp_point_off_body")
            details["grasp_point_off_body"] = (
                "something under a cup keeps the six from sealing (a fin, a step)"
                if fits
                else f"the six cups don't fit on it ({width_middle * 1000:.0f} mm wide)"
            )
        sizes = (
            ("length", fish.total_length, s.fish.length_range_m),
            ("width", width_middle, s.fish.width_range_m),
            ("height", middle_height, s.fish.thickness_range_m),
        )
        wrong = [item for item in sizes if s.check_size and not _within(item[1], item[2])]
        if wrong or not _in_area(center, self.pick_area):
            failed.add("geometry_out_of_range")
            details["geometry_out_of_range"] = (
                "; ".join(
                    f"{name} {value:.2f} m ({s.fish_type}: {low:.2f}-{high:.2f})"
                    for name, value, (low, high) in wrong
                )
                or "outside the pick area"
            )

        fish.heading = heading
        fish.grasp_pose = pose_along(surface, heading, approach)
        middle = np.array([center[0], center[1], table_z + middle_height])
        fish.target_pose = pose_along(middle, heading, approach)
        return _finish(fish, failed, details)

    def _grasp_offsets(self, length: float, heading_known: bool) -> list[float]:
        """Where along the body (m from the outline's middle, + toward the head) the cups are
        tried, in order: from the balance point back to the middle, then behind it."""
        s = self.settings
        ahead = s.fish.balance_ahead_share * length if heading_known else 0.0
        offsets = [float(d) for d in np.arange(ahead, 0.0, -s.grasp_step_m)] if ahead > 0 else []
        offsets.append(0.0)
        for d in np.arange(s.grasp_step_m, s.grasp_behind_m + 1e-9, s.grasp_step_m):
            offsets += [-float(d)] if heading_known else [float(d), -float(d)]
        return offsets

    def _table_around(self, frame, polygon, t_base_camera) -> float | None:
        """The table's height (z in {B}) just outside a fish's outline, from depth: between
        table_gap_m and table_ring_m out, where the table shows (TABLE_PERCENTILE of the
        heights there). None with too little of it in view."""
        s = self.settings
        h, w = frame.depth.shape[:2]
        distance = max(t_base_camera[2, 3] - self.table_z, 0.05)  # about, for the sizes in px
        gap = max(1, round(s.table_gap_m * frame.intrinsics.fx / distance))
        ring = max(gap + 2, round(s.table_ring_m * frame.intrinsics.fx / distance))
        x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int) - ring, 0)
        x1, y1 = np.minimum(np.ceil(polygon.max(axis=0)).astype(int) + ring + 1, (w, h))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        inside = polygon_mask(polygon - (x0, y0), (y1 - y0, x1 - x0))
        near = cv2.dilate(inside, np.ones((2 * gap + 1,) * 2, np.uint8))
        band = cv2.dilate(inside, np.ones((2 * ring + 1,) * 2, np.uint8)) & ~near.astype(bool)
        step = max(1, int(max(x1 - x0, y1 - y0)) // TABLE_GRID)
        band = band[::step, ::step].astype(bool)
        values = frame.depth[y0:y1:step, x0:x1:step]
        keep = band & (values >= s.depth_range_m[0]) & (values <= s.depth_range_m[1])
        rows, columns = np.nonzero(keep)
        if len(rows) < MIN_TABLE_POINTS:
            return None
        pixels = np.column_stack([x0 + columns * step, y0 + rows * step])
        points = to_frame(t_base_camera, frame.intrinsics.rays(pixels) * values[keep][:, None])
        return float(np.percentile(points[:, 2], TABLE_PERCENTILE))

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

    def _surface_points(self, frame, polygon, t_base_camera) -> np.ndarray:
        """(n, 3) points in {B} on the fish's upper surface: the valid depth pixels inside
        its outline, on a grid of at most about SURFACE_GRID x SURFACE_GRID (about 30
        points under each cup at a close look, enough to find a fin)."""
        s = self.settings
        h, w = frame.depth.shape[:2]
        x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(polygon.max(axis=0)).astype(int) + 1, (w, h))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 3))
        step = max(1, int(max(x1 - x0, y1 - y0)) // SURFACE_GRID)
        inside = polygon_mask(polygon - (x0, y0), (y1 - y0, x1 - x0))[::step, ::step].astype(bool)
        depth = frame.depth[y0:y1:step, x0:x1:step]
        keep = inside & (depth >= s.depth_range_m[0]) & (depth <= s.depth_range_m[1])
        rows, columns = np.nonzero(keep)
        if len(rows) == 0:
            return np.zeros((0, 3))
        pixels = np.column_stack([x0 + columns * step, y0 + rows * step])
        return to_frame(t_base_camera, frame.intrinsics.rays(pixels) * depth[keep][:, None])

    def _surface_shape(self, points, center, axis, length):
        """
        From surface points: (how much higher the surface is a quarter length toward +axis
        than toward -axis, or None; (upward normal, height at the grasp point) of the plane
        fitted under the cup footprint, or None if the fit can't be trusted).
        """
        s = self.settings
        if len(points) == 0:
            return None, None
        offset = points[:, :2] - center
        along, across = offset @ axis, offset @ np.array([-axis[1], axis[0]])
        half_long, half_wide = (v / 2 for v in s.cup_footprint_m)
        strip = np.abs(across) <= half_wide

        def height(low: float, high: float) -> float | None:
            band = points[strip & (along >= low * length) & (along <= high * length), 2]
            return float(np.median(band)) if len(band) >= 10 else None

        front, back = height(0.15, 0.35), height(-0.35, -0.15)
        rise = None if front is None or back is None else front - back
        under = points[strip & (np.abs(along) <= half_long)]
        if len(under) < 30:
            return rise, None
        normal, middle, rms = fit_plane(under)
        if rms > s.surface_fit_rms_m or normal[2] < 0.5:
            return rise, None
        away = center - middle[:2]
        return rise, (normal, middle[2] - (normal[0] * away[0] + normal[1] * away[1]) / normal[2])

    def _cup_contact(self, points: np.ndarray | None, point_xy, axis):
        """
        Whether all six cups (3 x 2, filling the footprint) would seal with the footprint's
        middle at `point_xy` (None: the depth can't tell), and the body's plane under them as
        (upward normal, height at `point_xy`), None without it. The plane goes through the
        body under each cup (its median); the cups lean onto it (if allowed and not too
        steep, else straight down). Each cup's lip meets the highest surface under it first
        (a high percentile, so one stray point doesn't count): something standing well above
        the body there, like a fin, keeps that cup off the fish. Pressed until the last cup
        touches, the first is compressed by the difference between their heights: that must
        be within the stroke. A cup over a hole in the depth (shiny skin, an edge) isn't
        judged; with fewer than min_cups_measured judged the depth can't tell.
        """
        s = self.settings
        if points is None or len(points) == 0:
            return True, None  # no depth: the footprint check alone
        half_long, half_wide = (v / 2 - s.cup_diameter_m / 2 for v in s.cup_footprint_m)
        side = np.array([-axis[1], axis[0]])
        offset = points[:, :2] - point_xy
        along, across = offset @ axis, offset @ side
        under = []
        for a in (-half_long, 0.0, half_long):
            for b in (-half_wide, half_wide):
                near = np.hypot(along - a, across - b) <= s.cup_diameter_m / 2
                if near.sum() >= 3:
                    under.append((a, b, near))
        if len(under) < max(3, s.min_cups_measured):
            return None, None
        middles = np.array([(a, b, np.median(points[near, 2])) for a, b, near in under])
        design = np.column_stack([middles[:, 0], middles[:, 1], np.ones(len(middles))])
        fit, *_ = np.linalg.lstsq(design, middles[:, 2], rcond=None)
        gradient = fit[0] * axis + fit[1] * side  # dz/dx, dz/dy in {B}
        normal = np.array([-gradient[0], -gradient[1], 1.0])
        normal /= np.linalg.norm(normal)
        tilt = math.acos(min(1.0, normal[2]))
        lean = s.align_to_surface and tilt <= math.radians(s.max_surface_tilt_deg)
        slope = fit[:2] if lean else np.zeros(2)  # the plane the cups are pressed onto
        lips = []
        for a, b, near in under:
            bump = _lip_bump(points[near, 2], along[near] - a, across[near] - b, fit)
            limit = s.cup_lip_conform_m if s.strict_seal_check else s.cup_stroke_m
            if bump > limit:  # a fin or a step under this cup's lip (past the bellows' stroke:
                return False, None  # surely; past what a lip conforms to: maybe)
            pressed = points[near, 2] - np.column_stack([along[near], across[near]]) @ slope
            lips.append(float(np.percentile(pressed, 90)))
        return max(lips) - min(lips) <= s.cup_stroke_m, (normal, float(fit[2]))

    def _footprint_fits(self, outline_xy, point_xy, axis) -> bool:
        """Whether all six cups, their footprint centered on the grasp point along the body,
        are on the fish seen from above: each cup's middle at least its radius less
        cup_overhang_m inside the outline. (The footprint's corners lie outside the round
        cups: requiring them inside refused fish the cups fit.)"""
        s = self.settings
        radius = s.cup_diameter_m / 2
        half_long, half_wide = (v / 2 - radius for v in s.cup_footprint_m)
        normal = np.array([-axis[1], axis[0]])
        cups = [
            point_xy + a * axis + b * normal
            for a in (-half_long, 0.0, half_long)
            for b in (-half_wide, half_wide)
        ]
        margin = max(radius - s.cup_overhang_m, 0.0)
        # pointPolygonTest works in float32: millimeters keep the precision
        inside = inside_polygon(outline_xy * 1000.0, np.array(cups) * 1000.0, margin * 1000.0)
        return bool(inside.all())


def _finish(fish: Fish, failed: set[str], details: dict[str, str] | None = None) -> Fish:
    """Set the result fields from the failed checks (VS-60, VS-68, VS-69), and what the
    first one measured."""
    fish.failures = tuple(code for code in FAILURE_ORDER if code in failed)
    fish.vision_ok = not fish.failures
    fish.failure_code = fish.failures[0] if fish.failures else "none"
    fish.detail = (details or {}).get(fish.failure_code, "") if fish.failures else ""
    if not fish.vision_ok:
        fish.grasp_pose = fish.target_pose = fish.heading = None
    return fish


def _lip_bump(heights, along, across, fit) -> float:
    """How far something under one cup stands above the surface its lip rests on (m): the
    heights' 90th percentile above their median, along the plane through that cup's own
    points (a round body's shoulder slopes under an outer cup, which the bellows' lip
    follows), or with too few points along the plane through the six cups. `along`,
    `across`: from the cup's middle."""
    if len(heights) >= 8:
        design = np.column_stack([along, across, np.ones(len(heights))])
        local, *_ = np.linalg.lstsq(design, heights, rcond=None)
        rest = heights - design @ local
    else:
        rest = heights - np.column_stack([along, across]) @ fit[:2]
    return float(np.percentile(rest, 90) - np.median(rest))


def _within(value: float, limits: tuple[float, float]) -> bool:
    return limits[0] <= value <= limits[1]


def _in_area(xy, area: Area | None) -> bool:
    if area is None:
        return True
    (x0, x1), (y0, y1) = area
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def _overlaps(polygons: list[np.ndarray], margin_px: float, trusted: list[bool]) -> list[bool]:
    """For each outline, whether it touches or overlaps another trusted one (Table VS-1)."""
    flags = [False] * len(polygons)
    boxes = [(p.min(axis=0) - margin_px, p.max(axis=0) + margin_px) for p in polygons]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if not (trusted[i] and trusted[j]):
                continue
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
    unless it is valid in every list at the same place (its middle: where the cups go can
    move a search step between frames without the fish moving) and heading (VS-52).
    """
    result = lists[-1]
    for fish in result.valid:
        for earlier in lists[:-1]:
            same = match_fish(fish, earlier.valid, settings.match_distance_m)
            if (
                same is None
                or np.linalg.norm(same.middle - fish.middle) > settings.stable_position_m
                or heading_difference(same, fish) > math.radians(settings.stable_heading_deg)
            ):
                _finish(fish, set(fish.failures) | {"unstable"})
                break
    return result
