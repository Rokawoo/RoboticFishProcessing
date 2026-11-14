"""
Head direction of each detected fish, from the geometry of its segmentation mask, kept
while the camera moves closer and the head leaves the frame (Requirement Spec VS-40,
VS-60).

Status: measuring one outline (H-6 to H-11) is implemented; its cue weights and scales
are first values, to be tuned on labeled real footage (H-23). Stricter than H-10 for
now: an outline with a cut end gives no measurement. Not implemented yet: keypoints
(H-12, no pose model) and HeadingMemory (H-13 to H-18).

Requirements
============

Conventions
    H-1  Heading = direction from tail to head. Image: unit vector and angle in pixels
         (x right, y down). Base frame {B}: yaw about the table normal, radians,
         counter-clockwise from +x of {B}, in (-pi, pi].
    H-2  All lengths in pixels (image) or meters ({B}); angles in radians. Thresholds
         live in HeadingSettings (later Vision/configs), each with its unit.
    H-3  This module is plain Python + NumPy/OpenCV, no ROS: the vision node calls it,
         and it is testable offline on recorded frames (model_viewer, labeled sets).

Input (per frame)
    H-4  Per fish: its outline from the segmentation model (polygon, image px) and
         detection confidence. Plus the image size and the frame's timestamp.
    H-5  Optional per frame: the camera pose in {B} (hand-eye calibration + arm
         kinematics), camera intrinsics and depth. Without them only image-space
         tracking (H-17) is possible.

Measuring the heading in one frame (geometry only)
    H-6  Body axis: the fish's centroid and long axis from the outline (second-order
         moments / PCA). The axis has no sign yet: it says along which line the fish
         lies, not which end is the head.
    H-7  Head vs tail, from the catfish's shape along the axis:
           - width profile: the head end is wide and blunt; toward the tail the body
             narrows to the caudal peduncle and flares again at the tail fin;
           - mass distribution: more of the area lies on the head side of the
             centroid (skewness of the outline's projection onto the axis);
           - tail fork: concave notch (convexity defect) at the tail end.
         Each cue gives a signed vote with a weight; the sum gives the head end and a
         confidence in [0, 1].
    H-8  The fish's ends are found as the outline's extreme points along the axis;
         head point and tail point are reported in image px (and in {B} with H-5).
    H-9  Bent fish (curved body) must still give the right end: cues are measured near
         each end, not only on the straight axis.
    H-10 Fish cut by the frame edge: detect which ends touch the image border (within
         a margin). Cues from a cut end are not used; a cut end may be the head. If
         neither end can be judged, the frame gives no measurement (H-14).
    H-11 Touching or overlapping fish: each outline is measured on its own; the cues
         must not use pixels of a neighbor's outline.
    H-12 Pose model (VS-30): when head/tail keypoints from a pose model are available
         and confident, they take precedence; geometry is the fallback and a cross-
         check. Disagreement above a threshold is flagged (H-19).

Keeping the heading (camera moving closer, head out of frame)
    H-13 The heading belongs to the fish, not to the image: fish on the table don't
         move between the observation pose and the grasp, so the last confident
         heading is stored per fish in {B}.
    H-14 Each frame, detections are matched to stored fish by position in {B} (and
         outline overlap). A matched fish whose heading can't be measured in this
         frame (head out of frame, low confidence) keeps its stored heading.
    H-15 A kept heading follows the current view: its axis is re-measured from the
         current outline (more precise as the camera gets closer) and its sign is the
         one closest to the stored heading (positive dot product).
    H-16 A confident new measurement updates the stored heading (weighted by
         confidence); one that contradicts it (about 180 degrees off) doesn't flip it
         alone: it needs N consecutive confident frames, else the fish is flagged.
    H-17 Without a camera pose (offline videos): match fish frame to frame in the image
         (centroid distance, overlap) and keep the heading for at most a set time.
    H-18 The memory of a fish is dropped when the fish moved (position change or axis
         rotation above a threshold: picked, nudged, flipped) or wasn't seen for a
         set time; it is re-measured, or reported unknown.

Output (per fish, per frame)
    H-19 HeadingEstimate: image angle and unit vector, yaw in {B} (with H-5), head and
         tail points, confidence, source ("measured", "keypoints", "kept", "unknown"),
         flags (end cut by the frame, ambiguous shape, keypoints disagree, kept for
         too long), and the age of a kept heading.
    H-20 Never guess (VS-60): below the confidence threshold the heading is "unknown".
         ControlSys must not grasp or place a fish without a known heading.

Performance and testing
    H-21 Fast enough for every frame of the vision node (target in the Requirement
         Spec; budget a few ms per fish on the CPU).
    H-22 Deterministic: same outline + same settings = same result.
    H-23 Tests: synthetic catfish outlines (body + forked tail) at every heading, bent,
         cut by each frame edge, touching another fish; a camera approach sequence
         where the head leaves the frame. Real frames: compare with hand-marked head
         points; report the share of correct ends and the angle error.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HeadingSettings:
    """Thresholds (H-2). Values TBD from tests on real footage."""

    min_confidence: float = 0.6  # H-20: below this the heading is unknown
    edge_margin_px: float = 3.0  # H-10: an outline this close to the border is cut
    max_keypoint_disagreement_rad: float = 0.5  # H-12
    flip_confirm_frames: int = 3  # H-16
    max_move_m: float = 0.03  # H-18: fish moved more than this: memory dropped
    max_turn_rad: float = 0.35  # H-18
    max_unseen_s: float = 2.0  # H-17, H-18


@dataclass
class FishObservation:
    """One detected fish in one frame (H-4)."""

    polygon: np.ndarray  # (n, 2) image px
    confidence: float
    keypoints: np.ndarray | None = None  # pose model (H-12): head, tail, ... (k, 3)


@dataclass
class HeadingEstimate:
    """Result per fish per frame (H-19)."""

    angle_image: float | None  # rad, image coordinates; None if unknown
    direction_image: np.ndarray | None  # unit vector tail -> head
    yaw_base: float | None  # rad in {B}; None without a camera pose
    head_px: np.ndarray | None
    tail_px: np.ndarray | None
    confidence: float
    source: str  # "measured", "keypoints", "kept", "unknown"
    flags: set[str] = field(default_factory=set)
    kept_for_s: float = 0.0


# Cues of head_vote: weight, and the raw value that counts as a full vote (H-7). First
# values, from the outlines of the 3D-printed stand-in fish seen from above: centroid
# 0.02-0.03 toward the head, end widths differing by 0.17-0.20, tail fin 2x its peduncle.
CUE_WEIGHTS = {"mass": 1.0, "width": 2.0, "waist": 2.0}
MASS_FULL_VOTE = 0.04  # centroid offset from the middle of the fish, share of half its length
WIDTH_FULL_VOTE = 0.15  # (head-end width - tail-end width) / their sum, 10-20% from each end
WAIST_FULL_VOTE = 0.25  # distance of the narrowest section from the middle, share of the length
FLARE_FULL_VOTE = 1.5  # widest section beyond the waist / the waist: a tail fin, not a taper
PROFILE_STATIONS = 64  # cross-sections measured along the fish


def polygon_moments(polygon: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(area, centroid, 2x2 covariance) of the region inside an outline, any vertex order."""
    points = np.asarray(polygon, float).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("an outline needs at least 3 points")
    origin = points.mean(axis=0)  # moments about a nearby point: no precision loss
    x, y = (points - origin).T
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        raise ValueError("the outline has no area")
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    sxx = ((x * x + x * xn + xn * xn) * cross).sum() / (12.0 * area) - cx * cx
    syy = ((y * y + y * yn + yn * yn) * cross).sum() / (12.0 * area) - cy * cy
    sxy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross).sum() / (24.0 * area) - cx * cy
    return abs(area), origin + (cx, cy), np.array([[sxx, sxy], [sxy, syy]])


def body_axis(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Centroid and unsigned unit long axis of an outline (H-6), from the area's second-order
    moments. Of the two directions, the one with x > 0 (y > 0 if x is 0) is returned, so
    the same outline always gives the same axis (H-22).
    """
    _, centroid, covariance = polygon_moments(polygon)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    return centroid, axis / np.linalg.norm(axis)


def along_axis(polygon: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """(n, 2): each outline point's distance from the centroid along and across the axis."""
    offset = np.asarray(polygon, float).reshape(-1, 2) - centroid
    normal = np.array([-axis[1], axis[0]])
    return np.column_stack([offset @ axis, offset @ normal])


def width_profile(
    polygon: np.ndarray, centroid: np.ndarray, axis: np.ndarray, stations: np.ndarray
) -> np.ndarray:
    """Width of the outline across the axis at each station (distance along the axis)."""
    a, b = along_axis(polygon, centroid, axis).T
    an, bn = np.roll(a, -1), np.roll(b, -1)
    widths = []
    for s in stations:
        crossing = ((a <= s) & (s < an)) | ((an <= s) & (s < a))
        slope = (bn[crossing] - b[crossing]) / (an[crossing] - a[crossing])
        hits = np.sort(b[crossing] + (s - a[crossing]) * slope)
        pairs = len(hits) // 2  # the line enters and leaves the outline in pairs
        widths.append(float((hits[1 : 2 * pairs : 2] - hits[0 : 2 * pairs : 2]).sum()))
    return np.array(widths)


def at_border(polygon: np.ndarray, image_size: tuple[int, int], margin_px: float) -> np.ndarray:
    """Which outline points lie within `margin_px` of the edge of a (width, height) image."""
    points = np.asarray(polygon, float).reshape(-1, 2)
    w, h = image_size
    return (
        (points[:, 0] <= margin_px)
        | (points[:, 1] <= margin_px)
        | (points[:, 0] >= w - 1 - margin_px)
        | (points[:, 1] >= h - 1 - margin_px)
    )


def touches(polygon: np.ndarray, image_size: tuple[int, int], margin_px: float) -> bool:
    """Whether the outline reaches the frame edge anywhere (VS-24)."""
    return bool(at_border(polygon, image_size, margin_px).any())


def cut_ends(
    polygon: np.ndarray, axis: np.ndarray, image_size: tuple[int, int], margin_px: float
) -> tuple[bool, bool]:
    """
    Whether the (+axis, -axis) ends touch the frame edge (H-10): an outline point within
    `margin_px` of the border of the (width, height) image, in that end's quarter of the fish.
    """
    points = np.asarray(polygon, float).reshape(-1, 2)
    border = at_border(points, image_size, margin_px)
    if not border.any():
        return False, False
    along = points @ axis
    low, high = along.min(), along.max()
    share = (along[border] - low) / max(high - low, 1e-9)
    return bool((share >= 0.75).any()), bool((share <= 0.25).any())


def head_vote(
    polygon: np.ndarray, centroid: np.ndarray, axis: np.ndarray, usable: tuple[bool, bool]
) -> tuple[float, float, dict[str, float]]:
    """
    (signed vote: > 0 = head toward +axis, confidence, per-cue votes) (H-7, H-9).

    Cues, each in [-1, 1]: "mass" (the centroid lies on the head side of the middle),
    "width" (the head is wider 10-20% from its end than the tail is 10-20% from its end,
    where the caudal peduncle lies) and "waist" (the narrowest section is near the tail
    and the outline flares again beyond it: the tail fin). The cues compare both ends, so
    with an unusable (cut) end there is no vote.
    """
    if not (usable[0] and usable[1]):
        return 0.0, 0.0, {}
    along = along_axis(polygon, centroid, axis)[:, 0]
    low, high = float(along.min()), float(along.max())
    length = high - low
    if length <= 0:
        return 0.0, 0.0, {}
    shares = (np.arange(PROFILE_STATIONS) + 0.5) / PROFILE_STATIONS  # 0: -axis end, 1: +axis end
    widths = width_profile(polygon, centroid, axis, low + shares * length)

    plus = widths[(shares >= 0.80) & (shares <= 0.90)].mean()
    minus = widths[(shares >= 0.10) & (shares <= 0.20)].mean()
    inner = np.flatnonzero((shares >= 0.05) & (shares <= 0.95))
    waist = int(inner[np.argmin(widths[inner])])
    beyond = widths[:waist] if shares[waist] < 0.5 else widths[waist + 1 :]
    flare = beyond.max() / max(widths[waist], 1e-9) if beyond.size else 0.0
    raw = {
        "mass": (-(low + high) / length) / MASS_FULL_VOTE,
        "width": ((plus - minus) / max(plus + minus, 1e-9)) / WIDTH_FULL_VOTE,
        "waist": (0.5 - shares[waist])
        / WAIST_FULL_VOTE
        * float(np.clip((flare - 1.0) / (FLARE_FULL_VOTE - 1.0), 0.0, 1.0)),
    }
    cues = {name: float(np.clip(value, -1.0, 1.0)) for name, value in raw.items()}
    vote = sum(CUE_WEIGHTS[name] * value for name, value in cues.items())
    vote /= sum(CUE_WEIGHTS.values())
    return float(vote), float(abs(vote)), cues


def measure_heading(
    fish: FishObservation, image_size: tuple[int, int], settings: HeadingSettings
) -> HeadingEstimate:
    """
    Heading of one fish from this frame alone (H-6 to H-11); image_size = (width, height).
    Keypoints (H-12) aren't used yet.
    """
    unknown = HeadingEstimate(None, None, None, None, None, 0.0, "unknown")
    try:
        centroid, axis = body_axis(fish.polygon)
    except ValueError:
        unknown.flags.add("ambiguous_shape")
        return unknown
    plus_cut, minus_cut = cut_ends(fish.polygon, axis, image_size, settings.edge_margin_px)
    vote, confidence, _ = head_vote(fish.polygon, centroid, axis, (not plus_cut, not minus_cut))
    unknown.confidence = confidence
    if plus_cut or minus_cut:
        unknown.flags.add("end_cut")
        return unknown
    if confidence < settings.min_confidence:  # H-20: never guess
        unknown.flags.add("ambiguous_shape")
        return unknown
    direction = axis if vote > 0 else -axis
    points = np.asarray(fish.polygon, float).reshape(-1, 2)
    along = points @ direction
    return HeadingEstimate(
        angle_image=float(np.arctan2(direction[1], direction[0])),
        direction_image=direction,
        yaw_base=None,
        head_px=points[int(np.argmax(along))],
        tail_px=points[int(np.argmin(along))],
        confidence=confidence,
        source="measured",
    )


class HeadingMemory:
    """Keeps each fish's heading across frames as the camera moves (H-13 to H-18)."""

    def __init__(self, settings: HeadingSettings | None = None):
        self.settings = settings or HeadingSettings()

    def update(
        self,
        fish: list[FishObservation],
        image_size: tuple[int, int],
        stamp_s: float,
        camera_pose: np.ndarray | None = None,  # 4x4 T_base_camera (H-5)
        intrinsics: np.ndarray | None = None,  # 3x3 K
        depth: np.ndarray | None = None,  # meters, aligned to the color image
    ) -> list[HeadingEstimate]:
        """One estimate per observation, in the same order."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget all fish (new observation of the table, e.g. after a pick)."""
        raise NotImplementedError
