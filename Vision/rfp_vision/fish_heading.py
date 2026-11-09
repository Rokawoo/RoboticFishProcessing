"""
Head direction of each detected fish, from the geometry of its segmentation mask, kept
while the camera moves closer and the head leaves the frame (Requirement Spec VS-40,
VS-60). Skeleton: requirements and interfaces only, no implementation yet.

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


def body_axis(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unsigned unit long axis of an outline (H-6)."""
    raise NotImplementedError


def cut_ends(
    polygon: np.ndarray, axis: np.ndarray, image_size: tuple[int, int], margin_px: float
) -> tuple[bool, bool]:
    """Whether the (+axis, -axis) ends touch the frame edge (H-10)."""
    raise NotImplementedError


def head_vote(
    polygon: np.ndarray, centroid: np.ndarray, axis: np.ndarray, usable: tuple[bool, bool]
) -> tuple[float, float, dict[str, float]]:
    """(signed vote: > 0 = head toward +axis, confidence, per-cue votes) (H-7, H-9)."""
    raise NotImplementedError


def measure_heading(
    fish: FishObservation, image_size: tuple[int, int], settings: HeadingSettings
) -> HeadingEstimate:
    """Heading of one fish from this frame alone (H-6 to H-12)."""
    raise NotImplementedError


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
