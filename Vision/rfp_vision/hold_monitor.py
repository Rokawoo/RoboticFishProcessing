"""
Vision check that a held fish is still on the gripper. The camera sits on the tool flange,
so a held fish stays where it is in the image while the arm moves; a fish that was not
picked up, or dropped, stays behind: its outline shrinks and moves, and the surface that
was close to the camera is gone.

Use: `set_background(frame)` with nothing held (the cups and bracket the camera always
sees), `start(frame)` when the cups are on the fish, then `update(frame)` for every frame
while the arm lifts and carries. Two cues, each used when it is available:

    mask   the detector's outline now, against the outline at the grasp (IoU)
    depth  how much of the close surface seen at the grasp is still at its distance

A frame counts as "fish there" if any available cue says so, and the fish counts as
dropped after `lost_frames` such frames in a row without it. Depth needs the fish at
least 7 cm from the D405's lens (its minimum range). Plain NumPy/OpenCV, no ROS.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from rfp_vision.camera import Frame
from rfp_vision.geometry import polygon_mask
from rfp_vision.settings import HoldSettings

WORK_WIDTH = 320  # masks and depth are compared at this width (px)


@dataclass
class HoldStatus:
    held: bool | None  # this frame: fish there, not there, or None if no cue could be read
    dropped: bool  # not there for `lost_frames` frames in a row
    mask_iou: float | None = None
    depth_share: float | None = None

    def __str__(self) -> str:
        parts = []
        if self.mask_iou is not None:
            parts.append(f"outline IoU {self.mask_iou:.2f}")
        if self.depth_share is not None:
            parts.append(f"close surface {self.depth_share:.0%}")
        state = {True: "held", False: "not seen", None: "unknown"}[self.held]
        return f"{state} ({', '.join(parts) or 'no cue'})"


class HoldMonitor:
    def __init__(self, settings: HoldSettings, detector=None):
        """`detector`: a FishDetector for the mask cue, or None for depth only."""
        self.settings = settings
        self._detector = detector if settings.use_mask else None
        self._background: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._region: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._lost = 0

    @property
    def active(self) -> bool:
        return self._mask is not None or self._region is not None

    @property
    def cues(self) -> list[str]:
        """The cues the current reference supports."""
        named = (("mask", self._mask), ("depth", self._region))
        return [name for name, reference in named if reference is not None]

    def set_background(self, frame: Frame) -> None:
        """What is close to the camera with nothing held: the gripper itself. Not a fish."""
        self._background = None
        if self.settings.use_depth and frame.depth is not None:
            self._background = self._near(self._small(frame.depth))

    def start(self, frame: Frame) -> list[str]:
        """Take the reference with the cups on the fish. Returns the usable cues ([] = none)."""
        self.stop()
        s = self.settings
        if self._detector is not None:
            found = self._detector.detect(frame.color)
            if found:  # largest outline first: the fish under the gripper fills the view
                self._mask = self._outline_mask(found[0].polygon, frame)
        if s.use_depth and frame.depth is not None:
            depth = self._small(frame.depth)
            region = self._near(depth)
            if self._background is not None:
                region &= ~self._background
            if region.mean() >= s.min_region_share:
                self._region, self._depth = region, depth
        return self.cues

    def update(self, frame: Frame) -> HoldStatus:
        """Compare a frame with the reference."""
        s = self.settings
        iou = share = None
        if self._mask is not None:
            iou = 0.0
            for detection in self._detector.detect(frame.color):
                mask = self._outline_mask(detection.polygon, frame)
                union = np.count_nonzero(mask | self._mask)
                if union:
                    iou = max(iou, np.count_nonzero(mask & self._mask) / union)
        if self._region is not None and frame.depth is not None:
            depth = self._small(frame.depth)
            same = np.abs(depth - self._depth) <= s.depth_tolerance_m
            share = float(np.count_nonzero(same & self._region & (depth > 0)))
            share /= np.count_nonzero(self._region)
        votes = []
        if iou is not None:
            votes.append(iou >= s.min_iou)
        if share is not None:
            votes.append(share >= s.min_depth_share)
        held = any(votes) if votes else None
        if held is False:
            self._lost += 1
        elif held:
            self._lost = 0
        return HoldStatus(held, self._lost >= s.lost_frames, iou, share)

    def stop(self) -> None:
        self._mask = self._region = self._depth = None
        self._lost = 0

    def _small(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape[:2]
        size = (WORK_WIDTH, max(1, round(h * WORK_WIDTH / w)))
        return cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)  # no mixing with holes

    def _near(self, depth: np.ndarray) -> np.ndarray:
        return (depth > 0) & (depth < self.settings.near_m)

    def _outline_mask(self, polygon: np.ndarray, frame: Frame) -> np.ndarray:
        w, h = frame.size
        scale = WORK_WIDTH / w
        return polygon_mask(polygon * scale, (max(1, round(h * scale)), WORK_WIDTH)).astype(bool)
