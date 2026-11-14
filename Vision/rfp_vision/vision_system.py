"""
The Vision System as the Control System uses it (VS-50): a camera, the detector and the
locator behind one `observe()` that returns a fish list, plus the hold monitor. The same
object works with a live D405, a recording or a plain video. No ROS.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rfp_vision.camera import Camera, Frame
from rfp_vision.detector import Detection
from rfp_vision.fish_list import FishList, FishLocator, merge_stable
from rfp_vision.hold_monitor import HoldMonitor
from rfp_vision.settings import VisionSettings


@dataclass
class Observation:
    fish_list: FishList
    frame: Frame  # the last frame used
    detections: list[Detection]


class VisionSystem:
    def __init__(self, camera: Camera, detector, locator: FishLocator, settings: VisionSettings):
        """`detector`: anything with detect(image) -> list[Detection] (a FishDetector)."""
        self.camera = camera
        self.detector = detector
        self.locator = locator
        self.settings = settings
        self.hold = HoldMonitor(settings.hold, detector)

    def frame(self) -> Frame:
        return self.camera.read()

    def observe(
        self, t_base_camera: np.ndarray, on_frame: Callable[[Frame], None] | None = None
    ) -> Observation:
        """
        One observation with the arm stopped (VS-13): `stable_frames` frames in a row, each
        turned into a fish list; a fish is valid only if it is in all of them at the same
        place (VS-52). `t_base_camera` is the camera's pose in {B} during the observation.
        """
        lists, frame, detections = [], None, []
        for _ in range(max(1, self.settings.stable_frames)):
            frame = self.camera.read()
            detections = self.detector.detect(frame.color)
            lists.append(self.locator.locate(frame, detections, t_base_camera))
            if on_frame is not None:
                on_frame(frame)
        return Observation(merge_stable(lists, self.settings), frame, detections)

    def close(self) -> None:
        self.camera.close()
