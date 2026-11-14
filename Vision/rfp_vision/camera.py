"""
Camera sources (VS-10): the RealSense D405 with depth aligned to color, RealSense
recordings, and plain OpenCV devices or video files (color only). All give Frames; the
rest of the vision code doesn't know which source it has. No ROS.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from rfp_vision.geometry import Intrinsics

REALSENSE_EXTS = frozenset({".bag", ".db3"})


class CameraError(RuntimeError):
    """The camera can't be opened or stopped delivering frames."""


@dataclass
class Frame:
    color: np.ndarray  # BGR, (height, width, 3) uint8
    depth: np.ndarray | None  # meters, aligned to color, 0 = no measurement; None: no depth
    intrinsics: Intrinsics
    stamp: float  # time.monotonic() at capture, for ages and timeouts
    wall_time: float  # time.time() at capture, for logs (IF-13)
    frame_id: int  # counts up per source (VS-12)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.color.shape[1], self.color.shape[0]


class Camera(Protocol):
    name: str
    has_depth: bool

    def read(self, timeout_s: float = 2.0) -> Frame:
        """The next frame. Raises CameraError if none arrives in time."""

    def close(self) -> None: ...


class RealSenseCamera:
    """A RealSense camera (first found, or by serial number) or a recording (.bag, .db3)."""

    def __init__(
        self,
        serial: str | None = None,
        recording: Path | None = None,
        size: tuple[int, int] = (1280, 720),
        fps: int = 30,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise CameraError("RealSense cameras need pyrealsense2 (the rfp environment)") from None
        self._rs = rs
        self._pipeline = rs.pipeline()
        self._count = 0
        self._to_bgr = {
            rs.format.rgb8: cv2.COLOR_RGB2BGR,
            rs.format.rgba8: cv2.COLOR_RGBA2BGR,
            rs.format.bgra8: cv2.COLOR_BGRA2BGR,
            rs.format.yuyv: cv2.COLOR_YUV2BGR_YUYV,
        }
        config = rs.config()
        if recording is not None:
            config.enable_device_from_file(str(recording), repeat_playback=True)
            self.name = f"RealSense recording {recording.name}"
        else:
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, fps)
            config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, fps)
            self.name = f"RealSense {serial or '(first found)'}"
        try:
            profile = self._pipeline.start(config)
        except RuntimeError as e:
            raise CameraError(f"can't start {self.name}: {e}") from None
        streams = {s.stream_type() for s in profile.get_streams()}
        if rs.stream.color not in streams:
            self._pipeline.stop()
            raise CameraError(f"{self.name} has no color stream")
        self.has_depth = rs.stream.depth in streams
        self._align = rs.align(rs.stream.color) if self.has_depth else None
        if recording is None:
            device = profile.get_device()
            self.name = (
                f"{device.get_info(rs.camera_info.name)} "
                f"{device.get_info(rs.camera_info.serial_number)}"
            )

    def read(self, timeout_s: float = 2.0) -> Frame:
        rs = self._rs
        try:
            frames = self._pipeline.wait_for_frames(int(timeout_s * 1000))
        except RuntimeError as e:
            raise CameraError(f"no frame from {self.name} in {timeout_s:g} s: {e}") from None
        stamp, wall_time = time.monotonic(), time.time()
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            raise CameraError(f"{self.name} sent a frame without a color image")
        data = np.asanyarray(color.get_data())
        fmt = color.get_profile().format()
        if fmt == rs.format.bgr8:
            image = data.copy()  # the SDK reuses the buffer for the next frame
        elif fmt in self._to_bgr:
            image = cv2.cvtColor(data, self._to_bgr[fmt])
        else:
            raise CameraError(f"unsupported color format {fmt}")
        k = color.get_profile().as_video_stream_profile().get_intrinsics()
        intrinsics = Intrinsics(k.fx, k.fy, k.ppx, k.ppy, k.width, k.height)
        depth = None
        if self.has_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:  # the D405 counts in 0.1 mm, other models in 1 mm: ask the frame
                raw = np.asanyarray(depth_frame.get_data())
                depth = raw.astype(np.float32) * depth_frame.get_units()
        self._count += 1
        return Frame(image, depth, intrinsics, stamp, wall_time, self._count)

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass


class OpenCVCamera:
    """A webcam (device index) or a video file, color only. A file starts over at its end."""

    has_depth = False

    def __init__(self, source: int | Path, hfov_deg: float, size: tuple[int, int] = (1280, 720)):
        self._source = source
        self._hfov_deg = hfov_deg
        self._is_file = not isinstance(source, int)
        self._capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not self._capture.isOpened():
            raise CameraError(f"OpenCV can't open camera or video '{source}'")
        if self._is_file:
            self.name = f"video {Path(source).name}"
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            self._interval = 1.0 / fps if fps and not math.isnan(fps) and fps <= 1000 else 1 / 30
        else:
            self.name = f"camera {source}"
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, where supported
            self._interval = 0.0
        self._count = 0
        self._next = time.monotonic()

    def read(self, timeout_s: float = 2.0) -> Frame:
        if self._is_file:  # play at the video's own rate, like a live camera
            time.sleep(max(0.0, self._next - time.monotonic()))
            self._next = max(self._next + self._interval, time.monotonic())
        ok, image = self._capture.read()
        if not ok and self._is_file:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
        if not ok:
            raise CameraError(f"no frame from {self.name}")
        self._count += 1
        h, w = image.shape[:2]
        intrinsics = Intrinsics.from_fov(w, h, self._hfov_deg)
        return Frame(image, None, intrinsics, time.monotonic(), time.time(), self._count)

    def close(self) -> None:
        self._capture.release()


def list_realsense() -> list[tuple[str, str]]:
    """(serial number, name) of each connected RealSense camera; [] without pyrealsense2."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    info = rs.camera_info
    return [
        (d.get_info(info.serial_number), d.get_info(info.name))
        for d in rs.context().query_devices()
    ]


def open_camera(
    device: str | None, hfov_deg: float = 87.0, size: tuple[int, int] = (1280, 720), fps: int = 30
) -> Camera:
    """
    Open a video input device:
        None or "realsense"     the first RealSense camera
        a serial number         that RealSense camera (see list_realsense)
        "0", "1", ...           an OpenCV camera index (color only)
        a .bag or .db3 file     a RealSense recording, replayed in a loop
        another file            a video, replayed in a loop (color only)
    `hfov_deg` is the horizontal field of view assumed for sources without intrinsics.
    """
    if device is None or device.lower() == "realsense":
        return RealSenseCamera(size=size, fps=fps)
    if device.isdigit() and len(device) <= 2:
        return OpenCVCamera(int(device), hfov_deg, size)
    path = Path(device)
    if path.suffix:
        if not path.is_file():
            raise CameraError(f"video not found: {path}")
        if path.suffix.lower() in REALSENSE_EXTS:
            return RealSenseCamera(recording=path)
        return OpenCVCamera(path, hfov_deg, size)
    return RealSenseCamera(serial=device, size=size, fps=fps)
