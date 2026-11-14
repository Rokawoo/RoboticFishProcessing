"""
Pixels to points: camera intrinsics, depth sampling and the mapping from the image to the
arm base frame {B} (Requirement Spec C.4). Plain NumPy/OpenCV, meters and radians, no ROS.

Frames: the camera optical frame has z forward, x right and y down. T_base_camera is the
4x4 pose of that frame in {B} (TCP pose at capture and the hand-eye calibration).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole model of the color image (pixels). Lens distortion is ignored."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """For cameras that don't report intrinsics: square pixels, center at the middle."""
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(f, f, width / 2, height / 2, width, height)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """(n, 3) directions (x, y, 1) in the camera frame through (n, 2) pixels."""
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        x = (pixels[:, 0] - self.cx) / self.fx
        y = (pixels[:, 1] - self.cy) / self.fy
        return np.column_stack([x, y, np.ones(len(pixels))])

    def project(self, points: np.ndarray) -> np.ndarray:
        """(n, 2) pixels of (n, 3) points in the camera frame (z > 0)."""
        points = np.asarray(points, float).reshape(-1, 3)
        u = self.fx * points[:, 0] / points[:, 2] + self.cx
        v = self.fy * points[:, 1] / points[:, 2] + self.cy
        return np.column_stack([u, v])


def deproject(intrinsics: Intrinsics, pixel: tuple[float, float], depth: float) -> np.ndarray:
    """Point in the camera frame of a pixel at `depth` (distance along the optical axis)."""
    return intrinsics.rays(np.array([pixel]))[0] * depth


def pose_down(xyz, yaw: float) -> np.ndarray:
    """4x4 pose at `xyz` whose z axis points straight down and whose x axis points along `yaw`."""
    c, s = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]]
    pose[:3, 3] = xyz
    return pose


def to_frame(t_a_b: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """(n, 3) points given in frame b, in frame a."""
    points_b = np.asarray(points_b, float).reshape(-1, 3)
    return points_b @ t_a_b[:3, :3].T + t_a_b[:3, 3]


def pixels_on_plane(
    intrinsics: Intrinsics, t_base_camera: np.ndarray, pixels: np.ndarray, plane_z: float
) -> np.ndarray:
    """
    (n, 3) points in {B} where the rays through `pixels` meet the horizontal plane z = plane_z.
    NaN for rays that don't reach the plane in front of the camera.
    """
    directions = intrinsics.rays(pixels) @ t_base_camera[:3, :3].T
    origin = t_base_camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (plane_z - origin[2]) / directions[:, 2]
    distance[~np.isfinite(distance) | (distance <= 0)] = np.nan
    return origin + directions * distance[:, None]


def depth_at(
    depth: np.ndarray,
    pixel: tuple[float, float],
    radius_px: float,
    valid_range: tuple[float, float],
) -> tuple[float | None, float]:
    """
    (median valid depth, share of valid pixels) in a disc around `pixel` (VS-45, VS-66).
    The depth is None if no pixel in the disc is valid.
    """
    h, w = depth.shape[:2]
    r = max(1, round(radius_px))
    u, v = round(float(pixel[0])), round(float(pixel[1]))
    x0, x1, y0, y1 = max(u - r, 0), min(u + r + 1, w), max(v - r, 0), min(v + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None, 0.0
    window = depth[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    values = window[disc]
    valid = values[(values >= valid_range[0]) & (values <= valid_range[1])]
    if valid.size == 0:
        return None, 0.0
    return float(np.median(valid)), valid.size / values.size


def polygon_mask(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled 0/1 uint8 mask of a pixel polygon in an image of `shape` (height, width)."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    return mask


def inside_polygon(polygon: np.ndarray, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Which (n, 2) points lie inside the polygon, at least `margin` from its outline."""
    contour = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    points = np.asarray(points, float).reshape(-1, 2)
    return np.array(
        [cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= margin for x, y in points]
    )


def touches_border(polygon: np.ndarray, size: tuple[int, int], margin_px: float) -> bool:
    """Whether a pixel polygon comes within `margin_px` of the edge of a (width, height) image."""
    polygon = np.asarray(polygon, float)
    w, h = size
    return bool(
        (polygon[:, 0] <= margin_px).any()
        or (polygon[:, 1] <= margin_px).any()
        or (polygon[:, 0] >= w - 1 - margin_px).any()
        or (polygon[:, 1] >= h - 1 - margin_px).any()
    )
