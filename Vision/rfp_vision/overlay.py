"""Drawing a fish list on its camera image: outlines, IDs, failure codes, grasp points,
headings (CS-73, and the image a VLM would be shown, Table IF-5). OpenCV only."""

import math

import cv2
import numpy as np

from rfp_vision.fish_list import FishList
from rfp_vision.geometry import Intrinsics, to_frame

VALID = (120, 220, 0)  # BGR
FAILED = (0, 140, 255)
CHOSEN = (255, 170, 0)
INK = (255, 255, 255)
ARROW_M = 0.12  # heading arrow length on the table


def _tag(image: np.ndarray, text: str, x: int, y: int, color, scale: float) -> None:
    """Text on a filled tag with its bottom-left corner at (x, y), kept inside the image."""
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, round(1.5 * scale))
    (tw, th), base = cv2.getTextSize(text, font, size, thickness)
    pad = max(2, round(3 * scale))
    x = min(max(x, 0), max(image.shape[1] - tw - 2 * pad, 0))
    y = max(y, th + base + 2 * pad)
    cv2.rectangle(image, (x, y - th - base - 2 * pad), (x + tw + 2 * pad, y), color, -1)
    ink = (0, 0, 0) if sum(color) > 380 else INK
    cv2.putText(image, text, (x + pad, y - base - pad), font, size, ink, thickness, cv2.LINE_AA)


def _pixel(intrinsics: Intrinsics, t_camera_base: np.ndarray, point) -> tuple[int, int] | None:
    in_camera = to_frame(t_camera_base, np.asarray(point, float))[0]
    if in_camera[2] <= 1e-6:
        return None
    u, v = intrinsics.project(in_camera)[0]
    return round(float(u)), round(float(v))


def draw_fish_list(
    image: np.ndarray,
    fish_list: FishList,
    intrinsics: Intrinsics,
    chosen_id: int | None = None,
) -> np.ndarray:
    """A copy of the image with every fish of the list drawn on it."""
    out = image.copy()
    scale = max(image.shape[:2]) / 1280
    line = max(1, round(2 * scale))
    t_camera_base = np.linalg.inv(fish_list.camera_pose)
    for fish in fish_list.fish:
        color = CHOSEN if fish.fish_id == chosen_id else VALID if fish.vision_ok else FAILED
        outline = np.round(fish.mask_polygon).astype(np.int32)
        cv2.polylines(out, [outline], True, color, line + (fish.fish_id == chosen_id), cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(outline)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, line)
        text = f"{fish.fish_id}  {fish.confidence:.2f}"
        if not fish.vision_ok:
            text += f"  {fish.failure_code}"
        elif not fish.heading_known:
            text += "  head?"
        _tag(out, text, x, y, color, scale)
        if fish.grasp_pose is None:
            continue
        point = fish.grasp_pose[:3, 3]
        center = _pixel(intrinsics, t_camera_base, point)
        if center is None:
            continue
        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, round(24 * scale), line + 1)
        ahead = point + ARROW_M * np.array([math.cos(fish.heading), math.sin(fish.heading), 0.0])
        tip = _pixel(intrinsics, t_camera_base, ahead)
        if tip is not None and fish.heading_known:
            cv2.arrowedLine(out, center, tip, color, line + 1, cv2.LINE_AA, tipLength=0.3)
    return out


def add_status(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """The image under a dark bar of text lines."""
    h, w = image.shape[:2]
    scale = max(h, w) / 1280
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(1.4 * scale))
    pad = round(8 * scale)
    size = 0.55 * scale
    widest = max(cv2.getTextSize(t, font, size, thickness)[0][0] for t in lines)
    size *= min(1.0, (w - 2 * pad) / max(widest, 1))
    height = max(cv2.getTextSize(t, font, size, thickness)[0][1] for t in lines)
    step = height + pad
    bar = np.full((step * len(lines) + pad, w, 3), 32, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (pad, pad + step * i + height), font, size, INK, thickness,
                    cv2.LINE_AA)  # fmt: skip
    return np.vstack([bar, image])
