"""
Which end of a fish is its head, from its outline (VS-40, VS-60): the body axis from the
outline's moments, then votes from the width profile (the head end is wide and blunt), where
the area and the girth lie (catfish are fattest toward the head) and the tail fork, plus the
body's slope from depth when given. A cut-off end isn't used. Weights and scales are first
values, to be tuned on labeled real footage. Not done: keypoints from a pose model, a heading
kept per fish across frames. No ROS.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HeadingSettings:
    min_confidence: float = 0.6  # below this the heading is unknown
    edge_margin_px: float = 3.0  # an outline this close to the image's border is cut off


@dataclass
class FishObservation:
    """One detected fish in one frame."""

    polygon: np.ndarray  # (n, 2) image px
    confidence: float
    keypoints: np.ndarray | None = None  # from a pose model (not used yet): (k, 3)


@dataclass
class HeadingEstimate:
    """The heading of one fish in one frame."""

    angle_image: float | None  # rad, image coordinates; None if unknown
    direction_image: np.ndarray | None  # unit vector tail -> head
    yaw_base: float | None  # rad in {B}; None without a camera pose
    head_px: np.ndarray | None
    tail_px: np.ndarray | None
    confidence: float
    source: str  # "measured", "keypoints", "kept", "unknown"
    flags: set[str] = field(default_factory=set)


# Cues of head_vote: weight, and the raw value that counts as a full vote. First
# values, from the outlines of the 3D-printed stand-in fish seen from above: centroid
# 0.02-0.03 toward the head, end widths differing by 0.17-0.20, tail fin 2x its peduncle.
CUE_WEIGHTS = {"mass": 1.0, "width": 2.0, "waist": 2.0, "girth": 1.0, "height": 2.0}
MASS_FULL_VOTE = 0.04  # centroid offset from the middle of the fish, share of half its length
GIRTH_FULL_VOTE = 0.1  # the girth's middle this share of the length from the fish's middle
GIRTH_SPAN = (0.15, 0.85)  # where the girth is looked for: the tail fin flares beyond
GIRTH_SHARE = 0.9  # the girth: the sections at least this share of the widest one
WIDTH_FULL_VOTE = 0.15  # (head-end width - tail-end width) / their sum, 10-20% from each end
WAIST_FULL_VOTE = 0.25  # distance of the narrowest section from the middle, share of the length
FLARE_FULL_VOTE = 1.5  # widest section beyond the waist / the waist: a tail fin, not a taper
HEIGHT_FULL_VOTE = 0.02  # m the fish's surface is higher a quarter length toward the head
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
    Centroid and unsigned unit long axis of an outline, from the area's second-order
    moments. Of the two directions, the one with x > 0 (y > 0 if x is 0) is returned, so
    the same outline always gives the same axis.
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
    Whether the (+axis, -axis) ends touch the frame edge: an outline point within
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
    polygon: np.ndarray,
    centroid: np.ndarray,
    axis: np.ndarray,
    usable: tuple[bool, bool],
    height_rise: float | None = None,
) -> tuple[float, float, dict[str, float]]:
    """
    (signed vote: > 0 = head toward +axis, confidence, per-cue votes).

    Cues, each in [-1, 1]: "mass" (the centroid on the head side), "width" (wider near the
    head end than near the tail end), "waist" (narrowest near the tail, the tail fin flaring
    beyond), "girth" (widest toward the head) and, with depth, "height" (`height_rise`: how
    much higher the surface is a quarter length toward +axis than toward -axis). The cues
    compare both ends: with a cut end there is no vote.
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
    body = np.flatnonzero((shares >= GIRTH_SPAN[0]) & (shares <= GIRTH_SPAN[1]))
    smooth = np.convolve(widths, np.ones(3) / 3, mode="same")  # one jagged station isn't girth
    girth = body[smooth[body] >= GIRTH_SHARE * smooth[body].max()]  # a flat top: its middle
    raw = {
        "mass": (-(low + high) / length) / MASS_FULL_VOTE,
        "width": ((plus - minus) / max(plus + minus, 1e-9)) / WIDTH_FULL_VOTE,
        "waist": (0.5 - shares[waist])
        / WAIST_FULL_VOTE
        * float(np.clip((flare - 1.0) / (FLARE_FULL_VOTE - 1.0), 0.0, 1.0)),
        "girth": (float(shares[girth].mean()) - 0.5) / GIRTH_FULL_VOTE,
    }
    if height_rise is not None:
        raw["height"] = height_rise / HEIGHT_FULL_VOTE
    cues = {name: float(np.clip(value, -1.0, 1.0)) for name, value in raw.items()}
    vote = sum(CUE_WEIGHTS[name] * value for name, value in cues.items())
    # A level surface says nothing about the head: the height cue weighs as much as it says.
    weights = sum(CUE_WEIGHTS[name] for name in cues if name != "height")
    vote /= weights + CUE_WEIGHTS["height"] * abs(cues.get("height", 0.0))
    return float(vote), float(abs(vote)), cues


def measure_heading(
    fish: FishObservation, image_size: tuple[int, int], settings: HeadingSettings
) -> HeadingEstimate:
    """
    Heading of one fish from this frame alone; image_size = (width, height).
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
    if confidence < settings.min_confidence:  # never guess
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
