"""
Color segmentor for auto_labeler.py: sampled colors -> outlines. Needs fish and table
colors to differ; on the current footage (white fish, gray table) use SAM.

1. CIELAB, not HSV: hue is noise at low saturation (catfish are dark gray-olive).
2. Each click adds a Gaussian cluster (patch mean + covariance): a dark back and a
   light belly get one each. Right-clicks add not-fish clusters.
3. Fish pixel: Mahalanobis distance to the nearest fish cluster <= tolerance and,
   if not-fish colors were sampled, more likely fish than not-fish.
4. Clean-up, keep the clicked regions (or all), split a region with several clicks
   at its narrowest points (touching fish).
5. Optional GrabCut on a band around each edge, then mask_to_polygon.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Vision/data
from pipeline_common import mask_to_polygon

SAMPLE_RADIUS = 4  # px around a click
MIN_SPREAD = 4.0  # minimum cluster std (Lab units), so one patch isn't too strict
MAX_CLUSTERS = 16  # older clusters are dropped beyond this


@dataclass
class Settings:
    tolerance: float = 3.0  # Mahalanobis distance; 3 keeps ~97% of a cluster's own colors
    keep_all: bool = False  # keep every matching region, not just clicked ones
    refine_edges: bool = True  # GrabCut on a band around each edge
    min_area_fraction: float = 0.0005  # smaller regions (share of the image) are dropped
    max_error_px: float = 1.5  # polygon may deviate this far from the mask edge...
    min_iou: float = 0.98  # ...while still matching the mask this well


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing, then Lab (L 0-100, a/b about -127..127)."""
    smooth = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=25, sigmaSpace=7)
    return cv2.cvtColor(smooth.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


class Cluster:
    """One Gaussian color cluster in Lab."""

    def __init__(self, pixels_lab: np.ndarray):
        self.mean = pixels_lab.mean(axis=0).astype(np.float32)
        cov = np.cov(pixels_lab.T) if len(pixels_lab) > 3 else np.zeros((3, 3))
        cov = cov + np.eye(3) * MIN_SPREAD**2
        # Cholesky factor of the inverse covariance: distance^2 = |(x - mean) @ factor|^2
        self.factor = np.linalg.cholesky(np.linalg.inv(cov)).astype(np.float32)
        self.log_det = float(np.log(np.linalg.det(cov)))

    def distance(self, lab: np.ndarray) -> np.ndarray:
        diff = (lab - self.mean) @ self.factor
        return np.sqrt(np.einsum("...i,...i->...", diff, diff))

    def log_likelihood(self, distance: np.ndarray) -> np.ndarray:
        return -0.5 * distance**2 - 0.5 * self.log_det


def sample_patch(lab: np.ndarray, x: float, y: float) -> np.ndarray | None:
    """Lab pixels around (x, y), minus those far from the patch median (edge clicks)."""
    h, w = lab.shape[:2]
    cx, cy = round(x), round(y)
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    r = SAMPLE_RADIUS
    patch = lab[max(cy - r, 0) : cy + r + 1, max(cx - r, 0) : cx + r + 1].reshape(-1, 3)
    close = np.linalg.norm(patch - np.median(patch, axis=0), axis=1) < 10.0
    return patch[close] if close.sum() >= 3 else patch


def lab_to_rgb(color_lab: np.ndarray) -> tuple[int, int, int]:
    rgb = cv2.cvtColor(color_lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)
    return tuple(int(v) for v in np.clip(rgb.reshape(3) * 255, 0, 255))


class ColorSegmentor:
    """Segments one image; make one per image. Clusters are passed in."""

    def __init__(self, image_bgr: np.ndarray):
        self.image = image_bgr
        self.lab = to_lab(image_bgr)
        self.short = min(image_bgr.shape[:2])
        self._cache_key = None
        self._maps = None

    def maps(self, objects: list[Cluster], background: list[Cluster]):
        """(distance to the nearest object cluster, object-beats-background mask), cached."""
        key = (tuple(objects), tuple(background))
        if key != self._cache_key:
            distances = [c.distance(self.lab) for c in objects]
            nearest = np.minimum.reduce(distances)
            wins = np.ones(self.lab.shape[:2], bool)
            if background:
                obj_ll = np.maximum.reduce(
                    [c.log_likelihood(d) for c, d in zip(objects, distances)]
                )
                for c in background:
                    wins &= obj_ll > c.log_likelihood(c.distance(self.lab))
            self._maps, self._cache_key = (nearest, wins), key
        return self._maps

    def enhanced(self, objects: list[Cluster], background: list[Cluster], tolerance: float):
        """Grayscale image: sampled colors bright, everything else dark."""
        distance, wins = self.maps(objects, background)
        value = np.clip(1.0 - distance / (2.0 * tolerance), 0.0, 1.0)
        value[~wins] *= 0.25
        return (value * 255).astype(np.uint8)

    def segment(
        self,
        objects: list[Cluster],
        background: list[Cluster],
        seeds: list[tuple[float, float]],
        settings: Settings,
    ) -> list[np.ndarray]:
        """One 0/1 mask per object found."""
        if not objects:
            return []
        h, w = self.lab.shape[:2]
        distance, wins = self.maps(objects, background)
        mask = ((distance <= settings.tolerance) & wins).astype(np.uint8)

        k = max(3, int(self.short * 0.006) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)  # fills holes (eyes, glare)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = settings.min_area_fraction * h * w
        seeds_in = {i: [] for i in range(1, n)}
        for x, y in seeds:
            hit = self._snap(labels, round(x), round(y))
            if hit:
                seeds_in[labels[hit[1], hit[0]]].append(hit)

        regions = []
        for i in range(1, n):
            if seeds_in[i]:
                region = (labels == i).astype(np.uint8)
                if len(seeds_in[i]) > 1:
                    regions += self._split(region, seeds_in[i])
                else:
                    regions.append(region)
            elif settings.keep_all and stats[i, cv2.CC_STAT_AREA] >= min_area:
                regions.append((labels == i).astype(np.uint8))

        if settings.refine_edges:
            regions = [self._refine(r, regions) for r in regions]
        return [r for r in regions if r.sum() >= max(min_area, 1)]

    def _snap(self, labels: np.ndarray, x: int, y: int) -> tuple[int, int] | None:
        """The click, or the nearest region pixel close by (a click on glare or an edge)."""
        h, w = labels.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        if labels[y, x]:
            return x, y
        r = max(5, int(self.short * 0.02))
        x0, y0 = max(x - r, 0), max(y - r, 0)
        ys, xs = np.nonzero(labels[y0 : y + r + 1, x0 : x + r + 1])
        if len(xs) == 0:
            return None
        nearest = np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2)
        return int(xs[nearest] + x0), int(ys[nearest] + y0)

    def _split(self, region: np.ndarray, seeds: list[tuple[int, int]]) -> list[np.ndarray]:
        """
        Split a region between several clicks at its narrowest points: pixels deeper
        than t (distance to the edge) form cores, which separate first where touching
        fish meet. The smallest t giving each click its own core wins, then every pixel
        joins the nearest core (no such t: the nearest click). Not color edges: a belly
        edge is often stronger than the seam. Not cv2.watershed: it floods by neighbor
        color differences, not depth.
        """
        x, y, w, h = cv2.boundingRect(region)
        crop = region[y : y + h, x : x + w]
        local = [(sx - x, sy - y) for sx, sy in seeds]
        depth = cv2.distanceTransform(crop, cv2.DIST_L2, 5)

        cores = np.zeros(crop.shape, np.int32)  # labeled click index + 1
        for t in np.arange(1.0, float(depth.max()), 1.0):
            core = (depth > t).astype(np.uint8)
            _, components = cv2.connectedComponents(core)
            ys, xs = np.nonzero(core)
            owners = [
                components[ys, xs][np.argmin((xs - sx) ** 2 + (ys - sy) ** 2)] for sx, sy in local
            ]
            if len(set(owners)) == len(local):
                for i, c in enumerate(owners):
                    cores[components == c] = i + 1
                break
        else:
            for i, (sx, sy) in enumerate(local):
                cv2.circle(cores, (sx, sy), 2, i + 1, -1)

        # Each region pixel takes the label of its nearest core pixel.
        not_core = np.where(cores > 0, 0, 255).astype(np.uint8)
        _, nearest = cv2.distanceTransformWithLabels(
            not_core, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        zy, zx = np.nonzero(cores)
        lookup = np.zeros(int(nearest.max()) + 1, np.int32)
        lookup[nearest[zy, zx]] = cores[zy, zx]
        owner = lookup[nearest] * (crop > 0)

        parts = []
        for i in range(len(seeds)):
            part = np.zeros_like(region)
            part[y : y + h, x : x + w] = owner == i + 1
            parts.append(part)
        return parts

    def _refine(self, region: np.ndarray, all_regions: list[np.ndarray]) -> np.ndarray:
        """GrabCut re-decides a band around the edge; the core and the far outside stay fixed."""
        band = max(3, int(self.short * 0.012))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        outer = cv2.dilate(region, kernel)
        inner = cv2.erode(region, kernel)
        x, y, bw, bh = cv2.boundingRect(outer)
        pad = band * 2
        h, w = region.shape
        crop = (
            slice(max(y - pad, 0), min(y + bh + pad, h)),
            slice(max(x - pad, 0), min(x + bw + pad, w)),
        )

        gc = np.full(region[crop].shape, cv2.GC_BGD, np.uint8)
        gc[outer[crop] > 0] = cv2.GC_PR_BGD
        gc[region[crop] > 0] = cv2.GC_PR_FGD
        gc[inner[crop] > 0] = cv2.GC_FGD
        for other in all_regions:  # never grow into a neighboring object
            if other is not region:
                gc[(other[crop] > 0) & (region[crop] == 0)] = cv2.GC_BGD
        if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
            return region  # too thin, or no background around it
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(self.image[crop], gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return region
        refined = np.zeros_like(region)
        refined[crop] = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return region
        out = np.zeros_like(region)  # largest piece, holes filled
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        return out

    def polygons(
        self, regions: list[np.ndarray], settings: Settings
    ) -> list[list[tuple[float, float]]]:
        """Each region as a simplified, normalized polygon."""
        h, w = self.lab.shape[:2]
        min_area = settings.min_area_fraction * h * w
        out = []
        for region in regions:
            points = mask_to_polygon(region, settings.max_error_px, settings.min_iou, min_area)
            if points:
                out.append(points)
        return out
