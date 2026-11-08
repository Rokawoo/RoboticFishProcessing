"""
Click-to-segment with SAM 2 (Segment Anything) through Ultralytics, for auto_labeler.py.

Each fish's clicks are positive points and the right-clicks negative points; SAM
returns that fish's mask. Unlike color thresholds, SAM separates a white fish
from a reflective gray table. The image is encoded once (set_image) and reused
for every click: ~0.1 s per click; a new image costs ~10 ms on a GPU, ~1 s on a
CPU. The first prompt on a GPU takes ~15 s (CUDA warm-up).
"""

from pathlib import Path

import numpy as np

SAM_MODEL = "sam2.1_t.pt"  # tiny, 75 MB; sam2.1_b.pt is slower and slightly finer
MODEL_PATH = Path(__file__).resolve().parent / SAM_MODEL  # downloaded here on first use


class SamSegmenter:
    def __init__(self, model_path: Path | None = None):
        from ultralytics.models.sam import SAM2Predictor

        overrides = {
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "model": str(model_path or MODEL_PATH),
            "save": False,
            "verbose": False,
        }
        self.predictor = SAM2Predictor(overrides=overrides)
        self.image_key = None

    def set_image(self, image_bgr: np.ndarray, key) -> None:
        """Encode the image, unless `key` says it is already the current one."""
        if key != self.image_key:
            self.predictor.set_image(image_bgr)
            self.image_key = key

    def segment(
        self, objects: list[list[tuple[float, float]]], negatives: list[tuple[float, float]]
    ) -> list[np.ndarray]:
        """
        One 0/1 mask per object (list of its points). `negatives` apply to every
        object, and each object's first point is a negative for all the others,
        so touching fish don't merge.
        """
        masks = []
        for i, positives in enumerate(objects):
            others = [points[0] for j, points in enumerate(objects) if j != i]
            negative = [*negatives, *others]
            points = [[float(x), float(y)] for x, y in [*positives, *negative]]
            labels = [1] * len(positives) + [0] * len(negative)
            # One extra nesting level = all points belong to one object.
            result = self.predictor(points=[points], labels=[labels])[0]
            if result.masks is not None and len(result.masks.data):
                masks.append(result.masks.data[0].cpu().numpy().astype(np.uint8))
        return masks
