"""
Click-to-segment with SAM 2 (Ultralytics) for auto_labeler.py: a fish's clicks are
positive points, right-clicks negative; SAM returns that fish's mask. The image is
encoded once (set_image) for all its clicks. On a GPU: ~0.1 s (sam2.1_t) to ~0.5 s
(sam2.1_l) per click; the first use takes ~15 s (CUDA warm-up).
"""

import gc
from pathlib import Path

import numpy as np

SAM_MODEL = "sam2.1_l.pt"  # sam2.1_x.pt {x = t, s, b, l} smallest to largest
MODEL_PATH = Path(__file__).resolve().parent / SAM_MODEL  # downloaded here on first use


class SamSegmentor:
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

    def close(self) -> None:
        """Release the model and its GPU memory."""
        self.predictor = None
        self.image_key = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
