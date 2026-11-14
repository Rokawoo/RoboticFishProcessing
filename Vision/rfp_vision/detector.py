"""
Fish detection and segmentation (VS-20): an Ultralytics YOLO segmentation model turned
into a list of Detections (outline, box, confidence). PyTorch loads on first use.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "trained_cv_models"


@dataclass
class Detection:
    cls: int
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    polygon: np.ndarray  # (n, 2) px outline


def find_weights(path: Path) -> Path:
    """A run folder -> its weights/best.pt (else last.pt); a .pt file -> itself."""
    if path.is_dir():
        for name in ("best.pt", "last.pt"):
            if (path / "weights" / name).is_file():
                return path / "weights" / name
        raise FileNotFoundError(f"no weights in {path / 'weights'}")
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    return path


def list_runs(root: Path = MODELS_DIR) -> list[Path]:
    """Training runs with weights, newest first."""
    if not root.is_dir():
        return []
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir() and any((p / "weights" / n).is_file() for n in ("best.pt", "last.pt"))
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def model_version(weights: Path) -> str:
    """The run's name for <run>/weights/best.pt, otherwise the file's (VD-84)."""
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def polygon_area(polygon: np.ndarray) -> float:
    x, y = np.asarray(polygon, float).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class FishDetector:
    """
    `detect(image)` returns the fish in a BGR image. Detections whose outline covers less
    than `min_area_share` of the frame are dropped (bright strips, label slivers).
    """

    def __init__(
        self,
        weights: Path,
        conf: float = 0.25,
        min_area_share: float = 0.03,
        imgsz: int | None = None,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.weights = find_weights(Path(weights))
        self.version = model_version(self.weights)
        self.conf = conf
        self.min_area_share = min_area_share
        self._options = {"retina_masks": True, "verbose": False}
        if imgsz:
            self._options["imgsz"] = imgsz
        if device:
            self._options["device"] = device
        self._model = YOLO(str(self.weights))
        if self._model.task != "segment":
            raise ValueError(f"{self.weights} is a '{self._model.task}' model, not segmentation")
        self.names = dict(self._model.names)

    def warm_up(self, image: np.ndarray) -> None:
        """First prediction (CUDA setup takes seconds); Ultralytics' slow-NMS warning is muted."""
        log = logging.getLogger("ultralytics")
        level = log.level
        log.setLevel(logging.ERROR)
        try:
            self._model.predict(image, conf=self.conf, **self._options)
        finally:
            log.setLevel(level)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(image, conf=self.conf, **self._options)[0]
        return detections_of(result, image.shape[0] * image.shape[1] * self.min_area_share)


def detections_of(result, min_area_px: float = 0.0) -> list[Detection]:
    """Detections of one Ultralytics segmentation result, largest outline first."""
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        return []

    def numpy(values) -> np.ndarray:
        return values.cpu().numpy() if hasattr(values, "cpu") else np.asarray(values)

    boxes = numpy(result.boxes.xyxy)
    classes = numpy(result.boxes.cls).astype(int)
    confs = numpy(result.boxes.conf)
    found = []
    for i, polygon in enumerate(result.masks.xy):
        polygon = np.asarray(polygon, float).reshape(-1, 2)
        if len(polygon) < 3 or polygon_area(polygon) < max(min_area_px, 1e-9):
            continue
        box = tuple(float(v) for v in boxes[i])
        found.append(Detection(int(classes[i]), float(confs[i]), box, polygon))
    return sorted(found, key=lambda d: polygon_area(d.polygon), reverse=True)
