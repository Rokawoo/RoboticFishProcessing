"""
Paths, file naming, label I/O and popups shared by the Vision data pipeline.

Pipeline:
    1. preprocessing/extract_frames.py              video -> image_sets/<video name>/
    2. annotating/auto_labeler.py or Label Studio   labels (.txt) next to the images
    3. annotating/augment_images.py                 image_sets/<set> (augmented)/
    4. annotating/label_viewer.py                   review and fix labels
    5. dataset/train/train_val_data_splitter.py     image_sets/* -> dataset/train/
    6. training/train.py                            -> Vision/trained_cv_models/<run>/

Labels are YOLO segmentation polygons, one object per line, normalized to 0-1:
    <class id> <x1> <y1> <x2> <y2> ... <xn> <yn>
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
VISION_DIR = DATA_DIR.parent
DATASET_DIR = DATA_DIR / "dataset"
VIDEOS_DIR = DATASET_DIR / "video_set_raw"
IMAGE_SETS_DIR = DATASET_DIR / "image_sets"
SPLIT_DIR = DATASET_DIR / "train"
CLASSES_FILE = DATASET_DIR / "classes.txt"  # line N = class N
TRAINED_MODELS_DIR = VISION_DIR / "trained_cv_models"

# ---------------------------------------------------------------------------
# File types and names
# ---------------------------------------------------------------------------
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".bag", ".db3"})
REALSENSE_EXTS = frozenset({".bag", ".db3"})  # .db3: librealsense 2.59+
LABEL_EXT = ".txt"
SET_CLASSES_NAME = "classes.txt"  # a set's own class list (Label Studio export)
TRASH_DIR_NAME = "_deleted"  # inside a set: where the label viewer moves deleted pairs

# Augmented copies: "<set> (augmented)/<original name>__aug<n>-<effect>.<ext>"
AUGMENTED_SUFFIX = " (augmented)"
_AUGMENTED_STEM = re.compile(r"^(?P<source>.+)__aug\d+(?:-[a-z_]+)?$")

# Label Studio prefixes uploaded files with 8 hex digits: "1a2b3c4d-<name>.txt"
_LABEL_STUDIO_PREFIX = re.compile(r"^[0-9a-f]{8}-(?P<name>.+)$")


def augmented_stem(source_stem: str, index: int, effect: str) -> str:
    return f"{source_stem}__aug{index}-{effect}"


def source_of_augmented(stem: str) -> str | None:
    """Original image name of an augmented copy, or None if it isn't one."""
    m = _AUGMENTED_STEM.match(stem)
    return m["source"] if m else None


def is_augmented_set(name: str) -> bool:
    return name.endswith(AUGMENTED_SUFFIX)


def source_set_name(name: str) -> str:
    """'fish012_left (augmented)' -> 'fish012_left'."""
    return name[: -len(AUGMENTED_SUFFIX)] if is_augmented_set(name) else name


# ---------------------------------------------------------------------------
# Images and labels
# ---------------------------------------------------------------------------
Polygon = tuple[int, list[tuple[float, float]]]  # (class id, [(x, y), ...] normalized)


def set_dirs(set_folder: Path) -> tuple[Path, Path]:
    """(image folder, label folder): the set, or its images/ and labels/ (Label Studio)."""
    images, labels = set_folder / "images", set_folder / "labels"
    if images.is_dir():
        return images, labels if labels.is_dir() else set_folder
    return set_folder, set_folder


def list_images(folder: Path) -> list[Path]:
    """Images directly in `folder`, sorted by name (= frame order for extracted frames)."""
    if not folder.is_dir():
        return []
    return sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
        key=lambda f: f.name.casefold(),
    )


def label_index(folder: Path) -> dict[str, Path]:
    """Label files keyed by image name (lowercase, no extension), Label Studio prefix included."""
    exact: dict[str, Path] = {}
    prefixed: dict[str, Path] = {}
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if not f.is_file() or f.suffix.lower() != LABEL_EXT:
                continue
            if f.name.lower() == SET_CLASSES_NAME:
                continue
            exact[f.stem.casefold()] = f
            m = _LABEL_STUDIO_PREFIX.match(f.stem)
            if m:
                prefixed.setdefault(m["name"].casefold(), f)
    return {**prefixed, **exact}  # an exact match wins


def find_label(image: Path, index: dict[str, Path] | None = None) -> Path | None:
    """Label file of `image`, or None. Pass `index` when looking up many images."""
    if index is None:
        index = label_index(image.parent)
    return index.get(image.stem.casefold())


def parse_polygon(line: str) -> Polygon | None:
    """A polygon label line, or None for anything else (boxes, keypoints, junk)."""
    parts = line.split()
    try:
        cls = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except (IndexError, ValueError):
        return None
    if cls < 0 or len(values) < 6 or len(values) % 2:
        return None
    return cls, list(zip(values[0::2], values[1::2]))


def read_label(path: Path | None) -> tuple[list[Polygon], list[str]]:
    """
    (polygons, other lines). Lines that aren't polygons (e.g. boxes) come back
    unchanged in `other` so tools can write them back. A missing file = no labels.
    """
    polygons: list[Polygon] = []
    other: list[str] = []
    if path is None or not path.is_file():
        return polygons, other
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            polygon = parse_polygon(line)
            if polygon:
                polygons.append(polygon)
            else:
                other.append(line.strip())
    return polygons, other


def inside_image(points: list[tuple[float, float]]) -> bool:
    return all(0.0 <= v <= 1.0 for point in points for v in point)


def clip_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    The part of a normalized polygon inside the image (Sutherland-Hodgman): the
    outline is cut where it leaves the image and the two crossing points are
    joined along the edge. Unchanged if fully inside; [] if nothing is left.
    (YOLO skips images whose labels go past 0-1.)
    """
    pts = [(float(x), float(y)) for x, y in points]
    edges = ((0, 0.0, False), (0, 1.0, True), (1, 0.0, False), (1, 1.0, True))
    for axis, limit, keep_below in edges:

        def inside(p, axis=axis, limit=limit, keep_below=keep_below):
            return p[axis] <= limit if keep_below else p[axis] >= limit

        def crossing(a, b, axis=axis, limit=limit):
            t = (limit - a[axis]) / (b[axis] - a[axis])
            p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            p[axis] = limit
            return p[0], p[1]

        out = []
        for i, current in enumerate(pts):
            previous = pts[i - 1]
            if inside(current):
                if not inside(previous):
                    out.append(crossing(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(crossing(previous, current))
        pts = out
    kept = [p for i, p in enumerate(pts) if p != pts[i - 1]]  # no repeated points
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(kept, kept[1:] + kept[:1]))
    return kept if len(kept) >= 3 and abs(area2) > 1e-12 else []


def polygon_line(cls: int, points: list[tuple[float, float]]) -> str:
    """Label line; the clamp only stops float noise from printing as -0.000000."""
    return f"{cls} " + " ".join(
        f"{min(max(x, 0.0), 1.0):.6f} {min(max(y, 0.0), 1.0):.6f}" for x, y in points
    )


def write_label(path: Path, polygons: list[Polygon], other: list[str] = ()) -> None:
    """Polygons, cut at the image edge (clip_polygon), then the `other` lines unchanged."""
    lines = [polygon_line(cls, kept) for cls, points in polygons if (kept := clip_polygon(points))]
    lines += list(other)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def imread(path: Path):
    """cv2.imread that also works with non-ASCII paths on Windows. None if unreadable."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path: Path, image, params: list[int] | None = None) -> bool:
    """cv2.imwrite that also works with non-ASCII paths on Windows."""
    import cv2

    ok, encoded = cv2.imencode(path.suffix, image, params or [])
    if ok:
        encoded.tofile(str(path))
    return ok


def mask_to_polygon(
    mask, max_error_px: float = 1.5, min_iou: float = 0.98, min_area_px: float = 1.0
) -> list[tuple[float, float]] | None:
    """
    Largest region of a 0/1 mask as a normalized polygon with as few corners as
    possible, or None if it is smaller than `min_area_px`.

    Douglas-Peucker (cv2.approxPolyDP) with the largest epsilon <= max_error_px
    that keeps IoU(polygon, mask) >= min_iou.
    """
    import cv2
    import numpy as np

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None

    # Candidates are compared with the filled outline, in a crop around it.
    x, y, bw, bh = cv2.boundingRect(contour)
    shift = np.array([x - 2, y - 2], np.int32)
    target = np.zeros((bh + 4, bw + 4), np.uint8)
    cv2.drawContours(target, [contour - shift], -1, 1, cv2.FILLED)
    target_area = int(target.sum())

    def simplify(epsilon: float):
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        test = np.zeros_like(target)
        cv2.fillPoly(test, [approx - shift], 1)
        inter = int(np.count_nonzero(test & target))
        union = target_area + int(test.sum()) - inter
        return approx, inter / max(union, 1)

    best, iou = simplify(max_error_px)
    if iou < min_iou:  # binary search for the largest epsilon that still meets min_iou
        lo, hi = 0.0, max_error_px
        best, _ = simplify(0.5)
        for _ in range(7):
            mid = (lo + hi) / 2
            candidate, iou = simplify(mid)
            if iou >= min_iou:
                best, lo = candidate, mid
            else:
                hi = mid
    if len(best) < 3:
        return None
    return [(float(px) / w, float(py) / h) for px, py in best.reshape(-1, 2)]


def load_class_names(set_folder: Path | None = None) -> list[str]:
    """Class names from dataset/classes.txt, else from the set's own classes.txt."""
    candidates = [CLASSES_FILE]
    if set_folder is not None:
        candidates.append(set_folder / SET_CLASSES_NAME)
    for f in candidates:
        if f.is_file():
            names = [n.strip() for n in f.read_text(encoding="utf-8").splitlines() if n.strip()]
            if names:
                return names
    return []


# ---------------------------------------------------------------------------
# Popups
# ---------------------------------------------------------------------------
_popup_root = None


def enable_dpi_awareness() -> None:
    """Sharp Tk windows on Windows at display scaling > 100%. Call before creating Tk."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _hidden_root():
    """
    One hidden Tk window per run for all dialogs. Creating a second Tk after
    destroying one can fail on Windows ("Can't find a usable init.tcl").
    """
    global _popup_root
    import tkinter as tk

    if _popup_root is None:
        enable_dpi_awareness()
        try:
            _popup_root = tk.Tk()
        except tk.TclError:
            sys.exit("❌ No display for popups. Pass the paths and settings on the command line.")
        _popup_root.withdraw()
        _popup_root.attributes("-topmost", True)  # dialogs in front of the terminal
    return _popup_root


def ask_files(title: str, initial_dir: Path, types: list[tuple[str, str]]) -> list[Path]:
    from tkinter import filedialog

    names = filedialog.askopenfilenames(
        parent=_hidden_root(), title=title, initialdir=str(initial_dir), filetypes=types
    )
    return [Path(n) for n in names]


def ask_folder(title: str, initial_dir: Path) -> Path | None:
    from tkinter import filedialog

    name = filedialog.askdirectory(
        parent=_hidden_root(), title=title, initialdir=str(initial_dir), mustexist=True
    )
    return Path(name) if name else None


def ask_integer(title: str, prompt: str, default: int, low: int, high: int) -> int | None:
    """Whole number between low and high; the dialog rejects anything else. None if cancelled."""
    from tkinter import simpledialog

    return simpledialog.askinteger(
        title, prompt, parent=_hidden_root(), initialvalue=default, minvalue=low, maxvalue=high
    )


def ask_yes_no(title: str, message: str) -> bool:
    from tkinter import messagebox

    return messagebox.askyesno(title, message, parent=_hidden_root())


def positive_int(text: str) -> int:
    """argparse type: a whole number > 0."""
    import argparse

    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def fix_console() -> None:
    """Don't crash on emoji when output goes to a non-UTF-8 console or file."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
