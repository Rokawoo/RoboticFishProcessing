#!/usr/bin/env python3
"""
Preview a trained model on a recorded video: every frame goes through the model in
order, as frames from the live camera would, and the model's outlines, boxes and
confidences are drawn on it.

Usage:
    python model_viewer.py                          # pick a model and a video in popups
    python model_viewer.py --model <run folder or .pt> --video <video file>
    python model_viewer.py --show                   # also watch it (q: stop, space: pause)
    python model_viewer.py --conf 0.5 --no-boxes    # confidence threshold, outlines only
    python model_viewer.py --min-area 0             # also show detections too small to be fish

Models: the runs in Vision/trained_cv_models/ (their weights/best.pt). Videos: the
recordings in Vision/data/dataset/video_set_raw/ (.mp4 etc., RealSense .bag/.db3).
Writes Vision/tests/model_previews/"<video name> - <model name>.mp4" (MPEG-4; plays in
VLC or Windows Media Player: OpenCV can't write H.264 on Windows). A bar above the video
shows the model's time per frame, to judge if it keeps up with a live camera. Detections smaller
than --min-area (% of the frame) are hidden as not fish and counted in the bar. q, closing the
window or Ctrl+C stops early and keeps what was rendered.
"""

import argparse
import csv
import logging
import math
import sys
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))  # Vision/data
from pipeline_common import (
    DATA_DIR,
    REALSENSE_EXTS,
    TRAINED_MODELS_DIR,
    VIDEO_EXTS,
    VIDEOS_DIR,
    ask_choice,
    ask_yes_no,
    fix_console,
    require_ultralytics,
)

PREVIEWS_DIR = Path(__file__).resolve().parent / "model_previews"
DEFAULT_CONF = 0.25  # Ultralytics' default
# Smaller detections are hidden. Labeled fish fill 8-50% of the frame (median ~25%); the
# false positives seen so far (bright strips at the frame edge) ~2%.
DEFAULT_MIN_AREA = 3.0  # % of the frame
REALSENSE_FPS = 30  # D405 recordings; OpenCV can't read their frame rate
FOURCC = "mp4v"
FILL_ALPHA = 0.35
SPEED_WINDOW = 30  # frames averaged for the speed shown
SHOW_MAX = (1600, 900)  # the live window fits in this (px)
COLORS = [  # BGR per class id, same hues as the label viewer
    (120, 220, 0),
    (0, 170, 255),
    (255, 170, 0),
    (200, 80, 255),
    (255, 120, 170),
    (0, 230, 255),
    (230, 230, 0),
    (100, 100, 255),
    (80, 255, 140),
    (200, 200, 200),
]


@dataclass
class Detection:
    cls: int
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    polygon: np.ndarray | None = None  # (n, 2) px, segmentation models
    keypoints: np.ndarray | None = None  # (k, 2 or 3) px (+ confidence), pose models


# ---------------------------------------------------------------------------
# Models and videos
# ---------------------------------------------------------------------------
def model_weights(path: Path) -> Path:
    """A run folder -> its weights/best.pt (else last.pt); a .pt file -> itself."""
    if path.is_dir():
        for name in ("best.pt", "last.pt"):
            if (path / "weights" / name).is_file():
                return path / "weights" / name
        raise FileNotFoundError(f"no weights in {path / 'weights'} (still training?)")
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    return path


def model_name(weights: Path) -> str:
    """The run's name for <run>/weights/best.pt, otherwise the file's."""
    if weights.parent.name == "weights":
        run = weights.parent.parent.name
        return run if weights.stem == "best" else f"{run} ({weights.stem})"
    return weights.stem


def list_runs(root: Path = TRAINED_MODELS_DIR) -> list[Path]:
    """Runs with weights, newest first."""
    if not root.is_dir():
        return []
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir() and any((p / "weights" / n).is_file() for n in ("best.pt", "last.pt"))
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def run_summary(run: Path) -> str:
    """'val mask mAP50-95 0.912, 100 epochs' from the run's results.csv ('' if unknown)."""
    try:
        with open(run / "results.csv", encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    except (OSError, csv.Error):
        return ""
    for key, kind in (("M", "mask"), ("P", "pose"), ("B", "box")):
        column = f"metrics/mAP50-95({key})"
        scores = [float(r[column]) for r in rows if r.get(column)]
        if scores:
            return f"val {kind} mAP50-95 {max(scores):.3f}, {len(rows)} epochs"
    return ""


def list_videos(root: Path = VIDEOS_DIR) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
        key=lambda p: p.name.casefold(),
    )


def open_video(path: Path) -> tuple[Iterator[np.ndarray], float, int]:
    """(BGR frames in order, frames per second, frame count or 0 if unknown)."""
    if path.suffix.lower() in REALSENSE_EXTS:
        sys.path.insert(0, str(DATA_DIR / "preprocessing"))
        from extract_frames import read_realsense

        frames = (frame for _, frame in read_realsense(path, REALSENSE_FPS))
        return frames, float(REALSENSE_FPS), 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV can't open {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps > 1000:
        fps = 30.0
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

    def frames() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame
        finally:
            cap.release()

    return frames(), fps, total


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _numpy(values) -> np.ndarray:
    return values.cpu().numpy() if hasattr(values, "cpu") else np.asarray(values)


def detections(result) -> list[Detection]:
    """Detections of one Ultralytics result (any task: detect, segment, pose)."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = _numpy(result.boxes.xyxy)
    classes = _numpy(result.boxes.cls).astype(int)
    confs = _numpy(result.boxes.conf)
    n = len(boxes)
    polygons = list(result.masks.xy) if result.masks is not None else [None] * n
    points = _numpy(result.keypoints.data) if result.keypoints is not None else [None] * n
    return [
        Detection(
            int(classes[i]),
            float(confs[i]),
            tuple(float(v) for v in boxes[i]),
            polygons[i] if polygons[i] is not None and len(polygons[i]) >= 3 else None,
            points[i],
        )
        for i in range(n)
    ]


def area_share(d: Detection, frame_area: float) -> float:
    """Share of the frame the detection covers: its outline's area, else its box's."""
    if d.polygon is not None:
        x, y = d.polygon[:, 0].astype(float), d.polygon[:, 1].astype(float)
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    else:
        x1, y1, x2, y2 = d.box
        area = (x2 - x1) * (y2 - y1)
    return area / frame_area


def warm_up(model, frame: np.ndarray, conf: float) -> None:
    """
    First prediction: CUDA setup takes seconds, so NMS overruns its time limit and
    Ultralytics warns. The result is thrown away, so the warning is muted here only.
    """
    log = logging.getLogger("ultralytics")
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        model.predict(frame, conf=conf, retina_masks=True, verbose=False)
    finally:
        log.setLevel(level)


def class_color(cls: int) -> tuple[int, int, int]:
    return COLORS[cls % len(COLORS)]


def put_label(image: np.ndarray, text: str, x: int, y: int, color, scale: float) -> None:
    """Text on a filled tag whose bottom-left corner is at (x, y), kept inside the frame."""
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, round(1.5 * scale))
    (tw, th), base = cv2.getTextSize(text, font, size, thickness)
    pad = max(2, round(3 * scale))
    x = min(max(x, 0), max(image.shape[1] - tw - 2 * pad, 0))
    y = max(y, th + base + 2 * pad)
    cv2.rectangle(image, (x, y - th - base - 2 * pad), (x + tw + 2 * pad, y), color, -1)
    ink = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(image, text, (x + pad, y - base - pad), font, size, ink, thickness, cv2.LINE_AA)


def draw(
    frame: np.ndarray,
    found: list[Detection],
    names: dict[int, str],
    masks: bool = True,
    boxes: bool = True,
) -> np.ndarray:
    """The frame with each detection's filled outline, box and 'class confidence' tag."""
    out = frame.copy()
    scale = max(frame.shape[:2]) / 1280  # lines and text grow with the frame
    line = max(1, round(2 * scale))
    polygons = [(d, d.polygon.round().astype(np.int32)) for d in found if d.polygon is not None]
    if masks and polygons:
        overlay = out.copy()
        for d, points in polygons:
            cv2.fillPoly(overlay, [points], class_color(d.cls))
        out = cv2.addWeighted(overlay, FILL_ALPHA, out, 1 - FILL_ALPHA, 0)
        for d, points in polygons:
            cv2.polylines(out, [points], True, class_color(d.cls), line, cv2.LINE_AA)
    for d in found:
        color = class_color(d.cls)
        x1, y1, x2, y2 = (round(v) for v in d.box)
        if boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line)
        if d.keypoints is not None:
            for point in d.keypoints:
                if len(point) < 3 or point[2] >= 0.5:  # visible keypoints only
                    center = (round(point[0]), round(point[1]))
                    cv2.circle(out, center, max(3, round(5 * scale)), color, -1, cv2.LINE_AA)
                    cv2.circle(out, center, max(3, round(5 * scale)), (0, 0, 0), 1, cv2.LINE_AA)
        put_label(out, f"{names.get(d.cls, d.cls)} {d.conf:.2f}", x1, y1, color, scale)
    return out


def add_info_bar(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """The image under a dark bar of text lines, so the text never hides a detection."""
    h, w = image.shape[:2]
    scale = max(h, w) / 1280
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(1.4 * scale))
    pad = round(8 * scale)
    size = 0.55 * scale
    widest = max(cv2.getTextSize(t, font, size, thickness)[0][0] for t in lines)
    size *= min(1.0, (w - 2 * pad) / widest)  # shrink to fit narrow (portrait) frames
    height = max(cv2.getTextSize(t, font, size, thickness)[0][1] for t in lines)
    step = height + pad
    bar = np.full((step * len(lines) + pad, w, 3), 32, np.uint8)
    for i, text in enumerate(lines):
        y = pad + step * i + height
        cv2.putText(bar, text, (pad, y), font, size, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([bar, image])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(
    model,
    video: Path,
    out_path: Path,
    name: str,
    conf: float = DEFAULT_CONF,
    masks: bool = True,
    boxes: bool = True,
    show: bool = False,
    min_area: float = DEFAULT_MIN_AREA,
) -> dict:
    """
    Run `model` on every frame of `video` in order and write the annotated video.
    Detections covering less than `min_area` % of the frame aren't drawn.
    """
    frames, fps, total = open_video(video)
    names = dict(model.names)
    partial = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    writer = None
    window = "Model preview (q: stop, space: pause)"
    speeds: deque[float] = deque(maxlen=SPEED_WINDOW)
    stats = {"frames": 0, "with_detections": 0, "detections": 0, "hidden": 0, "stopped": False}
    stats["ms"] = 0.0
    started = time.perf_counter()
    try:
        for index, frame in enumerate(frames):
            if index == 0:
                warm_up(model, frame, conf)  # not counted in the speed
            tick = time.perf_counter()
            result = model.predict(frame, conf=conf, retina_masks=True, verbose=False)[0]
            ms = (time.perf_counter() - tick) * 1000
            speeds.append(ms)
            frame_area = frame.shape[0] * frame.shape[1]
            everything = detections(result)
            found = [d for d in everything if area_share(d, frame_area) * 100 >= min_area]
            hidden = len(everything) - len(found)
            stats["hidden"] += hidden
            stats["frames"] += 1
            stats["detections"] += len(found)
            stats["with_detections"] += bool(found)
            stats["ms"] += ms

            out = draw(frame, found, names, masks, boxes)
            average = sum(speeds) / len(speeds)
            frame_text = f"{index + 1}/{total}" if total else f"{index + 1}"
            small = f" (+{hidden} under {min_area:g}%)" if hidden else ""
            frame_line = f"frame {frame_text}   {index / fps:.1f} s   fish: {len(found)}{small}"
            speed = f"model {average:.0f} ms/frame = {1000 / average:.0f} fps (video {fps:.0f})"
            out = add_info_bar(out, [name, f"{frame_line}   conf >= {conf:.2f}", speed])
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*FOURCC), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"can't write {partial}")
            writer.write(out)

            if index % 15 == 0:
                done = f" ({(index + 1) / total:.0%})" if total else ""
                print(f"\r   frame {frame_text}{done}, {average:.0f} ms/frame   ", end="")
            if show and not show_frame(window, out, fps, tick):
                stats["stopped"] = True
                break
    except KeyboardInterrupt:
        stats["stopped"] = True
    except BaseException:
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)
        raise
    finally:
        if show:
            cv2.destroyAllWindows()
    print()
    if writer is None:
        raise RuntimeError(f"no frames could be read from {video.name}")
    writer.release()
    partial.replace(out_path)
    stats["seconds"] = time.perf_counter() - started
    return stats


def show_frame(window: str, image: np.ndarray, fps: float, tick: float) -> bool:
    """Show a frame at the video's pace. False when the viewer stops (q, Esc, closed)."""
    h, w = image.shape[:2]
    fit = min(SHOW_MAX[0] / w, SHOW_MAX[1] / h, 1.0)
    if fit < 1:
        image = cv2.resize(image, (round(w * fit), round(h * fit)), interpolation=cv2.INTER_AREA)
    cv2.imshow(window, image)
    wait = max(1, round(1000 / fps - (time.perf_counter() - tick) * 1000))
    key = cv2.waitKey(wait) & 0xFF
    while key == ord(" "):  # paused until space again
        key = cv2.waitKey(0) & 0xFF
        if key == ord(" "):
            key = 0
            break
    if key in (ord("q"), 27):
        return False
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def pick_model() -> Path:
    runs = list_runs()
    if not runs:
        sys.exit(
            f"❌ No trained models in {TRAINED_MODELS_DIR}\n"
            "   Train one: python Vision/data/training/train.py"
        )
    labels = [f"{run.name}    {run_summary(run)}".rstrip() for run in runs]
    choice = ask_choice("Model preview", "Pick a model (newest first):", labels)
    if choice is None:
        sys.exit("Cancelled.")
    return runs[choice]


def pick_video() -> Path:
    videos = list_videos()
    if not videos:
        sys.exit(f"❌ No videos in {VIDEOS_DIR}")
    choice = ask_choice("Model preview", "Pick a video:", [v.name for v in videos])
    if choice is None:
        sys.exit("Cancelled.")
    return videos[choice]


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description="Draw a trained model's detections on a video.")
    parser.add_argument("--model", type=Path, help="run folder or .pt file (popup if not given)")
    parser.add_argument("--video", type=Path, help="video file (popup if not given)")
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF, help=f"confidence threshold ({DEFAULT_CONF})"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=DEFAULT_MIN_AREA,
        help=f"hide detections under this %% of the frame ({DEFAULT_MIN_AREA:g}; 0: show all)",
    )
    parser.add_argument("--no-boxes", action="store_true", help="outlines only, no boxes")
    parser.add_argument("--no-masks", action="store_true", help="boxes only, no outlines")
    parser.add_argument("--show", action="store_true", help="also show it while it renders")
    parser.add_argument("-y", "--yes", action="store_true", help="replace an existing preview")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    if not 0 <= args.min_area < 100:
        parser.error("--min-area must be a % of the frame, 0 to 100")

    try:
        weights = model_weights(args.model or pick_model())
    except FileNotFoundError as e:
        sys.exit(f"❌ {e}")
    video = args.video or pick_video()
    if not video.is_file():
        sys.exit(f"❌ Video not found: {video}")

    name = model_name(weights)
    out_path = PREVIEWS_DIR / f"{video.stem} - {name}.mp4"
    question = f"'{out_path.name}' already exists. Replace it?"
    if out_path.exists() and not args.yes and not ask_yes_no("Replace preview?", question):
        sys.exit("Cancelled.")
    PREVIEWS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("MODEL PREVIEW")
    print("=" * 60)
    print(f"Model:  {weights}")
    print(f"Video:  {video}")
    print(f"Output: {out_path}")
    require_ultralytics()  # late: --help and the popups don't wait for PyTorch
    from ultralytics import YOLO

    try:
        stats = render(
            YOLO(weights),
            video,
            out_path,
            name,
            args.conf,
            masks=not args.no_masks,
            boxes=not args.no_boxes,
            show=args.show,
            min_area=args.min_area,
        )
    except RuntimeError as e:
        sys.exit(f"❌ {e}")

    n = stats["frames"]
    print(
        f"{'⏹️  Stopped' if stats['stopped'] else '✅ Done'}: {n} frames in {stats['seconds']:.0f} s"
    )
    print(
        f"   model {stats['ms'] / n:.1f} ms/frame on average; detections in "
        f"{stats['with_detections'] / n:.0%} of frames ({stats['detections'] / n:.2f} per frame)"
    )
    if stats["hidden"]:
        print(f"   {stats['hidden']} detection(s) under {args.min_area:g}% of the frame hidden")
    print(f"   {out_path}")


if __name__ == "__main__":
    main()
