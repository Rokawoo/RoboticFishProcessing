#!/usr/bin/env python3
"""
Step 1: extract video frames into image_sets/<video name>/.

Usage:
    python extract_frames.py                      # pick videos and frame rate in popups
    python extract_frames.py clip.mp4 --fps 2
    python extract_frames.py a.mp4 b.db3 --fps 1 --keep-blurry --keep-duplicates

Frames are named <video name>_<frame number:06d>.jpg. Re-extracting a video only
adds missing frames, so existing frames and their labels stay matched. Name
videos by fish ID and view (fish012_left.mp4).

Dropped unless --keep-duplicates / --keep-blurry:
    near duplicates  almost nothing changed since the last saved frame
    blurry           much less sharp than the video's median frame (arm motion)

Reads video files with OpenCV, and RealSense recordings (.bag; newer
librealsense such as 2.59 writes .db3; color stream) with pyrealsense2. Extract either the
recording or the .mp4 of a clip: both map to the same folder.
"""

import argparse
import math
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Vision/data
from pipeline_common import (
    IMAGE_SETS_DIR,
    REALSENSE_EXTS,
    VIDEO_EXTS,
    VIDEOS_DIR,
    ask_files,
    ask_integer,
    fix_console,
    imwrite,
    positive_int,
)

DEFAULT_FPS = 2
JPEG_QUALITY = 95

# Near duplicate: fewer than DUPLICATE_MAX_CHANGED of the pixels (compared at
# 160x90) changed by more than DUPLICATE_PIXEL_DIFF since the last saved frame.
DUPLICATE_PIXEL_DIFF = 12
DUPLICATE_MAX_CHANGED = 0.001
# Blurry: Laplacian variance below BLUR_RATIO x the video's median. Relative,
# because absolute sharpness depends on the scene.
BLUR_RATIO = 0.4


# ---------------------------------------------------------------------------
# Reading videos
# ---------------------------------------------------------------------------
def read_opencv(path: Path, fps: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame number, BGR frame) at about `fps`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV can't open this video")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or math.isnan(video_fps) or video_fps > 1000:
        video_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   {video_fps:.1f} fps, {total} frames ({total / video_fps:.0f} s)")

    step = max(video_fps / fps, 1.0)
    next_keep, index = 0.0, 0
    try:
        while cap.grab():  # grab() skips without decoding; retrieve() decodes kept frames
            if index + 1e-6 >= next_keep:
                ok, frame = cap.retrieve()
                if ok:
                    yield index, frame
                next_keep += step
            index += 1
    finally:
        cap.release()


def read_realsense(path: Path, fps: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (color frame number, BGR frame) from a RealSense recording at about `fps`."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError("reading RealSense recordings needs pyrealsense2") from None

    convert = {
        rs.format.rgb8: cv2.COLOR_RGB2BGR,
        rs.format.rgba8: cv2.COLOR_RGBA2BGR,
        rs.format.bgra8: cv2.COLOR_BGRA2BGR,
        rs.format.yuyv: cv2.COLOR_YUV2BGR_YUYV,
    }
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device_from_file(str(path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)  # every frame, as fast as possible

    interval_ms = 1000.0 / fps
    next_ms, index = None, 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError:  # end of the recording
                break
            color = frames.get_color_frame()
            if not color:
                continue
            t = color.get_timestamp()  # ms
            if next_ms is None:
                next_ms = t
            if t + 1e-3 >= next_ms:
                data = np.asanyarray(color.get_data())
                fmt = color.get_profile().format()
                if fmt == rs.format.bgr8:
                    frame = data.copy()  # the buffer is reused for the next frame
                elif fmt in convert:
                    frame = cv2.cvtColor(data, convert[fmt])
                else:
                    raise RuntimeError(f"unsupported color format {fmt}")
                yield index, frame
                while next_ms <= t:
                    next_ms += interval_ms
            index += 1
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def thumbnail(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)


def is_near_duplicate(small: np.ndarray, previous: np.ndarray | None) -> bool:
    if previous is None:
        return False
    # Largest change over the color channels: catches changes grayscale can't see.
    diff = cv2.absdiff(small, previous).max(axis=2)
    changed = np.count_nonzero(diff > DUPLICATE_PIXEL_DIFF)
    return changed < DUPLICATE_MAX_CHANGED * diff.size


def sharpness(frame: np.ndarray) -> float:
    """Laplacian variance at 640 px width, so it doesn't depend on resolution."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (640, max(1, round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def extract(
    video: Path,
    fps: int,
    out_root: Path = IMAGE_SETS_DIR,
    keep_blurry: bool = False,
    keep_duplicates: bool = False,
) -> dict[str, int]:
    """Extract one video into out_root/<video name>/. Returns counts."""
    out_dir = out_root / video.stem
    print(f"\n🎞️  {video.name} -> {out_dir}")
    reader = read_realsense if video.suffix.lower() in REALSENSE_EXTS else read_opencv

    counts = {"sampled": 0, "saved": 0, "existing": 0, "duplicates": 0, "blurry": 0}
    written: list[tuple[Path, float]] = []  # (file, sharpness) written by this run
    scores: list[float] = []  # sharpness of every kept frame, for the median
    previous = None
    for index, frame in reader(video, fps):
        counts["sampled"] += 1
        small = thumbnail(frame)
        if not keep_duplicates and is_near_duplicate(small, previous):
            counts["duplicates"] += 1
            continue
        previous = small
        score = sharpness(frame)
        scores.append(score)

        path = out_dir / f"{video.stem}_{index:06d}.jpg"
        if path.exists():
            counts["existing"] += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if not imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise RuntimeError(f"couldn't write {path}")
        written.append((path, score))
        if len(written) % 50 == 0:
            print(f"   {len(written)} frames saved...")
    if counts["sampled"] == 0:
        raise RuntimeError("no frames could be read from this video")

    # Blur needs the whole video's median, so it is checked last; only this run's files are removed.
    if not keep_blurry and len(scores) >= 5:
        limit = BLUR_RATIO * statistics.median(scores)
        for path, score in written:
            if score < limit:
                path.unlink()
                counts["blurry"] += 1
    counts["saved"] = len(written) - counts["blurry"]

    print(
        f"   sampled {counts['sampled']}, saved {counts['saved']} new, "
        f"{counts['existing']} already there, dropped {counts['duplicates']} near duplicates "
        f"and {counts['blurry']} blurry"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames into image_sets/.")
    parser.add_argument("videos", nargs="*", type=Path, help="video files (popup if none)")
    parser.add_argument(
        "--fps", type=positive_int, help="frames per second to save, 1 or more (popup if not set)"
    )
    parser.add_argument("--keep-blurry", action="store_true", help="don't drop blurry frames")
    parser.add_argument(
        "--keep-duplicates", action="store_true", help="don't drop near-duplicate frames"
    )
    args = parser.parse_args()

    videos = args.videos
    if not videos:
        patterns = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        start_dir = VIDEOS_DIR if VIDEOS_DIR.is_dir() else Path.cwd()
        videos = ask_files(
            "Pick videos to extract", start_dir, [("Videos", patterns), ("All", "*")]
        )
        if not videos:
            sys.exit("No videos picked.")
    for v in videos:
        if not v.is_file():
            sys.exit(f"❌ Not found: {v}")

    fps = args.fps
    if fps is None:
        fps = ask_integer(
            "Frame rate",
            f"Frames per second to extract from {len(videos)} video(s) (whole number, 1 or more):",
            DEFAULT_FPS,
            1,
            240,
        )
        if fps is None:
            sys.exit("Cancelled.")

    print("=" * 60)
    print(f"FRAME EXTRACTION at {fps} fps -> {IMAGE_SETS_DIR}")
    print("=" * 60)
    start = time.perf_counter()
    total, failed = 0, []
    for video in videos:
        try:
            counts = extract(
                video.resolve(), fps, IMAGE_SETS_DIR, args.keep_blurry, args.keep_duplicates
            )
            total += counts["saved"]
        except RuntimeError as e:
            print(f"   ❌ Skipped: {e}")
            failed.append(video.name)

    print(f"\n✅ Saved {total} new frames in {time.perf_counter() - start:.0f} s.")
    if failed:
        print(f"⚠️  Couldn't read: {', '.join(failed)}")
    print("Next: label the frames (annotating/auto_labeler.py or Label Studio).")


if __name__ == "__main__":
    fix_console()
    main()
