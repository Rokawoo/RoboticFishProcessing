"""Tests for the Vision/data pipeline scripts. Run with: pytest Vision/tests"""

import importlib.util
import itertools
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))  # the pipeline scripts aren't an installed package
import pipeline_common as pc


def load_script(relative: str):
    """Import a pipeline script by path."""
    path = DATA_DIR / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iou(a, b) -> float:
    return (a & b).sum() / max((a | b).sum(), 1)


def breaks(numbers: list[int]) -> int:
    """Number of gaps in a sorted list of frame numbers."""
    return sum(1 for a, b in itertools.pairwise(numbers) if b != a + 1)


# ---------------------------------------------------------------------------
# pipeline_common
# ---------------------------------------------------------------------------
def test_label_round_trip_keeps_other_lines_and_clips(tmp_path):
    path = tmp_path / "a.txt"
    pc.write_label(path, [(0, [(0.1, 0.2), (0.3, 0.4), (0.5, 1.2)])], ["1 0.5 0.5 0.2 0.2"])
    polygons, other = pc.read_label(path)
    assert polygons == [(0, [(0.1, 0.2), (0.3, 0.4), (0.5, 1.0)])]
    assert other == ["1 0.5 0.5 0.2 0.2"]


def test_read_label_missing_file_and_blank_lines(tmp_path):
    assert pc.read_label(None) == ([], [])
    assert pc.read_label(tmp_path / "missing.txt") == ([], [])
    path = tmp_path / "b.txt"
    path.write_text("\n0 0 0 1 0 1 1\n\n", encoding="utf-8")
    assert len(pc.read_label(path)[0]) == 1


def test_augmented_names_lead_back_to_the_original():
    stem = pc.augmented_stem("fish012_left_000150", 2, "motion_blur")
    assert stem == "fish012_left_000150__aug2-motion_blur"
    assert pc.source_of_augmented(stem) == "fish012_left_000150"
    assert pc.source_of_augmented("fish012_left_000150") is None
    assert pc.source_set_name("fish012_left (augmented)") == "fish012_left"
    assert pc.source_set_name("fish012_left") == "fish012_left"


def test_label_studio_prefix_is_matched(tmp_path):
    (tmp_path / "f_000001.jpg").write_bytes(b"")
    (tmp_path / "1a2b3c4d-f_000001.txt").write_text("", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("catfish\n", encoding="utf-8")
    index = pc.label_index(tmp_path)
    assert pc.find_label(tmp_path / "f_000001.jpg", index).name == "1a2b3c4d-f_000001.txt"
    assert "classes" not in index


def test_imread_imwrite_handle_non_ascii_paths(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "pez ñandú" / "frame_é.png"
    path.parent.mkdir()
    image = np.zeros((4, 5, 3), np.uint8)
    image[1, 2] = (10, 20, 30)
    assert pc.imwrite(path, image)
    assert (pc.imread(path) == image).all()
    assert pc.imread(tmp_path / "missing.png") is None


def test_mask_to_polygon_uses_few_corners_and_keeps_the_shape():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    mask = np.zeros((400, 600), np.uint8)
    cv2.ellipse(mask, (300, 200), (220, 70), 25, 0, 360, 1, -1)
    contour_px = int(cv2.arcLength(cv2.findContours(mask, 0, 1)[0][0], True))
    points = pc.mask_to_polygon(mask, max_error_px=1.5, min_iou=0.98)
    back = np.zeros_like(mask)
    cv2.fillPoly(back, [np.round(np.array(points) * [600, 400]).astype(np.int32)], 1)
    assert iou(mask, back) >= 0.98
    assert len(points) < contour_px / 15


def requirement_names(lines) -> set[str]:
    names = set()
    for line in lines:
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_requirements_txt_covers_every_pyproject_extra():
    root = DATA_DIR.parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[project.optional-dependencies]")[1].split("\n[")[0]
    extras = requirement_names(re.findall(r'"([^"]+)"', section))
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    missing = extras - requirement_names(requirements)
    assert not missing, f"in pyproject.toml but not requirements.txt: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def splitter():
    return load_script("dataset/train/train_val_data_splitter.py")


def frames(splitter, set_name: str, n: int) -> list:
    return [
        splitter.Pair(
            Path(f"{set_name}_{i:06d}.jpg"),
            Path(f"{set_name}_{i:06d}.txt"),
            set_name,
            Counter({0: 1}),
        )
        for i in range(n)
    ]


def test_split_keeps_blocks_of_consecutive_frames_together(splitter):
    train, val, test = splitter.split_pairs(frames(splitter, "tank", 100), 0.8, 0.1, 42)
    assert (len(train), len(val), len(test)) == (80, 10, 10)
    for split in (val, test):  # one block of 10 consecutive frames each
        assert breaks(sorted(int(p.image.stem[-6:]) for p in split)) == 0
    names = [{p.image.name for p in s} for s in (train, val, test)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])


def test_split_ratios_hold_over_many_small_sets(splitter):
    pairs = [p for k in range(30) for p in frames(splitter, f"s{k:02d}", 2)]
    train, val, test = splitter.split_pairs(pairs, 0.8, 0.0, 42)
    assert (len(train), len(val), len(test)) == (48, 12, 0)
    train, val, test = splitter.split_pairs(pairs, 0.7, 0.1, 42)
    assert (len(train), len(val), len(test)) == (42, 12, 6)


def test_augmented_images_follow_their_original_into_train_only(splitter):
    originals = frames(splitter, "tank", 50)
    train, val, test = splitter.split_pairs(originals, 0.8, 0.1, 42)
    augmented = [
        splitter.Pair(
            Path(pc.augmented_stem(p.image.stem, 1, "jpeg") + ".jpg"),
            Path("x.txt"),
            "tank (augmented)",
            Counter(),
        )
        for p in originals
    ]
    augmented.append(
        splitter.Pair(Path("gone__aug1-jpeg.jpg"), Path("y.txt"), "tank (augmented)", Counter())
    )
    keep, elsewhere, orphans = splitter.place_augmented(augmented, train, originals)
    assert {pc.source_of_augmented(p.image.stem) for p in keep} == {p.image.stem for p in train}
    assert (elsewhere, orphans) == (len(val) + len(test), 1)


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def augmenter():
    pytest.importorskip("cv2")
    pytest.importorskip("albumentations")
    return load_script("annotating/augment_images.py")


def test_polygon_survives_mask_round_trip(augmenter):
    polygon = [(0, [(0.2, 0.2), (0.6, 0.25), (0.7, 0.6), (0.3, 0.7)])]
    mask = augmenter.polygons_to_masks(polygon, 640, 480)[0]
    back = augmenter.polygons_to_masks([(0, augmenter.mask_to_points(mask))], 640, 480)[0]
    assert iou(mask, back) > 0.97


def test_every_effect_keeps_the_label_on_the_object(augmenter):
    import albumentations as A
    import cv2
    import numpy as np

    # Green fish on a gray table: detected by color (G - R), so lighting effects
    # and the gray borders of spatial effects can't be mistaken for it.
    image = np.full((240, 320, 3), 60, np.uint8)
    corners = np.array([[80, 70], [230, 90], [250, 160], [100, 170]])
    cv2.fillPoly(image, [corners], (0, 230, 0))
    polygon = [(0, [(x / 320, y / 240) for x, y in corners])]
    for name, transforms in augmenter.build_recipes().items():
        for seed in range(5):
            out, moved = augmenter.augment_one(image, polygon, A.Compose(transforms, seed=seed))
            assert len(moved) == 1, name
            label = augmenter.polygons_to_masks(moved, 320, 240)[0].astype(bool)
            fish = out[:, :, 1].astype(int) - out[:, :, 2].astype(int) > 60
            assert iou(label, fish) > 0.85, f"{name} (seed {seed})"


def test_label_studio_export_layout_is_understood(augmenter, tmp_path):
    import numpy as np

    folder = tmp_path / "export"
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir()
    pc.imwrite(folder / "images" / "1a2b3c4d-f_000001.jpg", np.zeros((60, 80, 3), np.uint8))
    pc.write_label(
        folder / "labels" / "1a2b3c4d-f_000001.txt", [(0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])]
    )
    assert pc.set_dirs(folder) == (folder / "images", folder / "labels")
    out = augmenter.augment_set(folder, copies=1, assume_yes=True)
    assert len(pc.list_images(out)) == 1


def test_augment_set_writes_labeled_copies_next_to_the_set(augmenter, tmp_path):
    import numpy as np

    folder = tmp_path / "fishA"
    folder.mkdir()
    for i in range(3):
        pc.imwrite(folder / f"fishA_{i:06d}.jpg", np.full((60, 80, 3), 100, np.uint8))
    pc.write_label(folder / "fishA_000000.txt", [(0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])])
    pc.write_label(folder / "fishA_000001.txt", [])  # background image: still augmented
    out = augmenter.augment_set(folder, copies=2, assume_yes=True)
    assert out.name == "fishA (augmented)"
    images = pc.list_images(out)
    assert len(images) == 4  # 2 labeled images x 2 copies; the unlabeled one is skipped
    assert all(pc.find_label(img) is not None for img in images)
    assert {pc.source_of_augmented(img.stem) for img in images} == {"fishA_000000", "fishA_000001"}


# ---------------------------------------------------------------------------
# Frame extractor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def extractor():
    pytest.importorskip("cv2")
    return load_script("preprocessing/extract_frames.py")


def write_video(path: Path, frame_images) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for img in frame_images:
        writer.write(img)
    writer.release()


def square_frame(x: int, blur: bool = False):
    import cv2
    import numpy as np

    img = np.full((240, 320, 3), 90, np.uint8)
    cv2.rectangle(img, (x, 100), (x + 40, 140), (255, 255, 255), -1)
    cv2.putText(img, "fish", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return cv2.GaussianBlur(img, (0, 0), 6) if blur else img


def test_extract_samples_drops_duplicates_and_adds_only_new_frames(extractor, tmp_path):
    # 4 s at 10 fps: a square moves for 2 s, then everything stands still.
    video = tmp_path / "fish001_left.avi"
    write_video(video, [square_frame(min(i, 20) * 10) for i in range(40)])
    counts = extractor.extract(video, fps=2, out_root=tmp_path / "sets")
    out = tmp_path / "sets" / "fish001_left"
    # Every 5th frame is sampled; the 3 samples after the square stops are duplicates.
    assert (counts["sampled"], counts["duplicates"], counts["saved"]) == (8, 3, 5)
    assert pc.list_images(out)[0].name == "fish001_left_000000.jpg"
    again = extractor.extract(video, fps=2, out_root=tmp_path / "sets")
    assert (again["saved"], again["existing"]) == (0, 5)


def test_extract_drops_blurry_frames(extractor, tmp_path):
    video = tmp_path / "fish002_top.avi"
    write_video(video, [square_frame(i * 15, blur=(i == 4)) for i in range(10)])
    counts = extractor.extract(video, fps=10, out_root=tmp_path / "sets")
    assert counts["blurry"] == 1
    assert not (tmp_path / "sets" / "fish002_top" / "fish002_top_000004.jpg").exists()


def test_unreadable_video_leaves_no_folder(extractor, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError):
        extractor.extract(bad, fps=2, out_root=tmp_path / "sets")
    assert not (tmp_path / "sets" / "broken").exists()


def record_realsense(path_stem: Path, n: int = 30, fps: int = 30) -> Path:
    """A synthetic RealSense recording (moving red square, RGB8) via a software device."""
    import numpy as np
    import pyrealsense2 as rs

    w, h = 160, 120
    device = rs.software_device()
    sensor = device.add_sensor("Color")
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy = w, h, w / 2, h / 2
    intrinsics.fx = intrinsics.fy = 150
    intrinsics.model, intrinsics.coeffs = rs.distortion.none, [0.0] * 5
    stream = rs.video_stream()
    stream.type, stream.index, stream.uid, stream.fmt = rs.stream.color, 0, 0, rs.format.rgb8
    stream.width, stream.height, stream.fps, stream.bpp = w, h, fps, 3
    stream.intrinsics = intrinsics
    profile = sensor.add_video_stream(stream)
    for ext in (".db3", ".bag"):  # newer librealsense records .db3, older .bag
        path = path_stem.with_suffix(ext)
        try:
            recorder = rs.recorder(str(path), device)
            break
        except RuntimeError:
            continue
    recorder_sensor = recorder.query_sensors()[0]
    recorder_sensor.open(profile)
    recorder_sensor.start(lambda f: None)
    images = []  # frames reference these buffers without copying: keep them alive
    for i in range(n):
        img = np.full((h, w, 3), (40, 120, 200), np.uint8)
        img[40:80, 4 * i : 4 * i + 30] = (250, 30, 30)
        images.append(img)
        frame = rs.software_video_frame()
        frame.pixels, frame.stride, frame.bpp = img, w * 3, 3
        frame.timestamp, frame.frame_number = i * 1000.0 / fps, i
        frame.domain = rs.timestamp_domain.hardware_clock
        frame.profile = profile.as_video_stream_profile()
        sensor.on_video_frame(frame)
        time.sleep(0.005)  # let the recorder keep up
    recorder_sensor.stop()
    recorder_sensor.close()
    del recorder_sensor, recorder  # writes the file
    return path


def test_extract_reads_realsense_recordings(extractor, tmp_path):
    pytest.importorskip("pyrealsense2")
    recording = record_realsense(tmp_path / "fish003_top")
    counts = extractor.extract(recording, fps=10, out_root=tmp_path / "sets")
    assert counts["sampled"] == 10 and counts["saved"] == 10  # every 3rd of 30 frames
    first = pc.imread(tmp_path / "sets" / "fish003_top" / "fish003_top_000000.jpg")
    assert tuple(first[60, 15]) == pytest.approx((30, 30, 250), abs=12)  # red square, as BGR


# ---------------------------------------------------------------------------
# Color segmenter
# ---------------------------------------------------------------------------
BACK, BELLY, TABLE, SHADOW = (70, 85, 95), (110, 122, 130), (185, 185, 185), (128, 132, 136)


def fish_scene(fish, shadows=False, w=640, h=400):
    """Textured table with dark fish (lighter belly): (BGR image, true masks)."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.int16)
    img[:] = TABLE
    truth = []
    for cx, cy, a, b, angle in fish:
        if shadows:  # cast down-right of the fish
            cv2.ellipse(img, (cx + 18, cy + 22), (a, b), angle, 0, 360, SHADOW, -1)
    for cx, cy, a, b, angle in fish:
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 1, -1)
        img[mask > 0] = BACK
        belly = np.zeros_like(mask)
        cv2.ellipse(belly, (cx, cy + b // 2), (int(a * 0.8), b // 2), angle, 0, 360, 1, -1)
        img[(belly > 0) & (mask > 0)] = BELLY
        truth.append(mask)
    img += rng.integers(-6, 7, img.shape, dtype=np.int16)  # sensor noise / table texture
    return np.clip(img, 0, 255).astype(np.uint8), truth


@pytest.fixture(scope="module")
def segmenter_module():
    pytest.importorskip("cv2")
    return load_script("annotating/segmentors/color.py")


def outline(seg_mod, image, fish_clicks, belly_clicks=(), not_fish=(), seeds=None, **settings):
    seg = seg_mod.ColorSegmentor(image)
    fish = [
        seg_mod.Cluster(seg_mod.sample_patch(seg.lab, x, y))
        for x, y in [*fish_clicks, *belly_clicks]
    ]
    other = [seg_mod.Cluster(seg_mod.sample_patch(seg.lab, x, y)) for x, y in not_fish]
    s = seg_mod.Settings(**settings)
    seeds = list(fish_clicks) if seeds is None else seeds
    return seg, seg.segment(fish, other, seeds, s), s


def test_one_click_on_back_and_one_on_belly_outlines_the_fish(segmenter_module):
    image, truth = fish_scene([(320, 200, 160, 45, 10)])
    _, regions, _ = outline(segmenter_module, image, [(320, 180)])
    back_only = iou(regions[0].astype(bool), truth[0].astype(bool))
    _, regions, _ = outline(segmenter_module, image, [(320, 180)], belly_clicks=[(325, 225)])
    assert len(regions) == 1
    assert iou(regions[0].astype(bool), truth[0].astype(bool)) > 0.95
    assert back_only < 0.9  # the belly color had to be sampled too


def test_a_not_fish_color_stops_the_outline_leaking_into_a_shadow(segmenter_module):
    image, truth = fish_scene([(300, 190, 150, 45, 10)], shadows=True)
    clicks = {"fish_clicks": [(300, 170)], "belly_clicks": [(305, 215)], "tolerance": 9.0}
    _, leaking, _ = outline(segmenter_module, image, **clicks)
    _, fixed, _ = outline(segmenter_module, image, **clicks, not_fish=[(320, 250)])  # on the shadow
    assert iou(leaking[0].astype(bool), truth[0].astype(bool)) < 0.9
    assert iou(fixed[0].astype(bool), truth[0].astype(bool)) > 0.93


def test_touching_fish_are_split_between_clicks(segmenter_module):
    fish = [(220, 200, 130, 40, 0), (440, 215, 120, 40, 0)]  # ends overlap
    image, truth = fish_scene(fish)
    for clicks in ([(180, 190), (480, 205)], [(300, 195), (360, 210)]):  # wherever the clicks are
        _, regions, _ = outline(segmenter_module, image, clicks, belly_clicks=[(200, 225)])
        assert len(regions) == 2
        for region, true in zip(regions, truth):
            assert iou(region.astype(bool), true.astype(bool)) > 0.9


def test_label_all_matching_regions_needs_no_clicks(segmenter_module):
    image, _ = fish_scene([(170, 120, 110, 35, 20), (450, 280, 120, 38, -15)])
    seg = segmenter_module.ColorSegmentor(image)
    fish = [
        segmenter_module.Cluster(segmenter_module.sample_patch(seg.lab, x, y))
        for x, y in [(170, 110), (175, 130)]
    ]
    s = segmenter_module.Settings(keep_all=True)
    regions = seg.segment(fish, [], [], s)
    assert len(regions) == 2
    assert len(seg.polygons(regions, s)) == 2


# ---------------------------------------------------------------------------
# GUI: label editor and auto labeler (need a display)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tk_root():
    """
    One Tk for all GUI tests, a Toplevel per test: creating a new Tk after
    destroying one fails now and then on Windows ("Can't find a usable init.tcl").
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def editor(tmp_path, monkeypatch, tk_root):
    import tkinter as tk

    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    module = load_script("annotating/label_viewer.py")
    folder = tmp_path / "fishA"
    folder.mkdir()
    for i in range(3):
        Image.fromarray(np.full((200, 400, 3), 120, np.uint8)).save(folder / f"fishA_{i:06d}.jpg")
    pc.write_label(
        folder / "fishA_000000.txt", [(0, [(0.1, 0.1), (0.4, 0.1), (0.4, 0.5), (0.1, 0.5)])]
    )
    window = tk.Toplevel(tk_root)
    ed = module.LabelViewer(window, folder)
    window.update()
    ed.fit()
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *a, **k: True)
    yield ed
    if window.winfo_exists():
        window.destroy()


def click(ed, x: float, y: float, shift: bool = False):
    """Mouse event at image pixel (x, y)."""
    return SimpleNamespace(x=x * ed.scale + ed.ox, y=y * ed.scale + ed.oy, state=int(shift))


def test_editor_draw_drag_undo_and_save(editor):
    ed = editor
    assert len(ed.polys) == 1
    ed.start_drawing()
    for x, y in [(250, 40), (350, 40), (300, 150)]:
        ed.on_press(click(ed, x, y))
    ed.on_press(click(ed, 250, 40))  # clicking the first corner closes the polygon
    assert len(ed.polys) == 2 and ed.drawing is None

    ed.on_press(click(ed, 160, 20))  # drag a corner of the first polygon
    ed.on_drag(click(ed, 180, 30))
    ed.on_release(click(ed, 180, 30))
    assert ed.polys[0][1][1] == pytest.approx([180, 30], abs=1)
    ed.undo_last()
    assert ed.polys[0][1][1] == pytest.approx([160, 20], abs=1)

    ed.selected = 0
    a, b = ed.polys[0][1][0], ed.polys[0][1][1]
    ed.on_press(click(ed, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2))  # mid-edge dot: new corner
    ed.on_release(None)
    assert len(ed.polys[0][1]) == 5

    ed.go(1)  # changing image saves
    polygons, _ = pc.read_label(ed.folder / "fishA_000000.txt")
    assert len(polygons) == 2 and len(polygons[0][1]) == 5
    ed.render()


def test_editor_delete_polygon_and_image(editor):
    ed = editor
    ed.on_press(click(ed, 100, 60))  # inside the polygon: selects it
    assert ed.selected == 0
    ed.delete_polygon()
    ed.save()
    assert pc.read_label(ed.folder / "fishA_000000.txt") == ([], [])  # empty = background

    ed.delete_image()
    trash = ed.folder / pc.TRASH_DIR_NAME
    assert sorted(f.name for f in trash.iterdir()) == ["fishA_000000.jpg", "fishA_000000.txt"]
    assert len(ed.images) == 2


def test_zoom_keeps_the_point_under_the_mouse(editor):
    ed = editor
    point = (200.0, 100.0)
    sx, sy = point[0] * ed.scale + ed.ox, point[1] * ed.scale + ed.oy
    for _ in range(4):
        ed.on_wheel(SimpleNamespace(x=sx, y=sy, num=0, delta=120))
    assert ed.zoom > 2
    assert ed.to_image(sx, sy) == pytest.approx(point, abs=1.0)
    ed.on_resize(SimpleNamespace(width=ed.canvas_size[0], height=ed.canvas_size[1]))
    assert ed.zoom > 2  # an event without a size change doesn't reset the view
    ed.on_pan_start(SimpleNamespace(x=100, y=100))
    ed.on_pan(SimpleNamespace(x=150, y=120))
    ed.render()
    ed.fit()
    assert ed.zoom == 1.0


def test_polygon_iou():
    pytest.importorskip("cv2")
    module = load_script("annotating/auto_labeler.py")
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert module.polygon_iou(square, square) == pytest.approx(1.0)
    assert module.polygon_iou(square, [[50, 50], [60, 50], [60, 60]]) == 0.0
    half = [[5, 0], [15, 0], [15, 10], [5, 10]]
    assert module.polygon_iou(square, half) == pytest.approx(1 / 3, abs=0.05)


@pytest.fixture
def auto_labeler(tmp_path, monkeypatch, tk_root):
    import tkinter as tk

    pytest.importorskip("cv2")
    module = load_script("annotating/auto_labeler.py")
    folder = tmp_path / "fishB"
    folder.mkdir()
    for i, (cx, cy) in enumerate([(320, 200), (300, 180), (340, 210)]):
        image, _ = fish_scene([(cx, cy, 150, 45, 10)])
        pc.imwrite(folder / f"fishB_{i:06d}.png", image)
    window = tk.Toplevel(tk_root)
    app = module.AutoLabeler(window, folder)
    window.update()
    app.fit()
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *a, **k: True)
    yield app
    if window.winfo_exists():
        app.close()


def test_auto_labeler_color_engine_click_enter_next_image_and_label_the_rest(auto_labeler):
    app = auto_labeler
    app.set_engine("color")
    app.set_auto(True)
    app.on_press(click(app, 320, 180))  # the fish's back
    app.on_press(click(app, 325, 225, shift=True))  # its belly: same fish, another color
    app.update_preview()
    assert len(app.preview) == 1 and len(app.fish_colors) == 2
    assert len(app.objects) == 1 and len(app.objects[0]) == 2
    app.on_enter()
    assert len(app.polys) == 1 and app.dirty and not app.preview
    app.on_press(click(app, 320, 180))  # label the same fish again: replaces the old outline
    app.on_enter()
    assert len(app.polys) == 1
    app.toggle_enhanced()
    assert app.display_image() is app.enhanced_image
    app.render()

    app.go(1)  # saves image 0; colors carry over
    assert len(pc.read_label(app.folder / "fishB_000000.txt")[0]) == 1
    app.all_var.set(True)  # every matching region, no clicks
    app.update_preview()
    assert len(app.preview) == 1

    app.label_all_unlabeled()  # images 1 and 2 have no label yet
    for i in (1, 2):
        assert len(pc.read_label(app.folder / f"fishB_{i:06d}.txt")[0]) == 1


# ---------------------------------------------------------------------------
# SAM engine (runs when the SAM weights are present: they download on the
# auto labeler's first use, or set RFP_SAM_WEIGHTS to a sam2.1_t.pt file)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sam_weights():
    pytest.importorskip("ultralytics")
    sys.path.insert(0, str(DATA_DIR / "annotating"))
    from segmentors import sam

    path = Path(os.environ.get("RFP_SAM_WEIGHTS", sam.MODEL_PATH))
    if not path.is_file():
        pytest.skip(f"SAM weights not found: {path}")
    return path


def test_sam_outlines_one_fish_and_separates_touching_fish(sam_weights):
    from segmentors.sam import SamSegmentor

    sam = SamSegmentor(sam_weights)
    image, truth = fish_scene([(320, 200, 160, 45, 10)])
    sam.set_image(image, "one")
    mask = sam.segment([[(320, 185)]], [])[0]
    assert iou(mask.astype(bool), truth[0].astype(bool)) > 0.95

    image, truth = fish_scene([(220, 200, 130, 40, 0), (440, 215, 120, 40, 0)])
    sam.set_image(image, "two")
    masks = sam.segment([[(180, 190), (200, 222)], [(480, 205), (460, 238)]], [])  # back + belly
    for mask, true in zip(masks, truth):
        assert iou(mask.astype(bool), true.astype(bool)) > 0.85


def test_auto_labeler_sam_engine(auto_labeler, sam_weights, monkeypatch):
    monkeypatch.setattr(sys.modules["segmentors.sam"], "MODEL_PATH", sam_weights)
    app = auto_labeler
    assert app.engine == "sam"
    app.set_auto(True)
    app.on_press(click(app, 320, 185))
    app.update_preview()
    assert len(app.preview) == 1
    app.on_enter()
    assert len(app.polys) == 1 and app.dirty
    app.on_press(click(app, 320, 185))  # again: replaces, doesn't duplicate
    app.on_enter()
    assert len(app.polys) == 1
    sam = app.sam
    app.release_sam()  # also runs on close
    assert app.sam is None and sam.predictor is None
