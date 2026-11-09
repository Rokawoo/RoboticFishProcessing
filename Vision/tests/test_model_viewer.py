"""Tests for model_viewer.py, with a fake model (no trained weights or GPU needed)."""

import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
import model_viewer as mv  # after the skips: it needs cv2


def make_run(root: Path, name: str, weights=("best.pt",), maps=(0.5, 0.8, 0.7)) -> Path:
    run = root / name
    (run / "weights").mkdir(parents=True)
    for w in weights:
        (run / "weights" / w).write_bytes(b"")
    rows = [f"{i + 1},{m}" for i, m in enumerate(maps)]
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50-95(M)\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return run


def test_models_are_found_named_and_summarized(tmp_path):
    old = make_run(tmp_path, "fish_seg_yolo26m-seg_2026-11-01_100000")
    new = make_run(tmp_path, "fish_seg_yolo26s-seg_2026-11-02_100000", weights=("last.pt",))
    (tmp_path / "still_training" / "weights").mkdir(parents=True)  # no weights: not listed
    os.utime(old, (1, 1))
    assert mv.list_runs(tmp_path) == [new, old]  # newest first
    assert mv.model_weights(old) == old / "weights" / "best.pt"
    assert mv.model_weights(new) == new / "weights" / "last.pt"
    assert mv.model_name(old / "weights" / "best.pt") == old.name
    assert mv.model_name(new / "weights" / "last.pt") == f"{new.name} (last)"
    assert mv.run_summary(old) == "val mask mAP50-95 0.800, 3 epochs"
    with pytest.raises(FileNotFoundError):
        mv.model_weights(tmp_path / "still_training")


def test_draw_fills_the_outline_and_keeps_the_label_inside_the_frame():
    frame = np.full((200, 300, 3), 100, np.uint8)
    fish = mv.Detection(0, 0.93, (50, 0, 150, 120), np.array([[60, 10], [140, 10], [140, 110]]))
    out = mv.draw(frame, [fish], {0: "catfish"})
    assert out.shape == frame.shape and not np.array_equal(out[60, 120], frame[60, 120])
    assert np.array_equal(out[190, 280], frame[190, 280])  # untouched outside
    assert not np.array_equal(out[2:20, 50:60], frame[2:20, 50:60])  # tag drawn at the top
    bar = mv.add_info_bar(out, ["a very long model name " * 3, "frame 1/1"])
    assert bar.shape[1] == 300 and bar.shape[0] > 200  # text above the frame, same width


class FakeBoxes:
    def __init__(self, x: float):
        self.xyxy = np.array([[x, 20, x + 40, 50]], np.float32)
        self.cls = np.array([0.0])
        self.conf = np.array([0.9])

    def __len__(self):
        return len(self.xyxy)


class FakeModel:
    """Ultralytics-like model: one fish per frame, moving right by 2 px."""

    names = MappingProxyType({0: "catfish"})

    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        x = 20 + 2 * self.calls
        polygon = np.array([[x, 20], [x + 40, 20], [x + 40, 50], [x, 50]], np.float32)
        masks = SimpleNamespace(xy=[polygon])
        return [SimpleNamespace(boxes=FakeBoxes(x), masks=masks, keypoints=None)]


def test_render_writes_every_frame_with_overlays(tmp_path):
    video = tmp_path / "Test Fish.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(12):
        writer.write(np.full((120, 160, 3), 90, np.uint8))
    writer.release()

    out = tmp_path / "Test Fish - run.mp4"
    model = FakeModel()
    stats = mv.render(model, video, out, "run", conf=0.3)
    assert stats["frames"] == 12 and stats["with_detections"] == 12 and not stats["stopped"]
    assert model.calls == 13  # 12 frames + 1 warm-up
    assert [p.name for p in tmp_path.iterdir() if "partial" in p.name] == []
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    ok, frame = cap.read()
    assert ok and frame.shape[1] == 160 and frame.shape[0] > 120  # info bar on top
    cap.release()


def test_detections_too_small_to_be_fish_are_hidden_and_counted(tmp_path):
    import logging

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(4):
        writer.write(np.full((120, 160, 3), 90, np.uint8))
    writer.release()

    fish = mv.Detection(0, 0.9, (0, 0, 40, 30), np.array([[0, 0], [40, 0], [40, 30], [0, 30]]))
    assert mv.area_share(fish, 160 * 120) == pytest.approx(0.0625)  # outline: 6.25% of the frame
    log = logging.getLogger("ultralytics")
    level = log.level
    stats = mv.render(FakeModel(), video, tmp_path / "a.mp4", "run", min_area=10)
    assert stats["with_detections"] == 0 and stats["hidden"] == 4  # 6.25% < 10%: hidden
    assert log.level == level  # the warm-up's muting is undone
    stats = mv.render(FakeModel(), video, tmp_path / "b.mp4", "run")  # default 3%: shown
    assert stats["with_detections"] == 4 and stats["hidden"] == 0
