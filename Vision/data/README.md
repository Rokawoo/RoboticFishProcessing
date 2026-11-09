# Vision data pipeline

Tools for building the vision dataset and training the models. Implements Requirement Spec **Section 2.3 Vision Data and Models (VD)**. How the six steps fit together: [Vision/README.md](../README.md#building-the-fish-model).

| Folder | Stage | Spec |
|---|---|---|
| `preprocessing/` | Frames from recordings (`.bag`/`.db3` with color and depth, `.mp4`), named by fish and view, blurry and near-duplicate frames dropped | VD-10 to VD-37, Appendix C.1 |
| `annotating/` | Labeling (auto labeler and/or Label Studio), augmentation, label review | VD-40 to VD-65, Appendix C.2 |
| `dataset/train/`, `training/` | Train/val/test split, YOLO26 training with MLflow tracking, evaluation | VD-70 to VD-94, Appendix C.3 and C.10 |

```
Vision/
├── data/
│   ├── pipeline_common.py        paths, file names, label and image I/O, popups
│   ├── annotating/segmentors/    SAM and color engines of the auto labeler
│   └── dataset/
│       ├── classes.txt           class names, line N = class N (tracked in Git)
│       ├── video_set_raw/        recordings
│       ├── image_sets/<set>/     frames with their labels, one folder per video
│       └── train/                the split YOLO trains on (rebuilt by the splitter)
└── trained_cv_models/<run>/      one folder per training run; weights/best.pt is the model
```

Clips, frames, labels and weights live on the team's dataset store; `.gitignore` keeps them out of Git.

## Things to know

- **Name videos by fish ID and view** (`fish012_left.mp4`). Frames are named after the video (`fish012_left_000150.jpg`), so every image traces back to its fish.
- **Auto labeler engines.** SAM (Segment Anything, the default) outlines whatever you click and handles the current footage (white fish on a gray metal table). It downloads the model set in `annotating/segmentors/sam.py` on first use, runs fastest on a GPU, and is freed when you switch engine or close the window. The color engine needs fish and table colors to differ, but can then label whole sets without clicks.
- **Labeling over a label replaces it:** an outline that overlaps an existing label by more than half replaces that label (Ctrl+Z brings it back).
- **`classes.txt` must match Label Studio's label names and order.** The splitter stops if a set's own `classes.txt` disagrees.
- **Labels go next to their images.** Label Studio's export layout (`images/`, `labels/`) and its `1a2b3c4d-` label prefix also work.
- **Re-running is safe.** Re-extracting only adds missing frames, so labels stay matched. The splitter rebuilds `dataset/train/`. The augmenter asks before replacing a set, so fix labels in the original set and augment again.
- **The split:** 80% train / 10% val / 10% test (`TRAIN_RATIO`, `TEST_RATIO` in the splitter). Each set is split in blocks of consecutive frames, so near copies of train frames don't land in val or test; augmented images go to train only, when their original is in train. The same fish can appear in every split.
- **Training runs** are logged to MLflow in `training/mlflow/`; view them with `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` from `training/`.
