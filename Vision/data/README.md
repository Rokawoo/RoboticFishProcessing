# Vision data pipeline

Dataset and training tools; Requirement Spec **Section 2.3 (VD)**. How to run the six steps: [Vision/README.md](../README.md#building-the-fish-model).

| Folder | Stage | Spec |
|---|---|---|
| `preprocessing/` | Frames from recordings, named by fish and view, minus blurry and near-duplicate frames | VD-10 to VD-37, Appendix C.1 |
| `annotating/` | Labeling (auto labeler and/or Label Studio), augmentation, label review | VD-40 to VD-65, Appendix C.2 |
| `preprocessing/train_val_data_splitter.py`, `training/` | Train/val/test split, YOLO26 training with MLflow tracking, evaluation | VD-70 to VD-94, Appendix C.3 and C.10 |

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

- **Name videos by fish and view** (`fish012_left.mp4`): frames are named after the video (`fish012_left_000150.jpg`), so every image traces back to its fish.
- **Auto labeler engines:** SAM (default) handles the current footage (white fish, gray table); it downloads on first use (`SAM_MODEL` in `annotating/segmentors/sam.py`) and is freed on engine switch or close. The color engine needs color contrast but can label whole sets without clicks.
- **A new outline replaces** labels it overlaps by more than half (Ctrl+Z undoes).
- **`classes.txt` must match Label Studio's label names and order**; the splitter stops if a set's own `classes.txt` disagrees.
- **Labels go next to their images.** Label Studio's export layout (`images/`, `labels/`) and `1a2b3c4d-` label prefix also work.
- **Re-running is safe:** re-extracting only adds missing frames; the augmenter asks before replacing copies and keeps the old ones if it fails; the splitter rebuilds `dataset/train/`.
- **The split:** 80/10/10 (`TRAIN_RATIO`, `TEST_RATIO`), in blocks of consecutive frames per set so near copies don't straddle splits; augmented images only in train, with their original. The same fish can be in every split.
- **MLflow:** runs are in `training/mlflow/`; from `training/`: `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.
