# Vision data pipeline

Code and settings for each stage of building the vision dataset and models, in the order they run. Implements Requirement Spec **Section 2.3 Vision Data and Models (VD)**.

| Folder | Stage | Spec |
|---|---|---|
| `preprocessing/` | Record clips as `.bag` (color and depth; newer librealsense writes `.db3`) and `.mp4` (for review) with pyrealsense2, name every clip and frame by fish ID and view, extract frames and drop blurry and near-duplicate frames | VD-10 to VD-37, Appendix C.1 |
| `annotating/` | Label frames (auto labeler and/or Label Studio), review the labels, and export them in both the segmentation and pose label formats | VD-40 to VD-65, Appendix C.2 |
| `training/` | Split the dataset into train, validation and test sets by fish ID (no fish in two sets), train the YOLO26 segmentation and pose models with MLflow tracking, and evaluate them, including on fish held out as unseen | VD-70 to VD-94, Appendix C.3 and C.10 |

Clips, frames, labels and model weights live on the team's dataset store, not in Git; `.gitignore` keeps them out, so only code, configs and docs are committed.

## Running the pipeline

Without arguments every script asks for its inputs in popups, and each prints the next step. `python <script> --help` lists the options.

| Step | Run | Reads | Writes |
|---|---|---|---|
| 1. Video to frames | `python preprocessing/extract_frames.py` | videos (`.mp4`, `.bag`, `.db3`, ...) and a frame rate | `dataset/image_sets/<video name>/` |
| 2. Label | `python annotating/auto_labeler.py` and/or Label Studio | the frames | a `.txt` label next to each image |
| 3. Augment | `python annotating/augment_images.py` | a labeled set | `dataset/image_sets/<set> (augmented)/` |
| 4. Review | `python annotating/label_editor.py` | any set | fixed labels; deleted pairs go to `<set>/_deleted/` |
| 5. Split | `python dataset/train/train_val_data_splitter.py` | every set in `dataset/image_sets/` | `dataset/train/` and its `data.yaml` |
| 6. Train | `python training/train.py` | `dataset/train/data.yaml`, `training/fish_seg.yaml` | `../trained_cv_models/<name>_<model>_<date>_<time>/` |

```
Vision/
├── data/
│   ├── pipeline_common.py        paths, file names, label and image I/O, popups
│   └── dataset/
│       ├── classes.txt           class names, line N = class N (tracked in Git)
│       ├── video_set_raw/        recordings
│       ├── image_sets/<set>/     frames with their labels, one folder per video
│       └── train/                the split YOLO trains on (rebuilt by step 5)
└── trained_cv_models/<run>/      one folder per training run; weights/best.pt is the model
```

Things to know:

- **Name videos by fish ID and view** (`fish012_left.mp4`). Frames are named after the video (`fish012_left_000150.jpg`), and the splitter keeps every image of a fish in one split, so val and test measure fish the model hasn't seen.
- **Auto labeler** = the label editor plus click-to-label (G). Click each fish, Shift+click more of the same fish, right-click what isn't fish, Enter to keep the outlines. The default engine is SAM (Segment Anything; downloads a 75 MB model on first use, fastest with a GPU), which handles the current footage (white fish on a gray metal table). The color engine needs fish and table colors to differ, but can then label whole sets without clicks.
- **`classes.txt` must match Label Studio's label names and order.** The splitter stops if a set's own `classes.txt` disagrees.
- **Labels go next to their images.** Label Studio's export layout (`images/`, `labels/`) and its `1a2b3c4d-` label prefix also work.
- **Re-running is safe.** Re-extracting only adds missing frames, so labels stay matched. The splitter rebuilds `dataset/train/`. The augmenter asks before replacing a set, so fix labels in the original set and augment again.
- **The splitter keeps val and test honest:** whole fish per split, blocks of consecutive frames for names without a fish ID, augmented images in train only (when their original is). Set `TEST_RATIO` in the splitter (e.g. 0.15) once you have several fish, for a test set of unseen fish.
