# Vision

Turns camera frames into a list of fish, each with its mask, keypoints, position and heading in the arm base frame {B}, and builds the dataset and models that do it.

Implements Requirement Spec **Section 2.2 Vision System (VS)** and **Section 2.3 Vision Data and Models (VD)**; methods are in Appendix C.1 to C.4 and C.10.

| Folder | Contents |
|---|---|
| `rfp_vision/` | Runtime vision node: detection, keypoints, pose estimation, failure checks, fish list output |
| `data/` | The dataset and training tools below ([details](data/README.md)) |
| `trained_cv_models/` | One folder per training run; `weights/best.pt` is the model (not in Git) |
| `configs/` | Camera settings, model paths and thresholds |
| `tests/` | Tests (`pytest` from the repo root) |

## Building the fish model

Six scripts in `data/`, each feeding the next. Run them from the `rfp` environment; without arguments each asks for its inputs in popups and prints the next step (`--help` lists the options).

```mermaid
flowchart TD
    VID["Videos<br/>dataset/video_set_raw/fish012_left.mp4"] -->|"Step 1: extract_frames.py"| SET
    SET["Image set: one folder per video<br/>dataset/image_sets/fish012_left/"] -->|"Step 2: auto_labeler.py<br/>or Label Studio"| LAB
    LAB["Labeled set<br/>a .txt label next to each image"] -->|"Step 3: augment_images.py"| AUG
    AUG["Augmented copies<br/>image_sets/fish012_left (augmented)/"]
    REV["Step 4: label_viewer.py<br/>check, fix, delete"] <-.-> LAB
    REV <-.-> AUG
    LAB -->|"Step 5: train_val_data_splitter.py<br/>(every set in image_sets/)"| SPLIT
    AUG --> SPLIT
    SPLIT["dataset/train/<br/>train, val, test + data.yaml"] -->|"Step 6: train.py"| RUN
    RUN["trained_cv_models/run/<br/>weights/best.pt"] -.->|"used by"| NODE["rfp_vision<br/>runtime vision node"]
```

| Step | Run (from `Vision/data/`) | What you do | Result |
|---|---|---|---|
| 1. Frames | `python preprocessing/extract_frames.py` | Pick videos (`.mp4`, or RealSense `.bag`/`.db3`) and frames per second. Name videos by fish and view, e.g. `fish012_left.mp4`. | `dataset/image_sets/<video name>/`, blurry and near-duplicate frames dropped |
| 2. Label | `python annotating/auto_labeler.py` | Pick a set. Switch to **Auto-label** (G), click each fish, check the cyan outline, press Enter, go to the next image (Right). Or label in Label Studio and export YOLO labels next to the images. | A `.txt` label next to each image |
| 3. Augment | `python annotating/augment_images.py` | Pick a labeled set and how many copies per image (lighting, blur, noise, shadow, camera shift, ...). | `dataset/image_sets/<set> (augmented)/` |
| 4. Review | `python annotating/label_viewer.py` | Pick any set and step through it: drag corners, delete bad polygons or images. The auto labeler's Edit mode does the same. | Fixed labels; deleted images go to `<set>/_deleted/` |
| 5. Split | `python dataset/train/train_val_data_splitter.py` | Check the split it shows, confirm. | `dataset/train/` (80% train, 10% val, 10% test) and `data.yaml` |
| 6. Train | `python training/train.py` | Settings in `training/fish_seg.yaml`; override any on the command line (`epochs=5`). | `trained_cv_models/<name>_<model>_<date>_<time>/`, scored on train, val and test |

Labels are YOLO segmentation polygons, so the auto labeler, Label Studio and the label viewer all work on the same files. Fix labels in the original set and augment again: re-running step 3 replaces the copies. Step 5 rebuilds `dataset/train/` from scratch each time, so run it after every labeling session.

## Runtime (`rfp_vision`)

- **Image acquisition (VS-10):** aligned color and depth from the RealSense D405 through realsense2_camera
- **Detection and segmentation (VS-20):** YOLO26 segmentation model
- **Keypoints (VS-30):** YOLO26 pose model
- **Pose and geometry (VS-40):** position and heading in {B} from depth and the hand-eye calibration
- **Output (VS-50, IF-20):** fish list topic to ControlSys
- **Failure flagging (VS-60):** checks that mark unreliable detections
