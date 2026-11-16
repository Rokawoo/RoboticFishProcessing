# Vision

Turns camera frames into a fish list (mask, keypoints, position and heading in the arm's base frame) and builds the dataset and models for it. Requirement Spec **Section 2.2 (VS)** and **2.3 (VD)**, methods in Appendix C.1 to C.4 and C.10.

| Folder | Contents |
|---|---|
| `rfp_vision/` | Runtime vision (below) |
| `data/` | Dataset and training tools ([details](data/README.md)) |
| `trained_cv_models/` | One folder per training run; its `weights/best.pt` is the model, the one file of a run in Git |
| `configs/` | `vision.yaml`: camera, model and thresholds |

## Building the fish model

Six scripts in `data/`, each feeding the next. Without arguments they ask for inputs in popups and print the next step; `--help` lists options.

| Step | Run (from `Vision/data/`) | What you do | Result |
|---|---|---|---|
| 1. Frames | `python preprocessing/extract_frames.py` | Pick videos (`.mp4`, `.bag`, `.db3`), named by fish and view (`fish012_left.mp4`), and frames per second | `dataset/image_sets/<video name>/`, without blurry and near-duplicate frames |
| 2. Label | `python annotating/auto_labeler.py` | Pick a set; **Auto-label** (G), click each fish, check the cyan outline, Enter, Right. No fish: B. Or Label Studio, exporting YOLO labels next to the images | A `.txt` label next to each image |
| 3. Augment | `python annotating/augment_images.py` (`--all`: every set) | Pick a labeled set and copies per image: lighting, white balance, motion blur, noise, shadow. Pixels only, so labels copy unchanged | `dataset/image_sets/<set> (augmented)/` |
| 4. Review | `python annotating/label_viewer.py` | Drag corners, delete bad polygons or images | Fixed labels; deleted images in `<set>/_deleted/` |
| 5. Split | `python preprocessing/train_val_data_splitter.py` | Check the split, confirm | `dataset/train/` (80/10/10), `data.yaml` |
| 6. Train | `python training/train.py` | Settings: `training/fish_seg.yaml`; override on the command line (`epochs=5`) | `trained_cv_models/<name>_<model>_<date>_<time>/`, scored on train/val/test |
| Preview | `python ../tests/model_viewer.py` | Pick a model and a raw video | `Vision/tests/model_previews/<video> - <model>.mp4` with outlines, confidences and speed |

Fix labels in the original set, then re-run step 3 (it replaces the copies) and step 5 (it rebuilds `dataset/train/`).

## Runtime (`rfp_vision`)

Plain Python, no ROS: ControlSys calls it directly (`vision_system.py`). Settings: `configs/vision.yaml`.

- **Camera (VS-10):** `camera.py`: the D405 with depth aligned to color, RealSense recordings, or any OpenCV camera or video (color only: fish are put on the table plane)
- **Detection (VS-20):** `detector.py`: a training run's YOLO26 segmentation model
- **Keypoints (VS-30):** YOLO26 pose model, not trained yet
- **Pose (VS-40):** `fish_list.py`, `geometry.py`: middle, heading, size and the surface's slope under the cups, in the base frame. The cups go where the fish balances: `balance_ahead_share` of its length ahead of the middle, toward the heavier head. `fish_heading.py`: head end from the outline's shape and the body's slope (one frame at a time)
- **Fish types:** `fish_type` in `configs/vision.yaml` (or the control panel): `stand_in` (the printed fish) or `catfish` (real ones), each with its size limits and balance point
- **Output (VS-50, IF-20):** `FishList` with the fields of Tables IF-1 and IF-2
- **Failure flags (VS-60):** Table VS-2, except `covered` and `invalid_keypoints`
- **Hold check:** `hold_monitor.py`: whether a held fish is still on the gripper, from its outline and the depth near the camera while the arm moves
