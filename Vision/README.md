# Vision

Turns camera frames into a fish list (outline, position and heading in the arm's base frame) and builds the dataset and models for it. Requirement Spec Sections 2.2 (VS) and 2.3 (VD).

| Folder | Contents |
|---|---|
| `rfp_vision/` | Runtime vision (below) |
| `data/` | Dataset and training tools ([details](data/README.md)) |
| `trained_cv_models/` | One folder per training run; its `weights/best.pt` (the model) is in Git |

## Building the fish model

Six scripts in `data/`, each feeding the next. Without arguments they ask in popups and print the next step; `--help` lists the options.

| Step | Run (from `Vision/data/`) | What you do | Result |
|---|---|---|---|
| 1. Frames | `python preprocessing/extract_frames.py` | Pick videos (`.mp4`, `.bag`, `.db3`), named by fish and view (`fish012_left.mp4`), and frames per second | `dataset/image_sets/<video name>/`, without blurry and near-duplicate frames |
| 2. Label | `python annotating/auto_labeler.py` | **Auto-label** (G), click each fish, check the outline, Enter, Right. No fish: B. Or Label Studio, exporting YOLO labels next to the images | A `.txt` label next to each image |
| 3. Augment | `python annotating/augment_images.py` (`--all`: every set) | Pick a labeled set and copies per image (lighting, blur, noise, shadow; labels copy unchanged) | `dataset/image_sets/<set> (augmented)/` |
| 4. Review | `python annotating/label_viewer.py` | Drag corners, delete bad outlines or images | Fixed labels; deleted images in `<set>/_deleted/` |
| 5. Split | `python preprocessing/train_val_data_splitter.py` | Check the split, confirm | `dataset/train/` (80/10/10), `data.yaml` |
| 6. Train | `python training/train.py` | Settings: `training/fish_seg.yaml`; override on the command line (`epochs=5`) | `trained_cv_models/<name>_<model>_<date>_<time>/` |
| Preview | `python ../tests/model_viewer.py` | Pick a model and a video | `Vision/tests/model_previews/<video> - <model>.mp4` with outlines and confidences |

After fixing labels in the original set, run steps 3 and 5 again.

## Runtime (`rfp_vision`)

Plain Python, no ROS: ControlSys calls it directly (`vision_system.py`). Settings: `rfp_vision/settings.py`, changed in the control panel's `vision` section.

- `camera.py`: the D405 (depth aligned to color), RealSense recordings, or any OpenCV camera or video (color only: fish lie on the table plane).
- `detector.py`: the YOLO26 segmentation model. Keypoints (pose model): not trained yet.
- `fish_list.py`, `geometry.py`: each fish's middle, heading, size and the surface under the cups. Heights are measured from the table seen around each fish, so a wrong `table.z` doesn't shrink fish. The cups go where the fish balances (`balance_ahead_share` of its length ahead of the middle, toward the heavier head), else as near it as all six seal; they lean onto the body's slope. A fish whose head end is unclear is picked at its middle, placed either way round.
- `fish_heading.py`: the head end from the outline's shape (catfish are fattest toward the head) and the body's slope.
- **What isn't picked:** low detection confidence, cut off at the image's edge (the cycle looks closer), outside the pick area, no depth on the body, larger than the fish type allows (two fish seen as one). Fish too small or narrow for the cups are picked all the same: the arm is only given fish big enough. Each refused fish says what was measured (`Fish.detail`).
- `hold_monitor.py`: whether a held fish is still on the gripper, from its outline and the depth near the camera.
