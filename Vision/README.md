# Vision

Turns camera frames into a fish list (mask, keypoints, position and heading in the arm's base frame) and builds the dataset and models for it. Requirement Spec **Section 2.2 (VS)** and **2.3 (VD)**, methods in Appendix C.1 to C.4 and C.10.

| Folder | Contents |
|---|---|
| `rfp_vision/` | Runtime vision (below) |
| `data/` | Dataset and training tools ([details](data/README.md)) |
| `trained_cv_models/` | One folder per training run; its `weights/best.pt` is the model, the one file of a run in Git |

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

Plain Python, no ROS: ControlSys calls it directly (`vision_system.py`). Settings: `rfp_vision/settings.py` (defaults), changed in the control panel's `vision` section.

- **Camera (VS-10):** `camera.py`: the D405 with depth aligned to color, RealSense recordings, or any OpenCV camera or video (color only: fish are put on the table plane)
- **Detection (VS-20):** `detector.py`: a training run's YOLO26 segmentation model
- **Keypoints (VS-30):** YOLO26 pose model, not trained yet
- **Pose (VS-40):** `fish_list.py`, `geometry.py`: middle, heading, size and the surface's slope under the cups, in the base frame. The table around each fish is measured with depth (`measure_table`), and its height and outline are measured from there: a wrong `table.z` or a camera higher than the arm thinks no longer shrinks every fish (which had made them fail the size and cup checks). The cups go where the fish balances: `balance_ahead_share` of its length ahead of the middle, toward the heavier head, as far as all six still seal, else the middle, else up to `grasp_behind_m` behind it (nowhere: `grasp_point_off_body`). Each cup is checked as the circle it is (`cup_overhang_m` past the outline at most), not the footprint's corners. They lean onto the body's plane under them; a bump under one cup's lip taller than the bellows' stroke (judged against that cup's own plane, so a round body's sloping shoulders don't count), or more unevenness across the cups than the stroke (`cup_stroke_m`), rules a spot out; a smaller fin rules it out only with `strict_seal_check` (off: real fish differ from the prints, and the hold check tells whether the cups sealed); cups over holes in the depth (shiny skin) are left to the outline (`min_cups_measured`). A fish whose head end is unclear is still picked (`require_heading: false`), the cups at its middle, into the slot either way round. `fish_heading.py`: head end from the outline's shape (where it is wide and narrow, where its girth lies: catfish are fattest toward the head) and the body's slope (one frame at a time)
- **Fish types:** `fish_type` in the control panel: `stand_in` (printed fish, 0.5-0.8 m: limits that only check it is a fish, 0.35-1.00 m long) or `catfish` (real ones, Table SY-1 ± 10%), each with its size limits and balance point
- **Output (VS-50, IF-20):** `FishList` with the fields of Tables IF-1 and IF-2
- **Failure flags (VS-60):** Table VS-2, except `covered` and `invalid_keypoints`; each failed fish says what was measured (`Fish.detail`, e.g. `length 0.62 m (stand_in: 0.80-1.00)`), which ends ran out of the image (`cut_ends`) and where its middle probably is (`Fish.estimate`)
- **Hold check:** `hold_monitor.py`: whether a held fish is still on the gripper, from its outline and the depth near the camera while the arm moves
