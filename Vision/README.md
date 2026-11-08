# Vision

Turns camera frames into a list of fish, each with its mask, keypoints, position and heading in the arm base frame {B}, and builds the dataset and models that do it.

Implements Requirement Spec **Section 2.2 Vision System (VS)** and **Section 2.3 Vision Data and Models (VD)**; methods are in Appendix C.1 to C.4 and C.10.

## Layout

| Folder | Contents |
|---|---|
| `rfp_vision/` | Runtime vision node: detection, keypoints, pose estimation, failure checks, fish list output |
| `data/` | Dataset pipeline, one folder per stage: `preprocessing/`, `annotating/`, `training/`; see [data/README.md](data/README.md) for how to run it |
| `trained_cv_models/` | One folder per training run, with its weights (`weights/best.pt`), plots and scores; not in Git |
| `configs/` | Camera settings, model paths and thresholds |
| `tests/` | Tests |

## Runtime (`rfp_vision`)

- **Image acquisition (VS-10):** aligned color and depth from the RealSense D405 through realsense2_camera
- **Detection and segmentation (VS-20):** YOLO26 segmentation model
- **Keypoints (VS-30):** YOLO26 pose model
- **Pose and geometry (VS-40):** position and heading in {B} from depth and the hand-eye calibration
- **Output (VS-50, IF-20):** fish list topic to ControlSys
- **Failure flagging (VS-60):** checks that mark unreliable detections

## Data pipeline (`data/`)

See [data/README.md](data/README.md) for what each stage covers: preprocessing (capture, naming, frame extraction), annotating (labeling and export) and training (split by fish, training, evaluation).
