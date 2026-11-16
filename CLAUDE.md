# CLAUDE.md

How to work on this repo. Overview: README.md; each subsystem README lists what it owns and its Requirement Spec sections.

## Project

A FAIRINO FR5 arm with a suction gripper picks whole catfish off a table and places them with a uniform heading. **Vision** (RealSense D405, YOLO26 segmentation + pose) finds each fish's pose; **ControlSys** runs the cycle (Qwen3-VL picks the fish, fixed code validates every answer and owns every limit); **Arm** moves (MoveIt 2 with Pilz, frcobot_ros2). Fall 2026 is the design term. Working code: the Vision data pipeline in `Vision/data/`, and a first pick-and-place cycle in the `rfp_*` packages (below) that runs offline and in tests but has not been on the real arm.

Requirement IDs (Requirement Spec, Confluence) use these prefixes in code, docstrings and commits: SY system, VS vision, VD vision data and models, CS control, AS arm, SF safety, IF interfaces, HW/SW platform, HC hardware constraints.

## Environment

- `conda env create -f environment.yml`, `conda activate rfp`: Python 3.10, matching ROS 2 Humble on Ubuntu 22.04.
- **Python 3.10 code only** (no `tomllib`, `except*`, `typing.Self`, ...): the `rfp_*` packages run under ROS's Python.
- Dependencies go in the `pyproject.toml` extras (`vision`, `arm`, `control`, `analysis`, `dev`) **and** in `requirements.txt`, which lists every package explicitly (a test checks it covers every extra) and pins torch/torchvision to the CUDA index.
- ROS 2, MoveIt 2, frcobot_ros2, librealsense and vLLM are system installs on the robot PC. Label Studio needs its own environment (it pins `numpy<2`).
- Development is on Windows (VS Code, PowerShell or Git Bash); the robot PC runs Ubuntu. Keep code cross-platform: `pathlib`, no shell calls, tkinter for GUIs.

## Commands

- `pytest` from the repo root. GUI tests skip without a display, the RealSense test without pyrealsense2, SAM tests without the SAM weights (`RFP_SAM_WEIGHTS` points at a `sam2.1_t.pt`). `test_rfp_*.py` needs the repo installed (`pip install -e .`; the conda env does it).
- `ruff check .` and `ruff format .`: ruff pinned to 0.17.x (`dev` extra; default rules change between minor versions), line length 100 (E501). The pre-commit hook (rev must match the pin) runs both on commit; `.github/workflows/checks.yml` runs them and the light tests (dev extra only) on push.
- Every pipeline script supports `--help`; without arguments it asks for its inputs in popups.

## Vision data pipeline (`Vision/data/`)

| Step | Script | Writes |
|---|---|---|
| 1 | `preprocessing/extract_frames.py` | `dataset/image_sets/<video name>/` |
| 2 | `annotating/auto_labeler.py` (SAM or color engine) and/or Label Studio | `.txt` labels next to the images |
| 3 | `annotating/augment_images.py` (pixel-only effects) | `dataset/image_sets/<set> (augmented)/` |
| 4 | `annotating/label_viewer.py` | fixed labels; deleted pairs go to `<set>/_deleted/` |
| 5 | `preprocessing/train_val_data_splitter.py` | `dataset/train/` (train/val/test) and `data.yaml` |
| 6 | `training/train.py` (settings: `training/fish_seg.yaml`) | `Vision/trained_cv_models/<name>_<model>_<date>_<time>/` |
| Preview | `Vision/tests/model_viewer.py` | `Vision/tests/model_previews/<video> - <model>.mp4` (git-ignored) |

Rules the code depends on:

- **`pipeline_common.py` owns paths, file names, label I/O, image I/O and popups.** Use its helpers (`set_dirs`, `label_index`, `find_label`, `read_label`, `write_label`, `imread`, `imwrite`, `mask_to_polygon`, `ask_*`). The scripts aren't a package; each adds `Vision/data` to `sys.path`.
- **Labels** are YOLO segmentation polygons, normalized 0-1. `read_label` returns non-polygon lines separately (`other`); write them back unchanged. `write_label` cuts polygons at the image edge (`clip_polygon`, joining the crossing points along the edge): YOLO skips images whose labels go past 0-1.
- **Names carry meaning.** Frames: `<video name>_<frame number:06d>.jpg`; videos are named by fish ID and view (`fish012_left.mp4`). Augmented copies: `<original name>__aug<n>-<effect>`. Label Studio's `1a2b3c4d-` label prefix is matched.
- **The split** is 80/10/10 train/val/test (`TRAIN_RATIO`, `TEST_RATIO`), each set cut in blocks of consecutive frames so near-duplicate neighbors don't leak across splits; augmented images go to train only, when their original is in train. The same fish may be in every split (the team decided against splitting by fish, despite VD-70's wording). Keep the block and augmented rules.
- `Vision/data/dataset/classes.txt` is the class list (tracked); the splitter stops if a set's `classes.txt` disagrees.
- **Auto labeler** = `LabelViewer` (`annotating/label_viewer.py`) + click-to-label (`annotating/auto_labeler.py`, engines in `annotating/segmentors/`). The second toolbar row shows the current mode's tools (`edit_tools` / `auto_tools`). New outlines replace labels they overlap by IoU > `REPLACE_IOU`. SAM is freed on engine switch and close.
- **No data in Git.** `.gitignore` ignores `Vision/data/dataset/` except `.py`, `.gitkeep` and `classes.txt`, plus `Vision/trained_cv_models/*`, `Vision/data/training/mlflow/` and `*.pt`.
- End-to-end checks: copy `Vision/` to a temp folder and run the scripts there (paths are relative to `pipeline_common.py`, so real data stays untouched).

## Pick and place (`rfp_vision`, `rfp_arm`, `rfp_controlsys`)

`ControlSys/rfp_controlsys/pick_place.py --mode observe | offline | sim | check | dry-run | run` (its docstring explains each). Rules the code depends on:

- **Three layers behind interfaces.** `rfp_controlsys.cycle` talks to an `Arm` and a `Gripper` (`rfp_arm/interfaces.py`), a `VisionSystem` and a `FishSelector`. Modes only swap what is behind them (`SimArm`/`MoveItArm`, `SimGripper`/`NoGripper`/the real gripper from `make_gripper`). `rfp_vision` and `rfp_arm` don't import each other; `rfp_controlsys` imports both.
- **ROS only in `moveit_arm.py`, `fairino.py`, `ros_link.py`**, imported inside functions: every other module must import and run without ROS (Windows, CI).
- **Units:** meters and radians in code, 4x4 poses `T_a_b` (frame b in frame a), {B} = MoveIt's `base_link`. YAML files use meters and degrees, logs mm and degrees (IF-11).
- **Point-to-point moves go to joint goals** from the robot model's IK seeded with the present joints (`Chain.ik`), so the arm keeps its posture and the wrist ends where `Arm.wrist_angle` predicted; MoveIt's own IK (a pose goal) is only the fallback.
- **Every plan is checked in fixed code before it runs** (`safety.check_trajectory`: TCP speed, zones, joint travel), every target by `Workcell.pose_problem`. Don't add a motion path that skips them.
- **The VLM goes in as a `FishSelector`** (`selector.py`); its answers pass `validate_decision` and the fallback rule already. Don't let a selector reach the arm any other way (CS-30).
- **`control_panel.yaml` (repo root) overrides the three settings files** for everything that loads them (`load_workcell`, both `load_settings`; `panel=None` skips it). Its `run` section holds `pick_place.py`'s command-line defaults. A test checks that the shipped panel changes nothing, so a default changed in one place has to be changed in both.
- **Settings files mirror the dataclass defaults** (`vision.yaml`, `control.yaml`: a test compares them); unknown names are errors. `workcell.yaml` has placeholder values and `verified: false`: the real arm refuses it.
- **Taught demos** (`rfp_controlsys/demo.py`, files in `Demos/`): joint poses in degrees and `joints` / `line` / `vacuum` / `wait` steps, played through the same `Arm` and `Gripper`. A demo opens the workcell's zones (`demo_workcell`): its poses were reached on the arm when taught. Bare `on` / `off` in YAML are booleans.
- **3D simulation** (`Simulation/rfp_sim`, `--mode sim3d`, needs pyvista): a `VirtualArm`, `VirtualGripper` and `VirtualCamera` behind the real interfaces, so the real cycle and vision run on rendered color and depth. The fish model is `Simulation/models/catfish.3mf` (tracked; other models there are ignored). The FR5's meshes are FAIRINO's, downloaded on first use at a pinned frcobot_ros2 commit, checked by SHA-256 and cached in the ignored `Simulation/models/fr5/` (they have no license: never commit them). The tube and table body are drawn in the 3D window only, not in the camera's view. Its tests render offscreen at 320 x 180 and use the plain stand-in fish.
- **Tests without ROS:** `Arm/tests/fake_ros.py` is a fake rclpy, move_group and FAIRINO command server with message classes built from `ros_messages.json` (the real fields; rebuild with `make_ros_messages.py`). It catches wrong field names and types and logic errors, not what the real MoveIt rejects: that needs `--mode sim` on the robot PC.

## Conventions

- Comments and docstrings: short and to the point, only what isn't obvious from the code (the why, gotchas, units). No narration.
- User-facing scripts: docstring with a Usage block; every input on the command line, popups for anything missing; print what was done and the next step; `fix_console()` for emoji on Windows consoles.
- **Never destroy user data irreversibly.** Deletions go to `<set>/_deleted/`; ask (or `-y`) before replacing generated output; validate inputs before deleting anything.
- GUIs (tkinter): call `enable_dpi_awareness()` before creating the Tk root; every control gets a `Tooltip`; scale drawing sizes by `self.ui` (display scaling); return focus to the canvas after toolbar commands (`focus_after`), since keys are bound on the window.
- Tests: `Vision/tests/test_data_pipeline.py`, synthetic images and videos, scripts imported with `load_script`; GUI tests use the session `tk_root` fixture and a `Toplevel` per test.

## Gotchas found the hard way

- **Color segmentation fails on the current footage** (white printed fish, reflective gray perforated table): fish and table colors overlap. SAM (`annotating/segmentors/sam.py`) is the auto labeler's default; the color engine needs real color contrast (e.g. dark fish on a light table).
- SAM prompts: points for one object need an extra nesting level (`points=[[p1, p2]], labels=[[1, 0]]`); a flat list makes each point its own object. Each fish's first point is a negative for the other fish, or touching fish merge.
- `cv2.watershed` floods by neighbor color differences, not image brightness, so it can't split by a distance-transform landscape; `ColorSegmentor._split` (`annotating/segmentors/color.py`) uses distance-transform cores.
- pyrealsense2 2.59 records `.db3` (refuses `.bag`) and reads both; 2.55 only `.bag`. Software-device frames reference the NumPy pixel buffer without copying: keep it alive until recording ends.
- Duplicate-frame detection compares color thumbnails: grayscale misses changes between colors of equal brightness.
- `cv2.imread`/`imwrite` fail on non-ASCII paths on Windows: use `pipeline_common.imread`/`imwrite`.
- A second `tk.Tk()` after destroying one can fail on Windows ("Can't find a usable init.tcl"): reuse one root.
- Tk on Windows is blurry at display scaling > 100% unless the process is DPI aware. ttk's clam theme gives buttons a minimum width of 11 characters (`width=0` fixes it). Tk deletes a `tkfont.Font` when Python collects it: keep a reference.
- Albumentations' `MotionBlur` kernels are off-center (up to 2.8 px; 1.5 px with `allow_shifted=False`), moving the picture off its labels: `augment_images.CenteredMotionBlur` makes them point-symmetric. To check for shifts, phase correlation misreads symmetric blurs; use `cv2.matchTemplate`.
- Augmentation is pixel-only on purpose: labels copy byte for byte; geometry is Ultralytics' job at training time. (A spatial effect would need `BORDER_CONSTANT`, `fill_mask=0`: mirrored borders create unlabeled fish.) `PlanckianJitter`'s default range reaches 15000 K (strong blue); we use 4000-7500 K.
- Ultralytics: `settings` is one file per user shared by all projects (`train.py` restores the mlflow setting); downloads land in the working directory (`train.py` changes into `training/` and deletes the `.pt` files a run downloaded); `cache="ram"` isn't deterministic (we use `"disk"`); YOLO26 needs 8.4+.
- MLflow 3 refuses the plain-folder store: use SQLite (needs the full `mlflow` package, not `mlflow-skinny`).
- PyPI's Windows torch is CPU-only, and pip prefers PyPI when it has a newer version than the CUDA index: keep torch/torchvision pinned to versions on the chosen index.
- Dependencies currently resolve to OpenCV 5.0 (`opencv-python` and `opencv-python-headless` at the same version); the pipeline is tested on 4.13 and 5.0.

- The gripper as built is six cups under one vacuum, switched by a solenoid valve on an ESP32 (`firmware/valve_control`, `rfp_arm/valve.py`: lines `ON`/`OFF`/`PULSE <ms>`/`STATUS` at 115200 baud, answers `OK ON`/`OK OFF`/`ERR ...`; the same over HTTP), not the control box valves and vacuum switch of the Requirement Spec (`SuctionGripper`, kept as `driver: controller`). The board can't tell whether the cups hold: only the camera's hold check confirms a hold or sees a drop. Opening the port with DTR/RTS raised restarts the ESP32, and a restart closes the valve, so they are held low. Its Wi-Fi credentials live in the git-ignored `secrets.h`: never put them in a tracked file.
- rclpy messages are strict: an `int` assigned to a `float64` field raises. Convert with `float()` (NumPy scalars too).
- The FR5 URDF's last link (`wrist3_link`) has its origin on the j6 axis; the flange face is 100 mm further along z (no flange link). Checked against poses taught on the real arm: the URDF chain plus that offset gives the controller's TCP pose (tool 0) to 0.1 mm, so the controller's base frame is `base_link`, joint directions match, and its rx, ry, rz are roll, pitch, yaw about fixed axes. `tool.tcp` is measured from `wrist3_link`.
- A firm seal needs the cups pressed square onto the fish. Vision fits a plane to the depth under the cups (`fish_list._surface_shape`) and the grasp pose's z axis follows it; `GraspSequence._approach` then steps down until `Arm.contact_force` (joint torques through `contact.ContactEstimator`, no force sensor) shows the fish pushing back. The estimate is a size, read at rest, zeroed just above the fish; its thresholds are untested on the arm. Making one joint compliant needs the controller's impedance mode: only a hook exists (`grasp.press_start_commands`).
- Two orientations that are equal can differ by rounding (trace 2.9999999999999996): take a rotation's axis from its skew part and skip the turn when that is zero, or the result is NaN (`sim.interpolate`).
- The vacuum tube runs into the gripper, so the last joint (j6, +-175 deg by itself) must never wind up: `tool.wrist_range_deg` in the workcell file is checked on every planned move, `plan_grasp` takes, of the two ways the cups fit the fish, the one with the least twist (`Arm.wrist_angle` predicts j6 at a pose), and each cycle returns to the observation pose, which resets the wrist. Teach observation poses with j6 near 0.
- VTK's `get_image_depth()` is minus the distance along the optical axis (z-depth, like the D405), and needs the plotter shown first (`show(auto_close=False)`, also offscreen). A virtual camera inside its own housing mesh sees only the housing: the camera's scene leaves it out.
- A vtkOBBTree over a 100k-triangle mesh takes about a second to build: build it once per model, not per placed fish (`world._locator`).
- Lying on its side the printed catfish stands 0.29 x its length high with its fins; from 450 mm a fish that tall fits one camera view only up to about 0.55 m long (Simulation/README.md).
- The camera at 450 mm sees 0.85 x 0.50 m of table: a 0.6-0.75 m fish fits in the frame only lying along the image's long side. The outline is seen at about half the fish's height, not at its top: put its pixels on that plane (`outline_height_share`), or off-center fish shift by centimeters.
- A script or scratch file named like a standard-library module (`profile.py`, `operator.py`) in a folder that scripts run from shadows that module for everything imported after it.

## Robot safety (Arm, ControlSys)

Until the software limits are in place, the arm runs only at low speed with someone on the emergency stop. New motions run in simulation, then as a dry run with the vacuum off, then with vacuum or real fish. Say clearly when code moves the real arm, keep motion code out of module level (importing must not move the arm), and remember Ctrl+C doesn't cancel a trajectory the controller accepted. The VLM only chooses among candidates; fixed code owns every motion, safety and retry limit (CS-30).

## Working with the user

- Commit only when asked.
- Before finishing, run `pytest` and `ruff check`, and say what was and wasn't tested, especially anything that needs hardware (arm, camera, recordings) or a person clicking through a GUI.
