# CLAUDE.md

How to work on this repo. Overview: README.md; each subsystem README lists what it owns and its Requirement Spec sections.

## Project

A FAIRINO FR5 arm with a suction gripper picks whole catfish off a table and places them with a uniform heading. **Vision** (RealSense D405, YOLO26 segmentation + pose) finds each fish's pose; **ControlSys** runs the cycle (Qwen3-VL picks the fish, fixed code validates every answer and owns every limit); **Arm** moves (MoveIt 2 with Pilz, frcobot_ros2). Fall 2026 is the design term: `rfp_vision`, `rfp_arm` and `rfp_controlsys` are skeletons; the working code is the Vision data pipeline in `Vision/data/`.

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
| 5 | `dataset/train/train_val_data_splitter.py` | `dataset/train/` (train/val/test) and `data.yaml` |
| 6 | `training/train.py` (settings: `training/fish_seg.yaml`) | `Vision/trained_cv_models/<name>_<model>_<date>_<time>/` |

Rules the code depends on:

- **`pipeline_common.py` owns paths, file names, label I/O, image I/O and popups.** Use its helpers (`set_dirs`, `label_index`, `find_label`, `read_label`, `write_label`, `imread`, `imwrite`, `mask_to_polygon`, `ask_*`). The scripts aren't a package; each adds `Vision/data` to `sys.path`.
- **Labels** are YOLO segmentation polygons, normalized 0-1. `read_label` returns non-polygon lines separately (`other`); write them back unchanged.
- **Names carry meaning.** Frames: `<video name>_<frame number:06d>.jpg`; videos are named by fish ID and view (`fish012_left.mp4`). Augmented copies: `<original name>__aug<n>-<effect>`. Label Studio's `1a2b3c4d-` label prefix is matched.
- **The split** is 80/10/10 train/val/test (`TRAIN_RATIO`, `TEST_RATIO`), each set cut in blocks of consecutive frames so near-duplicate neighbors don't leak across splits; augmented images go to train only, when their original is in train. The same fish may be in every split (the team decided against splitting by fish, despite VD-70's wording). Keep the block and augmented rules.
- `Vision/data/dataset/classes.txt` is the class list (tracked); the splitter stops if a set's `classes.txt` disagrees.
- **Auto labeler** = `LabelViewer` (`annotating/label_viewer.py`) + click-to-label (`annotating/auto_labeler.py`, engines in `annotating/segmentors/`). The second toolbar row shows the current mode's tools (`edit_tools` / `auto_tools`). New outlines replace labels they overlap by IoU > `REPLACE_IOU`. SAM is freed on engine switch and close.
- **No data in Git.** `.gitignore` ignores `Vision/data/dataset/` except `.py`, `.gitkeep` and `classes.txt`, plus `Vision/trained_cv_models/*`, `Vision/data/training/mlflow/` and `*.pt`.
- End-to-end checks: copy `Vision/` to a temp folder and run the scripts there (paths are relative to `pipeline_common.py`, so real data stays untouched).

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
- Augmentation is pixel-only on purpose: labels copy byte for byte; geometry is Ultralytics' job at training time. (A spatial effect would need `BORDER_CONSTANT`, `fill_mask=0`: mirrored borders create unlabeled fish.) `PlanckianJitter`'s default range reaches 15000 K (strong blue); we use 4000-7500 K.
- Ultralytics: `settings` is one file per user shared by all projects (`train.py` restores the mlflow setting); downloads land in the working directory (`train.py` changes into `training/`); `cache="ram"` isn't deterministic (we use `"disk"`); YOLO26 needs 8.4+.
- MLflow 3 refuses the plain-folder store: use SQLite (needs the full `mlflow` package, not `mlflow-skinny`).
- PyPI's Windows torch is CPU-only, and pip prefers PyPI when it has a newer version than the CUDA index: keep torch/torchvision pinned to versions on the chosen index.
- Dependencies currently resolve to OpenCV 5.0 (`opencv-python` and `opencv-python-headless` at the same version); the pipeline is tested on 4.13 and 5.0.

## Robot safety (Arm, ControlSys)

Until the software limits are in place, the arm runs only at low speed with someone on the emergency stop. New motions run in simulation, then as a dry run with the vacuum off, then with vacuum or real fish. Say clearly when code moves the real arm, keep motion code out of module level (importing must not move the arm), and remember Ctrl+C doesn't cancel a trajectory the controller accepted. The VLM only chooses among candidates; fixed code owns every motion, safety and retry limit (CS-30).

## Working with the user

- Commit only when asked.
- Before finishing, run `pytest` and `ruff check`, and say what was and wasn't tested, especially anything that needs hardware (arm, camera, recordings) or a person clicking through a GUI.
