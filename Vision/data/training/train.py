#!/usr/bin/env python3
"""
Train a YOLO model on the split dataset in ../dataset/train.

Workflow:
    1. Split:  python ../dataset/train/train_val_data_splitter.py
               fills ../dataset/train/{images,labels}/{train,val} and writes data.yaml
    2. Train:  python train.py
               trains with fish_seg.yaml; results go to runs/fish_seg_<date>_<time>/
    3. Read the summary at the end: it scores the best weights on train, val
               (and test, if data.yaml has one) and flags overfitting.

Usage:
    python train.py                            # train with fish_seg.yaml
    python train.py epochs=5 batch=8           # override any setting as key=value
    python train.py fraction=0.1 epochs=3      # quick check on 10% of the data
    python train.py other_config.yaml          # train with another config
    python train.py --no-mlflow                # don't log this run to MLflow
    python train.py --resume runs/<run>/weights/last.pt   # continue a stopped run
    python train.py --resume runs/<run>/weights/last.pt batch=8   # ...with a smaller batch

Configs:
    A config is a YAML file of Ultralytics train arguments
    (https://docs.ultralytics.com/modes/train/#train-settings). Relative paths
    in it (model, data, project) are relative to the config file; relative
    paths given on the command line are relative to where you run from.
    fish_seg.yaml trains the fish segmentation model (spec VD-70 to VD-94).
    The keypoint model gets its own config (a yolo26*-pose.pt model and a
    data.yaml with kpt_shape) once pose labels exist; this script handles
    either, since it reads the task from the model.

    A resumed run keeps its own settings. Ultralytics only lets memory and
    hardware settings change on resume (batch, imgsz, device, workers, cache,
    patience, ...); other overrides are ignored.

MLflow:
    If mlflow is installed, each run is logged to runs/mlflow/mlflow.db:
    its settings, every epoch's metrics, the weights and plots, and the final
    train/val/test scores (eval/...). View the runs with (from this folder):
        mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db
    Set MLFLOW_TRACKING_URI to log to a shared server instead.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import yaml

TRAINING_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = TRAINING_DIR / "fish_seg.yaml"
SPLITTER = TRAINING_DIR.parent / "dataset" / "train" / "train_val_data_splitter.py"
MLFLOW_DIR = TRAINING_DIR / "runs" / "mlflow"

PATH_KEYS = ("model", "data", "project")  # config values that are file paths
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")

# The score that matters most for each task: M = mask, P = keypoints, B = box.
MAIN_METRIC = {
    "segment": ("M", "Mask"),
    "pose": ("P", "Pose"),
    "detect": ("B", "Box"),
    "obb": ("B", "Box"),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on ../dataset/train.",
        epilog="Example: python train.py epochs=5 batch=8 device=cpu",
    )
    parser.add_argument(
        "items",
        nargs="*",
        metavar="[config.yaml] [key=value ...]",
        help=f"config file (default {DEFAULT_CONFIG.name}) and setting overrides",
    )
    parser.add_argument(
        "--resume",
        metavar="LAST_PT",
        type=Path,
        help="continue a stopped run from its weights/last.pt",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="don't log this run to MLflow")
    args = parser.parse_args()

    args.config, args.overrides = DEFAULT_CONFIG, {}
    for item in args.items:
        if "=" in item:
            key, _, value = item.partition("=")
            args.overrides[key] = yaml.safe_load(value)  # "5" -> 5, "true" -> True, "cpu" -> "cpu"
        elif item.lower().endswith((".yaml", ".yml")):
            args.config = Path(item).resolve()
        else:
            parser.error(f"'{item}' is neither a .yaml config nor a key=value override")
    if args.resume:
        args.resume = args.resume.resolve()
    return args


def resolve_paths(cfg: dict, base: Path) -> None:
    """Make relative path settings absolute, relative to `base`."""
    for key in PATH_KEYS:
        value = cfg.get(key)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            continue
        if key == "model":
            # A model .yaml name (e.g. yolo26m-seg.yaml, training from scratch)
            # is looked up inside Ultralytics, so leave it alone.
            if not value.endswith(".pt"):
                continue
            # A bare name like yolo26m-seg.pt that isn't a file here is an
            # official model: Ultralytics downloads it to the path we give,
            # so point it at this folder wherever the script was run from.
            if Path(value).name == value and not (base / value).is_file():
                cfg[key] = str(TRAINING_DIR / value)
                continue
        cfg[key] = str((base / value).resolve())


def load_config(path: Path, overrides: dict) -> dict:
    if not path.is_file():
        sys.exit(f"❌ Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resolve_paths(cfg, path.parent)
    cfg.update(overrides)
    for key in ("model", "data"):
        if not cfg.get(key):
            sys.exit(f"❌ '{key}' is not set in {path.name} or on the command line")
    return cfg


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def count_images(data_yaml: Path) -> dict[str, tuple[int, Path]]:
    """{split: (image count, folder)} for each split data.yaml lists as a folder."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path") or data_yaml.parent)
    counts = {}
    for split in SPLITS:
        entry = data.get(split)
        if isinstance(entry, str) and entry:
            folder = root / entry
            n = 0
            if folder.is_dir():
                n = sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            counts[split] = (n, folder)
    return counts


def check_dataset(data_yaml: Path) -> None:
    """Fail early, with a useful message, if the split hasn't been made yet."""
    hint = f'Split the dataset first:  python "{SPLITTER}"'
    if not data_yaml.is_file():
        sys.exit(f"❌ Dataset file not found: {data_yaml}\n   {hint}")

    print(f"Dataset: {data_yaml}")
    counts = count_images(data_yaml)
    for split, (n, folder) in counts.items():
        print(f"   {split:<5} {n:>6} images   {folder}")
    for split in ("train", "val"):
        if counts.get(split, (0,))[0] == 0:
            # Also catches a data.yaml whose absolute path is stale because the
            # repo moved: rerunning the splitter rewrites it.
            sys.exit(f"❌ No {split} images found. {hint}")

    names = (yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names") or {}
    names = names.values() if isinstance(names, dict) else names
    print(f"   classes: {', '.join(map(str, names)) or '(none listed)'}")


def print_hardware() -> None:
    import torch

    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.0f} GB")
    else:
        print("⚠️  No CUDA GPU found: training will run on the CPU and be very slow.")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def start_mlflow(experiment: str, run_name: str) -> ModuleType | None:
    """
    Start an MLflow run for this training, or return None if MLflow isn't usable.

    We start the run ourselves (instead of letting Ultralytics do it) so the
    final train/val/test scores can be logged into the same run afterwards;
    Ultralytics logs into an already active run and leaves it open.

    By default runs go into a local SQLite database in runs/mlflow (MLflow no
    longer accepts its old plain-folder store). Environment variables you set
    yourself (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME) take priority.
    """
    # Silence a hint MLflow prints on import.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        print("ℹ️  mlflow isn't installed, so this run won't be logged (pip install mlflow).")
        return None

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", experiment)
    name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    try:
        if "MLFLOW_TRACKING_URI" not in os.environ:
            MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
            mlflow.set_tracking_uri(uri)
            # Create the experiment here so its files (weights, plots) go next
            # to the database rather than into whatever folder MLflow defaults to.
            if mlflow.get_experiment_by_name(name) is None:
                artifacts = (MLFLOW_DIR / "artifacts").as_uri()
                mlflow.create_experiment(name, artifact_location=artifacts)
            os.environ["MLFLOW_TRACKING_URI"] = uri  # Ultralytics reads it from here
        mlflow.set_experiment(name)
        mlflow.start_run(run_name=run_name)
    # Logging is a nice-to-have, so any failure here must not stop a training run.
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  MLflow setup failed, training without it: {type(e).__name__}")
        print(f"   {str(e)[:150]}")
        print("   The SQLite store needs the full package: pip install mlflow")
        return None
    print(f"MLflow: run '{run_name}' in experiment '{name}' at {mlflow.get_tracking_uri()}")
    return mlflow


def set_ultralytics_mlflow(enabled: bool) -> bool:
    """
    Switch Ultralytics' MLflow logging on or off and return the previous value.

    This is a setting in Ultralytics' settings file, shared by every project
    on this computer, so main() puts the old value back when it's done.
    """
    from ultralytics import settings

    previous = bool(settings["mlflow"])
    if previous != enabled:
        settings.update({"mlflow": enabled})
    return previous


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(best_pt: Path, data_yaml: str, run_dir: Path, splits: list[str]) -> dict:
    """Score the best weights on each split, print a summary, return the scores."""
    from ultralytics import YOLO

    model = YOLO(best_pt)
    key, label = MAIN_METRIC.get(model.task, ("B", "Box"))
    scores = {}
    for split in splits:
        print(f"\nScoring best.pt on {split}...")
        metrics = model.val(
            data=data_yaml,
            split=split,
            project=str(run_dir),
            name=f"eval_{split}",
            exist_ok=True,
            verbose=False,
        )
        results = metrics.results_dict
        scores[split] = {
            "mAP50": results.get(f"metrics/mAP50({key})", 0.0),
            "mAP50-95": results.get(f"metrics/mAP50-95({key})", 0.0),
        }

    print("\n" + "=" * 60)
    print(f"RESULTS ({label} mAP, best.pt)")
    print("=" * 60)
    print(f"{'Split':<8} {'mAP50':>8} {'mAP50-95':>10}")
    for split, s in scores.items():
        print(f"{split:<8} {s['mAP50']:>8.3f} {s['mAP50-95']:>10.3f}")
    print("mAP50 counts a prediction as right if it overlaps the label by 50%;")
    print("mAP50-95 averages stricter overlaps up to 95%, so it rewards tight masks.")

    # A model that scores much higher on images it trained on than on new ones
    # has memorized the training images instead of learning what a fish is.
    if "train" in scores and "val" in scores:
        gap = scores["train"]["mAP50-95"] - scores["val"]["mAP50-95"]
        print(f"\nTrain - val gap (mAP50-95): {gap:.3f}")
        if gap < 0.05:
            print("✅ Small gap: the model generalizes to the val images.")
        elif gap < 0.15:
            print("⚠️  Some overfitting. More fish and more varied images help most.")
        else:
            print("❌ Large gap: the model is memorizing. Add data, or try a smaller model")
            print("   or stronger augmentation.")
    print("ℹ️  Val only measures unseen fish if the split kept each fish on one side")
    print("   (GROUP_BY in the splitter). Otherwise these scores are optimistic.")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)

    # Overrides typed on the command line are relative to where you are.
    resolve_paths(args.overrides, Path.cwd())
    if args.resume:
        if not args.resume.is_file():
            sys.exit(f"❌ Checkpoint not found: {args.resume}")
        run_name = args.resume.parent.parent.name  # runs/<run>/weights/last.pt
        experiment = run_name.rsplit("_", 2)[0]  # fish_seg_<date>_<time> -> fish_seg
        print(f"Resuming: {args.resume}")
    else:
        cfg = load_config(args.config, args.overrides)
        print(f"Config:  {args.config}")
        check_dataset(Path(cfg["data"]))
        experiment = cfg.get("name") or args.config.stem
        # A date-time suffix gives every run its own folder and MLflow run name.
        run_name = cfg["name"] = f"{experiment}_{time.strftime('%Y-%m-%d_%H%M%S')}"
        cfg.setdefault("project", str(TRAINING_DIR / "runs"))

    print_hardware()
    mlflow = None if args.no_mlflow else start_mlflow(experiment, run_name)
    previous_setting = set_ultralytics_mlflow(mlflow is not None)

    # Ultralytics drops some downloads (e.g. the small model its AMP check
    # uses) into the current folder; keep them here instead of wherever the
    # script was started from. Every path above is absolute already.
    os.chdir(TRAINING_DIR)

    from ultralytics import YOLO

    status = "FAILED"
    model = None
    start = time.perf_counter()
    try:
        try:
            if args.resume:
                model = YOLO(args.resume)
                model.train(resume=True, **args.overrides)
            else:
                print(f"\nModel: {cfg['model']}")
                model = YOLO(cfg.pop("model"))
                model.train(**cfg)
        except KeyboardInterrupt:
            status = "KILLED"
            # Ultralytics saves weights/last.pt after every epoch, so a stopped run can continue.
            last = getattr(getattr(model, "trainer", None), "last", None)
            if last and Path(last).is_file():
                print(f'\n⏹️  Stopped. Continue with:\n   python train.py --resume "{last}"')
            else:
                print("\n⏹️  Stopped before the first epoch finished; nothing to resume.")
            sys.exit(130)

        trainer = model.trainer
        run_dir, best = Path(trainer.save_dir), Path(trainer.best)
        minutes, seconds = divmod(int(time.perf_counter() - start), 60)
        print(f"\n✅ Training finished in {minutes // 60}h {minutes % 60:02d}m {seconds:02d}s.")
        print(f"   Run folder:   {run_dir}")
        print(f"   Best weights: {best}")
        if not best.is_file():
            sys.exit("❌ No best.pt was saved, so there is nothing to evaluate.")

        # Only score splits that have images, so a missing test folder can't
        # crash the summary after hours of training.
        data_yaml = str(trainer.args.data)
        counts = count_images(Path(data_yaml))
        splits = [s for s in SPLITS if counts.get(s, (0,))[0] > 0]
        scores = evaluate(best, data_yaml, run_dir, splits)
        if mlflow:
            metrics = {f"eval/{split}/{k}": v for split, s in scores.items() for k, v in s.items()}
            if "train" in scores and "val" in scores:
                gap = scores["train"]["mAP50-95"] - scores["val"]["mAP50-95"]
                metrics["eval/train_val_gap"] = gap
            mlflow.log_metrics(metrics)
        status = "FINISHED"
    finally:
        if mlflow and mlflow.active_run():
            mlflow.end_run(status=status)
        set_ultralytics_mlflow(previous_setting)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"Plots: {run_dir}  (results.png, confusion matrix, val predictions)")
    if mlflow:
        print("MLflow: mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db  (from here)")
    print("Try it on new images:")
    print("   from ultralytics import YOLO")
    print(f'   model = YOLO(r"{best}")')
    print('   results = model.predict("image.jpg", retina_masks=True)  # full-resolution masks')


if __name__ == "__main__":
    # Avoid a crash on emoji if output goes to a non-UTF-8 console or file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C outside training (e.g. while scoring): no traceback
        sys.exit(130)
