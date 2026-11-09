#!/usr/bin/env python3
"""
Step 6: train YOLO on ../dataset/train (made by the splitter).

Usage:
    python train.py                            # train with fish_seg.yaml
    python train.py epochs=5 batch=8           # override any setting as key=value
    python train.py fraction=0.1 epochs=3      # quick check on 10% of the data
    python train.py other_config.yaml          # another config
    python train.py --no-mlflow                # don't log to MLflow
    python train.py --resume <run folder>/weights/last.pt          # continue a stopped run
    python train.py --resume <run folder>/weights/last.pt batch=8  # ...with a smaller batch

Each run gets Vision/trained_cv_models/<name>_<model>_<date>_<time>/
(weights/best.pt is the model). Afterwards best.pt is scored on train, val and
test (if the split has one), and the train-val gap flags overfitting.

Configs are YAML files of Ultralytics train arguments
(https://docs.ultralytics.com/modes/train/#train-settings). Relative paths in a
config are relative to it; on the command line, relative to the current folder.
The task (segment, pose, ...) comes from the model, so a pose config works too.
On resume, Ultralytics only accepts memory/hardware overrides (batch, imgsz,
device, workers, cache, patience, ...).

MLflow (if installed): runs are logged to mlflow/mlflow.db here, including the
final eval/... scores. View: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
(from this folder). MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME override.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Vision/data
from pipeline_common import IMAGE_EXTS, SPLIT_DIR, TRAINED_MODELS_DIR, fix_console

TRAINING_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = TRAINING_DIR / "fish_seg.yaml"
SPLITTER = SPLIT_DIR / "train_val_data_splitter.py"
MLFLOW_DIR = TRAINING_DIR / "mlflow"

PATH_KEYS = ("model", "data", "project")
SPLITS = ("train", "val", "test")
LOW_TRAIN_SCORE = 0.3  # train mAP50-95 below this = not learned yet (a gap means nothing then)
MAIN_METRIC = {  # task -> (results_dict suffix, label): M mask, P keypoints, B box
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
        "--resume", metavar="LAST_PT", type=Path, help="continue from weights/last.pt"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="don't log this run to MLflow")
    args = parser.parse_intermixed_args()  # key=value overrides before and after --options

    args.config, args.overrides = DEFAULT_CONFIG, {}
    for item in args.items:
        if "=" in item:
            key, _, value = item.partition("=")
            args.overrides[key] = yaml.safe_load(value)  # "5" -> 5, "true" -> True
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
            if not value.endswith(".pt"):
                continue  # model .yaml names are looked up inside Ultralytics
            if Path(value).name == value and not (base / value).is_file():
                # Official model name: Ultralytics downloads it to this path.
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
    """{split: (image count, folder)} for each split folder in data.yaml."""
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
    hint = f'Split the dataset first:  python "{SPLITTER}"'
    if not data_yaml.is_file():
        sys.exit(f"❌ Dataset file not found: {data_yaml}\n   {hint}")

    print(f"Dataset: {data_yaml}")
    counts = count_images(data_yaml)
    for split, (n, folder) in counts.items():
        print(f"   {split:<5} {n:>6} images   {folder}")
    for split in ("train", "val"):
        if counts.get(split, (0,))[0] == 0:  # also catches a stale path after moving the repo
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
    Start the MLflow run (None if MLflow isn't usable). Started here, not by
    Ultralytics, so the final scores can be logged into it: Ultralytics logs into
    an active run and leaves it open. MLflow 3 refuses its old folder store, so
    runs go to SQLite.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an import-time hint
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
            if mlflow.get_experiment_by_name(name) is None:  # keep artifacts next to the db
                artifacts = (MLFLOW_DIR / "artifacts").as_uri()
                mlflow.create_experiment(name, artifact_location=artifacts)
            os.environ["MLFLOW_TRACKING_URI"] = uri  # Ultralytics reads it from here
        mlflow.set_experiment(name)
        mlflow.start_run(run_name=run_name)
    except Exception as e:  # noqa: BLE001 - logging must never stop a training run
        print(f"⚠️  MLflow setup failed, training without it: {type(e).__name__}")
        print(f"   {str(e)[:150]}")
        print("   The SQLite store needs the full package: pip install mlflow")
        return None
    print(f"MLflow: run '{run_name}' in experiment '{name}' at {mlflow.get_tracking_uri()}")
    return mlflow


def set_ultralytics_mlflow(enabled: bool) -> bool:
    """Set Ultralytics' mlflow setting (shared by all projects); return the old value."""
    from ultralytics import settings

    previous = bool(settings["mlflow"])
    if previous != enabled:
        settings.update({"mlflow": enabled})
    return previous


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(best_pt: Path, data_yaml: str, run_dir: Path, splits: list[str]) -> dict:
    """Score best.pt on each split, print a summary, return the scores."""
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
    print("mAP50: >= 50% overlap counts as a hit. mAP50-95 also rewards tight masks.")

    if "train" in scores and "val" in scores:
        gap = scores["train"]["mAP50-95"] - scores["val"]["mAP50-95"]
        print(f"\nTrain - val gap (mAP50-95): {gap:.3f}")
        if scores["train"]["mAP50-95"] < LOW_TRAIN_SCORE:
            print("⚠️  Low score on train too: not learned yet. Train longer or check the labels.")
        elif gap < 0.05:
            print("✅ Small gap: the model generalizes to the val images.")
        elif gap < 0.15:
            print("⚠️  Some overfitting. More fish and more varied images help most.")
        else:
            print("❌ Large gap: the model is memorizing. Add data, or try a smaller model or")
            print("   stronger augmentation.")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)

    resolve_paths(args.overrides, Path.cwd())
    if args.resume:
        if not args.resume.is_file():
            sys.exit(f"❌ Checkpoint not found: {args.resume}")
        run_name = args.resume.parent.parent.name  # <run folder>/weights/last.pt
        experiment = run_name.rsplit("_", 3)[0]  # fish_seg_yolo26m-seg_<date>_<time> -> fish_seg
        print(f"Resuming: {args.resume}")
    else:
        cfg = load_config(args.config, args.overrides)
        print(f"Config:  {args.config}")
        check_dataset(Path(cfg["data"]))
        experiment = cfg.get("name") or args.config.stem
        model_name = Path(str(cfg["model"])).stem
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        run_name = cfg["name"] = f"{experiment}_{model_name}_{stamp}"
        cfg.setdefault("project", str(TRAINED_MODELS_DIR))

    print_hardware()
    mlflow = None if args.no_mlflow else start_mlflow(experiment, run_name)
    previous_setting = set_ultralytics_mlflow(mlflow is not None)

    # Ultralytics downloads some files (e.g. the AMP-check model) into the current
    # folder; keep them here. Every path above is already absolute.
    os.chdir(TRAINING_DIR)

    from ultralytics import YOLO  # late: --help and config errors don't wait for PyTorch

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
            last = getattr(getattr(model, "trainer", None), "last", None)  # saved every epoch
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

        # Only splits with images, so a missing test folder can't crash scoring.
        data_yaml = str(trainer.args.data)
        counts = count_images(Path(data_yaml))
        splits = [s for s in SPLITS if counts.get(s, (0,))[0] > 0]
        scores = evaluate(best, data_yaml, run_dir, splits)
        if mlflow:
            metrics = {f"eval/{split}/{k}": v for split, s in scores.items() for k, v in s.items()}
            if "train" in scores and "val" in scores:
                metrics["eval/train_val_gap"] = (
                    scores["train"]["mAP50-95"] - scores["val"]["mAP50-95"]
                )
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
        print(
            f'MLflow: cd "{TRAINING_DIR}" && mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db'
        )
    print("Try it on new images:")
    print("   from ultralytics import YOLO")
    print(f'   model = YOLO(r"{best}")')
    print('   results = model.predict("image.jpg", retina_masks=True)  # full-resolution masks')


if __name__ == "__main__":
    fix_console()
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C while scoring: no traceback
        sys.exit(130)
