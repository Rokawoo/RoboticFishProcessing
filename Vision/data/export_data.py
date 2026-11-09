#!/usr/bin/env python3
"""
Export the data and trained models (everything Git doesn't hold) to one zip.

Usage:
    python export_data.py                       # zip next to the repo folder
    python export_data.py -o D:/backup          # zip into another folder
    python export_data.py --with-split          # also dataset/train (else rebuilt by the splitter)

The zip holds RoboticFishProcessing/Vision/data/dataset/... and
RoboticFishProcessing/Vision/trained_cv_models/..., the same layout as the repo. To
import on another PC: unzip, then drag the RoboticFishProcessing folder onto the folder
that contains the cloned repo and choose to merge (replace files if asked).

Left out: files Git already has (scripts, .gitkeep, classes.txt), caches (.npy,
.cache, __pycache__), unfinished augmentation (.partial) and, unless --with-split, the
train/val/test split: its data.yaml holds this PC's paths, and the splitter rebuilds it
in seconds.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Vision/data
from pipeline_common import DATASET_DIR, SPLIT_DIR, TRAINED_MODELS_DIR, fix_console

REPO_DIR = DATASET_DIR.parents[2]
IN_GIT = {".py"}  # tracked by Git, like the names below
IN_GIT_NAMES = {"classes.txt", ".gitkeep"}
CACHES = {".npy", ".cache", ".pyc"}
STORED = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".pt", ".zip"}


def wanted(path: Path, with_split: bool) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or any(p.endswith(".partial") for p in path.parts):
        return False
    if path.name in IN_GIT_NAMES or path.suffix in IN_GIT or path.suffix in CACHES:
        return False
    return with_split or SPLIT_DIR not in path.parents


def collect(with_split: bool) -> tuple[list[Path], list[Path]]:
    """(files, empty folders) to export."""
    files, empty = [], []
    for root in (DATASET_DIR, TRAINED_MODELS_DIR):
        for path in sorted(root.rglob("*")):
            if path.is_file() and wanted(path, with_split):
                files.append(path)
            elif path.is_dir() and wanted(path, with_split) and not any(path.iterdir()):
                empty.append(path)
    return files, empty


def export(out_dir: Path, with_split: bool = False) -> Path:
    files, empty = collect(with_split)
    if not files:
        sys.exit("❌ Nothing to export: no data or models found.")
    zip_path = out_dir / f"{REPO_DIR.name}_data_{time.strftime('%Y-%m-%d_%H%M')}.zip"
    total = sum(f.stat().st_size for f in files)
    print(f"📦 {len(files)} files ({total / 1e6:.0f} MB) -> {zip_path}")
    partial = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(partial, "w") as z:
        for i, f in enumerate(files, 1):
            arc = Path(REPO_DIR.name) / f.relative_to(REPO_DIR)
            # Images, videos and weights are compressed already: storing them is much faster.
            kind = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            z.write(f, arc.as_posix(), compress_type=kind)
            if i % 200 == 0 or i == len(files):
                print(f"\r   {i}/{len(files)} files", end="")
        for d in empty:  # keep the folder layout
            z.writestr((Path(REPO_DIR.name) / d.relative_to(REPO_DIR)).as_posix() + "/", "")
    print()
    partial.replace(zip_path)  # a crash leaves only the .partial, never a broken zip
    return zip_path


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description="Zip the data and trained models.")
    parser.add_argument(
        "-o", "--out", type=Path, default=REPO_DIR.parent, help="folder for the zip"
    )
    parser.add_argument(
        "--with-split", action="store_true", help="include dataset/train (without caches)"
    )
    args = parser.parse_args()
    if not args.out.is_dir():
        sys.exit(f"❌ Not a folder: {args.out}")
    start = time.perf_counter()
    zip_path = export(args.out.resolve(), args.with_split)
    print(f"✅ Done in {time.perf_counter() - start:.0f} s: {zip_path}")
    print(
        f"   On the other PC: unzip, drag the {REPO_DIR.name} folder onto the folder that holds "
        "the cloned repo, merge. Then split the data again (preprocessing/"
        "train_val_data_splitter.py) before training."
    )


if __name__ == "__main__":
    main()
