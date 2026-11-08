#!/usr/bin/env python3
"""
Train/val splitter for YOLO training.

Gathers every labeled set in ../image_set_labeled into one YOLO dataset next
to this script:

    dataset/
    ├── image_set_labeled/          <- SOURCE (read only)
    │   ├── set 1/                  each set is either a Label Studio YOLO export
    │   │   ├── classes.txt           (images/ + labels/ subfolders) ...
    │   │   ├── images/  *.jpg
    │   │   └── labels/  *.txt
    │   └── set 2/                  ... or images and .txt labels side by side
    │       ├── frame_001.jpg
    │       └── frame_001.txt
    └── train/                      <- DESTINATION (this script's folder)
        ├── train_val_data_splitter.py
        ├── data.yaml               written by this script; pass it to YOLO
        ├── images/ train/ val/
        └── labels/ train/ val/

Steps:
    1. Find every set folder directly inside ../image_set_labeled.
       Folders starting with "." or "_" are skipped, so renaming a set to
       "_set 3" leaves it out without deleting it.
    2. Pair each image with its label (same file name, .txt extension) and
       check every label parses. Nothing is deleted until this all passes.
    3. Shuffle and split by TRAIN_RATIO. Each set is split on its own, so
       every set shows up in train and val in the same proportion.
    4. Delete everything in train/images and train/labels, copy the pairs
       into images/{train,val} and labels/{train,val}, write data.yaml.

Only the set folders and their images/ and labels/ subfolders are read.
Nothing above image_set_labeled or deeper inside a set is touched.

Usage:
    python train_val_data_splitter.py            # scan, show summary, ask, then build
    python train_val_data_splitter.py -y         # build without asking
    python train_val_data_splitter.py --dry-run  # scan and show summary only
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.8   # Share that goes to train; val gets the rest (0.7 -> 70/30).
SEED = 42           # Same seed + same files = same split, every run.

# Keep related images on the same side of the split. This is a regex run on
# each image's file name (without extension); its first capture group is the
# group key, and every image with the same key lands in train or in val,
# never both. Use it so frames of one fish don't end up in both sets, which
# makes val scores look better than they are (spec VD-70: split by fish ID).
#   e.g. r"^(fish\d+)"  ->  fish012_left_0003.jpg and fish012_top_0010.jpg stay together
# With grouping on, the split is made across all sets together instead of
# per set, because one fish can appear in several sets.
# None = no grouping: every image is split on its own.
GROUP_BY: str | None = None

SCRIPT_DIR = Path(__file__).parent.resolve()
SOURCE_DIR = SCRIPT_DIR.parent / "image_set_labeled"
YAML_PATH = SCRIPT_DIR / "data.yaml"
SPLITS = ("train", "val")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXT = ".txt"
CLASSES_FILE = "classes.txt"  # Label Studio writes this in each export: line N = class N.

COPY_WORKERS = 8    # Parallel copy threads. Copying is disk-bound, so threads speed it up.


class Pair(NamedTuple):
    image: Path
    label: Path
    set_name: str
    classes: Counter  # instances per class id in this label file


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def list_files(folder: Path) -> list[Path]:
    """Files directly in `folder` (no recursion). os.scandir is one directory read."""
    if not folder.is_dir():
        return []
    with os.scandir(folder) as entries:
        return [Path(e.path) for e in entries if e.is_file()]


def parse_label(path: Path) -> Counter | None:
    """
    Count instances per class in a YOLO label file, or None if it is malformed.

    Each non-blank line must be: <class id> <at least 4 numbers>. That covers
    detection, segmentation and pose labels. Coordinates aren't range-checked
    because pose keypoint visibility flags can be 2.
    An empty file is valid: YOLO treats it as a background image (no objects).
    """
    counts = Counter()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls < 0 or len(parts) < 5:
                    return None
                for value in parts[1:]:
                    float(value)
                counts[cls] += 1
    except (ValueError, UnicodeDecodeError):
        return None
    return counts


def collect_set(set_dir: Path) -> tuple[list[Pair], dict[str, list[str]]]:
    """
    Pair images with labels in one set.

    Uses images/ and labels/ subfolders if they exist, else the set folder itself.
    Names are matched ignoring case, as Windows does (F01.jpg pairs with f01.txt).
    Returns the pairs and a dict of problems found {description: [file names]}.
    """
    img_dir = set_dir / "images" if (set_dir / "images").is_dir() else set_dir
    lbl_dir = set_dir / "labels" if (set_dir / "labels").is_dir() else set_dir

    # Build name -> path lookups once, instead of an exists() check per image.
    images: dict[str, Path] = {}
    duplicates = []
    for f in sorted(list_files(img_dir)):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = f.stem.casefold()
        if key in images:
            duplicates.append(f.name)  # e.g. a.jpg and a.png share one label: keep the first
        else:
            images[key] = f
    labels = {
        f.stem.casefold(): f for f in list_files(lbl_dir)
        if f.suffix.lower() == LABEL_EXT and f.name.lower() != CLASSES_FILE
    }

    pairs, malformed = [], []
    for key in sorted(images.keys() & labels.keys()):
        counts = parse_label(labels[key])
        if counts is None:
            malformed.append(labels[key].name)
        else:
            pairs.append(Pair(images[key], labels[key], set_dir.name, counts))

    problems = {
        "images without a label (skipped)": sorted(images[k].name for k in images.keys() - labels.keys()),
        "labels without an image (skipped)": sorted(labels[k].name for k in labels.keys() - images.keys()),
        "images sharing a name with another image (skipped)": duplicates,
        "malformed labels (skipped)": malformed,
    }
    return pairs, problems


def read_class_names(set_dirs: list[Path]) -> list[str]:
    """
    Class names from the sets' classes.txt files.

    Every set's labels are merged as-is, so class 0 must mean the same thing in
    every set. If two classes.txt files disagree (e.g. two Label Studio projects
    with labels added in a different order), merging would silently mislabel
    data, so this stops the script instead.
    """
    found: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for d in set_dirs:
        f = d / CLASSES_FILE
        if f.is_file():
            names = tuple(n.strip() for n in f.read_text(encoding="utf-8").splitlines() if n.strip())
            found[names].append(d.name)

    if len(found) > 1:
        print(f"❌ The sets' {CLASSES_FILE} files don't match, so their class ids mean different things:")
        for names, sets in found.items():
            print(f"   {', '.join(sets)}: {list(names)}")
        sys.exit("   Make the class lists match (same names, same order) and relabel/export, then run again.")
    return list(next(iter(found))) if found else []


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_pairs(pairs: list[Pair], ratio: float, seed: int) -> tuple[list[Pair], list[Pair], int]:
    """
    Split into (train, val, number of images GROUP_BY didn't match).

    Images are put into groups (one image per group unless GROUP_BY is set),
    and groups into strata (one per set, or a single one when grouping).
    Each stratum's groups are shuffled and dealt to train until it reaches its
    share. The rounding error of each stratum carries into the next, so the
    overall split matches the ratio even when there are many small sets.
    Each stratum has its own seeded shuffle, so adding a new set doesn't
    reshuffle the existing ones.
    """
    pattern = re.compile(GROUP_BY) if GROUP_BY else None
    strata: dict[str, dict[str, list[Pair]]] = defaultdict(lambda: defaultdict(list))
    unmatched = 0
    for p in pairs:
        if pattern:
            m = pattern.search(p.image.stem)
            if m:
                key = m.group(1) if m.groups() else m.group(0)
            else:
                key, unmatched = f"{p.set_name}/{p.image.stem}", unmatched + 1
            strata["all"][key].append(p)
        else:
            strata[p.set_name][p.image.stem.casefold()].append(p)

    train, val = [], []
    seen = 0
    for name in sorted(strata):
        groups = [strata[name][k] for k in sorted(strata[name])]  # sorted first: shuffle needs a fixed start order
        random.Random(f"{seed}:{name}").shuffle(groups)
        n = sum(len(g) for g in groups)
        seen += n
        target = round(seen * ratio) - len(train)  # this stratum's train share, plus carried rounding error
        taken = 0
        for g in groups:
            if taken < target:
                train.extend(g)
                taken += len(g)
            else:
                val.extend(g)
    return train, val, unmatched


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def warn_list(title: str, items: list[str], limit: int = 5) -> None:
    if not items:
        return
    print(f"   ⚠️  {len(items)} {title}:")
    for name in items[:limit]:
        print(f"      - {name}")
    if len(items) > limit:
        print(f"      ... and {len(items) - limit} more")


def print_summary(train: list[Pair], val: list[Pair], class_names: list[str]) -> None:
    total = len(train) + len(val)
    print(f"\n📊 {len(train)} train ({len(train) / total:.0%}) / {len(val)} val ({len(val) / total:.0%})")

    per_set = defaultdict(lambda: [0, 0])
    for i, split in enumerate((train, val)):
        for p in split:
            per_set[p.set_name][i] += 1
    print(f"\n{'Set':<24} {'Train':>8} {'Val':>8}")
    print("-" * 42)
    for name in sorted(per_set):
        t, v = per_set[name]
        print(f"{name:<24} {t:>8} {v:>8}")

    train_counts = sum((p.classes for p in train), Counter())
    val_counts = sum((p.classes for p in val), Counter())
    print(f"\n{'Class':<24} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 51)
    for cls in sorted(train_counts.keys() | val_counts.keys()):
        t, v = train_counts[cls], val_counts[cls]
        name = class_names[cls] if cls < len(class_names) else f"class {cls}"
        print(f"{name:<24} {t:>8} {v:>8} {t + v:>8}")
        if t == 0 or v == 0:
            print(f"   ⚠️  '{name}' is missing from {'train' if t == 0 else 'val'}")
    print("-" * 51)
    t, v = sum(train_counts.values()), sum(val_counts.values())
    print(f"{'TOTAL instances':<24} {t:>8} {v:>8} {t + v:>8}")

    backgrounds = sum(1 for p in train + val if not p.classes)
    if backgrounds:
        print(f"\n   ℹ️  {backgrounds} image(s) have an empty label; YOLO uses them as background images.")


def write_yaml(class_names: list[str], pairs: list[Pair]) -> None:
    """Write data.yaml. YOLO needs a name for every id from 0 to the highest one used."""
    highest = max((c for p in pairs for c in p.classes), default=-1)
    names = class_names + [f"class_{i}" for i in range(len(class_names), highest + 1)]
    lines = [
        "# Written by train_val_data_splitter.py; rerunning the script overwrites it.",
        f"path: {json.dumps(SCRIPT_DIR.as_posix())}",  # JSON quoting is valid YAML and handles spaces
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)),
    ]
    YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ../image_set_labeled into a YOLO train/val split.")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask before deleting the old split")
    parser.add_argument("-n", "--dry-run", action="store_true", help="scan and show the split, change nothing")
    args = parser.parse_args()

    start = time.perf_counter()
    if not 0.0 < TRAIN_RATIO < 1.0:
        sys.exit(f"❌ TRAIN_RATIO must be between 0 and 1, got {TRAIN_RATIO}")
    if GROUP_BY:
        try:
            re.compile(GROUP_BY)
        except re.error as e:
            sys.exit(f"❌ GROUP_BY is not a valid regex: {e}")
    if not SOURCE_DIR.is_dir():
        sys.exit(f"❌ Source folder not found: {SOURCE_DIR}")

    print("=" * 60)
    print("YOLO TRAIN / VAL SPLITTER")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Dest:   {SCRIPT_DIR}")
    print(f"Split:  {TRAIN_RATIO:.0%} train / {1 - TRAIN_RATIO:.0%} val  (seed {SEED})")
    print(f"Groups: {GROUP_BY or 'off (each image split on its own)'}")

    # --- 1. Scan and check (nothing is changed yet) ---------------------
    set_dirs = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not set_dirs:
        sys.exit(f"❌ No set folders in {SOURCE_DIR}")
    class_names = read_class_names(set_dirs)

    print(f"\n🔍 Scanning {len(set_dirs)} set(s)...")
    pairs: list[Pair] = []
    for set_dir in set_dirs:
        set_pairs, problems = collect_set(set_dir)
        print(f"   {set_dir.name}: {len(set_pairs)} pairs")
        for title, items in problems.items():
            warn_list(title, items)
        pairs.extend(set_pairs)
    if not pairs:
        sys.exit("❌ No image/label pairs found.")

    if class_names:
        unknown = sorted({c for p in pairs for c in p.classes if c >= len(class_names)})
        if unknown:
            print(f"   ⚠️  Class ids {unknown} are used in labels but not listed in {CLASSES_FILE}")

    # --- 2. Split ---------------------------------------------------------
    train, val, unmatched = split_pairs(pairs, TRAIN_RATIO, SEED)
    if unmatched:
        print(f"   ⚠️  {unmatched} image name(s) didn't match GROUP_BY; each was split on its own")
    if not train or not val:
        sys.exit(f"❌ The {'train' if not train else 'val'} split would be empty "
                 f"({len(pairs)} pairs at {TRAIN_RATIO:.0%}). Add data or change TRAIN_RATIO.")

    # Two sets with the same file name would overwrite each other in the
    # merged folders, so those files get their set name as a prefix.
    name_counts = Counter(p.image.stem.casefold() for p in pairs)
    clashes = {stem for stem, n in name_counts.items() if n > 1}
    if clashes:
        print(f"   ⚠️  {len(clashes)} file name(s) appear in more than one set; "
              f"prefixing those with '<set>__'")

    print_summary(train, val, class_names)

    if args.dry_run:
        print("\nDry run: nothing was changed.")
        return
    if not args.yes:
        print(f"\nThis will DELETE everything in:\n   {SCRIPT_DIR / 'images'}\n   {SCRIPT_DIR / 'labels'}")
        if input("and rebuild them as shown above. Proceed? [y/N]: ").strip().lower() != "y":
            sys.exit("Cancelled.")

    # --- 3. Clear old output ----------------------------------------------
    print("\n🗑️  Clearing images/ and labels/...")
    for top in ("images", "labels"):
        folder = SCRIPT_DIR / top
        try:
            if folder.exists():
                shutil.rmtree(folder)  # also removes YOLO's *.cache files
        except OSError as e:
            # Failing loudly matters: a half-cleared folder would mix old files into the new split.
            sys.exit(f"❌ Couldn't clear {folder}: {e}\n   Close anything using these files and run again.")
        for split in SPLITS:
            (folder / split).mkdir(parents=True)
            (folder / split / ".gitkeep").touch()  # Git tracks the empty folder layout through these

    # --- 4. Copy and write data.yaml ---------------------------------------
    jobs: list[tuple[Path, Path]] = []
    for split, items in zip(SPLITS, (train, val)):
        for p in items:
            stem = f"{p.set_name}__{p.image.stem}" if p.image.stem.casefold() in clashes else p.image.stem
            jobs.append((p.image, SCRIPT_DIR / "images" / split / f"{stem}{p.image.suffix}"))
            jobs.append((p.label, SCRIPT_DIR / "labels" / split / f"{stem}{LABEL_EXT}"))

    print(f"📋 Copying {len(jobs)} files...")
    # copyfile copies only the bytes (no metadata), which is all training needs.
    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as pool:
        list(pool.map(lambda job: shutil.copyfile(*job), jobs))  # list() surfaces any copy error

    write_yaml(class_names, pairs)
    print(f"📝 Wrote {YAML_PATH.name}")
    print(f"\n✅ Done in {time.perf_counter() - start:.1f} s. Train with: data={YAML_PATH.as_posix()}")


if __name__ == "__main__":
    # Avoid a crash on emoji if output goes to a non-UTF-8 console or file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
