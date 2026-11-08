#!/usr/bin/env python3
"""
Step 5: split every labeled image in ../image_sets into YOLO train/val(/test).

Usage:
    python train_val_data_splitter.py            # scan, show the split, ask, then build
    python train_val_data_splitter.py -y         # build without asking
    python train_val_data_splitter.py --dry-run  # show the split only

Reads image_sets/<set>/ (images with labels next to them, or a Label Studio
export with images/ and labels/). Sets starting with "." or "_" are skipped;
images without a label are skipped. Writes images/ and labels/ with train, val
(and test) folders plus data.yaml here, after deleting the previous split.

Split rules (they keep val and test honest):
    - Images whose name has a fish ID (GROUP_BY, e.g. fish012_left_000150 ->
      fish012) are split by fish: all images of a fish go to one split. With
      fewer than 2 fish this falls back to blocks.
    - Other images are split per set in CHUNKS_PER_SET blocks of consecutive
      frames, because neighboring frames are near copies.
    - TEST_RATIO > 0 holds out whole fish (or whole sets, for names without a
      fish ID) as a test set of fish the model never sees.
    - Augmented images go to train only, and only if their original is in train.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Vision/data
from pipeline_common import (
    CLASSES_FILE,
    IMAGE_EXTS,
    IMAGE_SETS_DIR,
    LABEL_EXT,
    SET_CLASSES_NAME,
    SPLIT_DIR,
    fix_console,
    is_augmented_set,
    label_index,
    set_dirs,
    source_of_augmented,
    source_set_name,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.8  # share of the non-test images for train; val gets the rest
TEST_RATIO = 0.0  # share of all images held out as test (0 = no test set)
SEED = 42  # same seed + same files = same split
GROUP_BY: str | None = r"^(fish\d+)"  # fish ID in file names (first group); None = off
CHUNKS_PER_SET = 10  # blocks of consecutive frames per set; 0 = image by image

YAML_PATH = SPLIT_DIR / "data.yaml"
COPY_WORKERS = 8


class Pair(NamedTuple):
    image: Path
    label: Path
    set_name: str
    classes: Counter  # instances per class id


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def list_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    with os.scandir(folder) as entries:
        return [Path(e.path) for e in entries if e.is_file()]


def parse_label(path: Path) -> Counter | None:
    """
    Instances per class, or None if malformed. A line must be <class id> plus at
    least 4 numbers (detection, segmentation and pose all qualify; coordinates
    aren't range-checked because pose visibility flags can be 2). Empty = background.
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
    """Pairs of one set, and problems found {description: [file names]}."""
    img_dir, lbl_dir = set_dirs(set_dir)
    images: dict[str, Path] = {}
    duplicates = []
    for f in sorted(list_files(img_dir)):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = f.stem.casefold()
        if key in images:
            duplicates.append(f.name)  # a.jpg + a.png would share one label
        else:
            images[key] = f
    labels = label_index(lbl_dir)

    pairs, malformed, used = [], [], set()
    for key in sorted(images.keys() & labels.keys()):
        used.add(labels[key])
        counts = parse_label(labels[key])
        if counts is None:
            malformed.append(labels[key].name)
        else:
            pairs.append(Pair(images[key], labels[key], set_dir.name, counts))

    problems = {
        "images without a label (skipped)": sorted(
            images[k].name for k in images.keys() - labels.keys()
        ),
        "labels without an image (skipped)": sorted(
            {f.name for f in labels.values() if f not in used}
        ),
        "images sharing a name with another image (skipped)": duplicates,
        "malformed labels (skipped)": malformed,
    }
    return pairs, problems


def read_class_names(set_folders: list[Path]) -> list[str]:
    """
    Class names from dataset/classes.txt and the sets' own classes.txt. Stops if
    they disagree: merging labels whose class ids mean different things would
    silently mislabel data.
    """
    found: dict[tuple[str, ...], list[str]] = defaultdict(list)
    sources = [(CLASSES_FILE, "dataset/classes.txt")]
    sources += [(d / SET_CLASSES_NAME, d.name) for d in set_folders]
    for f, label in sources:
        if f.is_file():
            names = tuple(
                n.strip() for n in f.read_text(encoding="utf-8").splitlines() if n.strip()
            )
            found[names].append(label)

    if len(found) > 1:
        print("❌ The class lists don't match, so their class ids mean different things:")
        for names, where in found.items():
            print(f"   {', '.join(where)}: {list(names)}")
        sys.exit("   Make them match (same names, same order), then run again.")
    return list(next(iter(found))) if found else []


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def fish_of(p: Pair, pattern: re.Pattern | None) -> str | None:
    """Fish ID from the image name, or None."""
    m = pattern.search(p.image.stem) if pattern else None
    if not m:
        return None
    return m.group(1) if m.groups() else m.group(0)


def hold_out_test(
    pairs: list[Pair], ratio: float, seed: int, pattern: re.Pattern | None
) -> tuple[list[Pair], list[Pair]]:
    """
    (test, rest): whole fish (whole sets for names without a fish ID), about
    `ratio` of the images. At least 2 fish/sets stay for train and val, so
    fewer than 3 gives no test set.
    """
    units: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        fish = fish_of(p, pattern)
        units[f"fish:{fish}" if fish else f"set:{p.set_name}"].append(p)
    if ratio <= 0 or len(units) < 3:
        return [], pairs
    keys = sorted(units)
    random.Random(f"{seed}:test").shuffle(keys)
    target = round(ratio * len(pairs))
    test: list[Pair] = []
    for key in keys[: len(keys) - 2]:
        if len(test) >= target:
            break
        test += units[key]
    in_test = {id(p) for p in test}
    return test, [p for p in pairs if id(p) not in in_test]


def split_pairs(
    pairs: list[Pair], ratio: float, seed: int, pattern: re.Pattern | None = None
) -> tuple[list[Pair], list[Pair]]:
    """
    (train, val) of original images. Groups are whole fish (across all sets) or
    blocks of consecutive frames (per set). Each stratum (all fish, or one set)
    is shuffled with its own seed, so new sets don't reshuffle old ones, and
    dealt to train until it reaches its share; the rounding error carries into
    the next stratum, so the overall ratio holds even for many small sets.
    """
    strata: dict[str, dict[str, list[Pair]]] = defaultdict(lambda: defaultdict(list))
    by_set: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        fish = fish_of(p, pattern)
        if fish is not None:
            strata["fish"][fish].append(p)
        else:
            by_set[p.set_name].append(p)
    for set_name, items in by_set.items():
        items.sort(key=lambda p: p.image.name.casefold())  # frame order
        n = len(items)
        for i, p in enumerate(items):
            chunk = i * CHUNKS_PER_SET // n if CHUNKS_PER_SET > 0 else i
            strata[f"set:{set_name}"][f"{chunk:06d}"].append(p)

    train, val = [], []
    seen = 0
    for name in sorted(strata):
        groups = [strata[name][k] for k in sorted(strata[name])]
        random.Random(f"{seed}:{name}").shuffle(groups)
        seen += sum(len(g) for g in groups)
        target = round(seen * ratio) - len(train)
        taken = 0
        for g in groups:
            if taken < target:
                train.extend(g)
                taken += len(g)
            else:
                val.extend(g)

    # With only a few fish, every fish can land in train; give val the smallest one.
    fish_groups = list(strata.get("fish", {}).values())
    if not val and len(fish_groups) >= 2:
        smallest = {id(p) for p in min(fish_groups, key=len)}
        val = [p for p in train if id(p) in smallest]
        train = [p for p in train if id(p) not in smallest]
    return train, val


def place_augmented(
    augmented: list[Pair], train: list[Pair], originals: list[Pair]
) -> tuple[list[Pair], int, int]:
    """
    (augmented pairs whose original is in train, count left out because the
    original is in val/test, count left out because there's no labeled original).
    """
    in_train = {(p.set_name, p.image.stem.casefold()) for p in train}
    known = {(p.set_name, p.image.stem.casefold()) for p in originals}
    keep, elsewhere, orphans = [], 0, 0
    for p in augmented:
        stem = source_of_augmented(p.image.stem)
        key = (source_set_name(p.set_name), (stem or "").casefold())
        if key in in_train:
            keep.append(p)
        elif key in known:
            elsewhere += 1
        else:
            orphans += 1
    return keep, elsewhere, orphans


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


def print_summary(splits: dict[str, list[Pair]], class_names: list[str]) -> None:
    names = list(splits)
    n_aug = sum(1 for p in splits["train"] if is_augmented_set(p.set_name))
    originals = {s: len(items) - (n_aug if s == "train" else 0) for s, items in splits.items()}
    total = sum(originals.values())
    parts = " / ".join(f"{originals[s]} {s} ({originals[s] / total:.0%})" for s in names)
    print(f"\n📊 Originals: {parts}, plus {n_aug} augmented in train")

    per_set: dict[str, Counter] = defaultdict(Counter)
    for s, items in splits.items():
        for p in items:
            per_set[p.set_name][s] += 1
    width = max(24, *(len(n) + 2 for n in per_set))
    print(f"\n{'Set':<{width}}" + "".join(f"{s.title():>8}" for s in names))
    for set_name in sorted(per_set):
        print(f"{set_name:<{width}}" + "".join(f"{per_set[set_name][s]:>8}" for s in names))

    counts = {s: sum((p.classes for p in items), Counter()) for s, items in splits.items()}
    print(f"\n{'Class':<24}" + "".join(f"{s.title():>8}" for s in names))
    for cls in sorted(set().union(*counts.values())):
        name = class_names[cls] if cls < len(class_names) else f"class {cls}"
        print(f"{name:<24}" + "".join(f"{counts[s][cls]:>8}" for s in names))
        missing = [s for s in names if counts[s][cls] == 0]
        if missing:
            print(f"   ⚠️  '{name}' is missing from {', '.join(missing)}")

    backgrounds = sum(1 for items in splits.values() for p in items if not p.classes)
    if backgrounds:
        print(f"\n   ℹ️  {backgrounds} image(s) have an empty label (YOLO: background).")


def write_yaml(class_names: list[str], pairs: list[Pair], with_test: bool) -> None:
    """data.yaml; YOLO needs a name for every class id up to the highest used."""
    highest = max((c for p in pairs for c in p.classes), default=-1)
    names = class_names + [f"class_{i}" for i in range(len(class_names), highest + 1)]
    lines = [
        "# Written by train_val_data_splitter.py; overwritten on every run.",
        f"path: {json.dumps(SPLIT_DIR.as_posix())}",  # JSON quoting is valid YAML
        "train: images/train",
        "val: images/val",
        *(["test: images/test"] if with_test else []),
        "names:",
        *(f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)),
    ]
    YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Split image_sets into YOLO train/val(/test).")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask before rebuilding")
    parser.add_argument("-n", "--dry-run", action="store_true", help="show the split only")
    args = parser.parse_args()

    start = time.perf_counter()
    if not 0.0 < TRAIN_RATIO < 1.0 or not 0.0 <= TEST_RATIO < 1.0:
        sys.exit("❌ TRAIN_RATIO must be in (0, 1) and TEST_RATIO in [0, 1)")
    try:
        pattern = re.compile(GROUP_BY) if GROUP_BY else None
    except re.error as e:
        sys.exit(f"❌ GROUP_BY is not a valid regex: {e}")
    if not IMAGE_SETS_DIR.is_dir():
        sys.exit(f"❌ Source folder not found: {IMAGE_SETS_DIR}")

    print("=" * 60)
    print("YOLO TRAIN / VAL SPLITTER")
    print("=" * 60)
    print(f"Source: {IMAGE_SETS_DIR}")
    print(f"Dest:   {SPLIT_DIR}")
    test_note = f", {TEST_RATIO:.0%} test" if TEST_RATIO else ""
    print(f"Split:  {TRAIN_RATIO:.0%} train / {1 - TRAIN_RATIO:.0%} val{test_note} (seed {SEED})")

    # --- 1. Scan and check (nothing changes yet) --------------------------
    set_folders = sorted(
        p for p in IMAGE_SETS_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )
    if not set_folders:
        sys.exit(f"❌ No set folders in {IMAGE_SETS_DIR}")
    class_names = read_class_names(set_folders)

    print(f"\n🔍 Scanning {len(set_folders)} set(s)...")
    originals: list[Pair] = []
    augmented: list[Pair] = []
    for set_dir in set_folders:
        set_pairs, problems = collect_set(set_dir)
        print(f"   {set_dir.name}: {len(set_pairs)} pairs")
        for title, items in problems.items():
            warn_list(title, items)
        (augmented if is_augmented_set(set_dir.name) else originals).extend(set_pairs)
    if not originals:
        sys.exit("❌ No labeled (non-augmented) images found.")

    if class_names:
        used = {c for p in originals + augmented for c in p.classes}
        unknown = sorted(c for c in used if c >= len(class_names))
        if unknown:
            print(f"   ⚠️  Class ids {unknown} are used in labels but not in classes.txt")

    # --- 2. Split -----------------------------------------------------------
    fish = {f for p in originals if (f := fish_of(p, pattern))}
    without_id = sum(1 for p in originals if fish_of(p, pattern) is None)
    if pattern and len(fish) < 2:
        if fish:
            print(f"   ⚠️  Only 1 fish ({next(iter(fish))}): splitting in blocks of frames, so")
            print("      val shows the same fish as train. Record more fish for an honest val.")
        pattern = None
    elif pattern and without_id:
        print(
            f"   ⚠️  {without_id} image(s) have no fish ID in their name: split in blocks of frames"
        )
    grouping = f"{len(fish)} fish" if pattern else f"blocks of frames ({CHUNKS_PER_SET} per set)"
    print(f"   Grouping: {grouping}")

    test, rest = hold_out_test(originals, TEST_RATIO, SEED, pattern)
    if TEST_RATIO and not test:
        print("   ⚠️  No test set: it needs at least 3 fish (or sets).")
    train, val = split_pairs(rest, TRAIN_RATIO, SEED, pattern)
    if not train or not val:
        sys.exit(
            f"❌ The {'train' if not train else 'val'} split would be empty "
            f"({len(rest)} images). Add data or change TRAIN_RATIO."
        )
    if augmented:
        aug_train, elsewhere, orphans = place_augmented(augmented, train, originals)
        train += aug_train
        print(
            f"   Augmented: {len(aug_train)} to train, {elsewhere} left out (original not in train)"
        )
        if orphans:
            print(f"   ⚠️  {orphans} augmented image(s) left out: no labeled original found")

    splits = {"train": train, "val": val, **({"test": test} if test else {})}
    everything = [p for items in splits.values() for p in items]
    # Same file name in two sets: prefix with the set name so they don't overwrite each other.
    name_counts = Counter(p.image.stem.casefold() for p in everything)
    clashes = {stem for stem, n in name_counts.items() if n > 1}
    if clashes:
        print(f"   ⚠️  {len(clashes)} file name(s) in more than one set: prefixed with '<set>__'")

    print_summary(splits, class_names)

    if args.dry_run:
        print("\nDry run: nothing was changed.")
        return
    if not args.yes:
        print("\nThis will DELETE everything in:")
        print(f"   {SPLIT_DIR / 'images'}\n   {SPLIT_DIR / 'labels'}")
        if input("and rebuild them as shown above. Proceed? [y/N]: ").strip().lower() != "y":
            sys.exit("Cancelled.")

    # --- 3. Clear old output ------------------------------------------------
    print("\n🗑️  Clearing images/ and labels/...")
    for top in ("images", "labels"):
        folder = SPLIT_DIR / top
        try:
            if folder.exists():
                shutil.rmtree(folder)  # also removes YOLO's .cache and .npy files
        except OSError as e:
            # Stop: a half-cleared folder would mix old files into the new split.
            sys.exit(
                f"❌ Couldn't clear {folder}: {e}\n   Close anything using them and run again."
            )
        for split in splits:
            (folder / split).mkdir(parents=True)
        for split in ("train", "val"):
            (folder / split / ".gitkeep").touch()  # tracked folder layout

    # --- 4. Copy and write data.yaml -----------------------------------------
    jobs: list[tuple[Path, Path]] = []
    for split, items in splits.items():
        for p in items:
            stem = (
                f"{p.set_name}__{p.image.stem}"
                if p.image.stem.casefold() in clashes
                else p.image.stem
            )
            jobs.append((p.image, SPLIT_DIR / "images" / split / f"{stem}{p.image.suffix}"))
            jobs.append((p.label, SPLIT_DIR / "labels" / split / f"{stem}{LABEL_EXT}"))

    print(f"📋 Copying {len(jobs)} files...")
    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as pool:
        list(pool.map(lambda job: shutil.copyfile(*job), jobs))  # list() raises copy errors

    write_yaml(class_names, everything, with_test=bool(test))
    print(f"📝 Wrote {YAML_PATH.name}")
    print(f"\n✅ Done in {time.perf_counter() - start:.1f} s. Next: python ../../training/train.py")


if __name__ == "__main__":
    fix_console()
    main()
