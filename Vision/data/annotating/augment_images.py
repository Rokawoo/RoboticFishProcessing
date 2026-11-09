#!/usr/bin/env python3
"""
Step 3: augmented copies of a labeled image set.

Usage:
    python augment_images.py                                  # pick a set and copy count in popups
    python augment_images.py "../dataset/image_sets/fish012_left" --copies 2
    python augment_images.py <set folder> -y                  # replace old copies without asking

Writes "<set> (augmented)/" with COPIES versions of every labeled image (empty
labels = no fish included), one effect each, named <original name>__aug<n>-<effect>:
    lighting        brightness, contrast, gamma
    white_balance   warmer or cooler light, slight color shifts
    motion_blur     arm moving during the exposure
    sensor_noise    low-light noise
    shadow          arm or person blocking the light

Effects change pixels only, never geometry, so each copy gets its original's label
file unchanged. Flips, rotation, scale and mosaic happen at training time
(training/fish_seg.yaml). The splitter puts augmented images in train only, and only
when their original is in train. Re-running replaces the augmented set (asks
first): fix labels in the original set, then augment again.
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Vision/data
from pipeline_common import (
    AUGMENTED_SUFFIX,
    IMAGE_SETS_DIR,
    LABEL_EXT,
    ask_folder,
    ask_integer,
    ask_yes_no,
    augmented_stem,
    find_label,
    fix_console,
    imread,
    imwrite,
    is_augmented_set,
    label_index,
    list_images,
    positive_int,
    set_dirs,
)

DEFAULT_COPIES = 2  # few: Ultralytics augments again during training
MAX_COPIES = 20
SEED = 42
JPEG_QUALITY = 95


def build_effects() -> dict:
    """Effect name -> pixel-only transform."""
    return {
        "lighting": A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15),
                A.RandomGamma(gamma_limit=(75, 130)),
            ],
            p=1,
        ),
        "white_balance": A.OneOf(
            [
                # Indoor light only; the default range reaches 15000 K (strong blue).
                A.PlanckianJitter(mode="blackbody", temperature_limit=(4000, 7500)),
                A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=8),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            ],
            p=1,
        ),
        "motion_blur": A.MotionBlur(blur_limit=(3, 9), p=1),
        "sensor_noise": A.OneOf(
            [
                A.GaussNoise(std_range=(0.01, 0.04)),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
            ],
            p=1,
        ),
        "shadow": A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            shadow_intensity_range=(0.2, 0.45),
            p=1,
        ),
    }


def augment_set(folder: Path, copies: int, seed: int = SEED, assume_yes: bool = False) -> Path:
    """Write `copies` augmented versions of every labeled image to '<folder> (augmented)'."""
    if is_augmented_set(folder.name):
        sys.exit(f"❌ '{folder.name}' is already augmented. Pick the original set.")
    image_dir, label_dir = set_dirs(folder)
    labels = label_index(label_dir)
    pairs = [(image, find_label(image, labels)) for image in list_images(image_dir)]
    pairs = [(image, label) for image, label in pairs if label is not None]
    if not pairs:
        sys.exit(f"❌ No labeled images in {folder}")
    out_dir = folder.parent / f"{folder.name}{AUGMENTED_SUFFIX}"

    print("=" * 60)
    print("IMAGE AUGMENTATION")
    print("=" * 60)
    print(f"Set:    {folder}  ({len(pairs)} labeled images)")
    print(f"Output: {out_dir}  ({copies} copies each)")
    if out_dir.exists() and any(out_dir.iterdir()):
        question = (
            f"'{out_dir.name}' already has {sum(1 for _ in out_dir.iterdir())} files.\n\n"
            "Replace them? Edits made to the old copies are lost."
        )
        if not assume_yes and not ask_yes_no("Replace augmented set?", question):
            sys.exit("Cancelled.")

    # Built in a hidden folder (the splitter skips "." folders) and swapped in at the
    # end, so a crash or Ctrl+C leaves the previous augmented set as it was.
    partial = folder.parent / f".{out_dir.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)  # left over from an interrupted run
    partial.mkdir()
    effects = build_effects()
    names = list(effects)
    pipelines = {n: A.Compose([effects[n]], seed=seed + i) for i, n in enumerate(names)}
    rng = random.Random(seed)

    written, unreadable = 0, []
    start = time.perf_counter()
    for i, (path, label) in enumerate(pairs, 1):
        image = imread(path)
        if image is None:
            unreadable.append(path.name)
            continue
        # Each copy of an image gets a different effect, until all are used.
        chosen = rng.sample(names, min(copies, len(names)))
        chosen += rng.choices(names, k=copies - len(chosen))
        jpeg = path.suffix.lower() in (".jpg", ".jpeg")
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if jpeg else []
        for n, effect in enumerate(chosen, 1):
            stem = augmented_stem(path.stem, n, effect)
            out_image = pipelines[effect](image=image)["image"]
            if not imwrite(partial / f"{stem}{path.suffix}", out_image, params):
                raise RuntimeError(f"couldn't write {partial / stem}{path.suffix}")
            shutil.copyfile(label, partial / f"{stem}{LABEL_EXT}")
            written += 1
        if i % 25 == 0 or i == len(pairs):
            print(f"   {i}/{len(pairs)} images done...")

    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
    except OSError as e:
        sys.exit(
            f"❌ Couldn't replace '{out_dir.name}' ({e}). Close programs using it and run "
            f"again; this run's copies are in {partial}"
        )
    print(f"\n✅ Wrote {written} augmented images in {time.perf_counter() - start:.0f} s.")
    if unreadable:
        print(f"   ⚠️  Skipped {len(unreadable)} unreadable image(s): {', '.join(unreadable[:10])}")
    print("Next: spot-check them with annotating/label_viewer.py, then split the data")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment a labeled image set.")
    parser.add_argument(
        "folder", nargs="?", type=Path, help="set folder in image_sets (popup if not given)"
    )
    parser.add_argument(
        "--copies", type=positive_int, help=f"copies per image (default {DEFAULT_COPIES})"
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default {SEED})")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="replace an old augmented set without asking"
    )
    args = parser.parse_args()

    folder = args.folder
    if folder is None:
        folder = ask_folder("Pick a labeled image set to augment", IMAGE_SETS_DIR)
        if folder is None:
            sys.exit("No folder picked.")
    if not folder.is_dir():
        sys.exit(f"❌ Not a folder: {folder}")

    copies = args.copies
    if copies is None and args.folder is None:
        copies = ask_integer(
            "Copies",
            f"Augmented copies per image of '{folder.name}':",
            DEFAULT_COPIES,
            1,
            MAX_COPIES,
        )
        if copies is None:
            sys.exit("Cancelled.")
    augment_set(folder.resolve(), copies or DEFAULT_COPIES, args.seed, args.yes)


if __name__ == "__main__":
    fix_console()
    main()
