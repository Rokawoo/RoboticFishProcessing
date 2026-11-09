#!/usr/bin/env python3
"""
Step 3: augmented copies of a labeled image set.

Usage:
    python augment_images.py                                # pick a set and copy count in popups
    python augment_images.py "../dataset/image_sets/fish012_left" --copies 3
    python augment_images.py <set folder> --copies 3 -y     # replace old copies without asking

Writes "<set> (augmented)/" with COPIES versions of every labeled image, named
<original name>__aug<n>-<effect>. Effects cover what Ultralytics' training-time
augmentation (flips, rotation, scale, HSV; see training/fish_seg.yaml) doesn't:
    lighting, white_balance, motion_blur (arm moving), defocus, sensor_noise,
    shadow (arm or person), jpeg, camera_move (shift/turn/zoom), camera_tilt, mixed

Labels: each polygon is rasterized to a mask, transformed with the image, and
traced back to a polygon; fish pushed (almost) out of frame are dropped. Spatial
effects fill borders with gray, because mirrored borders create unlabeled fish.

Only labeled images are augmented; labels with non-polygon lines are skipped.
The splitter uses augmented images for train only, and only if their original
is in train. Re-running replaces the augmented set (asks first), so fix labels
in the original set, not in the copies.
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Vision/data
from pipeline_common import (
    AUGMENTED_SUFFIX,
    IMAGE_SETS_DIR,
    Polygon,
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
    mask_to_polygon,
    positive_int,
    read_label,
    set_dirs,
    write_label,
)

DEFAULT_COPIES = 3  # few: Ultralytics also augments during training
SEED = 42
JPEG_QUALITY = 95
BORDER_GRAY = (114, 114, 114)  # Ultralytics' padding gray
MIN_AREA_FRACTION = 0.0005  # fish with less visible area (share of the image) are dropped
POLYGON_MAX_ERROR_PX = 1.0
POLYGON_MIN_IOU = 0.98


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------
def build_recipes() -> dict[str, list]:
    """Effect name -> list of transforms."""
    border = {"border_mode": cv2.BORDER_CONSTANT, "fill": BORDER_GRAY, "fill_mask": 0}

    def lighting(p=1.0):
        return A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15),
                A.RandomGamma(gamma_limit=(75, 130)),
            ],
            p=p,
        )

    def white_balance(p=1.0):
        return A.OneOf(
            [
                # Indoor light only; the default range reaches 15000 K (strong blue).
                A.PlanckianJitter(mode="blackbody", temperature_limit=(4000, 7500)),
                A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=8),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            ],
            p=p,
        )

    def degrade(p=1.0):
        return A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 9)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(std_range=(0.01, 0.04)),
                A.ImageCompression(quality_range=(55, 85)),
            ],
            p=p,
        )

    return {
        "lighting": [lighting()],
        "white_balance": [white_balance()],
        "motion_blur": [A.MotionBlur(blur_limit=(3, 9), p=1)],
        "defocus": [A.GaussianBlur(blur_limit=(3, 5), p=1)],
        "sensor_noise": [
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.04)),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
                ],
                p=1,
            ),
            A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.0), contrast_limit=0, p=0.5),
        ],
        "shadow": [
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                shadow_intensity_range=(0.2, 0.45),
                p=1,
            )
        ],
        "jpeg": [A.ImageCompression(quality_range=(55, 85), p=1)],
        "camera_move": [
            A.Affine(
                scale=(0.95, 1.08), translate_percent=(-0.04, 0.04), rotate=(-8, 8), p=1, **border
            ),
            lighting(p=0.3),
        ],
        "camera_tilt": [A.Perspective(scale=(0.02, 0.05), p=1, **border), lighting(p=0.3)],
        "mixed": [
            A.Affine(scale=(0.95, 1.05), rotate=(-5, 5), p=0.5, **border),
            lighting(p=0.8),
            white_balance(p=0.5),
            degrade(p=0.5),
        ],
    }


# ---------------------------------------------------------------------------
# Labels <-> masks
# ---------------------------------------------------------------------------
def polygons_to_masks(polygons: list[Polygon], w: int, h: int) -> list[np.ndarray]:
    masks = []
    for _, points in polygons:
        mask = np.zeros((h, w), np.uint8)
        pts = np.round(np.array(points) * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


def mask_to_points(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Mask -> polygon (largest piece if a transform cut a fish in two), or None if too small."""
    h, w = mask.shape
    return mask_to_polygon(mask, POLYGON_MAX_ERROR_PX, POLYGON_MIN_IOU, MIN_AREA_FRACTION * w * h)


def augment_one(
    image: np.ndarray, polygons: list[Polygon], pipeline: A.Compose
) -> tuple[np.ndarray, list[Polygon]]:
    """Apply `pipeline` to an image and move its polygons with it."""
    h, w = image.shape[:2]
    if not polygons:
        return pipeline(image=image)["image"], []
    result = pipeline(image=image, masks=polygons_to_masks(polygons, w, h))
    out_h, out_w = result["image"].shape[:2]
    moved = []
    for (cls, _), mask in zip(polygons, result["masks"]):
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.shape != (out_h, out_w):
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        points = mask_to_points(mask)
        if points:
            moved.append((cls, points))
    return result["image"], moved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def augment_set(folder: Path, copies: int, seed: int = SEED, assume_yes: bool = False) -> Path:
    """Augment every labeled image in `folder` into '<folder> (augmented)'."""
    if is_augmented_set(folder.name):
        sys.exit(f"❌ '{folder.name}' is already augmented. Pick the original set.")
    out_dir = folder.parent / f"{folder.name}{AUGMENTED_SUFFIX}"
    image_dir, label_dir = set_dirs(folder)
    labels = label_index(label_dir)
    images = [img for img in list_images(image_dir) if img.stem.casefold() in labels]
    if not images:
        sys.exit(f"❌ No labeled images in {folder}")

    print("=" * 60)
    print("IMAGE AUGMENTATION")
    print("=" * 60)
    print(f"Set:    {folder}  ({len(images)} labeled images)")
    print(f"Output: {out_dir}  ({copies} copies each)")

    if out_dir.exists() and any(out_dir.iterdir()):
        n_old = sum(1 for _ in out_dir.iterdir())
        question = (
            f"'{out_dir.name}' already has {n_old} files.\n\n"
            "Replace them? Edits made to the old copies are lost."
        )
        if not assume_yes and not ask_yes_no("Replace augmented set?", question):
            sys.exit("Cancelled.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipes = build_recipes()
    names = list(recipes)
    pipelines = {n: A.Compose(recipes[n], seed=seed + i) for i, n in enumerate(names)}
    rng = random.Random(seed)

    written, skipped, dropped_fish = 0, [], 0
    start = time.perf_counter()
    for i, path in enumerate(images, 1):
        polygons, other = read_label(find_label(path, labels))
        if other:
            skipped.append(f"{path.name} (label has non-polygon lines)")
            continue
        image = imread(path)
        if image is None:
            skipped.append(f"{path.name} (can't read image)")
            continue

        # Each copy of an image gets a different effect, until all are used.
        chosen = rng.sample(names, min(copies, len(names)))
        chosen += rng.choices(names, k=copies - len(chosen))
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            if path.suffix.lower() in (".jpg", ".jpeg")
            else []
        )
        for n, recipe in enumerate(chosen, 1):
            aug_image, aug_polygons = augment_one(image, polygons, pipelines[recipe])
            dropped_fish += len(polygons) - len(aug_polygons)
            stem = augmented_stem(path.stem, n, recipe)
            out_path = out_dir / f"{stem}{path.suffix}"
            if not imwrite(out_path, aug_image, params):
                raise RuntimeError(f"couldn't write {out_path}")
            write_label(out_dir / f"{stem}.txt", aug_polygons)
            written += 1
        if i % 25 == 0 or i == len(images):
            print(f"   {i}/{len(images)} images done...")

    print(f"\n✅ Wrote {written} augmented images in {time.perf_counter() - start:.0f} s.")
    if dropped_fish:
        print(f"   {dropped_fish} fish were pushed out of frame and left out of those labels.")
    if skipped:
        print(f"   ⚠️  Skipped {len(skipped)} image(s):")
        for s in skipped[:10]:
            print(f"      - {s}")
    print("Next: check them with annotating/label_viewer.py")
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
    if copies is None:
        if args.folder is not None:
            copies = DEFAULT_COPIES
        else:
            copies = ask_integer(
                "Copies", f"Augmented copies per image of '{folder.name}':", DEFAULT_COPIES, 1, 50
            )
            if copies is None:
                sys.exit("Cancelled.")

    augment_set(folder.resolve(), copies, args.seed, args.yes)


if __name__ == "__main__":
    fix_console()
    main()
