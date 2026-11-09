#!/usr/bin/env python3
"""
Step 3: augmented copies of labeled image sets.

Usage:
    python augment_images.py                       # pick a set and copy count in popups
    python augment_images.py --all                 # every set in image_sets
    python augment_images.py "<set>" "<set 2>" --copies 3
    python augment_images.py --all -y              # replace old copies without asking

Writes "<set> (augmented)/" with COPIES versions of every labeled image (empty
labels = no fish included), one effect each, named <original name>__aug<n>-<effect>:
    lighting        brightness, contrast, gamma
    white_balance   warmer or cooler light, slight color shifts
    motion_blur     arm moving during the exposure
    sensor_noise    low-light noise
    shadow          arm or person blocking the light

Effects change pixels only, never geometry, so labels carry over exactly: each
line is copied as is, except a polygon reaching past the image edge, which is cut
at the edge (YOLO would skip the image). Flips, rotation, scale and mosaic happen
at training time (training/fish_seg.yaml). The splitter puts augmented images in
train only, with their original. Re-running replaces a set's copies (asks first):
fix labels in the original set, then augment again.
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
    clip_polygon,
    find_label,
    fix_console,
    imread,
    imwrite,
    inside_image,
    is_augmented_set,
    label_index,
    list_images,
    parse_polygon,
    polygon_line,
    positive_int,
    set_dirs,
)

DEFAULT_COPIES = 2  # few: Ultralytics augments again during training
MAX_COPIES = 20
SEED = 42
JPEG_QUALITY = 95


class Skipped(Exception):
    """A set that isn't augmented (no labels, already augmented, replacement declined)."""


class CenteredMotionBlur(A.MotionBlur):
    """
    MotionBlur with a point-symmetric kernel. Albumentations' kernels are off-center
    (measured up to 2.8 px, 1.5 px with allow_shifted=False): the picture moves off its labels.
    """

    def get_params(self) -> dict:
        kernel = super().get_params()["kernel"]
        kernel = kernel + kernel[::-1, ::-1]
        return {"kernel": kernel / kernel.sum()}


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
        "motion_blur": CenteredMotionBlur(blur_limit=(3, 9), p=1),
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


def carry_label(source: Path, target: Path) -> tuple[int, int]:
    """
    Copy a label file. Polygons reaching past the image edge are cut at the edge
    (dropped if nothing is left); every other line is copied unchanged.
    Returns (polygons cut, polygons dropped).
    """
    lines, cut, dropped = [], 0, 0
    for line in source.read_text(encoding="utf-8").splitlines():
        polygon = parse_polygon(line)
        if polygon is None or inside_image(polygon[1]):
            lines.append(line)
            continue
        kept = clip_polygon(polygon[1])
        if kept:
            lines.append(polygon_line(polygon[0], kept))
            cut += 1
        else:
            dropped += 1
    if cut or dropped:
        target.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    else:
        shutil.copyfile(source, target)  # byte for byte
    return cut, dropped


def labeled_images(folder: Path) -> list[tuple[Path, Path]]:
    """(image, label) of every labeled image in a set."""
    image_dir, label_dir = set_dirs(folder)
    labels = label_index(label_dir)
    pairs = [(image, find_label(image, labels)) for image in list_images(image_dir)]
    return [(image, label) for image, label in pairs if label is not None]


def augment_set(folder: Path, copies: int, seed: int = SEED, assume_yes: bool = False) -> Path:
    """Write `copies` augmented versions of every labeled image to '<folder> (augmented)'."""
    if is_augmented_set(folder.name):
        raise Skipped("already an augmented set")
    pairs = labeled_images(folder)
    if not pairs:
        raise Skipped("no labeled images")
    out_dir = folder.parent / f"{folder.name}{AUGMENTED_SUFFIX}"

    print("=" * 60)
    print(f"AUGMENTING {folder.name}")
    print("=" * 60)
    print(f"Set:    {folder}  ({len(pairs)} labeled images)")
    print(f"Output: {out_dir}  ({copies} copies each)")
    if out_dir.exists() and any(out_dir.iterdir()):
        question = (
            f"'{out_dir.name}' already has {sum(1 for _ in out_dir.iterdir())} files.\n\n"
            "Replace them? Edits made to the old copies are lost."
        )
        if not assume_yes and not ask_yes_no("Replace augmented set?", question):
            raise Skipped("kept the existing augmented set")

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

    written, unreadable, cut, dropped = 0, [], 0, 0
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
            if out_image.shape != image.shape:  # labels are only valid on the same pixels
                raise RuntimeError(f"'{effect}' changed the image size of {path.name}")
            if not imwrite(partial / f"{stem}{path.suffix}", out_image, params):
                raise RuntimeError(f"couldn't write {partial / stem}{path.suffix}")
            c, d = carry_label(label, partial / f"{stem}{LABEL_EXT}")
            cut, dropped = cut + c, dropped + d
            written += 1
        if i % 25 == 0 or i == len(pairs):
            print(f"   {i}/{len(pairs)} images done...")

    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
    except OSError as e:
        raise RuntimeError(
            f"couldn't replace '{out_dir.name}' ({e}). Close programs using it and run "
            f"again; this run's copies are in {partial}"
        ) from None
    print(f"✅ Wrote {written} augmented images in {time.perf_counter() - start:.0f} s.")
    if cut or dropped:
        print(f"   ✂️  {cut} polygon(s) cut at the image edge, {dropped} fully outside dropped")
    if unreadable:
        print(f"   ⚠️  Skipped {len(unreadable)} unreadable image(s): {', '.join(unreadable[:10])}")
    return out_dir


def all_sets() -> list[Path]:
    """Every original set in image_sets ("." and "_" folders and augmented sets excluded)."""
    if not IMAGE_SETS_DIR.is_dir():
        return []
    return sorted(
        p
        for p in IMAGE_SETS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and not is_augmented_set(p.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment labeled image sets.")
    parser.add_argument(
        "folders", nargs="*", type=Path, help="set folders in image_sets (popup if none)"
    )
    parser.add_argument("--all", action="store_true", help="every set in image_sets")
    parser.add_argument(
        "--copies", type=positive_int, help=f"copies per image (default {DEFAULT_COPIES})"
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default {SEED})")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="replace old augmented sets without asking"
    )
    args = parser.parse_args()

    folders = all_sets() if args.all else args.folders
    asked = not args.all and not args.folders
    if asked:
        folder = ask_folder("Pick a labeled image set to augment", IMAGE_SETS_DIR)
        if folder is None:
            sys.exit("No folder picked.")
        folders = [folder]
    if not folders:
        sys.exit(f"❌ No sets in {IMAGE_SETS_DIR}")
    for folder in folders:
        if not folder.is_dir():
            sys.exit(f"❌ Not a folder: {folder}")

    copies = args.copies
    if copies is None and asked:
        copies = ask_integer(
            "Copies",
            f"Augmented copies per image of '{folders[0].name}':",
            DEFAULT_COPIES,
            1,
            MAX_COPIES,
        )
        if copies is None:
            sys.exit("Cancelled.")

    done, skipped, failed = [], [], []
    for folder in folders:
        try:
            done.append(
                augment_set(folder.resolve(), copies or DEFAULT_COPIES, args.seed, args.yes)
            )
        except Skipped as e:
            print(f"⏭️  Skipped '{folder.name}': {e}")
            skipped.append(folder.name)
        except RuntimeError as e:
            print(f"❌ '{folder.name}': {e}")
            failed.append(folder.name)
        print()

    print(f"Augmented {len(done)} set(s): {', '.join(d.name for d in done) or '-'}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    if failed:
        sys.exit(f"❌ Failed: {', '.join(failed)}")
    print("Next: spot-check them with annotating/label_viewer.py, then split the data")


if __name__ == "__main__":
    fix_console()
    main()
