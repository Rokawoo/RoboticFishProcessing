#!/usr/bin/env python3
"""
Step 2: label fish by clicking them (with or instead of Label Studio).

Usage:
    python auto_labeler.py                 # pick a set folder in a popup
    python auto_labeler.py "<set folder>"

The label viewer plus auto-labeling. Switch to "Auto-label" (G), click each
fish, check the cyan outlines, press Enter to add them as labels, go to the next
image (Right). Labeling over an existing label replaces it. Every button
explains itself on hover; F1 lists all keys.

Engines:
    SAM (default)   Segment Anything (segmentors/sam.py) outlines the clicked
                    object on any background. Downloads the model (SAM_MODEL in
                    segmentors/sam.py) on first use; fastest with a GPU.
    Color           matches the clicked colors (segmentors/color.py). Needs fish
                    and table colors to differ (e.g. dark fish, light table).
                    Colors carry over to the next image, so it can label every
                    matching region without clicks, or every unlabeled image.

Labels are plain YOLO polygon files, the same as Label Studio's YOLO export, so
both tools can be used on the same set.
"""

import dataclasses
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))  # annotating/
from label_viewer import HELP, PANEL, LabelViewer, Tooltip, main
from pipeline_common import find_label, label_index, mask_to_polygon, write_label
from segmentors.color import (
    MAX_CLUSTERS,
    Cluster,
    ColorSegmentor,
    Settings,
    lab_to_rgb,
    sample_patch,
)
from segmentors.sam import SamSegmentor

PREVIEW_COLOR = (0, 255, 255)
MARK_COLORS = {"fish": (0, 255, 0), "same fish": (255, 255, 0), "not fish": (255, 60, 60)}
DEBOUNCE_MS = 150  # recompute this long after the last click or slider move
REPLACE_IOU = 0.5  # a new outline replaces existing labels it overlaps by more than this
AUTO_HELP = (
    "Auto-label (G on/off)",
    [
        ("Click", "a fish: the engine outlines it in cyan (click each fish)"),
        ("Shift+click", "more of the same fish (a fin or color it missed)"),
        ("Right-click", "something that is not fish (table, shadow)"),
        ("Enter", "add the outlines as labels of the current class (replaces overlapped labels)"),
        ("Esc", "clear this image's clicks"),
        ("E", "engine: SAM or Color"),
        ("[ ]  M  C", "color engine: tolerance down / up, enhanced view, forget colors"),
    ],
)
SAM_TIP = (
    "Segment Anything: outlines the clicked object on any background. "
    "Downloads the model on first use; fastest with a GPU."
)
COLOR_TIP = (
    "Matches the clicked colors. Needs fish and table colors to differ "
    "(dark fish, light table). Colors carry over to the next image."
)


def polygon_iou(a: list[list[float]], b: list[list[float]]) -> float:
    """Intersection over union of two pixel polygons (rasterized around both)."""
    pa, pb = np.round(np.array(a)).astype(np.int32), np.round(np.array(b)).astype(np.int32)
    lo = np.minimum(pa.min(axis=0), pb.min(axis=0))
    hi = np.maximum(pa.max(axis=0), pb.max(axis=0))
    if (
        np.maximum(pa.min(axis=0), pb.min(axis=0)) > np.minimum(pa.max(axis=0), pb.max(axis=0))
    ).any():
        return 0.0  # bounding boxes don't touch
    w, h = (hi - lo + 1).tolist()
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [pa - lo], 1)
    cv2.fillPoly(mb, [pb - lo], 1)
    union = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / union if union else 0.0


class AutoLabeler(LabelViewer):
    TITLE = "Auto labeler"

    def __init__(self, root: tk.Misc, folder: Path):
        # Set before LabelViewer.__init__, which loads the first image.
        self.auto = False
        self.engine = "sam"
        self.settings = Settings()
        self.objects: list[list[tuple[float, float]]] = []  # this image: points per fish
        self.negatives: list[tuple[float, float]] = []  # this image: not-fish points
        self.fish_colors: list[Cluster] = []  # color engine, kept across images
        self.not_fish_colors: list[Cluster] = []
        self.color_segmentor: ColorSegmentor | None = None  # color engine, per image
        self.sam: SamSegmentor | None = None
        self.bgr: np.ndarray | None = None
        self.preview: list[list[tuple[float, float]]] = []  # normalized outlines
        self.show_enhanced = False
        self.enhanced_image: Image.Image | None = None
        self.pending = None
        super().__init__(root, folder)

        keys = {
            "<Key-g>": lambda e: self.set_auto(not self.auto),
            "<Key-e>": lambda e: self.set_engine("color" if self.engine == "sam" else "sam"),
            "<Key-m>": lambda e: self.toggle_enhanced(),
            "<Key-c>": lambda e: self.forget_colors(),
            "<Return>": lambda e: self.on_enter(),
            "<Escape>": lambda e: self.on_escape(),
            "<bracketleft>": lambda e: self.nudge_tolerance(-0.25),
            "<bracketright>": lambda e: self.nudge_tolerance(0.25),
        }
        for key, handler in keys.items():
            root.bind(key, handler)  # replaces the editor's binding

    # --- toolbar ------------------------------------------------------------
    def build_toolbar(self) -> None:
        super().build_toolbar()
        self.auto_row = row = self.toolbar_row()
        self.label(row, "Mode", padx=(2, 4))
        self.mode_var = tk.StringVar(value="edit")
        self.choice(
            row,
            self.mode_var,
            [
                ("edit", "Edit", "Select polygons and fix them by hand."),
                ("auto", "Auto-label", "Click each fish; the engine outlines it."),
            ],
            lambda: self.set_auto(self.mode_var.get() == "auto"),
            "G switches between them.",
        )
        self.gap(row)
        self.label(row, "Engine", padx=(2, 4))
        self.engine_var = tk.StringVar(value=self.engine)
        self.choice(
            row,
            self.engine_var,
            [("sam", "SAM", SAM_TIP), ("color", "Color", COLOR_TIP)],
            lambda: self.set_engine(self.engine_var.get()),
            "E switches between them.",
        )
        self.gap(row)
        self.error_var = self.slider(
            row,
            "Max corner error",
            self.settings.max_error_px,
            0.5,
            4.0,
            0.25,
            "How far (image px) the outline may cut corners off the fish: higher = fewer "
            "corners, lower = a closer fit.",
        )
        self.gap(row)
        self.add_button = self.button(
            row,
            "Add as labels",
            self.on_enter,
            "Add the cyan outlines as labels of the current class (Enter). An outline over "
            "an existing label replaces it; Ctrl+Z brings it back.",
        )
        self.clear_button = self.button(
            row, "Clear clicks", self.on_escape, "Forget this image's clicks (Esc)"
        )

        self.color_row = row = self.toolbar_row()  # shown for the color engine only
        self.label(row, "Color engine", padx=(2, 4)).configure(style="Heading.TLabel")
        self.tolerance_var = self.slider(
            row,
            "Tolerance",
            self.settings.tolerance,
            1.0,
            10.0,
            0.25,
            "How different a pixel may be from the clicked colors and still count as fish "
            "([ and ] change it).",
        )
        self.all_var = self.check(
            row,
            "All matching regions",
            self.settings.keep_all,
            self.on_setting,
            "Outline every region that matches the fish colors, not only the clicked ones.",
        )
        self.refine_var = self.check(
            row,
            "Refine edges",
            self.settings.refine_edges,
            self.on_setting,
            "Snap the outlines to the image edges (GrabCut). A bit slower.",
        )
        self.enhanced_var = self.check(
            row,
            "Enhanced view",
            False,
            self.toggle_enhanced,
            "Show how fish-like each pixel is: white = fish color (M).",
        )
        self.button(
            row, "Forget colors", self.forget_colors, "Forget the sampled colors and start over (C)"
        )
        self.button(
            row,
            "Label all unlabeled images",
            self.label_all_unlabeled,
            "Label every image of the set that has no label yet, with all regions matching "
            "the sampled colors. Labeled images are not changed.",
        )
        self.swatches = tk.Canvas(
            row, height=int(18 * self.ui), width=1, bg=PANEL, highlightthickness=0
        )
        self.swatches.pack(side="left", padx=8)
        Tooltip(
            self.swatches,
            f"Sampled colors: green frame = fish, red = not fish (the newest {MAX_CLUSTERS} "
            "of each are kept).",
        )
        self.show_color_row()

    def slider(self, row, text, value, low, high, step, tip) -> tk.DoubleVar:
        self.label(row, text, padx=(2, 4))
        var = tk.DoubleVar(value=value)
        shown = ttk.Label(row, width=5)
        var.trace_add("write", lambda *_: shown.configure(text=f"{var.get():.2f}"))

        def moved(_value):
            snapped = round(round(var.get() / step) * step, 4)
            if snapped != var.get():
                var.set(snapped)
            self.on_setting()

        scale = ttk.Scale(
            row,
            from_=low,
            to=high,
            variable=var,
            command=moved,
            length=int(110 * self.ui),
            takefocus=False,
        )
        scale.pack(side="left")
        shown.pack(side="left", padx=(4, 0))
        shown.configure(text=f"{value:.2f}")
        Tooltip(scale, tip)
        return var

    def show_color_row(self) -> None:
        if self.engine == "color":
            self.color_row.pack(side="top", fill="x", pady=(0, 4), after=self.auto_row)
        else:
            self.color_row.pack_forget()

    def update_controls(self) -> None:
        super().update_controls()
        clicks = self.auto and bool(self.objects or self.negatives)
        self.add_button.state(["!disabled"] if self.auto and self.preview else ["disabled"])
        self.clear_button.state(["!disabled"] if clicks else ["disabled"])

    def help_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        return [AUTO_HELP, *HELP]

    def hint(self) -> tuple[str, str]:
        if not self.auto or self.drawing is not None:
            badge, hint = super().hint()
            return badge, hint + "  G: auto-label." if badge == "EDIT" else hint
        if self.engine == "sam":
            return "AUTO-LABEL · SAM", (
                "Click a fish.  Shift+click: more of it.  Right-click: not fish.  "
                "Enter: add as labels.  Esc: clear clicks.  G: back to editing."
            )
        return "AUTO-LABEL · COLOR", (
            "Click a fish to sample its colors.  Shift+click: another color of it.  "
            "Right-click: not fish.  Enter: add as labels.  [ ]: tolerance."
        )

    def read_settings(self) -> None:
        self.settings = dataclasses.replace(
            self.settings,
            tolerance=self.tolerance_var.get(),
            max_error_px=self.error_var.get(),
            keep_all=self.all_var.get(),
            refine_edges=self.refine_var.get(),
        )

    def on_setting(self) -> None:
        self.canvas.focus_set()  # arrow keys go to the image, not the slider
        self.schedule()

    def nudge_tolerance(self, step: float) -> None:
        self.tolerance_var.set(min(max(self.tolerance_var.get() + step, 1.0), 10.0))
        self.schedule()

    def update_swatches(self) -> None:
        """One square per sampled color: green frame = fish, red = not fish."""
        self.swatches.delete("all")
        u = self.ui
        x = 2 * u
        for clusters, outline in ((self.fish_colors, "#00ff00"), (self.not_fish_colors, "#ff3c3c")):
            for c in clusters:
                fill = "#{:02x}{:02x}{:02x}".format(*lab_to_rgb(c.mean))
                self.swatches.create_rectangle(
                    x, 2 * u, x + 14 * u, 16 * u, fill=fill, outline=outline, width=round(2 * u)
                )
                x += 18 * u
        self.swatches.configure(width=max(int(x), 1))

    # --- modes -----------------------------------------------------------------
    def set_auto(self, on: bool) -> None:
        if self.drawing is not None:
            self.say("Finish (Enter) or cancel (Esc) the polygon first")
            self.mode_var.set("auto" if self.auto else "edit")
            return
        self.auto = on
        self.mode_var.set("auto" if on else "edit")
        self.canvas.focus_set()
        if on:
            self.selected = None
            self.say(f"Auto-label on ({self.engine_name()}): click a fish")
            self.schedule()
        else:
            self.preview = []
            self.set_enhanced(False)
            self.update_view()
            self.say("Auto-label off: editing")

    def set_engine(self, engine: str) -> None:
        self.engine = engine
        self.engine_var.set(engine)
        self.canvas.focus_set()
        self.objects, self.negatives, self.preview = [], [], []  # clicks mean different things
        self.set_enhanced(False)
        if engine != "sam":
            self.release_sam()
        self.show_color_row()
        self.update_view()
        self.say(f"Engine: {self.engine_name()}")

    def engine_name(self) -> str:
        return "SAM" if self.engine == "sam" else "Color"

    # --- clicks -----------------------------------------------------------------
    def image_bgr(self) -> np.ndarray:
        if self.bgr is None:
            self.bgr = np.ascontiguousarray(np.asarray(self.base)[:, :, ::-1])
        return self.bgr

    def get_color_segmentor(self) -> ColorSegmentor:
        if self.color_segmentor is None:
            self.color_segmentor = ColorSegmentor(self.image_bgr())
        return self.color_segmentor

    def get_sam(self) -> SamSegmentor | None:
        try:
            if self.sam is None:
                self.say("Loading SAM (the first use downloads it, ~15 s)...")
                self.root.update_idletasks()
                self.sam = SamSegmentor()
            self.sam.set_image(self.image_bgr(), str(self.images[self.index]))
        except Exception as e:  # noqa: BLE001 - e.g. no ultralytics or no internet: fall back
            self.sam = None
            self.set_engine("color")
            self.say(f"SAM unavailable ({type(e).__name__}: {str(e)[:80]}); using the color engine")
            return None
        return self.sam

    def add_color(self, x: float, y: float, fish: bool) -> None:
        pixels = sample_patch(self.get_color_segmentor().lab, x, y)
        if pixels is None:
            return
        colors = self.fish_colors if fish else self.not_fish_colors
        colors.append(Cluster(pixels))
        del colors[:-MAX_CLUSTERS]  # the swatches show which are kept
        self.update_swatches()

    def inside(self, x: float, y: float) -> bool:
        w, h = self.base.size
        return 0 <= x < w and 0 <= y < h

    def on_press(self, event) -> None:
        self.canvas.focus_set()
        if not self.auto or self.drawing is not None:
            super().on_press(event)
            return
        x, y = self.to_image(event.x, event.y)
        if not self.inside(x, y):
            return
        if event.state & 0x0001 and self.objects:  # Shift: same fish
            self.objects[-1].append((x, y))
        else:
            self.objects.append([(x, y)])
        if self.engine == "color":
            self.add_color(x, y, fish=True)
        self.schedule()

    def on_right_click(self, event) -> None:
        if not self.auto or self.drawing is not None:
            super().on_right_click(event)
            return
        x, y = self.to_image(event.x, event.y)
        if not self.inside(x, y):
            return
        self.negatives.append((x, y))
        if self.engine == "color":
            self.add_color(x, y, fish=False)
        self.schedule()

    # --- preview -----------------------------------------------------------------
    def schedule(self) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(DEBOUNCE_MS, self.update_preview)

    def outlines(self, masks: list[np.ndarray]) -> list[list[tuple[float, float]]]:
        w, h = self.base.size
        min_area = self.settings.min_area_fraction * w * h
        out = []
        for mask in masks:
            points = mask_to_polygon(
                mask, self.settings.max_error_px, self.settings.min_iou, min_area
            )
            if points:
                out.append(points)
        return out

    def update_preview(self) -> None:
        self.pending = None
        self.read_settings()
        self.preview = []
        if not self.auto:
            self.redraw()
            return
        if self.engine == "sam":
            if not self.objects:
                self.say("Click each fish")
                return
            sam = self.get_sam()
            if sam is None:
                return
            try:
                self.preview = self.outlines(sam.segment(self.objects, self.negatives))
            except Exception as e:  # noqa: BLE001 - e.g. out of GPU memory: report, don't crash
                self.say(f"SAM failed: {type(e).__name__}: {str(e)[:80]}")
                return
        else:
            if not self.fish_colors:
                self.say("Click a fish to sample its colors")
                return
            seg = self.get_color_segmentor()
            seeds = [points[0] for points in self.objects]
            regions = seg.segment(self.fish_colors, self.not_fish_colors, seeds, self.settings)
            self.preview = self.outlines(regions)
            if self.show_enhanced:
                self.make_enhanced()
        if self.preview:
            corners = sum(len(p) for p in self.preview)
            self.say(f"{len(self.preview)} outline(s), {corners} corners. Enter: add as labels")
        elif self.engine == "color" and not self.objects and not self.settings.keep_all:
            self.say("Click each fish (colors are kept from earlier images)")
        else:
            self.say(
                "Nothing outlined: add clicks"
                + (" or raise the tolerance" if self.engine == "color" else "")
            )

    def on_enter(self) -> None:
        if not self.auto or self.drawing is not None:
            self.finish_drawing()
            return
        if self.pending is not None:  # apply the latest clicks first
            self.root.after_cancel(self.pending)
            self.update_preview()
        if not self.preview:
            self.say("Nothing outlined yet: click a fish")
            return
        self.push_undo()
        w, h = self.base.size
        new = [[[x * w, y * h] for x, y in points] for points in self.preview]
        # Labeling over an existing outline replaces it (Ctrl+Z brings it back).
        kept = [p for p in self.polys if all(polygon_iou(p[1], n) <= REPLACE_IOU for n in new)]
        replaced = len(self.polys) - len(kept)
        self.polys = kept + [[self.current_class, n] for n in new]
        self.objects, self.negatives, self.preview, self.dirty = [], [], [], True
        note = f", replaced {replaced} old one(s)" if replaced else ""
        self.say(f"Added {len(new)} label(s){note}. Next image: Right or D")

    def on_escape(self) -> None:
        if self.auto and self.drawing is None and (self.objects or self.negatives):
            self.objects, self.negatives = [], []
            self.schedule()
            self.say("Clicks cleared")
        else:
            self.escape()

    # --- color engine extras --------------------------------------------------
    def forget_colors(self) -> None:
        self.canvas.focus_set()
        self.fish_colors, self.not_fish_colors = [], []
        self.preview, self.enhanced_image = [], None
        self.set_enhanced(False)
        self.update_swatches()
        self.update_view()
        self.say("Colors forgotten")

    def set_enhanced(self, on: bool) -> None:
        self.show_enhanced = on
        self.enhanced_var.set(on)

    def toggle_enhanced(self) -> None:
        if not (self.auto and self.engine == "color" and self.fish_colors):
            self.set_enhanced(False)
            self.say("Enhanced view: in auto-label mode, after clicking a fish")
            return
        self.set_enhanced(not self.show_enhanced)
        if self.show_enhanced:
            self.make_enhanced()
        else:
            self.update_view()

    def make_enhanced(self) -> None:
        gray = self.get_color_segmentor().enhanced(
            self.fish_colors, self.not_fish_colors, self.settings.tolerance
        )
        self.enhanced_image = Image.fromarray(gray).convert("RGB")
        self.update_view()

    def label_all_unlabeled(self) -> None:
        self.canvas.focus_set()
        if not self.fish_colors:
            self.say("Uses the color engine: click a fish in auto-label mode first")
            return
        labels = label_index(self.label_dir)
        todo = [p for p in self.images if find_label(p, labels) is None]
        if not todo:
            self.say("Every image already has a label")
            return
        question = (
            f"Label {len(todo)} image(s) without a label file, using every region that matches "
            "the sampled colors?\n\nLabeled images are not changed. Images where nothing "
            "matches stay unlabeled."
        )
        if not messagebox.askyesno("Label all unlabeled images", question, parent=self.root):
            return
        self.save()
        self.read_settings()
        settings = dataclasses.replace(self.settings, keep_all=True)
        labeled = 0
        for i, path in enumerate(todo, 1):
            self.message_label.configure(text=f"Auto-labeling {i}/{len(todo)}: {path.name}")
            self.root.update_idletasks()  # redraw only; no input events mid-batch
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            seg = ColorSegmentor(np.ascontiguousarray(rgb[:, :, ::-1]))
            regions = seg.segment(self.fish_colors, self.not_fish_colors, [], settings)
            outlines = seg.polygons(regions, settings)
            if outlines:  # no match: leave unlabeled, don't mark it as background
                label = self.label_dir / f"{path.stem}.txt"
                write_label(label, [(self.current_class, p) for p in outlines])
                labeled += 1
        self.load(self.index)
        self.say(f"Labeled {labeled} of {len(todo)} image(s). Check them; U finds the rest")

    # --- editor hooks -----------------------------------------------------------
    def display_image(self) -> Image.Image:
        if self.auto and self.show_enhanced and self.enhanced_image is not None:
            return self.enhanced_image
        return self.base

    def load(self, index: int) -> None:
        self.objects, self.negatives, self.preview = [], [], []  # clicks are per image
        self.color_segmentor, self.bgr, self.enhanced_image = None, None, None
        super().load(index)
        if self.auto and self.engine == "color" and self.fish_colors:
            self.schedule()  # sampled colors carry over

    def release_sam(self) -> None:
        """Free SAM and its GPU memory (reloaded on the next SAM click)."""
        if self.sam is not None:
            self.sam.close()
            self.sam = None

    def close(self) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.release_sam()
        super().close()

    def render(self) -> Image.Image:
        img = super().render()
        if not self.auto:
            return img
        s, u, (x0, y0), (w, h) = self.scale, self.ui, self.view_origin, self.base.size

        def screen(points):
            return [((x * w - x0) * s, (y * h - y0) * s) for x, y in points]

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for points in self.preview:
            d.polygon(screen(points), fill=PREVIEW_COLOR + (55,))
        img = Image.alpha_composite(img, overlay)
        d = ImageDraw.Draw(img)
        for points in self.preview:
            sp = screen(points)
            d.line(sp + sp[:1], fill=PREVIEW_COLOR, width=round(2 * u))
            r = 2 * u
            for x, y in sp:
                d.rectangle((x - r, y - r, x + r, y + r), fill=PREVIEW_COLOR)
        marks = [
            (x, y, "fish" if i == 0 else "same fish")
            for pts in self.objects
            for i, (x, y) in enumerate(pts)
        ]
        marks += [(x, y, "not fish") for x, y in self.negatives]
        r = 7 * u
        for x, y, kind in marks:  # a cross per click, outlined in black
            sx, sy = (x - x0) * s, (y - y0) * s
            for width, color in ((round(4 * u), (0, 0, 0)), (round(2 * u), MARK_COLORS[kind])):
                d.line((sx - r, sy, sx + r, sy), fill=color, width=width)
                d.line((sx, sy - r, sx, sy + r), fill=color, width=width)
        return img


if __name__ == "__main__":
    main(AutoLabeler, "Label fish by clicking them, and edit the labels.")
