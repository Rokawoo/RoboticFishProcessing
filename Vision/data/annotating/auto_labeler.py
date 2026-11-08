#!/usr/bin/env python3
"""
Step 2: label fish by clicking them (with or instead of Label Studio).

Usage:
    python auto_labeler.py                 # pick a set folder in a popup
    python auto_labeler.py "<set folder>"

The label editor (all its tools and keys) plus auto-labeling:
    G               auto-labeling on/off (off = normal editing)
    click           a fish: it is outlined in cyan (click each fish)
    Shift+click     more of the same fish (a fin or color it missed)
    right-click     something that is not fish (table, shadow)
    Enter           add the outlines as labels (current class, 0-9 to change)
    Esc             clear this image's clicks
    E               engine: SAM or Color

Engines:
    SAM (default)   Segment Anything (sam_segmenter.py) outlines the clicked
                    object. Works on any contrast; downloads a 75 MB model on
                    first use; fastest with a GPU.
    Color           matches the clicked colors (color_segmenter.py). Needs fish
                    and table colors to differ (e.g. dark fish, light table).
                    Colors carry over to the next image, so it can label every
                    matching region without clicks ("Label all matching regions",
                    "Label all unlabeled images"). [ ]: tolerance, M: enhanced
                    view, C: forget colors.

Labels are plain YOLO polygon files, the same as Label Studio's YOLO export, so
both tools can be used on the same set.
"""

import dataclasses
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))  # annotating/
from color_segmenter import (
    MAX_CLUSTERS,
    Cluster,
    ColorSegmenter,
    Settings,
    lab_to_rgb,
    sample_patch,
)
from label_editor import LabelEditor, main
from pipeline_common import find_label, label_index, mask_to_polygon, write_label
from sam_segmenter import SamSegmenter

PREVIEW_COLOR = (0, 255, 255)
MARK_COLORS = {"fish": (0, 255, 0), "same fish": (255, 255, 0), "not fish": (255, 60, 60)}
DEBOUNCE_MS = 150  # recompute this long after the last click or slider move
AUTO_HELP = (
    "Auto-label (G): click a fish   Shift+click: more of the same fish   "
    "Right-click: not fish   Enter: add outlines as labels   Esc: clear clicks   "
    "E: engine (SAM / Color)   Color engine: [ ] tolerance, M enhanced view, C forget colors"
)


class AutoLabeler(LabelEditor):
    def __init__(self, root: tk.Misc, folder: Path):
        # Set before LabelEditor.__init__, which loads the first image.
        self.auto = False
        self.engine = "sam"
        self.settings = Settings()
        self.objects: list[list[tuple[float, float]]] = []  # this image: points per fish
        self.negatives: list[tuple[float, float]] = []  # this image: not-fish points
        self.fish_colors: list[Cluster] = []  # color engine, kept across images
        self.not_fish_colors: list[Cluster] = []
        self.segmenter: ColorSegmenter | None = None  # color engine, per image
        self.sam: SamSegmenter | None = None
        self.bgr: np.ndarray | None = None
        self.preview: list[list[tuple[float, float]]] = []  # normalized outlines
        self.show_enhanced = False
        self.enhanced_image: Image.Image | None = None
        self.pending = None
        super().__init__(root, folder)

        root.title(f"Auto labeler - {folder.name}")
        self.build_toolbar()
        tk.Label(
            root, text=AUTO_HELP, fg="#7fd8d8", bg="#2b2b2b", anchor="w", justify="left", padx=8
        ).pack(fill="x")
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
        look = {"bg": "#2b2b2b", "fg": "#e8e8e8", "highlightthickness": 0}
        check = {**look, "selectcolor": "#3c3c3c", "activebackground": "#2b2b2b"}
        button = {"bg": "#3c3c4c", "fg": "#e8e8e8", "relief": "flat", "padx": 8}
        rows = []
        for _ in range(2):
            row = tk.Frame(self.root, bg="#2b2b2b", padx=6, pady=2)
            row.pack(side="top", fill="x", before=self.canvas)
            rows.append(row)

        def checkbox(row, text, value, command):
            var = tk.BooleanVar(value=value)
            tk.Checkbutton(row, text=text, variable=var, command=command, **check).pack(
                side="left", padx=4
            )
            return var

        def slider(row, text, value, low, high, step):
            tk.Label(row, text=text, **look).pack(side="left", padx=(10, 2))
            var = tk.DoubleVar(value=value)
            tk.Scale(
                row,
                from_=low,
                to=high,
                resolution=step,
                orient="horizontal",
                length=130,
                variable=var,
                command=lambda _: self.on_setting(),
                troughcolor="#444",
                **look,
            ).pack(side="left")
            return var

        top, color = rows
        self.auto_var = checkbox(
            top, "Auto-label (G)", self.auto, lambda: self.set_auto(self.auto_var.get())
        )
        tk.Label(top, text="Engine (E):", **look).pack(side="left", padx=(10, 2))
        self.engine_var = tk.StringVar(value=self.engine)
        for value, text in (("sam", "SAM"), ("color", "Color")):
            tk.Radiobutton(
                top,
                text=text,
                value=value,
                variable=self.engine_var,
                command=lambda: self.set_engine(self.engine_var.get()),
                **check,
            ).pack(side="left")
        self.error_var = slider(
            top, "Max corner error (px)", self.settings.max_error_px, 0.5, 4.0, 0.25
        )

        tk.Label(color, text="Color engine:", **look).pack(side="left")
        self.tolerance_var = slider(color, "Tolerance", self.settings.tolerance, 1.0, 10.0, 0.25)
        self.all_var = checkbox(
            color, "Label all matching regions", self.settings.keep_all, self.on_setting
        )
        self.refine_var = checkbox(
            color, "Refine edges", self.settings.refine_edges, self.on_setting
        )
        tk.Button(color, text="Forget colors (C)", command=self.forget_colors, **button).pack(
            side="left", padx=4
        )
        tk.Button(
            color, text="Label all unlabeled images", command=self.label_all_unlabeled, **button
        ).pack(side="left", padx=4)
        self.swatches = tk.Canvas(color, height=18, width=300, bg="#2b2b2b", highlightthickness=0)
        self.swatches.pack(side="left", padx=8)

    def read_settings(self) -> None:
        if not hasattr(self, "tolerance_var"):
            return  # toolbar not built yet (first image)
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
        x = 2
        for clusters, outline in ((self.fish_colors, "#00ff00"), (self.not_fish_colors, "#ff3c3c")):
            for c in clusters:
                fill = "#{:02x}{:02x}{:02x}".format(*lab_to_rgb(c.mean))
                self.swatches.create_rectangle(
                    x, 2, x + 14, 16, fill=fill, outline=outline, width=2
                )
                x += 18

    # --- modes -----------------------------------------------------------------
    def set_auto(self, on: bool) -> None:
        if self.drawing is not None:
            self.say("Finish (Enter) or cancel (Esc) the polygon first")
            self.auto_var.set(self.auto)
            return
        self.auto = on
        self.auto_var.set(on)
        self.canvas.focus_set()
        if on:
            self.selected = None
            self.say(f"Auto-label on ({self.engine_name()}): click a fish")
            self.schedule()
        else:
            self.preview, self.show_enhanced = [], False
            self.update_view()
            self.say("Auto-label off: editing")

    def set_engine(self, engine: str) -> None:
        self.engine = engine
        self.engine_var.set(engine)
        self.canvas.focus_set()
        self.objects, self.negatives, self.preview = [], [], []  # clicks mean different things
        self.show_enhanced = False
        self.update_view()
        self.say(f"Engine: {self.engine_name()}")

    def engine_name(self) -> str:
        return "SAM" if self.engine == "sam" else "Color"

    # --- clicks -----------------------------------------------------------------
    def image_bgr(self) -> np.ndarray:
        if self.bgr is None:
            self.bgr = np.ascontiguousarray(np.asarray(self.base)[:, :, ::-1])
        return self.bgr

    def get_color_segmenter(self) -> ColorSegmenter:
        if self.segmenter is None:
            self.segmenter = ColorSegmenter(self.image_bgr())
        return self.segmenter

    def get_sam(self) -> SamSegmenter | None:
        try:
            if self.sam is None:
                self.status.configure(
                    text="Loading SAM (first use downloads it; the first click can take ~15 s)..."
                )
                self.root.update_idletasks()
                self.sam = SamSegmenter()
            self.sam.set_image(self.image_bgr(), str(self.images[self.index]))
        except Exception as e:  # noqa: BLE001 - e.g. no ultralytics or no internet: fall back
            self.sam = None
            self.set_engine("color")
            self.say(f"SAM unavailable ({type(e).__name__}: {str(e)[:80]}); using the color engine")
            return None
        return self.sam

    def add_color(self, x: float, y: float, fish: bool) -> None:
        pixels = sample_patch(self.get_color_segmenter().lab, x, y)
        if pixels is None:
            return
        colors = self.fish_colors if fish else self.not_fish_colors
        colors.append(Cluster(pixels))
        if len(colors) > MAX_CLUSTERS:
            del colors[0]
            self.message = f"Oldest color dropped (keeps {MAX_CLUSTERS})"
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
            seg = self.get_color_segmenter()
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
        for points in self.preview:
            self.polys.append([self.current_class, [[x * w, y * h] for x, y in points]])
        added = len(self.preview)
        self.objects, self.negatives, self.preview, self.dirty = [], [], [], True
        self.say(f"Added {added} label(s). Next image: Right or D")

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
        self.preview, self.show_enhanced, self.enhanced_image = [], False, None
        self.update_swatches()
        self.update_view()
        self.say("Colors forgotten")

    def toggle_enhanced(self) -> None:
        if not (self.auto and self.engine == "color" and self.fish_colors):
            self.say("Enhanced view: color engine, after sampling a fish color")
            return
        self.show_enhanced = not self.show_enhanced
        if self.show_enhanced:
            self.make_enhanced()
        else:
            self.update_view()

    def make_enhanced(self) -> None:
        gray = self.get_color_segmenter().enhanced(
            self.fish_colors, self.not_fish_colors, self.settings.tolerance
        )
        self.enhanced_image = Image.fromarray(gray).convert("RGB")
        self.update_view()

    def label_all_unlabeled(self) -> None:
        self.canvas.focus_set()
        if not self.fish_colors:
            self.say("Batch labeling uses the color engine: sample fish colors first (E, G, click)")
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
            self.status.configure(text=f"Auto-labeling {i}/{len(todo)}: {path.name}")
            self.root.update_idletasks()  # redraw only; no input events mid-batch
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            seg = ColorSegmenter(np.ascontiguousarray(rgb[:, :, ::-1]))
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
        self.segmenter, self.bgr, self.enhanced_image = None, None, None
        super().load(index)
        if self.auto and self.engine == "color" and self.fish_colors:
            self.schedule()  # sampled colors carry over

    def close(self) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        super().close()

    def render(self) -> Image.Image:
        img = super().render()
        if not self.auto:
            return img
        s, (x0, y0), (w, h) = self.scale, self.view_origin, self.base.size

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
            d.line(sp + sp[:1], fill=PREVIEW_COLOR, width=2)
            for x, y in sp:
                d.rectangle((x - 2, y - 2, x + 2, y + 2), fill=PREVIEW_COLOR)
        marks = [
            (x, y, "fish" if i == 0 else "same fish")
            for pts in self.objects
            for i, (x, y) in enumerate(pts)
        ]
        marks += [(x, y, "not fish") for x, y in self.negatives]
        for x, y, kind in marks:  # a cross per click
            sx, sy = (x - x0) * s, (y - y0) * s
            for width, color in ((4, (0, 0, 0)), (2, MARK_COLORS[kind])):
                d.line((sx - 6, sy, sx + 6, sy), fill=color, width=width)
                d.line((sx, sy - 6, sx, sy + 6), fill=color, width=width)
        return img


if __name__ == "__main__":
    main(AutoLabeler, "Label fish by clicking them, and edit the labels.")
