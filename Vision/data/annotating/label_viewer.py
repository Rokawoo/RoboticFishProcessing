#!/usr/bin/env python3
"""
Step 4: view, fix and delete labels.

Usage:
    python label_viewer.py                 # pick a set folder in a popup
    python label_viewer.py "<set folder>"

Changes are saved when you change image or close the window (Ctrl+S saves now).

Mouse
    click a fish                 select its polygon
    drag a corner                move it
    drag the dot mid-edge        add a corner there
    right-click a corner         remove it
    wheel / middle-drag          zoom at the mouse / pan (R resets the view)
Keys
    Right / Left (or D / A)      next / previous image
    U                            next image without a label
    N                            new polygon: click corners, then Enter or click the
                                 first corner (Backspace: remove last corner, Esc: cancel)
    Delete                       delete the selected polygon
    0-9                          class of the selected polygon and of new ones
    Ctrl+Z / Ctrl+S              undo / save
    Ctrl+Delete                  delete the image and its label (moved to <set>/_deleted/)
    H / F                        hide labels / toggle the fill

An image whose polygons are all deleted keeps an empty label file: YOLO reads
that as "no fish" (background), unlike an image without a label file.
"""

import argparse
import copy
import math
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Vision/data
from pipeline_common import (
    IMAGE_SETS_DIR,
    LABEL_EXT,
    TRASH_DIR_NAME,
    find_label,
    fix_console,
    label_index,
    list_images,
    load_class_names,
    read_label,
    set_dirs,
    write_label,
)

COLORS = [  # per class id
    (0, 220, 120),
    (255, 170, 0),
    (0, 170, 255),
    (255, 80, 200),
    (170, 120, 255),
    (255, 230, 0),
    (0, 230, 230),
    (255, 100, 100),
    (140, 255, 80),
    (200, 200, 200),
]
FILL_ALPHA = 60
HIT_RADIUS = 9  # screen px
UNDO_LIMIT = 100
MAX_ZOOM = 40.0  # x fit-to-window

HELP = (
    "Click: select   Drag corner: move   Drag mid-edge dot: add corner   "
    "Right-click corner: remove   Del: delete polygon   0-9: class\n"
    "N: new polygon (click corners; Enter or first corner: finish; Backspace; Esc)   "
    "Ctrl+Z: undo   Ctrl+S: save   Ctrl+Del: delete image\n"
    "Left/Right or A/D: image   U: next unlabeled   H: hide labels   F: fill   "
    "Wheel: zoom   Middle-drag: pan   R: reset view"
)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def point_in_polygon(x: float, y: float, pts: list[list[float]]) -> bool:
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        (xi, yi), (xj, yj) = pts[i], pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class LabelViewer:
    def __init__(self, root: tk.Misc, folder: Path):
        self.root = root
        self.folder = folder
        self.image_dir, self.label_dir = set_dirs(folder)
        self.images = list_images(self.image_dir)
        self.class_names = load_class_names(folder)
        self.index = 0
        self.current_class = 0
        self.show_labels = True
        self.fill = True
        self.message = ""
        self.font = _font(14)

        # Current image. Polygons are in image pixels.
        self.polys: list[list] = []  # [[class id, [[x, y], ...]], ...]
        self.other: list[str] = []  # non-polygon label lines, written back unchanged
        self.label_path: Path | None = None
        self.dirty = False
        self.undo: list[list] = []
        self.selected: int | None = None
        self.drag: tuple[int, int] | None = None  # (polygon, corner)
        self.moved = False
        self.drawing: list[list[float]] | None = None
        self.mouse = (0.0, 0.0)  # image px

        # Image pixel (x, y) is drawn at canvas (ox + x * scale, oy + y * scale),
        # scale = fit_scale * zoom. Only the visible part is resized (`view`,
        # top-left at image pixel `view_origin`), so zooming stays fast.
        self.base: Image.Image | None = None
        self.view: Image.Image | None = None
        self.view_origin = (0, 0)
        self.fit_scale, self.zoom = 1.0, 1.0
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.canvas_size = (0, 0)
        self.pan_start = None
        self.photo = None

        root.title(f"Label viewer - {folder.name}")
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{int(w * 0.85)}x{int(h * 0.85)}+{int(w * 0.05)}+{int(h * 0.03)}")
        root.configure(bg="#1e1e1e")
        self.canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        bar = {"anchor": "w", "justify": "left", "bg": "#2b2b2b", "padx": 8}
        self.status = tk.Label(root, fg="#e8e8e8", pady=2, **bar)
        self.status.pack(fill="x")
        tk.Label(root, text=HELP, fg="#9a9a9a", pady=4, **bar).pack(fill="x")
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<MouseWheel>", self.on_wheel)  # Windows, macOS
        self.canvas.bind("<Button-4>", self.on_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan)
        keys = {
            "<Right>": lambda e: self.go(self.index + 1),
            "<Left>": lambda e: self.go(self.index - 1),
            "<Key-d>": lambda e: self.go(self.index + 1),
            "<Key-a>": lambda e: self.go(self.index - 1),
            "<Key-u>": lambda e: self.next_unlabeled(),
            "<Key-n>": lambda e: self.start_drawing(),
            "<Return>": lambda e: self.finish_drawing(),
            "<Escape>": lambda e: self.escape(),
            "<BackSpace>": lambda e: self.remove_last_corner(),
            "<Delete>": lambda e: self.delete_polygon(),
            "<Control-Delete>": lambda e: self.delete_image(),
            "<Control-z>": lambda e: self.undo_last(),
            "<Control-s>": lambda e: self.save(announce=True),
            "<Key-h>": lambda e: self.toggle("show_labels"),
            "<Key-f>": lambda e: self.toggle("fill"),
            "<Key-r>": lambda e: self.fit(),
        }
        for key, handler in keys.items():
            root.bind(key, handler)
        for digit in "0123456789":
            root.bind(f"<Key-{digit}>", lambda e, d=int(digit): self.set_class(d))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.load(0)

    # --- coordinates ------------------------------------------------------
    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    def clamp(self, x: float, y: float) -> list[float]:
        w, h = self.base.size
        return [min(max(x, 0.0), w), min(max(y, 0.0), h)]

    def tolerance(self) -> float:
        return HIT_RADIUS / self.scale  # image px

    # --- loading, saving, navigation --------------------------------------
    def load(self, index: int) -> None:
        self.index = index % len(self.images)
        path = self.images[self.index]
        # `with` releases the file (so it can be moved); exif_transpose matches OpenCV/YOLO.
        with Image.open(path) as im:
            self.base = ImageOps.exif_transpose(im).convert("RGB")
        self.label_path = find_label(path, label_index(self.label_dir))
        polygons, self.other = read_label(self.label_path)
        w, h = self.base.size
        self.polys = [[cls, [[x * w, y * h] for x, y in pts]] for cls, pts in polygons]
        self.dirty, self.undo, self.selected = False, [], None
        self.drag, self.drawing = None, None
        if self.other:
            self.message = f"{len(self.other)} non-polygon label line(s) kept unchanged"
        self.fit()

    def save(self, announce: bool = False) -> None:
        if not self.dirty:
            if announce:
                self.say("Nothing to save")
            return
        path = self.images[self.index]
        target = self.label_path or self.label_dir / f"{path.stem}{LABEL_EXT}"
        w, h = self.base.size
        polygons = [(cls, [(x / w, y / h) for x, y in pts]) for cls, pts in self.polys]
        write_label(target, polygons, self.other)
        self.label_path, self.dirty = target, False
        if announce:
            self.say(f"Saved {target.name}")

    def go(self, index: int) -> None:
        if self.drawing is not None:
            self.say("Finish (Enter) or cancel (Esc) the polygon first")
            return
        self.save()
        self.load(index)

    def next_unlabeled(self) -> None:
        labels = label_index(self.label_dir)
        n = len(self.images)
        for step in range(1, n + 1):
            i = (self.index + step) % n
            if find_label(self.images[i], labels) is None:
                self.go(i)
                return
        self.say("Every image has a label")

    def delete_image(self) -> None:
        path = self.images[self.index]
        question = (
            f"Delete {path.name} and its label?\n\nThey are moved to {TRASH_DIR_NAME}/ in this set."
        )
        if not messagebox.askyesno("Delete image", question, parent=self.root):
            return
        trash = self.folder / TRASH_DIR_NAME
        trash.mkdir(exist_ok=True)
        for f in (path, self.label_path):
            if f is not None and f.exists():
                dest = trash / f.name
                n = 1
                while dest.exists():  # keep earlier deletions of the same name
                    dest = trash / f"{f.stem} ({n}){f.suffix}"
                    n += 1
                shutil.move(str(f), str(dest))
        del self.images[self.index]
        self.dirty = False
        if not self.images:
            messagebox.showinfo("Label viewer", "No images left in this folder.", parent=self.root)
            self.root.destroy()
            return
        self.load(min(self.index, len(self.images) - 1))
        self.say(f"Deleted {path.name} (moved to {TRASH_DIR_NAME}/)")

    def close(self) -> None:
        self.save()
        self.root.destroy()

    # --- editing ----------------------------------------------------------
    def push_undo(self) -> None:
        self.undo.append(copy.deepcopy(self.polys))
        del self.undo[:-UNDO_LIMIT]

    def undo_last(self) -> None:
        if not self.undo:
            self.say("Nothing to undo")
            return
        self.polys = self.undo.pop()
        self.selected, self.dirty = None, True
        self.redraw()

    def start_drawing(self) -> None:
        self.drawing, self.selected = [], None
        self.say(f"New '{self.class_name(self.current_class)}' polygon: click its corners")

    def finish_drawing(self) -> None:
        if self.drawing is None:
            return
        if len(self.drawing) < 3:
            self.say("A polygon needs at least 3 corners")
            return
        self.push_undo()
        self.polys.append([self.current_class, self.drawing])
        self.selected, self.drawing, self.dirty = len(self.polys) - 1, None, True
        self.redraw()

    def remove_last_corner(self) -> None:
        if self.drawing:
            self.drawing.pop()
            self.redraw()

    def escape(self) -> None:
        if self.drawing is not None:
            self.drawing = None
            self.say("Polygon cancelled")
        else:
            self.selected = None
            self.redraw()

    def delete_polygon(self) -> None:
        if self.selected is None:
            self.say("Click a polygon first")
            return
        self.push_undo()
        del self.polys[self.selected]
        self.selected, self.dirty = None, True
        self.redraw()

    def set_class(self, cls: int) -> None:
        if self.class_names and cls >= len(self.class_names):
            self.say(f"No class {cls}: classes.txt lists 0-{len(self.class_names) - 1}")
            return
        self.current_class = cls
        if self.selected is not None and self.polys[self.selected][0] != cls:
            self.push_undo()
            self.polys[self.selected][0] = cls
            self.dirty = True
        self.say(f"Class: {cls} {self.class_name(cls)}")

    def toggle(self, attr: str) -> None:
        setattr(self, attr, not getattr(self, attr))
        self.redraw()

    # --- hit tests (image px) -----------------------------------------------
    def hit_corner(self, x: float, y: float) -> tuple[int, int] | None:
        """Nearest corner in reach, selected polygon first."""
        order = list(range(len(self.polys)))
        if self.selected is not None:
            order.remove(self.selected)
            order.insert(0, self.selected)
        tol2 = self.tolerance() ** 2
        for p in order:
            best = None
            for v, (px, py) in enumerate(self.polys[p][1]):
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 <= tol2 and (best is None or d2 < best[0]):
                    best = (d2, v)
            if best:
                return p, best[1]
        return None

    def hit_midpoint(self, x: float, y: float) -> int | None:
        """Edge of the selected polygon whose midpoint dot is under (x, y)."""
        if self.selected is None:
            return None
        pts = self.polys[self.selected][1]
        tol2 = self.tolerance() ** 2
        for i in range(len(pts)):
            (ax, ay), (bx, by) = pts[i], pts[(i + 1) % len(pts)]
            if ((ax + bx) / 2 - x) ** 2 + ((ay + by) / 2 - y) ** 2 <= tol2:
                return i
        return None

    def hit_polygon(self, x: float, y: float) -> int | None:
        for p in reversed(range(len(self.polys))):  # last drawn is on top
            if point_in_polygon(x, y, self.polys[p][1]):
                return p
        return None

    # --- mouse --------------------------------------------------------------
    def on_press(self, event) -> None:
        x, y = self.to_image(event.x, event.y)
        if self.drawing is not None:
            first = self.drawing[0] if self.drawing else None
            if (
                first
                and len(self.drawing) >= 3
                and (first[0] - x) ** 2 + (first[1] - y) ** 2 <= self.tolerance() ** 2
            ):
                self.finish_drawing()
            else:
                self.drawing.append(self.clamp(x, y))
                self.redraw()
            return
        if not self.show_labels:
            return
        corner = self.hit_corner(x, y)
        if corner:
            self.push_undo()  # dropped on release if the corner didn't move
            self.selected, self.drag, self.moved = corner[0], corner, False
            self.redraw()
            return
        edge = self.hit_midpoint(x, y)
        if edge is not None:
            self.push_undo()
            self.polys[self.selected][1].insert(edge + 1, self.clamp(x, y))
            self.drag, self.moved, self.dirty = (self.selected, edge + 1), True, True
            self.redraw()
            return
        self.selected = self.hit_polygon(x, y)
        self.redraw()

    def on_drag(self, event) -> None:
        if self.drag is None:
            return
        p, v = self.drag
        self.polys[p][1][v] = self.clamp(*self.to_image(event.x, event.y))
        self.moved = self.dirty = True
        self.redraw()

    def on_release(self, event) -> None:
        if self.drag is not None and not self.moved:
            self.undo.pop()
        self.drag = None

    def on_right_click(self, event) -> None:
        if self.drawing is not None or not self.show_labels:
            return
        corner = self.hit_corner(*self.to_image(event.x, event.y))
        if corner is None:
            return
        p, v = corner
        if len(self.polys[p][1]) <= 3:
            self.say("A polygon needs 3 corners. Delete removes the whole polygon.")
            return
        self.push_undo()
        del self.polys[p][1][v]
        self.selected, self.dirty = p, True
        self.redraw()

    def on_motion(self, event) -> None:
        self.mouse = self.to_image(event.x, event.y)
        if self.drawing is not None:
            self.redraw()  # line to the mouse
        elif self.show_labels:
            over = self.hit_corner(*self.mouse) or self.hit_midpoint(*self.mouse) is not None
            self.canvas.configure(cursor="fleur" if over else "crosshair")

    # --- view ------------------------------------------------------------------
    def on_resize(self, event) -> None:
        if (event.width, event.height) != self.canvas_size:  # ignore events that don't resize
            self.canvas_size = (event.width, event.height)
            self.fit()

    def fit(self) -> None:
        """Fit the image to the window (zoom 1)."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.base is None or cw < 2 or ch < 2:
            return
        w, h = self.base.size
        self.fit_scale = self.scale = min(cw / w, ch / h)
        self.zoom = 1.0
        self.ox, self.oy = (cw - w * self.scale) / 2, (ch - h * self.scale) / 2
        self.update_view()

    def clamp_offsets(self) -> None:
        """Keep a zoomed image covering the window; center it if it is smaller."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        w, h = self.base.size
        size_w, size_h = w * self.scale, h * self.scale
        self.ox = (cw - size_w) / 2 if size_w <= cw else min(0.0, max(cw - size_w, self.ox))
        self.oy = (ch - size_h) / 2 if size_h <= ch else min(0.0, max(ch - size_h, self.oy))

    def update_view(self) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        w, h = self.base.size
        s = self.scale
        x0, y0 = max(0, int(-self.ox / s)), max(0, int(-self.oy / s))
        x1 = min(w, math.ceil((cw - self.ox) / s) + 1)
        y1 = min(h, math.ceil((ch - self.oy) / s) + 1)
        size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
        resample = (
            Image.Resampling.NEAREST if s >= 3 else Image.Resampling.BILINEAR
        )  # sharp pixels when zoomed in
        self.view = self.display_image().crop((x0, y0, x1, y1)).resize(size, resample)
        self.view_origin = (x0, y0)
        self.redraw()

    def display_image(self) -> Image.Image:
        """Image shown under the labels (the auto labeler overrides this)."""
        return self.base

    def on_wheel(self, event) -> None:
        zoom_in = event.num == 4 or getattr(event, "delta", 0) > 0
        new_zoom = min(max(self.zoom * (1.25 if zoom_in else 0.8), 1.0), MAX_ZOOM)
        if new_zoom == self.zoom or self.base is None:
            return
        ix, iy = self.to_image(event.x, event.y)  # keep this point under the mouse
        self.zoom = new_zoom
        self.scale = self.fit_scale * new_zoom
        self.ox, self.oy = event.x - ix * self.scale, event.y - iy * self.scale
        self.clamp_offsets()
        self.update_view()

    def on_pan_start(self, event) -> None:
        self.pan_start = (event.x, event.y, self.ox, self.oy)

    def on_pan(self, event) -> None:
        if self.pan_start is None or self.base is None:
            return
        x, y, ox, oy = self.pan_start
        self.ox, self.oy = ox + event.x - x, oy + event.y - y
        self.clamp_offsets()
        self.update_view()

    # --- drawing ----------------------------------------------------------------
    def class_name(self, cls: int) -> str:
        return self.class_names[cls] if cls < len(self.class_names) else f"class {cls}"

    def say(self, message: str) -> None:
        self.message = message
        self.redraw()

    def render(self) -> Image.Image:
        """Visible part of the image with the labels, at screen size."""
        img = self.view.convert("RGBA")
        s = self.scale
        x0, y0 = self.view_origin

        def screen(pts):
            return [((x - x0) * s, (y - y0) * s) for x, y in pts]

        if self.show_labels and self.polys:
            if self.fill:
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(overlay)
                for i, (cls, pts) in enumerate(self.polys):
                    alpha = FILL_ALPHA * 2 if i == self.selected else FILL_ALPHA
                    d.polygon(screen(pts), fill=COLORS[cls % len(COLORS)] + (alpha,))
                img = Image.alpha_composite(img, overlay)
            d = ImageDraw.Draw(img)
            for i, (cls, pts) in enumerate(self.polys):
                color, sp = COLORS[cls % len(COLORS)], screen(pts)
                selected = i == self.selected
                d.line(sp + sp[:1], fill=color, width=3 if selected else 2)
                r = 4 if selected else 2
                for x, y in sp:
                    d.rectangle(
                        (x - r, y - r, x + r, y + r),
                        fill=(255, 255, 255) if selected else color,
                        outline=color,
                    )
                if selected:  # midpoint dots (drag to add a corner)
                    for j in range(len(sp)):
                        (ax, ay), (bx, by) = sp[j], sp[(j + 1) % len(sp)]
                        mx, my = (ax + bx) / 2, (ay + by) / 2
                        d.ellipse((mx - 3, my - 3, mx + 3, my + 3), outline=(255, 255, 255))
                tx, ty = min(sp, key=lambda p: p[1])
                d.text((tx + 4, ty - 18), self.class_name(cls), fill=color, font=self.font)

        if self.drawing is not None:
            d = ImageDraw.Draw(img)
            color = COLORS[self.current_class % len(COLORS)]
            sp = screen(self.drawing + [list(self.mouse)])
            if len(sp) >= 2:
                d.line(sp, fill=color, width=2)
            for x, y in sp[:-1]:
                d.rectangle((x - 3, y - 3, x + 3, y + 3), fill=color)
            if len(self.drawing) >= 3:  # ring on the first corner: click it to finish
                x, y = sp[0]
                d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 255, 255), width=2)
        return img

    def redraw(self) -> None:
        if self.view is None:
            return
        self.photo = ImageTk.PhotoImage(self.render())
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        x0, y0 = self.view_origin
        self.canvas.coords(self.image_item, self.ox + x0 * self.scale, self.oy + y0 * self.scale)

        path = self.images[self.index]
        if self.label_path is None and not self.polys:
            label = "no label yet"
        else:
            label = f"{len(self.polys)} polygon(s)"
        state = "unsaved changes" if self.dirty else "saved"
        cls = self.current_class
        text = (
            f"{self.index + 1}/{len(self.images)}   {path.name}   |   {label}   |   "
            f"new polygons: {cls} {self.class_name(cls)}   |   {state}"
        )
        if self.message:
            text += f"   |   {self.message}"
            self.message = ""
        self.status.configure(text=text)


def main(
    viewer_class: type[LabelViewer] = LabelViewer,
    description: str = "View, fix and delete the labels of an image set.",
) -> None:
    fix_console()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "folder", nargs="?", type=Path, help="image set folder (popup if not given)"
    )
    folder = parser.parse_args().folder
    root = tk.Tk()
    if folder is None:
        from tkinter import filedialog

        root.withdraw()
        start = IMAGE_SETS_DIR if IMAGE_SETS_DIR.is_dir() else Path.cwd()
        name = filedialog.askdirectory(parent=root, title="Pick an image set", initialdir=start)
        if not name:
            sys.exit("No folder picked.")
        folder = Path(name)
        root.deiconify()
    if not folder.is_dir():
        sys.exit(f"❌ Not a folder: {folder}")
    if not list_images(set_dirs(folder)[0]):
        sys.exit(f"❌ No images in {folder}")
    viewer_class(root, folder.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
