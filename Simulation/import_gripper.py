#!/usr/bin/env python3
"""
Import the gripper's CAD model (cups, plate, camera and bracket, from the flange down) for
the 3D simulation: turned into the FR5's tip link frame, cut down to 100k triangles and
written to Simulation/models/gripper.stl. Prints the TCP and cup layout measured on it, for
the workcell and simulation settings.

Usage:
    python Simulation/import_gripper.py "gripper assembly.stl"
    python Simulation/import_gripper.py model.stl --up +y --units meter
    python Simulation/import_gripper.py model.stl --turn 90    # bolted on a quarter turn round

The face farthest along --up (the CAD axis from the cups toward the arm) is the one bolted
to the flange; its middle goes on the flange's axis.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: fix_console, popups
try:
    from pipeline_common import fix_console
except ImportError:

    def fix_console() -> None:
        pass


OUT = HERE / "models" / "gripper.stl"


def ask_file() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    name = filedialog.askopenfilename(
        title="The gripper's CAD model", filetypes=[("Meshes", "*.stl *.ply *.obj"), ("All", "*")]
    )
    root.destroy()
    return Path(name) if name else None


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("model", nargs="?", type=Path, help="the CAD export (.stl, .ply, .obj)")
    parser.add_argument("--units", default="millimeter",
                        choices=("millimeter", "centimeter", "meter", "inch"),
                        help="the file's units (default millimeter)")  # fmt: skip
    parser.add_argument("--up", default="+z", choices=("+x", "-x", "+y", "-y", "+z", "-z"),
                        help="CAD axis from the cups toward the arm (default +z)")  # fmt: skip
    parser.add_argument("--turn", type=float, default=0.0,
                        help="degrees about the flange's axis (default 0)")  # fmt: skip
    parser.add_argument("--out", type=Path, default=OUT, help=f"default {OUT}")
    args = parser.parse_args()
    model = args.model or ask_file()
    if model is None or not model.is_file():
        sys.exit(f"❌ No model file: {model}")

    import numpy as np
    import pyvista as pv

    sys.path.insert(0, str(HERE))
    from rfp_sim.assets import FLANGE_Z, import_gripper

    print(f"Reading {model.name} ...")
    gripper = import_gripper(pv.read(model), args.units, args.up, args.turn)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gripper.mesh.save(args.out, binary=True)
    low, high = np.array(gripper.mesh.bounds).reshape(3, 2).T
    rows = sorted({round(float(x), 4) for x in gripper.cups[:, 0]})
    columns = sorted({round(float(y), 4) for y in gripper.cups[:, 1]})
    cup_mm = gripper.cup_radius * 2000
    print(f"✅ Written: {args.out} ({gripper.mesh.n_cells} triangles)")
    print(f"   {len(gripper.cups)} cups, {cup_mm:.0f} mm across, faces "
          f"{(gripper.tcp_z - FLANGE_Z) * 1000:.1f} mm from the flange")  # fmt: skip
    print("\nPut these in the settings (they must match the drawing):")
    print(f"  Arm/configs/workcell.yaml  tool.tcp: {{xyz: [0.0, 0.0, {gripper.tcp_z:.4f}]}}")
    print(f"                             tool.body: x {_range(low[0], high[0])}, "
          f"y {_range(low[1], high[1])}, z {_range(low[2], gripper.tcp_z - 0.02)}")  # fmt: skip
    print(f"  Simulation/sim.yaml        cup_rows: {rows}, cup_columns: {columns}")
    span = gripper.cups.max(axis=0) - gripper.cups.min(axis=0) + 2 * gripper.cup_radius
    print(f"  Vision/configs/vision.yaml cup_footprint_m: [{span[0]:.3f}, {span[1]:.3f}], "
          f"cup_diameter_m: {gripper.cup_radius * 2:.3f}")  # fmt: skip
    print("  and camera.mount (workcell.yaml) at the camera's lens, in the same frame:")
    print(f"  the flange's middle at z {FLANGE_Z}, z toward the cups, x along the cup rows.")
    print("Next: python ControlSys/rfp_controlsys/pick_place.py --mode sim3d")


def _range(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


if __name__ == "__main__":
    main()
