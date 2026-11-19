#!/usr/bin/env python3
"""
Benchmark: the real pick-and-place cycle in the 3D virtual cell over many seeded scenes,
scored against the simulator's truth. Run it before and after a change to vision, planning
or the cycle: the change is better if these numbers are. Nothing real moves; no ROS.

Usage:
    python Simulation/benchmark.py                 # 20 scenes, fish anywhere (sim.yaml's size)
    python Simulation/benchmark.py --scenes 50 --label my-change
    python Simulation/benchmark.py --compare Simulation/benchmarks/<run>   # with the change
    python Simulation/benchmark.py --fish-length 0.55 --placement view     # the old easy case
    python Simulation/benchmark.py --detector truth   # without the trained model
    python Simulation/benchmark.py --any-size         # without vision's size checks

Scenes are the same for the same --seed, --fish, --fish-length and --placement, so two runs
differ only by the code and settings. Settings come from the control panel like a real run
(--panel), except the fish, which come from the options above. Each run writes fish.csv
(one row per fish) and summary.json to Simulation/benchmarks/<date>_<time>[_label]/.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: fix_console
try:
    from pipeline_common import fix_console
except ImportError:

    def fix_console() -> None:
        pass


OUT_DIR = HERE / "benchmarks"


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--scenes", type=int, default=20, help="layouts to run (default 20)")
    parser.add_argument("--fish", type=int, default=1, help="fish per scene (default 1)")
    parser.add_argument(
        "--fish-length", type=float, help="m (default: the simulation settings'; big prints: 0.79)"
    )
    parser.add_argument(
        "--placement",
        choices=("anywhere", "view"),
        default="anywhere",
        help="anywhere: as fish really lie (default); view: the easy case",
    )
    parser.add_argument(
        "--detector",
        choices=("model", "truth"),
        default="model",
        help="model: the newest trained model (default); truth: the renderer's",
    )
    parser.add_argument("--model", type=Path, help="run folder or .pt (default: the newest run)")
    parser.add_argument("--seed", type=int, default=1, help="first scene's seed (default 1)")
    parser.add_argument(
        "--no-hold-check",
        action="store_true",
        help="confirm holds by a vacuum switch, not the camera (faster)",
    )
    parser.add_argument(
        "--any-size",
        action="store_true",
        help="skip vision's size checks (to measure other changes alone)",
    )
    parser.add_argument("--label", default="", help="added to the output folder's name")
    parser.add_argument("--compare", type=Path, help="an earlier run (folder or summary.json)")
    parser.add_argument("--panel", type=Path, help="control panel (default: control_panel.yaml)")
    args = parser.parse_args()
    if args.scenes < 1 or args.fish < 1:
        parser.error("--scenes and --fish must be 1 or more")

    from rfp_arm.workcell import WorkcellError, load_workcell
    from rfp_controlsys.settings import PANEL_FILE
    from rfp_controlsys.settings import load_settings as load_control
    from rfp_sim.benchmark import build_bench, report_lines, run_benchmark, summarize
    from rfp_sim.settings import load_settings as load_sim
    from rfp_vision.settings import load_settings as load_vision

    panel = args.panel or PANEL_FILE
    baseline = None
    if args.compare:
        path = args.compare / "summary.json" if args.compare.is_dir() else args.compare
        if not path.is_file():
            sys.exit(f"❌ No summary to compare with: {path}")
        baseline = json.loads(path.read_text(encoding="utf-8"))["summary"]
    try:
        cell = load_workcell(panel=panel)
        vision = load_vision(panel=panel)
        control = load_control(panel=panel)
        sim = load_sim(panel=panel)
    except (WorkcellError, FileNotFoundError, ValueError, TypeError) as e:
        sys.exit(f"❌ {e}")
    if args.any_size:
        vision.check_size = False
    args.fish_length = args.fish_length or sim.fish_length
    sim = dataclasses.replace(
        sim, fish_count=args.fish, fish_length=args.fish_length, placement=args.placement,
        speed=0.0, drop="", seed=args.seed,
    )  # fmt: skip

    detector = None
    if args.detector == "model":
        from rfp_vision.detector import FishDetector, find_weights, list_runs

        runs = list_runs()
        if args.model is None and not runs:
            sys.exit("❌ No trained model: train one, pass --model, or use --detector truth")
        weights = find_weights(args.model or runs[0])
        detector = FishDetector(weights, vision.predict_conf, vision.min_area_share)

    print("=" * 60)
    print(f"BENCHMARK: {args.scenes} scene(s), {args.fish} fish of {args.fish_length} m each, "
          f"placed {args.placement}")  # fmt: skip
    print(f"Detector:  {detector.version if detector else 'the renderer (truth)'}")
    print("=" * 60)
    started = time.monotonic()
    try:
        bench = build_bench(cell, vision, control, sim, detector, not args.no_hold_check)
    except ValueError as e:
        sys.exit(f"❌ {e}")
    rows = run_benchmark(bench, args.scenes, args.seed)
    summary = summarize(rows)

    name = time.strftime("%Y-%m-%d_%H%M%S") + (f"_{args.label}" if args.label else "")
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=False)
    import csv

    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "fish.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "summary": summary,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "detector": detector.version if detector else "truth",
    }
    (out / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print()
    for line in report_lines(summary, baseline):
        print(line)
    print(f"\n✅ {len(rows)} fish in {time.monotonic() - started:.0f} s. Written: {out}")
    print(f"Next: change something, then: python Simulation/benchmark.py --compare {out}")


if __name__ == "__main__":
    main()
