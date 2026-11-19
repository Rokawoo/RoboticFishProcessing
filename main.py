#!/usr/bin/env python3
"""
Root shortcut for the pick-and-place cycle.

Runs the same program as
    python ControlSys/rfp_controlsys/pick_place.py [options]
from the repository root. It makes the repository's rfp_* packages importable
(no pip install required) and hands over to pick_place.main(), which takes the
same command-line options:

    python main.py --mode check
    python main.py --mode sim3d
    python main.py --mode dry-run --step
    python main.py --mode run
"""

import sys
from pathlib import Path

# Put the repository's package roots on sys.path so this works whether or not
# the project was pip-installed in the current Python environment.
REPO = Path(__file__).resolve().parent
for sub in ("Vision/data", "Vision", "Arm", "ControlSys", "Simulation"):
    directory = REPO / sub
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def main() -> None:
    from rfp_controlsys.pick_place import main as pick_place_main

    pick_place_main()


if __name__ == "__main__":
    main()
