#!/usr/bin/env bash
# Sets up RoboticFishProcessing on Linux: tools/setup_rfp.py, run with the system's Python.
#   ./setup.sh                 a development PC (conda environment "rfp")
#   ./setup.sh --robot         the robot PC (Ubuntu 22.04): also ROS 2, MoveIt 2, librealsense...
#   ./setup.sh --check         only report what is missing
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null; then
    echo "Python 3 is needed first: sudo apt install python3"
    exit 1
fi
exec python3 tools/setup_rfp.py "$@"
