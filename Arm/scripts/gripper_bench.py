#!/usr/bin/env python3
"""
Gripper bench: switch the vacuum by hand, to check the gripper before a run (AS-30) or to
let go of a fish after a stopped run. The arm doesn't move.

Usage:
    python gripper_bench.py                 # pick the action in a popup
    python gripper_bench.py status          # the gripper's state
    python gripper_bench.py on              # vacuum on, then the state for 3 s
    python gripper_bench.py off             # vacuum off
    python gripper_bench.py pulse 500       # valve board only: vacuum on for 500 ms
    python gripper_bench.py watch           # print the state until Ctrl+C

workcell.gripper in control_panel.yaml (or --workcell) says what switches the vacuum:
    driver: serial      the ESP32 valve board over USB; needs pyserial, no ROS
    driver: http        the same board over Wi-Fi; no ROS
    driver: controller  the FR5 control box I/O with a vacuum switch (Table AS-4); needs
                        ROS 2 and: ros2 run fairino_hardware ros2_cmd_server
status and watch only read the vacuum; on, off and pulse turn it off first.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: popups
from pipeline_common import ask_choice, fix_console
from rfp_arm.gripper import make_gripper
from rfp_arm.interfaces import ArmError
from rfp_arm.workcell import WorkcellError, load_workcell

ACTIONS = {
    "status": "the gripper's state",
    "on": "vacuum on, then the state for 3 s",
    "off": "vacuum off",
    "pulse": "valve board only: vacuum on for some milliseconds",
    "watch": "print the state until Ctrl+C",
}


def watch(gripper, seconds: float | None) -> None:
    """Print the gripper state whenever it changes."""
    last, end = None, None if seconds is None else time.monotonic() + seconds
    while end is None or time.monotonic() < end:
        state = gripper.state()
        if state != last:
            print(f"   {time.strftime('%H:%M:%S')}  {state.value}")
            last = state
        time.sleep(0.02)


def controller_io():
    """The control box I/O through ROS (driver: controller). Returns (link, io)."""
    try:
        from rfp_arm.fairino import FairinoIO
        from rfp_arm.ros_link import RosLink

        link = RosLink("rfp_gripper_bench")
        io = FairinoIO(link)
    except ImportError as e:
        sys.exit(f"❌ The controller gripper needs ROS 2 and fairino_msgs ({e}).")
    io.connect()
    return link, io


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description="Switch and watch the suction gripper by hand.")
    parser.add_argument("action", nargs="?", choices=list(ACTIONS), help="popup if not given")
    parser.add_argument("ms", nargs="?", type=int, default=500, help="pulse length (500)")
    parser.add_argument("--workcell", type=Path, help="workcell file (Arm/configs/workcell.yaml)")
    args = parser.parse_args()
    action = args.action
    if action is None:
        names = list(ACTIONS)
        choice = ask_choice("Gripper bench", "Action:", [f"{n:7s} {ACTIONS[n]}" for n in names])
        if choice is None:
            sys.exit("Cancelled.")
        action = names[choice]
    try:
        workcell = load_workcell(args.workcell)
    except WorkcellError as e:
        sys.exit(f"❌ {e}")
    settings = workcell.gripper
    link = io = None
    try:
        if settings.driver == "controller":
            link, io = controller_io()
            print(f"Gripper: control box, vacuum valve DO{settings.vacuum_do}")
        else:
            where = settings.serial_port if settings.driver == "serial" else settings.http_host
            print(f"Gripper: valve board ({settings.driver}) at {where}")
        gripper = make_gripper(
            settings,
            io,
            workcell.grasp.release_timeout,
            start_off=action not in ("status", "watch"),
        )
        if action == "on":
            gripper.vacuum_on()
            print("✅ Vacuum on. Press the cups on a fish: they should hold it.")
            watch(gripper, 3.0)
            print("   Next: python gripper_bench.py off")
        elif action == "off":
            gripper.release()
            watch(gripper, 1.5)
            print("✅ Vacuum off.")
        elif action == "pulse":
            if not hasattr(gripper, "pulse"):
                sys.exit("❌ pulse needs the valve board (gripper.driver: serial or http)")
            gripper.pulse(args.ms)
            print(f"✅ Vacuum on for {args.ms} ms.")
        elif action == "watch":
            print("Watching the gripper (Ctrl+C to stop):")
            watch(gripper, None)
        else:
            print(f"   state: {gripper.state().value}")
            for line in io.report() if io else []:
                print(f"   {line}")
    except ArmError as e:
        sys.exit(f"❌ {e}")
    except KeyboardInterrupt:
        print("\nStopped. The vacuum stays as it is.")
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
