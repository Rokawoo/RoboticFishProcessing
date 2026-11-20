#!/usr/bin/env python3
"""
Taught demos: play back hand-taught arm poses with the vacuum switched on and off between
them, or record a new one. No camera: the arm goes where it was taught. Files: Demos/.

Usage:
    python demo.py                              # pick a demo and a mode in popups
    python demo.py play fish_pickup --mode offline          # simulated arm, no ROS
    python demo.py play fish_pickup --mode sim              # MoveIt's mock hardware (RViz)
    python demo.py play fish_pickup --mode real             # THE ARM MOVES, Enter per step
    python demo.py play fish_pickup --mode real --auto      # THE ARM MOVES, straight through
    python demo.py record my_demo               # teach a new one (the arm is only read)

Taught poses are trusted: the zone checks are off for a demo; the speed, wrist and joint
limits still apply. Ctrl+C stops the arm.

Recording: bring the arm to a pose by hand or with the pendant, then type
    j <name>   go to this pose with a joint move        on / off    switch the vacuum
    l <name>   go to this pose in a straight line       w <s>       wait
    done       save                                     undo        drop the last step

sim, real and record need MoveIt running (see pick_place.py).
"""

import argparse
import dataclasses
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: popups
try:
    from pipeline_common import ask_choice, fix_console
except ImportError:  # installed without the repository around it
    ask_choice = None

    def fix_console() -> None:
        pass


from rfp_arm.gripper import SimGripper, make_gripper
from rfp_arm.interfaces import ArmError, MotionAborted
from rfp_arm.kinematics import FR5
from rfp_arm.sim import SimArm
from rfp_arm.workcell import Box, Workcell, WorkcellError, load_workcell

from rfp_controlsys.console import ask

DEMOS_DIR = REPO / "Demos"
MODES = {
    "offline": "simulated arm and gripper, no ROS",
    "sim": "MoveIt's mock hardware, shown in RViz",
    "real": "THE REAL ARM MOVES, with the real gripper",
}
STEP_KINDS = ("joints", "line", "vacuum", "wait")


class DemoError(ValueError):
    """A demo file that can't be played."""


@dataclasses.dataclass
class Demo:
    name: str
    description: str
    line_speed: float  # m/s
    poses: dict[str, list[float]]  # joint angles in rad
    steps: list[tuple[str, object]]  # (kind, value)


def demo_path(name: str) -> Path:
    """Demos/<name>.yaml for a bare name; a path as given."""
    path = Path(name)
    return path if path.suffix else DEMOS_DIR / f"{name}.yaml"


def load_demo(path: Path) -> Demo:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        poses = {
            str(k): [math.radians(float(v)) for v in joints] for k, joints in data["poses"].items()
        }
        steps = [next(iter(step.items())) for step in data["steps"]]
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise DemoError(f"{path.name}: not a demo file ({e})") from None
    for kind, value in steps:
        if kind not in STEP_KINDS:
            raise DemoError(f"{path.name}: unknown step '{kind}'")
        if kind in ("joints", "line") and value not in poses:
            raise DemoError(f"{path.name}: step '{kind}: {value}' names no pose")
        if kind in ("joints", "line") and len(poses[value]) != len(FR5.joint_names):
            raise DemoError(f"{path.name}: pose '{value}' needs {len(FR5.joint_names)} joints")
    if not steps:
        raise DemoError(f"{path.name}: no steps")
    speed = float(data.get("line_speed", 0.10))
    return Demo(path.stem, str(data.get("description", "")), speed, poses, steps)


def save_demo(path: Path, demo: Demo) -> None:
    data = {
        "description": demo.description,
        "line_speed": demo.line_speed,
        "poses": {k: [round(math.degrees(v), 3) for v in q] for k, q in demo.poses.items()},
        "steps": [{kind: value} for kind, value in demo.steps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=None), encoding="utf-8"
    )


def demo_workcell(workcell: Workcell) -> Workcell:
    """The workcell with its zones opened: a demo's poses were taught on the arm, so only
    the speed, wrist and joint-travel limits are kept."""
    everywhere = Box("anywhere", (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0))
    return dataclasses.replace(workcell, work_zone=everywhere, exclusion_zones=[], table_z=-10.0)


def play(
    demo: Demo,
    arm,
    gripper,
    workcell: Workcell,
    confirm: Callable[[str], bool] = lambda text: True,
    say: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    tick=None,
) -> bool:
    """Run the demo's steps. `confirm(step text)` False stops it. True if it ran to the end."""
    speed = min(demo.line_speed, workcell.motion.tcp_speed_max)
    for number, (kind, value) in enumerate(demo.steps, 1):
        on = value is True or str(value).lower() == "on"  # YAML reads a bare on as True
        shown = ("on" if on else "off") if kind == "vacuum" else value
        text = f"{number}/{len(demo.steps)}  {kind}: {shown}"
        if not confirm(text):
            say("   stopped before " + text)
            return False
        say("   " + text)
        if kind == "joints":
            arm.move_joints(demo.poses[value], tick)
        elif kind == "line":
            pose = workcell.tcp_of_tip(FR5.fk(demo.poses[value]))
            arm.move_lin(pose, speed, workcell.motion.tcp_accel_max, tick)
        elif kind == "vacuum":
            gripper.vacuum_on() if on else gripper.release()
        else:
            sleep(float(value))
    return True


def record(
    read_joints: Callable[[], list[float]],
    answer: Callable[[str], str | None],
    name: str,
    say: Callable[[str], None] = print,
) -> Demo | None:
    """Build a demo from typed commands (see the module docstring). None if nothing was taught."""
    demo = Demo(name, f"Taught on {time.strftime('%Y-%m-%d')}", 0.10, {}, [])
    say("Bring the arm to a pose, then: j <name> | l <name> | on | off | w <s> | undo | done")
    while True:
        reply = answer(f"   step {len(demo.steps) + 1} > ")
        words = ["done"] if reply is None else reply.split()  # input closed: save
        command = words[0] if words else ""
        if command == "done":
            break
        if command == "undo" and demo.steps:
            say(f"   dropped {demo.steps.pop()}")
        elif command in ("j", "l") and len(words) == 2:
            demo.poses[words[1]] = [float(v) for v in read_joints()]
            demo.steps.append(("joints" if command == "j" else "line", words[1]))
            degrees = ", ".join(f"{math.degrees(v):.1f}" for v in demo.poses[words[1]])
            say(f"   pose '{words[1]}': {degrees}")
        elif command in ("on", "off"):
            demo.steps.append(("vacuum", command))
        elif command == "w" and len(words) == 2 and words[1].replace(".", "", 1).isdigit():
            demo.steps.append(("wait", float(words[1])))
        else:
            say("   ? j <name> | l <name> | on | off | w <seconds> | undo | done")
    return demo if demo.steps else None


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def pick(title: str, prompt: str, options: list[str]) -> int:
    if ask_choice is None:
        sys.exit(f"❌ {prompt} Pass it on the command line (--help).")
    choice = ask_choice(title, prompt, options)
    if choice is None:
        sys.exit("Cancelled.")
    return choice


def ros_arm(workcell: Workcell, real: bool):
    """(link, arm, io) on MoveIt; io is the controller's state on the real arm."""
    try:
        from rfp_arm.fairino import FairinoIO
        from rfp_arm.moveit_arm import MoveItArm
        from rfp_arm.ros_link import RosLink

        link = RosLink("rfp_demo")
        io = None
        if FairinoIO.present(link):
            if not real:
                sys.exit("❌ The FAIRINO command server is running: the real arm. Use --mode real")
            io = FairinoIO(link)
            io.connect()
        arm = MoveItArm(link, workcell, fault=io.fault if io else None)
        arm.connect()
    except ImportError as e:
        sys.exit(f"❌ This needs ROS 2 with MoveIt ({e}): see pick_place.py's docstring.")
    return link, arm, io


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description="Play or record a taught demo (Demos/).")
    parser.add_argument("action", nargs="?", choices=("play", "record"), help="popup: play")
    parser.add_argument("name", nargs="?", help="demo name or file (popup if not given)")
    parser.add_argument("--mode", choices=list(MODES), help="where to play it (popup if not given)")
    parser.add_argument("--auto", action="store_true", help="don't wait for Enter before each step")
    parser.add_argument("--workcell", type=Path, help="workcell file (Arm/configs/workcell.yaml)")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask before moving the arm")
    args = parser.parse_args()
    try:
        workcell = demo_workcell(load_workcell(args.workcell))
    except WorkcellError as e:
        sys.exit(f"❌ {e}")
    link = None
    try:
        if args.action == "record":
            if not args.name:
                parser.error("record needs a name for the new demo")
            path = demo_path(args.name)
            if (
                path.exists()
                and not args.yes
                and ask(f"{path.name} exists. Replace it? [y/N] ") != "y"
            ):
                sys.exit("Cancelled.")
            link, arm, _ = ros_arm(workcell, real=True)
            demo = record(lambda: [arm.joints()[j] for j in workcell.joints], ask, path.stem)
            if demo is None:
                sys.exit("Nothing taught: no file written.")
            save_demo(path, demo)
            print(f"✅ Saved {path}\n   Next: python demo.py play {path.stem} --mode sim")
            return

        files = sorted(DEMOS_DIR.glob("*.yaml"))
        if not args.name and not files:
            sys.exit(f"❌ No demos in {DEMOS_DIR}. Teach one: python demo.py record <name>")
        path = (
            demo_path(args.name)
            if args.name
            else files[pick("Demo", "Demo to play:", [f.stem for f in files])]
        )
        try:
            demo = load_demo(path)
        except DemoError as e:
            sys.exit(f"❌ {e}")
        names = list(MODES)
        mode = args.mode or names[pick("Demo", "Mode:", [f"{n:8s} {MODES[n]}" for n in names])]
        print(f"DEMO {demo.name} ({mode}): {demo.description}")
        if mode == "offline":
            arm, gripper = SimArm(workcell), SimGripper()
        else:
            link, arm, io = ros_arm(workcell, real=mode == "real")
            gripper = make_gripper(workcell.gripper, io) if mode == "real" else SimGripper()
        if mode == "real":
            print(
                "!" * 60 + "\nTHE REAL ARM WILL MOVE through the demo's taught poses. Someone is at"
            )
            print("the emergency stop; zone checks are off for demos.\n" + "!" * 60)
            if not args.yes and ask('Type "move" to start: ') != "move":
                sys.exit("Cancelled.")

        def confirm(step: str) -> bool:
            return args.auto or ask(f"   next: {step}   Enter: go, q: stop > ") not in ("q", None)

        finished = play(demo, arm, gripper, workcell, confirm)
        print("✅ Demo done." if finished else "⏹️  Demo stopped. The vacuum stays as it is.")
    except (ArmError, MotionAborted) as e:
        sys.exit(f"❌ {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped. The vacuum stays as it is.")
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
