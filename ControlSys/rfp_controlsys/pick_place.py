#!/usr/bin/env python3
"""
Pick and place: look over the table with the camera on the arm, find a fish, pick it up
with the suction gripper and set it down in the next placement slot with a uniform heading,
watching with the camera that the fish stays on the gripper. Repeats until the pick area
is clear.

Usage:
    python pick_place.py                                  # pick the mode (model, camera) in popups
    python pick_place.py --mode observe                   # camera, detections, fish positions
    python pick_place.py --mode offline --device clip.mp4 --any-size --cycles 2
    python pick_place.py --mode sim --cycles 2            # MoveIt's mock hardware in RViz
    python pick_place.py --mode check                     # real arm: state, I/O, camera
    python pick_place.py --mode dry-run --step            # THE ARM MOVES, vacuum off
    python pick_place.py --mode run                       # THE ARM MOVES, with vacuum

Modes, in the order to bring a cell up (SF-43):
    observe   Nothing moves. Shows what the camera and the model see and where each fish is
              in the arm's frame (with MoveIt running: from the arm's real pose).
    offline   The whole cycle with a simulated arm and gripper, no ROS: for recorded videos.
    sim       The whole cycle on MoveIt with mock hardware; watch the arm in RViz (CS-92).
    check     Real arm, nothing moves: prints joint state, TCP pose, the controller's flange
              pose and I/O, the gripper state and what the camera sees.
    dry-run   THE REAL ARM MOVES. Vacuum off, every approach stops 50 mm short (CS-93).
    run       THE REAL ARM MOVES and picks fish.

--device: "realsense" (first one found), a RealSense serial number, an OpenCV camera index
(0, 1, ...: color only), or a recording (.bag/.db3 with depth, any video without). Without
depth, fish are placed on the table plane (Vision/configs/vision.yaml).

Before sim, check, dry-run and run (ROS 2 Humble sourced, one terminal each):
    ros2 launch fairino5_v6_moveit2_config demo.launch.py      # MoveIt, RViz, the controller
    ros2 run fairino_hardware ros2_cmd_server                  # real arm only: I/O and state
On the real arm the launch has to load fairino_hardware/FairinoHardwareInterface instead of
mock_components/GenericSystem (config/fairino5_v6_robot.ros2_control.xacro).

Moving the real arm: low speed and someone on the emergency stop (Arm/README.md). q or Esc
in the camera window, or Ctrl+C, stops the arm where it is; a move that MoveIt no longer
answers for needs the emergency stop. Each run is logged in ControlSys/runs/.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: popups
try:
    from pipeline_common import ask_choice, fix_console
except ImportError:  # installed without the repository around it
    ask_choice = None

    def fix_console() -> None:
        pass


from rfp_arm.fairino import FairinoIO
from rfp_arm.gripper import NoGripper, SimGripper, make_gripper
from rfp_arm.interfaces import ArmError, GripperState
from rfp_arm.sim import SimArm
from rfp_arm.workcell import Workcell, WorkcellError, describe, load_workcell
from rfp_vision.camera import CameraError, list_realsense, open_camera
from rfp_vision.detector import MODELS_DIR, FishDetector, find_weights, list_runs
from rfp_vision.fish_list import FishLocator
from rfp_vision.overlay import draw_fish_list
from rfp_vision.settings import VisionSettings
from rfp_vision.settings import load_settings as load_vision_settings
from rfp_vision.vision_system import VisionSystem

from rfp_controlsys.console import Console, ask
from rfp_controlsys.cycle import PickPlaceCycle
from rfp_controlsys.run_log import RunLog
from rfp_controlsys.selector import RuleSelector
from rfp_controlsys.settings import ControlSettings
from rfp_controlsys.settings import load_settings as load_control_settings

MODES = {
    "observe": "camera, detections and fish positions; nothing moves",
    "offline": "the whole cycle with a simulated arm and gripper, no ROS (recorded videos)",
    "sim": "the whole cycle on MoveIt's mock hardware, shown in RViz",
    "check": "real arm: read its state, I/O and the camera; nothing moves",
    "dry-run": "THE REAL ARM MOVES: vacuum off, approaches stop 50 mm short",
    "run": "THE REAL ARM MOVES: pick and place with vacuum",
}
ROS_MODES = ("sim", "check", "dry-run", "run")
REAL_MODES = ("check", "dry-run", "run")
MOVING_MODES = ("dry-run", "run")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def pick_mode() -> str:
    if ask_choice is None:
        sys.exit("❌ Pass --mode (one of: " + ", ".join(MODES) + ")")
    names = list(MODES)
    choice = ask_choice("Pick and place", "Mode:", [f"{n:8s}  {MODES[n]}" for n in names])
    if choice is None:
        sys.exit("Cancelled.")
    return names[choice]


def pick_model() -> Path:
    runs = list_runs()
    if not runs:
        sys.exit(
            f"❌ No trained models in {MODELS_DIR}\n"
            "   Train one (python Vision/data/training/train.py) or pass --model <run or .pt>"
        )
    if len(runs) == 1 or ask_choice is None:
        return runs[0]
    choice = ask_choice("Pick and place", "Fish model (newest first):", [r.name for r in runs])
    if choice is None:
        sys.exit("Cancelled.")
    return runs[choice]


def pick_device() -> str:
    cameras = list_realsense()
    if not cameras:
        sys.exit(
            "❌ No RealSense camera found. Plug in the D405, or pass --device "
            "(camera index, serial number, or a recording)"
        )
    if len(cameras) == 1 or ask_choice is None:
        return cameras[0][0]
    labels = [f"{name}  {serial}" for serial, name in cameras]
    choice = ask_choice("Pick and place", "Camera:", labels)
    if choice is None:
        sys.exit("Cancelled.")
    return cameras[choice][0]


def build_vision(
    device: str, weights: Path, settings: VisionSettings, workcell: Workcell
) -> VisionSystem:
    """Camera, model and locator. The first frame warms the model up."""
    try:
        camera = open_camera(device, settings.hfov_deg, (settings.width, settings.height),
                             settings.fps)  # fmt: skip
    except CameraError as e:
        sys.exit(f"❌ {e}")
    print(f"Camera:   {camera.name}" + ("" if camera.has_depth else "  (no depth: table plane)"))
    try:
        from pipeline_common import require_ultralytics

        require_ultralytics()  # clear message if PyTorch/Ultralytics is missing or too old
    except ImportError:
        pass
    try:
        detector = FishDetector(weights, settings.predict_conf, settings.min_area_share)
        detector.warm_up(camera.read().color)
    except (CameraError, ValueError) as e:
        camera.close()
        sys.exit(f"❌ {e}")
    locator = FishLocator(
        settings, workcell.table_z, workcell.pick_area.area, detector.version,
        workcell.calibration_version,
    )  # fmt: skip
    return VisionSystem(camera, detector, locator, settings)


def ros_link():
    try:
        from rfp_arm.ros_link import RosLink

        return RosLink()
    except ImportError as e:
        sys.exit(
            f"❌ This mode needs ROS 2 ({e}).\n"
            "   On the robot PC: source /opt/ros/humble/setup.bash and the workspace with\n"
            "   MoveIt 2 and frcobot_ros2, then run this script with ROS's Python."
        )


def moveit_arm(link, workcell: Workcell, console: Console, io=None):
    """The arm through MoveIt, connected. Nothing moves."""
    try:
        from rfp_arm.moveit_arm import MoveItArm

        arm = MoveItArm(link, workcell, console.confirm_move, io.fault if io else None)
        arm.connect()
    except ImportError as e:
        sys.exit(f"❌ MoveIt 2's Python messages aren't available ({e}): install ros-humble-moveit")
    return arm


def fairino_io(link) -> FairinoIO:
    try:
        io = FairinoIO(link)
    except ImportError as e:
        sys.exit(f"❌ fairino_msgs isn't available ({e}): build and source frcobot_ros2")
    io.connect()
    return io


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def observe(vision: VisionSystem, workcell: Workcell, console: Console, camera_pose) -> None:
    """Show detections and fish positions until q, Esc or Ctrl+C. Nothing moves."""
    print("\nObserving (q or Esc in the window, or Ctrl+C, to stop). Fish positions in {B}:")
    last_print = 0.0
    console.state("observing", "nothing moves")
    while not console.stop_requested():
        frame = vision.frame()
        fish_list = vision.locator.locate(frame, vision.detector.detect(frame.color), camera_pose())
        image = draw_fish_list(frame.color, fish_list, frame.intrinsics)
        console.show(image, [f"{len(fish_list.valid)} valid of {len(fish_list.fish)} fish"])
        if time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            print("   " + ("; ".join(_fish_text(f) for f in fish_list.fish) or "no fish"))
        if not console.window:
            time.sleep(0.2)


def _fish_text(fish) -> str:
    if not fish.vision_ok:
        return f"fish {fish.fish_id}: {fish.failure_code}"
    x, y, z = fish.grasp_point * 1000
    heading = f"{np.degrees(fish.heading):.0f} deg" + ("" if fish.heading_known else " (head?)")
    return (
        f"fish {fish.fish_id}: x {x:.0f} y {y:.0f} z {z:.0f} mm, heading {heading}, "
        f"length {fish.total_length * 1000:.0f} mm"
    )


def check(arm, io: FairinoIO, vision: VisionSystem, workcell: Workcell) -> None:
    """Print what the arm, the controller, the gripper and the camera report. Nothing moves."""
    joints = arm.joints()
    tcp = arm.tcp_pose()
    tip = workcell.tip_of_tcp(tcp)[:3, 3]
    print("\nArm (MoveIt):")
    print("   joints (deg): " + ", ".join(f"{np.degrees(joints[j]):.1f}" for j in workcell.joints))
    print("   tip link (mm): " + ", ".join(f"{v * 1000:.1f}" for v in tip))
    print("   TCP (mm):      " + ", ".join(f"{v * 1000:.1f}" for v in tcp[:3, 3]))
    print(f"   {arm.limits()}")
    problem = workcell.pose_problem(tcp)
    print(f"   TCP now: {'inside the work zone' if problem is None else problem}")
    print("Controller (ros2_cmd_server):")
    for line in io.report():
        print(f"   {line}")
    flange = io.flange_position()
    if flange is not None:
        offset = np.linalg.norm(np.array(flange) - tip) * 1000
        print(
            f"   controller flange to MoveIt tip link: {offset:.0f} mm apart. Expect the flange\n"
            "   offset along the tool axis (about 100 mm); anything else means the two base\n"
            "   frames differ: sort that out before trusting positions."
        )
    fault = io.fault()
    print(f"   fault: {fault or 'none'}")
    try:
        gripper = make_gripper(workcell.gripper, io, workcell.grasp.release_timeout)
        switch = "" if gripper.has_switch else " (no vacuum switch: the camera confirms a hold)"
        print(f"Gripper ({workcell.gripper.driver}): {gripper.state().value}{switch}")
    except ArmError as e:
        print(f"Gripper ({workcell.gripper.driver}): NOT AVAILABLE: {e.message}")
    frame = vision.frame()
    print(f"Camera: {frame.size[0]} x {frame.size[1]} px, fx {frame.intrinsics.fx:.0f}")
    if frame.depth is not None:
        low, high = vision.settings.depth_range_m
        valid = float(((frame.depth >= low) & (frame.depth <= high)).mean())
        print(f"   depth: {valid:.0%} of pixels between {low:g} and {high:g} m (VS-17 wants 95%)")
    fish_list = vision.locator.locate(
        frame, vision.detector.detect(frame.color), workcell.camera_of_tcp(tcp)
    )
    print("   sees: " + ("; ".join(_fish_text(f) for f in fish_list.fish) or "no fish"))
    print("\n✅ Check done. Nothing moved. Next: --mode dry-run --step")


def start_checks(mode: str, workcell: Workcell, io: FairinoIO | None, gripper) -> None:
    """SF-41: don't start moving the real arm unless these pass."""
    problems = []
    if not workcell.verified:
        problems.append(
            f"{workcell.source.name} is not verified: measure the cell, set its values, "
            "then set 'verified: true'"
        )
    if mode == "run" and not workcell.calibrated:
        problems.append("the camera is not calibrated (camera.calibrated in the workcell file)")
    fault = io.fault() if io else None
    if fault:
        problems.append(f"the controller reports: {fault}")
    if mode == "run":
        state = gripper.state()
        if state == GripperState.ERROR:
            problems.append("the gripper's I/O can't be read")
        elif state != GripperState.VACUUM_OFF:
            problems.append(f"the gripper reports '{state.value}': turn the vacuum off first")
    if problems:
        sys.exit("❌ Start checks failed (SF-41):\n   - " + "\n   - ".join(problems))


def confirm_real_motion(mode: str, yes: bool) -> None:
    print("\n" + "!" * 60)
    print(f"THE REAL ARM WILL MOVE ({mode}).")
    print("   - someone is at the emergency stop, nobody is inside the work zone")
    print("   - the controller's speed limit and collision detection are set (SF-14, SF-22):")
    print("     this program doesn't set them yet")
    print("   - q or Esc in the camera window, or Ctrl+C, stops the arm where it is")
    print("!" * 60)
    if yes:
        return
    if not sys.stdin.isatty():
        sys.exit("❌ Not confirmed (no terminal). Pass -y to start without the question.")
    if ask('Type "move" to start: ') != "move":
        sys.exit("Cancelled.")


def run_cycles(
    mode: str,
    arm,
    gripper,
    vision: VisionSystem,
    workcell: Workcell,
    settings: ControlSettings,
    console: Console,
    max_cycles: int | None,
) -> None:
    log = RunLog(
        {
            "mode": mode,
            "camera": vision.camera.name,
            "model_version": vision.detector.version,
            "calibration_version": workcell.calibration_version,
            "workcell": str(workcell.source),
        }
    )
    cycle = PickPlaceCycle(
        arm, gripper, vision, RuleSelector(), workcell, settings, console, log,
        dry_run=mode == "dry-run", vision_hold=mode == "run",
    )  # fmt: skip
    print(f"\nRunning ({mode}). Log: {log.folder}")
    stopped = ""
    try:
        cycle.run(max_cycles)
    except KeyboardInterrupt:
        stopped = "stopped with Ctrl+C"
        arm.stop()
    report = cycle.report
    report.ended = stopped or report.ended
    log.close(report.as_record())

    print("\n" + "=" * 60)
    print(f"{'⏹️ ' if stopped or report.ended.startswith('stopped') else '🏁'} {report.ended}")
    print(f"   {report.cycles} cycle(s): {report.successes} placed", end="")
    print("".join(f", {n} x {code}" for code, n in report.failures.items()))
    for where in report.skipped:
        print(f"   skipped fish at {where} (attempt limit, CS-62)")
    print(f"   log: {log.folder}")
    if cycle.sequence is not None and cycle.sequence.holding:
        print("⚠️  The vacuum is still on: a fish may be on the gripper.")
        if console.interactive and ask("   Turn the vacuum off now? [y/N] ") == "y":
            cycle.sequence.let_go()
        else:
            print("   Turn it off with: python Arm/scripts/gripper_bench.py off")
    if mode == "offline":
        print("Next: --mode sim on the robot PC (MoveIt's mock hardware in RViz)")
    elif mode == "sim":
        print("Next: --mode check on the real arm, then --mode dry-run --step")
    elif mode == "dry-run":
        print("Next: --mode run, once the dry run looks right (SF-43)")


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(
        description="Find fish with the camera on the arm, pick them and place them in slots.",
        epilog="Modes: " + "; ".join(f"{name}: {text}" for name, text in MODES.items()),
    )
    parser.add_argument("--mode", choices=list(MODES), help="what to run (popup if not given)")
    parser.add_argument("--device", help="camera: realsense, a serial number, an index, a video")
    parser.add_argument("--model", type=Path, help="run folder or .pt (default: the newest run)")
    parser.add_argument("--workcell", type=Path, help="workcell file (Arm/configs/workcell.yaml)")
    parser.add_argument("--vision", type=Path, help="vision settings (Vision/configs/vision.yaml)")
    parser.add_argument("--control", type=Path, help="limits (ControlSys/configs/control.yaml)")
    parser.add_argument(
        "--cycles", type=int, help="stop after this many grasps (default: no limit)"
    )
    parser.add_argument("--step", action="store_true", help="wait for Enter before every move")
    parser.add_argument("--no-window", action="store_true", help="don't show the camera image")
    parser.add_argument(
        "--any-size", action="store_true", help="skip fish size checks (videos from other cameras)"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask before moving the arm")
    args = parser.parse_args()
    if args.cycles is not None and args.cycles < 1:
        parser.error("--cycles must be 1 or more")
    mode = args.mode or pick_mode()

    try:
        workcell = load_workcell(args.workcell)
        vision_settings = load_vision_settings(args.vision)
        control_settings = load_control_settings(args.control)
        weights = find_weights(args.model or pick_model())
    except (WorkcellError, FileNotFoundError, ValueError, TypeError) as e:
        sys.exit(f"❌ {e}")
    if args.any_size:
        vision_settings.check_size = False
    device = args.device or pick_device()

    print("=" * 60)
    print(f"PICK AND PLACE: {mode} ({MODES[mode]})")
    print("=" * 60)
    print(f"Workcell: {workcell.source}" + ("" if workcell.verified else "  (NOT verified)"))
    for line in describe(workcell):
        print(f"          {line}")
    print(f"Model:    {weights}")
    vision = build_vision(device, weights, vision_settings, workcell)
    console = Console(window=not args.no_window, step=args.step)
    link = None
    try:
        if mode == "offline":
            run_cycles(mode, SimArm(workcell), SimGripper(), vision, workcell, control_settings,
                       console, args.cycles)  # fmt: skip
            return
        if mode == "observe":
            observe(vision, workcell, console, _camera_pose_source(workcell, console))
            return

        link = ros_link()
        real = FairinoIO.present(link)
        if mode == "sim" and real:
            sys.exit(
                "❌ The FAIRINO command server is running: this looks like the real arm, not a "
                "simulation. Use --mode dry-run, or stop ros2_cmd_server."
            )
        io = fairino_io(link) if mode in REAL_MODES else None
        arm = moveit_arm(link, workcell, console, io)
        print(f"Arm:      MoveIt group '{workcell.group}'; {arm.limits()}")
        if mode == "check":
            check(arm, io, vision, workcell)
            return
        if mode == "sim":
            gripper = SimGripper()
        elif mode == "dry-run":
            gripper = NoGripper()
        else:
            gripper = make_gripper(workcell.gripper, io, workcell.grasp.release_timeout)
        if mode in MOVING_MODES:
            start_checks(mode, workcell, io, gripper)
            confirm_real_motion(mode, args.yes)
        run_cycles(mode, arm, gripper, vision, workcell, control_settings, console, args.cycles)
    except ArmError as e:
        sys.exit(f"❌ {e}")
    except CameraError as e:
        sys.exit(f"❌ Camera: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")
    finally:
        console.close()
        vision.close()
        if link is not None:
            link.close()


def _camera_pose_source(workcell: Workcell, console: Console):
    """For observe mode: the camera's pose from the arm if MoveIt is up, else the first
    observation pose as written in the workcell file."""
    try:
        from rfp_arm.moveit_arm import MoveItArm
        from rfp_arm.ros_link import RosLink

        arm = MoveItArm(RosLink("rfp_observe"), workcell)
        arm.connect(timeout=5.0)
        print("Camera pose: from the arm (MoveIt's joint states)")
        return lambda: workcell.camera_of_tcp(arm.tcp_pose())
    except (ImportError, ArmError) as e:
        first = workcell.observation_poses[0]
        if first.camera is None:
            sys.exit(
                f"❌ The arm's pose isn't available ({e}) and observation pose '{first.name}' "
                "is taught joints: start MoveIt, or give that pose as camera_xy."
            )
        print(f"Camera pose: observation pose '{first.name}' from the workcell file "
              f"(the arm isn't available: {e})")  # fmt: skip
        return lambda: first.camera


if __name__ == "__main__":
    main()
