# Arm

Moves the FAIRINO FR5, runs the gripper, places fish and keeps the arm safe; ControlSys decides when each step runs. Requirement Spec **Section 2.5 (AS)** and **2.6 (SF)**, methods in Appendix C.5 to C.7.

Workcell setup and bring-up on the real arm: [top README](../README.md#set-up-the-workcell).

## Layout

| Folder | Contents |
|---|---|
| `rfp_arm/` | The modules below; `basic_move_demo.py` nods and swings the real arm (read its docstring first) |
| `scripts/` | `gripper_bench.py`: vacuum `on`, `off`, `pulse <ms>`, `status`, `watch` by hand |
| `firmware/valve_control/` | Valve board sketch (Adafruit Feather ESP32 V2). Wi-Fi credentials go in `secrets.h`, copied from `secrets.example.h`, never committed |
| `configs/` | `workcell.yaml`: frames, tool, camera mount, zones, observation poses, slots, motion and gripper limits |

## Modules (`rfp_arm/`)

ControlSys sees only `interfaces.py`: an `Arm`, a `Gripper` and the error codes of Table AS-2. Real and simulated versions are interchangeable.

| Module | What it does | ROS |
|---|---|---|
| `workcell.py` | Loads and checks `workcell.yaml`; says why a tool pose isn't allowed: work zone, exclusion zones, table clearance, ends of a held fish (SF-10) | no |
| `grasp.py` | `plan_grasp`: the poses of Table AS-1 for a fish and a slot, with the cups the way round that turns j6 least (vacuum tube). `GraspSequence`: runs them: approach along the fish's surface normal (up to 15°), the last centimeters felt by joint torque, hold confirmed by the camera; stops at a failed step with its error code (AS-20, AS-70) | no |
| `gripper.py` | `ValveGripper`: the ESP32 valve board (`driver: serial` or `http`), on/off only. `SuctionGripper`: control box I/O with a vacuum switch, the spec's variant (`driver: controller`, Table AS-4). Simulated and dry-run grippers | no |
| `valve.py` | The valve board over USB (`SerialValve`) or Wi-Fi (`HttpValve`) | no |
| `contact.py` | Force along the tool axis from joint torques (no force sensor); tune with `pick_place.py --mode feel` | no |
| `safety.py` | Checks every planned trajectory before it runs: TCP speed, zones, joint travel, wrist range | no |
| `kinematics.py`, `poses.py` | FR5 kinematics from the URDF: forward, inverse (seeded, so the arm keeps its posture), j6 at a pose; pose math | no |
| `sim.py` | Simulated arm for tests and offline runs | no |
| `moveit_arm.py` | MoveIt 2 with Pilz: PTP/LIN plans from `/move_action`, checked by `safety.py`, run by `/execute_trajectory`; table, zones and tool in the planning scene. Cancels the move on Ctrl+C, a stop or a controller fault | yes |
| `fairino.py` | Digital I/O, emergency stop and error state through frcobot_ros2's `ros2_cmd_server` | yes |

Not done yet: hand-eye calibration (AS-60; `camera.mount` is typed in), the controller's speed limit and collision level from software (SF-14, SF-41: set them on the pendant), payload update (CS-45), a compliant wrist while pressing (only a hook: `grasp.press_start_commands`).
