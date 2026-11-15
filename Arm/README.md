# Arm

Moves the FAIRINO FR5, runs the suction gripper, places fish and keeps the arm safe. ControlSys decides when each step runs; Arm provides the steps and reports arm and gripper status.

Implements Requirement Spec **Section 2.5 Arm System (AS)** and **Section 2.6 Safety (SF)**; methods are in Appendix C.5 to C.7.

## Layout

| Folder | Contents |
|---|---|
| `rfp_arm/` | Motion, grasp sequence steps, gripper I/O, placement poses, failure detection and status (modules below); `basic_move_demo.py` nods and swings the arm (moves the real arm: read its docstring first) |
| `scripts/` | `gripper_bench.py`: switch the vacuum by hand (`on`, `off`, `pulse`, `status`). Calibration tools come here |
| `firmware/valve_control/` | Arduino sketch of the valve board (Adafruit Feather ESP32 V2). Wi-Fi names and passwords go in `secrets.h`, copied from `secrets.example.h` and never committed |
| `configs/` | `workcell.yaml`: frames, tool and camera mount, zones, observation poses, placement slots, motion and gripper limits |

## Modules (`rfp_arm/`)

The Control System sees only `interfaces.py`: an `Arm` (move the TCP, stop), a `Gripper` (vacuum on, release, state) and the error codes of Table AS-2. Real and simulated versions are interchangeable.

| Module | What it does | Needs ROS |
|---|---|---|
| `workcell.py` | Loads `configs/workcell.yaml`; says why a tool pose is not allowed (work zone, exclusion zones, table clearance, ends of a held fish) | no |
| `grasp.py` | `plan_grasp`: the poses of Table AS-1 for a fish and a slot. `GraspSequence`: runs them, stops at the failed step with its error code | no |
| `gripper.py` | The gripper in each form: `ValveGripper` (vacuum switched by the ESP32 valve board of `valve.py`: on/off only, the camera confirms the hold), `SuctionGripper` (control box I/O with a vacuum switch, Table AS-4), simulated and dry-run grippers | no |
| `valve.py` | The valve board from the PC: `SerialValve` (USB) and `HttpValve` (Wi-Fi), every answer checked | no |
| `contact.py` | Feels contact without a force sensor: the force along the tool axis from the change in joint torques | no |
| `safety.py` | Checks every planned trajectory in fixed code before it runs: TCP speed, zones, joint travel | no |
| `kinematics.py`, `poses.py` | FR5 forward kinematics (from the URDF) and pose math | no |
| `sim.py` | Simulated arm for tests and offline runs | no |
| `moveit_arm.py` | The arm through MoveIt 2: Pilz PTP/LIN plans from `/move_action` (plan only), checked by `safety.py`, run by `/execute_trajectory`; loads the table, zones and tool into the planning scene | yes |
| `fairino.py` | Digital outputs (`SetDO`) and inputs, emergency stop and error state through frcobot_ros2's `ros2_cmd_server` | yes |

`workcell.yaml` ships with placeholder values and `verified: false`: dry runs and real runs refuse to start until someone has measured the cell and set it to `true`.

## Responsibilities

- **Motion (AS-10, AS-40):** MoveIt 2 with the Pilz planner (straight LIN and PTP moves) within the fish handling limits
- **Grasp sequence (AS-20):** pre-grasp, vacuum on, straight approach, confirm the hold, set the payload, lift, carry, lower, release with blow-off, retreat. The approach follows the fish's surface normal from the camera (up to 15 degrees off vertical), and on the real arm the last centimeters are felt: down in 3 mm steps until the joint torques show the fish pushing back, then pressed on (`grasp.feel_*`, `contact_force`, `press_force_max` in the workcell file; `pick_place.py --mode feel` shows the force live for tuning)
- **Suction gripper (AS-30):** six cups (3 x 2) under one vacuum, switched by a solenoid valve on an ESP32 board (firmware in `firmware/valve_control/`, PC side `rfp_arm/valve.py`) over USB serial or Wi-Fi: `gripper.driver: serial` or `http` in the workcell file. The Requirement Spec's variant, vacuum and blow-off valves on DO0/DO1 and a vacuum switch on DI0/DI1 through frcobot_ros2's `ros2_cmd_server`, is `driver: controller`
- **Placement (AS-50):** placement slots outside the pick area, uniform heading
- **Calibration (AS-60):** eye-in-hand hand-eye calibration with a ChArUco board and OpenCV
- **Failure detection, status and logging (AS-70, AS-80):** no seal, dropped fish, collision, planning failure
- **Safety (SF-10 to SF-40):** workspace zones, controller speed limit and collision detection, emergency and software stops, stop on link loss

## Before you run the arm

Until the software limits are in place: low speed only, someone on the emergency stop. New motions run in simulation, then as a dry run with the vacuum off, then with vacuum or real fish. Ctrl+C doesn't stop a trajectory the controller accepted: use the emergency stop. (`moveit_arm.py` cancels its trajectory on Ctrl+C, a stop request or a controller fault; if the program is killed, the move in progress still runs to its end.)

Bring-up order with [pick_place.py](../ControlSys/README.md#running-pick-and-place): `--mode sim`, `--mode check` (compare the controller's flange pose and I/O with what the program reads), `python scripts/gripper_bench.py status`, `--mode dry-run --step`, then `--mode run`.

Not done yet: hand-eye calibration tool (AS-60; `camera.mount` in the workcell file is typed in by hand), controller safety settings from software (SF-14, SF-41: set the speed limit and collision level on the pendant), payload update (CS-45), approach along the surface normal (only straight down).
