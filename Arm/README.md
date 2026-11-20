# Arm

Moves the FAIRINO FR5, runs the gripper, places fish and keeps the arm safe; ControlSys decides when each step runs. Requirement Spec Sections 2.5 (AS) and 2.6 (SF). Setting up the cell and the real arm: [top README](../README.md#set-up-the-workcell).

| Folder | Contents |
|---|---|
| `rfp_arm/` | The modules below |
| `scripts/` | `gripper_bench.py`: the vacuum by hand (`on`, `off`, `pulse <ms>`, `status`, `watch`) |
| `firmware/valve_control/` | Valve board sketch (ESP32). Wi-Fi credentials go in `secrets.h` (from `secrets.example.h`), never committed |
| `configs/` | `workcell.yaml`: the cell as measured. `collision_model.yaml`: the arm and gripper as spheres (generated) |
| `ros/` | `fr5_bringup.launch.py`: FAIRINO's MoveIt config with the real driver (or `hardware:=mock`) and their command server; `joint_limits.yaml`: the accelerations it plans with |

## Modules (`rfp_arm/`)

ControlSys sees only `interfaces.py`: an `Arm`, a `Gripper` and the error codes. Real and simulated ones are interchangeable. Only `moveit_arm.py` and `fairino.py` need ROS.

| Module | What it does |
|---|---|
| `workcell.py` | Loads and checks the cell; says why a tool pose isn't allowed (zones, table clearance, a held fish's ends) |
| `layout.py` | The cell's layout: checked against what the arm reaches and the camera sees (`check_layout`, printed by every mode), saved line by line |
| `grasp.py` | `plan_grasp`: the poses for a fish and a slot, carried just over what lies on the way, the wrist turned the way that keeps the vacuum tube from winding up. `GraspSequence` runs them: down onto the fish until it's felt, hold confirmed by the camera, lift, carry and lower as one move |
| `gripper.py`, `valve.py` | The ESP32 valve board over USB or Wi-Fi (on/off only); the controller's I/O with a vacuum switch (`driver: controller`); simulated and dry-run grippers |
| `contact.py` | Force along the tool's axis from the joint torques (no force sensor) |
| `safety.py` | Checks every trajectory before it runs: the speed of every point of the arm, zones, joint travel, the wrist's range, the arm against itself |
| `collision.py` | The arm and gripper as spheres: a pose or path running the arm into itself, or into a box |
| `kinematics.py`, `poses.py` | FR5 kinematics (inverse seeded, so the arm keeps its posture), pose math |
| `sim.py` | A simulated arm for the cycle tests |
| `moveit_arm.py` | MoveIt 2 with Pilz: point-to-point and straight moves, blended sequences, each checked by `safety.py`; cancels on Ctrl+C, a stop or a fault |
| `fairino.py` | I/O, emergency stop and errors through FAIRINO's command server, for firmware 3.7 and newer |
| `bringup.py` | Starts MoveIt and the driver for a run when nothing is running, stops them after |

Not done yet: the controller's speed limit and collision level from software (set them on the pendant), a payload update, a compliant wrist while pressing (only a hook: `grasp.press_start_commands`).
