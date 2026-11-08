# Arm

Moves the FAIRINO FR5, runs the suction gripper, places fish and keeps the arm safe. The control node in ControlSys decides when each step runs; this subsystem provides the steps and reports arm and gripper status back.

Implements Requirement Spec **Section 2.5 Arm System (AS)** and **Section 2.6 Safety (SF)**; methods are in Appendix C.5 to C.7.

## Layout

| Folder | Contents |
|---|---|
| `rfp_arm/` | Motion, grasp sequence steps, gripper I/O, placement poses, failure detection and status |
| `scripts/` | Calibration and gripper bench tools |
| `configs/` | Workspace zones, motion limits, observation poses, placement slots, calibration results |
| `tests/` | Tests |

## Responsibilities

- **Motion (AS-10, AS-40):** MoveIt 2 with the Pilz planner (straight LIN and PTP moves) within the fish handling limits
- **Grasp sequence (AS-20):** pre-grasp, vacuum on, straight approach, confirm the hold, set the payload, lift, carry, lower, release with blow-off, retreat
- **Suction gripper (AS-30):** vacuum valve on DO0, blow-off valve on DO1, vacuum switch holding and released outputs on DI0 and DI1, through frcobot_ros2's `ros2_cmd_server`
- **Placement (AS-50):** placement slots outside the pick area, uniform heading
- **Calibration (AS-60):** eye-in-hand hand-eye calibration with a ChArUco board and OpenCV
- **Failure detection, status and logging (AS-70, AS-80):** no seal, dropped fish, collision, planning failure
- **Safety (SF-10 to SF-40):** workspace zones, controller speed limit and collision detection, emergency and software stops, stop on link loss

## Before you run the arm

Until the software limits are in place, the arm runs only at low speed with someone on the emergency stop. New motions run in simulation, then as a dry run with the vacuum off, before they run with vacuum or real fish.
