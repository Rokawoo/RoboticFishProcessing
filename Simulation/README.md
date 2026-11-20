# Simulation

A virtual cell running the real pick-and-place code: a virtual FR5, gripper and RealSense D405, and catfish from a 3D model. No ROS; Windows works. How to run it: [top README](../README.md#run-it-in-the-3d-simulation).

Settings: the `simulation` section of [control_panel.yaml](../control_panel.yaml); all of them: [rfp_sim/settings.py](rfp_sim/settings.py). Needs `pyvista`. The fish is [models/catfish.3mf](models/README.md), the file the stand-ins were printed from. The FR5's meshes are FAIRINO's ([frcobot_ros2](https://github.com/FAIR-INNOVATION/frcobot_ros2)), downloaded on the first run and kept out of Git (they have no license); offline, the arm is drawn as tubes.

## What is real and what is simulated

| Real code | Simulated (`rfp_sim/`) |
|---|---|
| The cycle, selector, failure handling and log (`rfp_controlsys`) | `world.py`: the arm's joints, the fish, the cups' contact |
| Vision: the trained model, depth, fish list, hold check (`rfp_vision`) | `render.py`: the camera's color and depth, the 3D window |
| Grasp planning and sequence, every safety check (`rfp_arm`) | `devices.py`: the arm and the vacuum |

- **Camera:** color and depth as the D405 gives them, with the lens of `vision.hfov_deg`, 0.5 mm depth noise, no depth outside 0.07-1.0 m or beside a depth step. Not modeled: holes on shiny skin, exposure, blur, shadows. `detector: truth` uses the renderer's outlines instead of the model.
- **Motion:** timed as MoveIt's Pilz planner times it on the FR5 (speeding up and slowing down at the joints' limits); lift, carry and lower as one blended move.
- **Contact:** each cup touches within 2 mm and pushes back at 500 N/m, which is what the arm feels; the vacuum holds a fish when 5 of 6 cups touch. A placed fish is taken away once the next is on the cups. `drop: lift` or `carry` loses the fish on the way, to test drop detection.
- **Fish:** `placement: anywhere` lays each anywhere in the pick area at any heading, as real fish lie; `view` only where one look sees all of it (the easy case).
- **3D window:** the cell, the vacuum tube coiled round the wrist as j6 turns, the camera's view, and what vision found drawn over the true fish (outline, cups, heading, the depth as points).
- **Twin of the real arm** (`twin.py`): the same drawing, following the real arm's joint states, beside a run on the robot PC (`run.twin: false` leaves it out).
- **Layout editor** (`--mode setup`, `layout_editor.py`): [top README](../README.md#lay-it-out-in-3d).

## Benchmark

`python Simulation/benchmark.py` runs the real cycle on 20 seeded scenes with nobody at the console and scores each fish against the simulator's truth: where it ended up, its distance from its slot, its heading error, where the cups took it relative to its true center of mass, failures, looks, wrist travel and simulated time. The same seed gives the same scenes, so two runs differ by code and settings alone. Results go to `Simulation/benchmarks/` (not in Git).

| Option | |
|---|---|
| `--compare <earlier run>` | The difference to an earlier run |
| `--fish 2` | Two fish a scene: anything about placing |
| `--fish-length 0.79` | The big prints (default: the panel's `fish_length`, 0.60 m) |
| `--placement view`, `--any-size`, `--no-hold-check`, `--detector truth`, `--panel` | Easier scenes, no size refusals, holds confirmed by a vacuum switch (faster), the renderer's outlines, another control panel |

GPU inference can differ by a pixel between runs, which can tip a fish lying right at a limit.

## Other simulators

- **MoveIt with mock hardware** (`--mode sim` on the robot PC): the real ROS and MoveIt path; no camera, fish or vacuum.
- **FAIRINO SimMachine** (the controller software in a VM): the real controller for the driver and error codes; no camera, fish or vacuum.
