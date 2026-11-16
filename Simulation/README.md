# Simulation

A virtual workcell in which the real pick-and-place code runs: a virtual FR5, suction gripper and RealSense D405, and catfish made from a 3D model, shown in a 3D window. It runs on Windows without ROS, so the vision and cycle code can be tried before the arm is free.

```bash
python ControlSys/rfp_controlsys/pick_place.py --mode sim3d
python ControlSys/rfp_controlsys/pick_place.py --mode sim3d --video run.mp4   # also record the 3D view
```

Settings: the `simulation` section of [control_panel.yaml](../control_panel.yaml) (every setting, explained: [sim.yaml](sim.yaml)). Needs `pyvista` (the `sim` extra). The fish model is [models/catfish.3mf](models/README.md), the file the stand-in fish were printed from. The FR5 is drawn with FAIRINO's own link meshes from [frcobot_ros2](https://github.com/FAIR-INNOVATION/frcobot_ros2): on the first run they are downloaded at a fixed commit, checked against their SHA-256 and kept in `models/fr5/`. They carry no license, so they stay out of Git. A sourced ROS workspace's `fairino_description`, or a folder set as `robot_meshes`, is used first; with `download_robot_meshes: false` or no network the arm is drawn as plain tubes.

## What runs for real and what is simulated

| Real code | Simulated (`rfp_sim/`) |
|---|---|
| The cycle, the fish selector, failure handling, the log (`rfp_controlsys`) | `world.py`: the arm's joints, the fish, the cups' contact |
| Vision: the trained YOLO model on the camera's images, depth, the fish list, head/tail, surface slope, hold check (`rfp_vision`) | `render.py`: the camera's color and depth images, the 3D window |
| Grasp planning and sequence, the fixed-code trajectory check, wrist and zone limits (`rfp_arm`) | `devices.py`: the arm (plans by inverse kinematics, then the same check as on the real arm) and the vacuum |

**The virtual camera** renders the scene from the camera's pose on the gripper (`camera.mount` in the workcell file, moved by the arm) as the D405 would deliver it: 1280 × 720 color and aligned depth, an 87° pinhole lens (no distortion), depth in meters along the optical axis from the GPU's depth buffer, with 0.5 mm noise, no depth outside 0.07–1.0 m, and none within 3 px of a depth step of more than about 1 cm (where one of the stereo pair can't see past an edge). The table is drawn as the perforated breadboard of the real cell, the fish in the white of the printed stand-ins, so the trained model sees images close to the footage it learned from (it finds the rendered fish at about 0.9 confidence). Not modeled: holes on shiny skin, exposure, motion blur, shadows. `detector: truth` uses the renderer's own fish outlines (smoothed as a segmentation model's are) instead of the model, to test everything else.

**The 3D window** also shows what the camera's view leaves out: the vacuum tube, which winds round the wrist by as much as the wrist (j6) has turned from where the tube hangs free and then runs along the forearm, the table's slab and legs, and the wrist angle against the tube's range. The gripper is drawn as built: flange adapter, plate, six bellows cups, the D405 on its bracket.

**The contact:** each of the six cups is a ray along the tool axis. A cup touches when the fish's surface is within 2 mm of its face, each compressed cup pushes back at 500 N/m (what the arm "feels" while pressing on), and the vacuum holds the fish only when 5 of 6 cups touch: a firm, square contact. A held fish follows the cups once they lift and lies down flat where it is let go. There is no other physics: nothing falls over, slides or collides. `drop: lift` or `carry` loses the fish on the way, to test the camera's drop detection.

## Found with it

- The FR5's wrist (j6) depends on how the camera is turned at an observation pose: with the placeholder cell's camera at yaw 90° it needed about 170°, past the vacuum tube's range, so the observation poses now face −90° (wrist about −15°).
- Lying on its side, the printed catfish stands 0.29 × its length high with its fins. Seen from 450 mm (AS-11), a fish that tall and longer than about 0.55 m doesn't fit whole in one camera view, so VS-15 can't be met for the spec's 0.60–0.75 m fish there. Hence `fish_length: 0.55`; for longer fish the cell needs a higher observation pose or more views.
- A fish already placed, seen at the edge of the frame, kept the run from ending; fish outside the pick area no longer count (SY-22).
- The rendered fish matches the camera model to the pixel, but vision measures this fish about 18% long: it puts the outline at half the fish's thickness, while the ends of the printed fish (lying on its side) are about 12 cm up. Length checks and slot spacing should allow for it, or the outline's points should be placed with their own depth.
- Turning the fish to the placement heading while the base swings to the slot costs the wrist up to about 180° when the head end is known, whichever way round the cups grasp it. Over 1000 random fish the wrist turned 193° per cycle on average (there and back) and reached 170° at most: the tube's ±170° range is enough, ±135° would refuse about one fish in five, ±90° two in five (fish with an unknown head end always fit in ±90°).

## Other simulators

- **MoveIt with mock hardware in RViz** (`pick_place.py --mode sim`, robot PC): the real ROS and MoveIt path, but no camera, fish or vacuum. RViz shows the robot, the planning scene and the planned trajectories; it can display camera images and point clouds that something else publishes, but it doesn't render a camera itself.
- **FAIRINO SimMachine** (FAIRINO's controller software in a VMware image): the real controller, so the driver, the command server's answers, ServoJ streaming, joint limits and error codes can be tested; no camera, fish or vacuum either. It is the best stand-in for the arm before the real one: point frcobot_ros2 at the VM's address instead of the robot's.
- **Gazebo**: physics and simulated depth cameras on ROS topics, but this code reads the camera through pyrealsense2, not ROS, and suction needs a custom plugin.

This simulation covers what those don't (the camera, the fish and the cycle) and SimMachine covers what this one can't (the real controller).
