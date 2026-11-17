# Simulation

A virtual cell running the real pick-and-place code: a virtual FR5, gripper and RealSense D405, and catfish from a 3D model, in a 3D window. No ROS; Windows works. How to run it: [top README](../README.md#run-it-in-the-3d-simulation).

Settings: the `simulation` section of [control_panel.yaml](../control_panel.yaml); all of them, explained: [sim.yaml](sim.yaml). Needs `pyvista` (the `sim` extra). The fish is [models/catfish.3mf](models/README.md), the file the stand-ins were printed from. The FR5 is drawn with FAIRINO's link meshes from [frcobot_ros2](https://github.com/FAIR-INNOVATION/frcobot_ros2), downloaded on the first run at a fixed commit, checked by SHA-256 and kept in `models/fr5/`. They carry no license, so they stay out of Git. A sourced ROS workspace's `fairino_description` or a `robot_meshes` folder is used first; offline or with `download_robot_meshes: false` the arm is plain tubes.

## Real and simulated

| Real code | Simulated (`rfp_sim/`) |
|---|---|
| Cycle, selector, failure handling, log (`rfp_controlsys`) | `world.py`: the arm's joints, the fish, the cups' contact |
| Vision: the trained model on the camera's images, depth, fish list, head/tail, surface slope, hold check (`rfp_vision`) | `render.py`: the camera's color and depth, the 3D window |
| Grasp planning and sequence, trajectory check, wrist and zone limits (`rfp_arm`) | `devices.py`: the arm (inverse kinematics, then the real arm's checks) and the vacuum |

**Camera:** rendered from the camera's pose on the gripper (`camera.mount`) as a D405 delivers it: 1280 × 720 color and aligned depth, 87° pinhole lens, depth along the optical axis with 0.5 mm noise, none outside 0.07–1.0 m, and none within 3 px of a depth step over about 1 cm (the stereo pair can't see past an edge). The table is the cell's perforated breadboard, the fish the stand-ins' white, so the model sees footage like its training data (it finds rendered fish at about 0.9 confidence). Not modeled: holes on shiny skin, exposure, motion blur, shadows. `detector: truth` uses the renderer's outlines instead of the model.

**Layout editor** (`--mode setup`, `rfp_sim/layout_editor.py`): the same arm and gripper with the workcell's zones, slots and observation poses as boxes and balls to drag, checked as they change (`rfp_arm/layout.py`); see the [top README](../README.md#lay-it-out-in-3d).

**3D window:** also the vacuum tube, coiled round the wrist by as much as j6 has turned, the table's body, and j6 against the tube's range. The gripper is its CAD model (`models/gripper.stl`, from `import_gripper.py`); the camera's own view leaves out its housing.

**Contact:** each cup is a ray along the tool axis, touching within 2 mm of its face and pushing back at 500 N/m when compressed (what the arm feels while pressing). The vacuum holds the fish only when 5 of 6 cups touch. A held fish follows the cups once they lift and lies flat where it's let go; no other physics. `drop: lift` or `carry` loses the fish on the way, to test drop detection.

**Fish placement:** `placement: anywhere` (default) lays each fish anywhere in the pick area at any heading, as real fish lie; `view` only where an observation pose sees all of it along the image's long side, the easy case.

## Benchmark

`python Simulation/benchmark.py` runs the real cycle on seeded scenes with nobody at the console and scores each fish against the simulator's truth (`rfp_sim/benchmark.py`): where it ended up, its middle's distance from its slot, its heading error, where the cups took it relative to its true center of mass (the model's volume centroid), failures with their messages, looks, wrist travel, simulated time. Default: 20 scenes, one 0.79 m stand-in lying anywhere, the trained model. `--compare <earlier run>` shows the difference; `--any-size`, `--no-hold-check` (a vacuum switch confirms holds; faster), `--placement view`, `--fish-length`, `--detector truth`. The same seed gives the same scenes, so two runs differ by code and settings alone; each behavior has a setting to switch it off (`close_looks: 0`, `balance_ahead_share: 0`).

Full-size stand-ins lying anywhere, as the changes went in:

| | Placed | Slot error | Cups from the center of mass |
|---|---|---|---|
| Before (size check off) | 0 / 20: all cut off at the image's edge | – | – |
| + closer looks from 0.62 m | 16 / 20 | 34 mm | 113 mm toward the tail |
| + stand-in size limits | 15 / 20 | 34 mm | 113 mm toward the tail |
| + cups where it balances | 13 / 20 | 16 mm | 1 mm |
| + point-to-point carry when the straight one can't, held fish checked from its middle | **35 / 40** | 15 mm | 0 mm |

With the gripper as built (its CAD model: six 40 mm cups 160 x 60 mm apart on 4.5-fold bellows), 40 scenes:

| | Placed | Slot error | Cups from the center of mass |
|---|---|---|---|
| The CAD gripper, the old 12 mm stroke | 7 / 40: 62 lost seals | 13 mm | 0 mm |
| + 20 mm stroke, no cup on a fin, the cups lean onto the plane through them | 34 / 40 | 15 mm | 37 mm toward the tail |
| + back from the balance point in 10 mm steps (each scene's camera noise from its seed from here) | 34 / 40 | 13 mm | 32 mm |
| + stand-in width limit 0.25 m, refused plans don't pause the run | **37 / 40** | 13 mm | 31 mm |
| + the bellows' 30 mm (40 mm to about 10), the wrist rules: a held fish turned at most 180 deg, the empty gripper within 90 of the middle where it can, no move over 180 | 35 / 40 (two fish vision never picked flip between runs) | 13 mm | 31 mm |

The 0.55 m small fish (`--fish-length 0.55 --placement view --any-size`) with the stand-in's own balance share: 6 of 20 with the old gripper (lost on the lift), 19 of 20 now, the search backing off from the fins by itself. Two fish a scene (`--fish 2`, 20 scenes): 23 of 40 placed (16 with the old slots), no move refused for the wrist.

## Found with it

- **The cups bottomed out on the fins:** with a 12 mm stroke one cup sat on a printed fin 25 mm above the body and the others never touched; vision took each cup's median depth, which hides a fin. Each cup's lip now meets the highest surface under it, against the plane through the cups.
- **Two fish a scene showed what one never did:** slot 2 was out of reach at the carry height, the tool being up to 126 mm past the slot (the cups hold a fish ahead of its middle). `check_layout` now allows for it; with two fish a scene, carries that found no joint solution fell from 28 (old slots) to none. The second fish is still often left cut off at the image's edge: the weakest part with several fish.
- **Tried and removed:** moving a closer look toward the end cut off (no full-size fish more, five of 20 small fish fewer); refusing plans whose predicted wrist move passes the joint travel limit (the predictions are up to 13 deg off: it refused plans that ran).

- **Observation poses face −90°:** at +90° the FR5's j6 needed about 170°, past the tube's range; at −90° it's about −15°.
- **The easy layouts hid the main problem:** fish laid only where a 450 mm view sees all of them hid that a full-size fish lying anywhere almost never fits one view. Closer looks centered on the fish (`observation.close_height`, 0.62 m: raised fins fill more of the image the nearer they are) fixed it.
- **Vision measures the printed fish about 14% long** (its raised ends) and 174 mm tall: hence size limits per fish type.
- **It balances 98 mm ahead of its outline's middle,** outside the 140 mm cup array centered there; on a smaller fish the array reaches the pectoral fins before the balance point (all six cups seal up to 0.094 of the length ahead at 0.79 m, 0.062 at 0.55 m), and depth doesn't see the thin fin. Hence a share per fish type, short of the fins: 0.55 m printed fish with the 0.79 m stand-in's 0.126 placed 6 of 20 (lost on the lift), with 0.06 all 20, the cups 39 mm behind the balance point instead of 77.
- **A straight carry can wind the wrist into its stop** where the planned turn goes the other way round: carried point-to-point instead.
- **Placed heading is a steady 5° off** the model's axis (the outline's main axis leans with the fins), the same for every fish.
- **Wrist:** j6 has one angle per tool angle, so a plan only chooses which way round the cups hold the fish. Placed at heading 0 the wrist ends at about -30 or 150 deg; a known-head fish that must arrive the far way round, about one in six, needs the empty gripper past 90 deg (up to ~150) for its held turn to stay within 180 deg: the other way would turn it ~245 deg, past the range. With a slot's two angles at +-90 none would (`check_layout` names the heading). Closer looks that took whichever way turned least from where the wrist was made the arm turn round to look and back to grab; they now take the way round nearest the middle, as the grasp does.

## Other simulators

- **MoveIt with mock hardware in RViz** (`--mode sim`, robot PC): the real ROS and MoveIt path; no camera, fish or vacuum. RViz can show camera images others publish but renders none.
- **FAIRINO SimMachine** (the controller software in a VMware image): the real controller, for the driver, the command server, ServoJ, joint limits and error codes; no camera, fish or vacuum. The best stand-in for the arm: point frcobot_ros2 at the VM.
- **Gazebo:** physics and depth cameras on ROS topics, but this code reads the camera through pyrealsense2, and suction needs a custom plugin.
