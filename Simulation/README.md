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

**Layout editor** (`--mode setup`, `rfp_sim/layout_editor.py`): the same arm and gripper with the workcell's table, pick area and placement zone (corner balls to drag), slots (balls; an added one goes where a whole fish fits), keep-out zones (a ball each; the operator's side goes from table edge to table edge) and the camera's views (they follow the pick area), checked as they change (`rfp_arm/layout.py`); see the [top README](../README.md#lay-it-out-in-3d).

**3D window:** also the vacuum tube, coiled round the wrist by as much as j6 has turned, the table's body, and j6 against the tube's range. The gripper is its CAD model (`models/gripper.stl`, from `import_gripper.py`); the camera's own view leaves out its housing.

**Twin of the real arm** (`rfp_sim/twin.py`, the moving and observing modes on the robot PC): the same arm and gripper following the real arm's joint states, the cell's zones and keep-out zones, each fish vision finds (outline; for a fish that can be picked the six cups where the grasp puts them and its heading) and the camera's depth as points colored by height. It only reads; `run.twin: false` in the control panel leaves it out.

**Motion, as on the arm:** every move is timed as MoveIt's Pilz planner times it on the FR5: it speeds up and slows down at the joints' limits ([Arm/ros/joint_limits.yaml](../Arm/ros/joint_limits.yaml), times the workcell's ptp scales) or, on straight moves, at the move's speed and `tcp_accel_max`, and is slowed wherever any point of the arm would pass `tcp_speed_max`. At constant speed, as before, a 20° move took 0.37 s here and takes 1.4 s on the arm, so the empty arm looked much faster than it will be. Lift, carry and lower run as one blended move with rounded corners (`VirtualArm.move_through`), as the arm runs them with Pilz's sequence: one at a time, the arm stopped at every waypoint of a carry. The camera's view is a pyramid from the lens down to the table, here and in the twin.

**Contact:** each cup is a ray along the tool axis, touching within 2 mm of its face and pushing back at 500 N/m when compressed (what the arm feels while pressing). The vacuum holds the fish only when 5 of 6 cups touch. A held fish follows the cups once they lift and lies flat where it's let go; no other physics. `drop: lift` or `carry` loses the fish on the way, to test drop detection.

**Fish placement:** `placement: anywhere` (default) lays each fish anywhere in the pick area at any heading, as real fish lie; `view` only where an observation pose sees all of it along the image's long side, the easy case.

## Benchmark

`python Simulation/benchmark.py` runs the real cycle on seeded scenes with nobody at the console and scores each fish against the simulator's truth (`rfp_sim/benchmark.py`): where it ended up, its middle's distance from its slot, its heading error, where the cups took it relative to its true center of mass (the model's volume centroid), failures with their messages, looks, wrist travel, simulated time. Default: 20 scenes, one stand-in of the simulation's `fish_length` (0.60 m, about the real fish; the big prints are 0.79) lying anywhere, the trained model. `--compare <earlier run>` shows the difference; `--any-size`, `--no-hold-check` (a vacuum switch confirms holds; faster), `--placement view`, `--fish-length`, `--detector truth`. The same seed gives the same scenes, so two runs differ by code and settings alone; each behavior has a setting to switch it off (`focus_steps: 1`, `carry_over_seen: false`, `measure_table: false`, `balance_ahead_share: 0`).

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
| + the bellows' 30 mm (40 mm to about 10), the wrist rules: a held fish turned at most 180 deg, the empty gripper within 90 of the middle where it can, no move over 180 | 35-37 / 40 (two fish vision never picks flip between runs) | 13 mm | 31 mm |

The 0.55 m small fish (`--fish-length 0.55 --placement view --any-size`) with the stand-in's own balance share: 6 of 20 with the old gripper (lost on the lift), 19 of 20 now, the search backing off from the fins by itself. Two fish a scene (`--fish 2`, 20 scenes): 23 of 40 placed (16 with the old slots), no move refused for the wrist. The largest single wrist move: 173 deg with one fish a scene, 177 with two (at most 180 planned, 200 allowed).

The real cell's layout (turned: pick area beside the arm, placement zone to its right), two fish a scene, 20 scenes (`--fish 2`):

| | Placed | Failures | Scenes ending "pick area clear" | Simulated s per placed fish |
|---|---|---|---|---|
| 0.79 m prints, before (fixed looks, fixed lift) | 21 / 40 | | | 65.0 |
| + one focus look over each fish, as high as it needs; carry with the wrist turning evenly | 32 / 40 | 2 drops, 3 refused moves | 9 | 26.9 |
| 0.60 m fish (as the real ones), the same code | 38 / 40 | 4 drops (false), 1 refused move | 13 | 23.8 |
| + hold check knows the table under a thin fish; mount check needs a jump of the camera's whole move | 38 / 40 | 1 refused move | 17 | 25.3 |
| + the arm keeps its side of the base; the arm isn't a fish; re-check timed by the decision alone | 39 / 40 | none | 19 | 24.5 |
| + the whole plan checked before anything moves; looks-only checks advisory; the pick area's edge within 3 cm (pick area 2 cm nearer the arm) | 40 / 40 | none | 20 | 24.4 |
| Moves timed as the arm's planner times them (speeding up and slowing down; the times above are too short) | 40 / 40 | none | 20 | 33.0 |
| + one look over the pick area instead of two; lift, carry and lower as one blended move | 40 / 40, 39 head end right | none | 20 | 25.1 |
| + an unclear head end placed as vision guessed, not the way that turns the wrist least | **40 / 40**, all head end right | none | **20** | **25.1** |

With the last row the search takes 3.6 looks per fish (5.2 with two looks) and the wrist turns 219° per fish (329); the median slot error is 3 mm (at most 10), the heading 5° (the outline's lean). The 0.79 m prints with the same code: 35 of 40 placed and 17 of 20 scenes clear (33 and 12 before this round), 5.1 looks and 291° of wrist per fish (7.3 and 470); the rest is the layout's margins for so long a fish: a held fish's end would reach the operator's zone (2 scenes), and two picked at the pick area's edge were set down over 5 cm off slot 2, which is 5 cm inside the placement zone's edge. Two benchmarks run side by side used to place different fish (the re-check before a grasp counted vision's own time); now only the fish list's age limit (1 s, CS-53) depends on how busy the computer is, and the GPU's model can differ by a pixel between runs, which tips a fish lying right at a limit.

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
