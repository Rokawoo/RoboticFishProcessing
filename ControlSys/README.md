# ControlSys

Runs each pick-and-place cycle: checks Vision's fish list, has the selector (later the VLM) choose a fish, validates the choice, runs the Arm's grasp sequence, handles failures, logs. Requirement Spec Section 2.4 (CS) and the run behavior of 2.1 (SY).

| Folder | Contents |
|---|---|
| `rfp_controlsys/` | `pick_place.py` (the program; `--help` lists the modes), `cycle.py` (the cycle), `selector.py` (which fish next), `calibrate.py` (the camera's mount), `console.py` (operator), `run_log.py`, `demo.py` |
| `prompts/` | VLM prompt templates |
| `runs/` | One folder per run: `cycles.csv`, each decision's image, `report.json` (not in Git) |

Running it: [the simulation](../README.md#run-it-in-the-3d-simulation), [the arm](../README.md#run-it-on-the-arm). Every mode runs the same cycle; only the arm, gripper and camera behind it change. What vision makes of a recording: `pick_place.py --mode check --device clip.mp4 --any-size`.

One cycle (`cycle.py`):

- **Search:** the observation poses in turn until one sees a fish. Every fish seen is remembered, so after a place the arm goes straight to the next.
- **Focus:** the camera straight over the fish, high enough to see all of it, again while it's cut off at the image's edge. A fish that can't be picked after `max_looks_per_fish` looks is set aside; the operator is told why.
- **Checks:** a fish that seems to jump as the camera moves means the camera isn't mounted where the workcell says: the run stops. The whole grasp plan is checked before anything moves, so no pick stops half way.
- **Decide:** the selector chooses among the candidates; its answer is validated.
- **Grasp:** a table measured more than `table_tolerance_m` off `table.z` next to the fish stops the run first (the cups' depth would be off by as much); carried just over what lies on the way; a failure is retried, skipped or handed to the operator; one CSV row.
- **Slots** are used in turn: each placed fish is taken away before its slot comes round again.

**Taught demos:** `demo.py` plays hand-taught poses with vacuum steps (no camera), on the simulated or the real arm; `demo.py record <name>` teaches one from the arm's joints. Files: [Demos/](../Demos/).

**Adding the VLM:** a class with `choose(context) -> Decision` (`selector.py`), passed to `PickPlaceCycle` instead of `RuleSelector`. Its answers go through `validate_decision`, one retry and the fallback rule: it can only choose among the candidates.

Not done yet: the VLM, payload update, rosbag logging, an operator interface beyond the terminal and the camera window.
