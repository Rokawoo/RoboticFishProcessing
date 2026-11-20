# ControlSys

Runs each pick-and-place cycle: checks Vision's fish list, has the selector (later the VLM) choose a fish, validates the choice, runs the Arm's grasp sequence, handles failures, logs. Requirement Spec **Section 2.4 (CS)** and the run behavior of **2.1 (SY)**, methods in Appendix C.8 and C.9.

## Layout

| Folder | Contents |
|---|---|
| `rfp_controlsys/` | `pick_place.py` (the script), `cycle.py` (state machine, input checks, failure handling), `selector.py` (which fish next), `calibrate.py` (the camera's mount from small moves: dry-run and run do it first while it isn't calibrated, `--calibrate` again), `console.py` (operator interface), `run_log.py`, `demo.py` |
| `prompts/` | VLM prompt templates |
| `runs/` | One folder per run: `cycles.csv`, each decision's image, `report.json` (not in Git) |

## Running

In the [simulation](../README.md#run-it-in-the-3d-simulation) and [on the arm](../README.md#run-it-on-the-arm); `pick_place.py --help` lists every mode. What vision makes of a recording: `pick_place.py --mode check --device clip.mp4 --any-size` (`--any-size`: skip the size checks, for other cameras). Every mode runs the same cycle code; only the arm, gripper and camera behind the interfaces change (CS-90).

One cycle (`cycle.py`):

- **Search:** the observation poses in turn, a quick look (`search_frames`) from each, until one sees a fish. Every fish seen is remembered (`Sighting`: where its middle is as best known, its axis, length, why it couldn't be picked), so after a place the arm goes to the next it knows of, and a fish is never left for another view while it can still be worked on.
- **Focus:** the camera straight over the fish's middle, level, the image's long side along it, as high as `look_margin` times its expected length needs (between `observation.height` and `close_height`), the way round nearest the wrist's middle; again (up to `focus_steps`) while it is cut off at the image's edge, its middle then half its expected length from the end that shows and the look higher (it didn't fit the last). `max_looks_per_fish` looks without a candidate: set aside; all set aside: the operator is told why, fish by fish, with what vision measured.
- **Mount check:** a fish seen whole from two places may not jump by much and by the camera's move or more (`max_jump_m`; skipped when another fish lies near where it was expected): the camera isn't where the workcell has it (turned on its mount), the run stops and says to calibrate.
- **One command on the arm:** real modes check that MoveIt drives this arm (its joints and the controller's agree: not a MoveIt left running with mock hardware) and that FAIRINO's driver was built for the controller's firmware; `dry-run` and `run` calibrate the camera first while `camera.calibrated` is false.
- **Checked before anything moves:** the grasp plan, pose by pose from the arm's joints (`grasp.plan_problem`: a joint solution in its posture, the arm clear of itself): a pick that would have to stop half way isn't begun; the fish is tried again from a new look, then skipped, and the run goes on.
- **The arm isn't a fish:** what a look sees where the arm's own body is between the camera and the table (the shoulder, looking near the base) isn't remembered as a fish.
- **Decide:** the selector chooses, its answer validated; the look is taken again before the grasp (CS-41) only if the decision took longer than `recheck_after_s`.
- **Grasp:** the carry height from what lies on the way (`carry_over_seen`: other fish seen, fish placed already), the grasp sequence, a failure answered as in Table CS-1, a CSV row.

While the arm moves the camera window shows what vision finds every `live_view_s`; in the moving modes a 3D twin of the arm (Simulation/rfp_sim/twin.py) shows it too, with the depth as points.

**Taught demos:** `demo.py` plays hand-taught poses with vacuum steps between them (no camera) on the simulated arm, in RViz or on the real arm, waiting for Enter at each step unless `--auto`. `demo.py record <name>` teaches a new one from the arm's joints. Files: [Demos/](../Demos/) (`fish_pickup.yaml`: the pickup taught on the arm).

**Adding the VLM:** Qwen3-VL-8B served by vLLM (OpenAI-compatible API) answers JSON `{action, fish_id, reason}`. Write a class with `choose(context) -> Decision` (`selector.py`; the context holds the annotated image, the candidates and the last result, Table IF-5) and pass it to `PickPlaceCycle` instead of `RuleSelector`. Its answers already go through `validate_decision`, one retry and the fallback rule: it can only choose among candidates (CS-30).

Not done yet: the VLM (CS-20 to CS-27), payload update (CS-45), rosbag logging (CS-84), an operator interface beyond the terminal and the camera window (CS-70).
