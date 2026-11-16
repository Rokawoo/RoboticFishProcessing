# ControlSys

Runs each pick-and-place cycle: checks Vision's fish list, has the selector (later the VLM) choose a fish, validates the choice, runs the Arm's grasp sequence, handles failures, logs. Requirement Spec **Section 2.4 (CS)** and the run behavior of **2.1 (SY)**, methods in Appendix C.8 and C.9.

## Layout

| Folder | Contents |
|---|---|
| `rfp_controlsys/` | `pick_place.py` (the script), `cycle.py` (state machine, input checks, failure handling), `selector.py` (which fish next), `console.py` (operator interface), `run_log.py`, `demo.py` |
| `prompts/` | VLM prompt templates |
| `configs/` | `control.yaml`: attempt limits, timeouts, hold confirmation |
| `runs/` | One folder per run: `cycles.csv`, each decision's image, `report.json` (not in Git) |

## Running

In the [simulation](../README.md#run-it-in-the-3d-simulation) and [on the arm](../README.md#run-it-on-the-arm); `pick_place.py --help` lists every mode. On a recording: `pick_place.py --mode offline --device clip.mp4 --any-size --cycles 2` (`--any-size`: skip the size checks, for other cameras). Every mode runs the same cycle code; only the arm, gripper and camera behind the interfaces change (CS-90).

One cycle: go to the observation pose, get a fish list, let the selector choose, validate the choice, look again from the same pose (CS-41), run the grasp sequence, answer a failure as in Table CS-1, log a CSV row.

**Taught demos:** `demo.py` plays hand-taught poses with vacuum steps between them (no camera) on the simulated arm, in RViz or on the real arm, waiting for Enter at each step unless `--auto`. `demo.py record <name>` teaches a new one from the arm's joints. Files: [Demos/](../Demos/) (`fish_pickup.yaml`: the pickup taught on the arm).

**Adding the VLM:** Qwen3-VL-8B served by vLLM (OpenAI-compatible API) answers JSON `{action, fish_id, reason}`. Write a class with `choose(context) -> Decision` (`selector.py`; the context holds the annotated image, the candidates and the last result, Table IF-5) and pass it to `PickPlaceCycle` instead of `RuleSelector`. Its answers already go through `validate_decision`, one retry and the fallback rule: it can only choose among candidates (CS-30).

Not done yet: the VLM (CS-20 to CS-27), payload update (CS-45), rosbag logging (CS-84), an operator interface beyond the terminal and the camera window (CS-70).
