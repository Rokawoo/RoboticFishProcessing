# ControlSys

Runs each pick-and-place cycle: checks Vision's fish list, asks the VLM which fish to pick, validates the answer, commands the Arm, handles failures, logs the cycle.

Implements Requirement Spec **Section 2.4 Control System (CS)** and owns the run and cycle behavior of **Section 2.1 System (SY)**; methods are in Appendix C.8 and C.9. The interfaces between the three subsystems are in Section 2.7.

## Layout

| Folder | Contents |
|---|---|
| `rfp_controlsys/` | `pick_place.py` (the script), `cycle.py` (state machine, input checks, failure handling), `selector.py` (which fish next: fixed rules now, the VLM later), `console.py` (operator interface), `run_log.py` |
| `prompts/` | VLM prompt templates |
| `configs/` | `control.yaml`: attempt limits, timeouts, hold confirmation |
| `runs/` | One folder per run: `cycles.csv`, the observation image of each cycle, `report.json` (not in Git) |

## Running pick and place

```bash
python ControlSys/rfp_controlsys/pick_place.py --mode observe      # camera, detections, fish positions
python ControlSys/rfp_controlsys/pick_place.py --mode offline --device clip.mp4 --any-size --cycles 2
python ControlSys/rfp_controlsys/pick_place.py --mode sim --cycles 2   # MoveIt's mock hardware, RViz
python ControlSys/rfp_controlsys/pick_place.py --mode check        # real arm: state, I/O, camera
python ControlSys/rfp_controlsys/pick_place.py --mode dry-run --step   # THE ARM MOVES, vacuum off
python ControlSys/rfp_controlsys/pick_place.py --mode run          # THE ARM MOVES, with vacuum
```

`observe` and `offline` run anywhere (Windows too). The others need ROS 2 Humble with MoveIt running (`ros2 launch fairino5_v6_moveit2_config demo.launch.py`), and on the real arm the command server (`ros2 run fairino_hardware ros2_cmd_server`); the script's docstring has the details. The same cycle code runs in every mode: only the arm, gripper and camera behind their interfaces change (CS-90).

One cycle: move to the next observation pose, get a fish list, let the selector choose, validate the answer, look again from the same pose (CS-41), run the grasp sequence, answer a failure as in Table CS-1, log a CSV row.

**Adding the VLM:** write a class with `choose(context) -> Decision` (`selector.py`: the context holds the annotated image, the candidates and the last result, as in Table IF-5) and pass it to `PickPlaceCycle` in place of `RuleSelector`. Its answers already go through `validate_decision`, one retry and the fallback rule.

## Responsibilities

- **Inputs (CS-10):** check the fish list and use the next observation pose for each new observation
- **VLM decisions (CS-20):** send an annotated image and the candidate fish to Qwen3-VL-8B (served locally by vLLM through an OpenAI-compatible API); the answer is JSON `{action, fish_id, reason}`
- **Validation and fixed rules (CS-30):** reject any answer that is not a valid candidate fish, and fall back to a fixed rule when the VLM fails; the VLM only chooses among candidates, and fixed code owns every limit
- **Grasp commands (CS-40):** run the Arm's grasp sequence and assign placement slots
- **Timing (CS-50):** answer timeouts and the staleness of the fish list
- **Failure handling (CS-60):** a defined response to every failure: retry, skip the fish or pause for the operator
- **Operator interface (CS-70):** start, pause, resume and a stop control that is always visible
- **Logging (CS-80):** every topic in rosbag2 (MCAP), plus one CSV row per cycle
- **Test modes (CS-90):** replay of recorded observations, simulation with mock hardware and RViz 2, and dry run on the real arm with the vacuum off

Not done yet: the VLM (CS-20 to CS-27), payload update (CS-45), rosbag logging (CS-84), an operator interface beyond the terminal and the camera window (CS-70).
