# ControlSys

Runs each pick-and-place cycle: checks the fish list from Vision, asks the VLM which fish to pick, validates the answer, commands the Arm, handles failures and logs every cycle.

Implements Requirement Spec **Section 2.4 Control System (CS)** and owns the run and cycle behavior of **Section 2.1 System (SY)**; methods are in Appendix C.8 and C.9. The interfaces between the three subsystems are in Section 2.7.

## Layout

| Folder | Contents |
|---|---|
| `rfp_controlsys/` | Control node: state machine, input checks, VLM client, validation, failure handling, logging, operator interface |
| `prompts/` | VLM prompt templates |
| `configs/` | Fixed limits, timeouts, attempt limits and model configuration |
| `tests/` | Tests |

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
