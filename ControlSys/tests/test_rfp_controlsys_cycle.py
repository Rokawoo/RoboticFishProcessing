"""Tests for the pick-and-place cycle on a simulated arm, gripper and table of fish:
the selector rules, every failure response of Table CS-1, the log and the script's checks."""

import csv
import json
import math
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from rfp_arm.gripper import NoGripper, SimGripper
from rfp_arm.interfaces import E_PROTECTIVE_STOP, ArmError, GripperState
from rfp_arm.sim import SimArm
from rfp_arm.workcell import load_workcell
from rfp_controlsys import pick_place
from rfp_controlsys.console import Console
from rfp_controlsys.cycle import PickPlaceCycle
from rfp_controlsys.run_log import RunLog
from rfp_controlsys.selector import (
    Decision,
    DecisionContext,
    RuleSelector,
    fallback_choice,
    validate_decision,
)
from rfp_controlsys.settings import ControlSettings, load_settings
from rfp_vision.camera import Frame
from rfp_vision.fish_list import Fish, FishList
from rfp_vision.geometry import Intrinsics, pose_down
from rfp_vision.vision_system import Observation

CELL = load_workcell()
K = Intrinsics.from_fov(320, 180, 87.0)


def fish(fish_id, x, y, heading_deg=90.0, confidence=0.9, code="none", overlap=False, known=True):
    ok = code == "none"
    pose = pose_down((x, y, 0.10), math.radians(heading_deg)) if ok else None
    outline = np.array([[100, 60], [220, 60], [220, 120], [100, 120]], float)
    heading = math.radians(heading_deg) if ok else None
    return Fish(fish_id, ok, code, confidence, pose, pose, heading, 0.60, 0.12, 0.10, None,
                overlap, outline, heading_known=known)  # fmt: skip


class Hold:
    """Stands in for the vision hold monitor."""

    def __init__(self):
        self.cues, self.lost_at, self.updates, self.active = ["depth"], None, 0, False

    def set_background(self, frame):
        pass

    def start(self, frame):
        self.active, self.updates = bool(self.cues), 0
        return self.cues

    def update(self, frame):
        self.updates += 1
        gone = self.lost_at is not None and self.updates >= self.lost_at
        return type("Status", (), {"dropped": gone, "__str__": lambda s: "not seen"})()

    def stop(self):
        self.active = False


class Table:
    """Fish lying in {B}, and the vision system that sees them: (x, y) or (x, y, options)."""

    def __init__(self, *fish_at):
        self.fish = [(*f, {}) if len(f) == 2 else f for f in fish_at]
        self.hold = Hold()
        self.frames = 0
        self.observations = 0
        self.before_observation = {}  # observation number -> action

    def frame(self) -> Frame:
        self.frames += 1
        now = time.monotonic()
        return Frame(np.zeros((180, 320, 3), np.uint8), None, K, now, time.time(), self.frames)

    def observe(self, t_base_camera, on_frame=None) -> Observation:
        self.observations += 1
        if action := self.before_observation.get(self.observations):
            action()
        frame = self.frame()
        seen = [fish(i + 1, x, y, **options) for i, (x, y, options) in enumerate(self.fish)]
        fish_list = FishList(frame.frame_id, frame.stamp, frame.wall_time, t_base_camera, seen)
        return Observation(fish_list, frame, [])


class TableGripper(SimGripper):
    """Picks up the fish under the arm, and takes it off the table when it is placed."""

    def __init__(self, table: Table, arm: SimArm, seals: bool = True):
        super().__init__(seals)
        self.table, self.arm, self.carried = table, arm, None

    def vacuum_on(self) -> None:
        super().vacuum_on()
        x, y = self.arm.tcp_pose()[:2, 3]
        near = [f for f in self.table.fish if math.hypot(f[0] - x, f[1] - y) < 0.05]
        self.carried = near[0] if near and self.seals else None

    def release(self) -> None:
        placed = self.sealed and CELL.placement_zone.contains(self.arm.tcp_pose()[:2, 3])
        if placed and self.carried in self.table.fish:
            self.table.fish.remove(self.carried)
        self.carried = None
        super().release()


class Clock:
    """Time that passes only when the simulated arm moves: 50 ms per camera frame."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Operator(Console):
    """A console that answers alerts from a list (no answer left: stop). `at_step` runs an
    action when a grasp step starts."""

    def __init__(self, clock: Clock, answers=()):
        super().__init__(window=False, interactive=False)
        self.clock = clock
        self.answers, self.alerts, self.states = list(answers), [], []
        self.at_step = {}  # Table AS-1 step name -> action
        self.stop = False

    def state(self, state, detail=""):
        self.states.append(state)
        if action := self.at_step.get(detail):
            action()

    def info(self, text):
        self.states.append(text)

    def alert(self, message):
        self.alerts.append(message)
        return self.answers.pop(0) if self.answers else False

    def show(self, image, lines=None):
        self.clock.now += 0.05

    def stop_requested(self):
        return self.stop


class Rig:
    def __init__(self, *fish_at, answers=(), gripper=None, selector=None, log=None, **options):
        self.table = Table(*fish_at)
        self.arm = SimArm(CELL)
        self.gripper = gripper or TableGripper(self.table, self.arm)
        self.clock = Clock()
        self.operator = Operator(self.clock, answers)
        settings = options.pop("settings", ControlSettings(settle_s=0.0))
        options.setdefault("vision_hold", False)
        self.cycle = PickPlaceCycle(
            self.arm, self.gripper, self.table, selector or RuleSelector(), CELL, settings,
            self.operator, log, clock=self.clock, sleep=self.clock.sleep, **options,
        )  # fmt: skip

    def run(self, max_cycles=None):
        return self.cycle.run(max_cycles)


A, B, C = (0.30, -0.10), (0.50, -0.10), (0.40, 0.10)


def once(action):
    """`action`, but only the first time it is called."""
    calls = []

    def run():
        if not calls:
            calls.append(1)
            action()

    return run


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
def test_answers_are_validated_before_use():
    candidates = [fish(1, *A), fish(2, *B)]
    assert validate_decision(Decision("pick", 2), candidates) is None
    assert validate_decision(Decision("reobserve"), candidates) is None
    assert validate_decision(Decision("end"), candidates) is None
    assert "not a candidate" in validate_decision(Decision("pick", 3), candidates)
    assert "without a fish ID" in validate_decision(Decision("pick", None), candidates)
    assert "without a fish ID" in validate_decision(Decision("pick", "2"), candidates)
    assert "unknown action" in validate_decision(Decision("grab", 1), candidates)
    assert validate_decision({"action": "pick", "fish_id": 1}, candidates) == "not a decision"


def test_fixed_rules_choose_a_free_far_confident_fish():
    crowded = [fish(1, 0.30, -0.10, overlap=True, confidence=0.99), fish(2, 0.50, -0.10)]
    assert fallback_choice(crowded).fish_id == 2  # CS-61: the most confident free fish
    assert (
        fallback_choice(crowded[:1]).fish_id == 1 and fallback_choice(crowded).source == "fallback"
    )
    assert fallback_choice([]).action == "reobserve"

    def choose(candidates, **context):
        fish_list = FishList(1, 0.0, 0.0, np.eye(4), candidates)
        return RuleSelector().choose(DecisionContext(fish_list, candidates, **context))

    assert choose(crowded).fish_id == 2  # Table CS-2, goal 3
    row = [fish(1, 0.30, -0.10), fish(2, 0.36, -0.10), fish(3, 0.55, -0.10, confidence=0.6)]
    assert choose(row).fish_id == 3  # goal 4: farthest from every other fish
    assert choose(row, last_failed_id=3).fish_id in (1, 2)  # goal 5: another fish first
    assert choose(row[2:], last_failed_id=3).fish_id == 3  # unless it is the only one
    assert choose([]).action == "reobserve"
    assert validate_decision(choose(row), row) is None


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------
def test_run_places_every_fish_then_reports_the_pick_area_clear(tmp_path):
    log = RunLog({"mode": "test", "model_version": "m1"}, tmp_path)
    rig = Rig(A, B, log=log)
    report = rig.run()
    log.close(report.as_record())
    assert report.ended == "pick area clear" and report.cycles == 2 and report.successes == 2
    assert rig.table.fish == [] and rig.gripper.events == ["vacuum on", "release"] * 2
    assert rig.operator.alerts == [] and report.failures == {} and report.skipped == []
    # SY-22: three passes over the two observation poses without a fish
    assert rig.table.observations == 2 * 2 + 3 * 2
    for state in ("observing", "deciding", "approaching", "holding", "carrying", "placing", "idle"):
        assert state in rig.operator.states  # CS-72
    with open(log.folder / "cycles.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["result"] for row in rows] == ["success", "success"]
    assert [row["slot"] for row in rows] == ["1", "2"] and rows[0]["decision_source"] == "rule"
    assert float(rows[0]["grasp_z_mm"]) == 100.0 and rows[0]["heading_known"] == "True"
    summary = json.loads((log.folder / "report.json").read_text(encoding="utf-8"))
    assert summary["successes"] == 2 and summary["model_version"] == "m1"
    assert (log.folder / "cycle_001.jpg").is_file()  # the observation the decision used (CS-82)


def test_only_valid_fish_inside_the_pick_area_are_candidates():
    rig = Rig((0.30, -0.10, {"code": "low_confidence"}), (0.40, 0.25))  # the 2nd: outside
    report = rig.run()
    assert report.cycles == 0 and rig.gripper.events == []
    assert "none can be picked: fish 1: low_confidence" in rig.operator.alerts[0]
    assert report.ended.startswith("stopped at: Fish are on the table")


def test_no_seal_is_retried_then_the_fish_is_skipped():
    rig = Rig(A, answers=[True])
    rig.gripper.seals = False
    report = rig.run()
    assert report.failures == {"E_NO_SEAL": 3} and report.cycles == 3 and report.successes == 0
    assert len(report.skipped) == 1 and "x 300, y -100" in report.skipped[0]  # CS-62
    assert "3 cycles failed in a row" in rig.operator.alerts[0]  # CS-63, acknowledged
    assert "attempt limit reached" in rig.operator.alerts[1]  # nothing left to pick: stop
    assert rig.gripper.state() == GripperState.VACUUM_OFF and rig.table.fish  # still on the table
    assert rig.arm.tcp_pose()[2, 3] > 0.15  # retreated from the fish each time


def test_a_dropped_fish_inside_the_pick_area_is_picked_again():
    rig = Rig(A)
    rig.operator.at_step["lift"] = once(rig.gripper.drop)  # still over the pick area
    report = rig.run()
    assert report.failures == {"E_DROP": 1} and report.successes == 1 and report.cycles == 2
    assert rig.operator.alerts == [] and rig.table.fish == []


def test_a_fish_dropped_outside_the_pick_area_needs_the_operator():
    rig = Rig(A)

    rig.operator.at_step["lower"] = once(rig.gripper.drop)  # above its slot
    report = rig.run()
    assert report.failures == {"E_DROP": 1}
    assert "dropped outside the pick area" in rig.operator.alerts[0]
    assert report.ended.startswith("stopped at")


def test_vision_reports_a_drop_when_no_switch_can():
    class Blind(TableGripper):
        has_switch = False

    rig = Rig(A, vision_hold=True)
    rig.gripper = rig.cycle.gripper = Blind(rig.table, rig.arm)
    rig.table.hold.lost_at = 6
    report = rig.run(max_cycles=1)
    assert report.failures == {"E_DROP": 1} and rig.gripper.state() == GripperState.VACUUM_OFF
    assert any("vision: the fish is gone" in str(s) for s in rig.operator.states)

    rig = Rig(A, vision_hold=True)
    rig.gripper = rig.cycle.gripper = Blind(rig.table, rig.arm)
    rig.table.hold.cues = []  # the camera can't tell here, and there is no switch
    assert rig.run(max_cycles=1).failures == {"E_NO_SEAL": 1}
    lenient = ControlSettings(settle_s=0.0, require_hold_confirmation=False)
    blind = Rig(A, vision_hold=True, settings=lenient)
    blind.gripper = blind.cycle.gripper = Blind(blind.table, blind.arm)
    blind.table.hold.cues = []
    assert blind.run(max_cycles=1).successes == 1  # lifting blind, when the settings allow it


def test_arm_faults_stop_the_run_until_the_operator_resets():
    rig = Rig(A)
    moves = []

    def protective_stop(pose, speed, accel, tick=None):
        moves.append(pose)
        raise ArmError(E_PROTECTIVE_STOP, "emergency stop pressed")

    rig.arm.move_lin = protective_stop
    report = rig.run()
    assert len(moves) == 1 and report.failures == {"E_PROTECTIVE_STOP": 1}
    assert "Reset the arm" in report.ended and rig.operator.alerts == []


def test_unreachable_fish_are_skipped_and_the_others_picked():
    rig = Rig(A, B)
    rig.arm.fail = lambda kind, target: (
        "no IK solution" if kind == "ptp" and abs(target[0, 3] - A[0]) < 0.01 else None
    )
    rig.table.before_observation[14] = lambda: rig.table.fish.clear()  # then end the run
    report = rig.run()
    assert report.failures == {"E_MOTION": 3} and report.successes == 1
    assert len(report.skipped) == 1 and B not in [f[:2] for f in rig.table.fish]


def test_a_full_placement_zone_pauses_the_run():
    rig = Rig(A, B, C, answers=[True])
    report = rig.run()
    assert report.successes == 3 and "placement zone is full" in rig.operator.alerts[0]
    stopped = Rig(A, B, C).run()
    assert stopped.successes == 2 and stopped.ended.startswith("stopped at: The placement zone")


class Scripted:
    """A selector that gives prepared answers, then behaves like the rules."""

    def __init__(self, *answers):
        self.answers, self.contexts = list(answers), []

    def choose(self, context):
        self.contexts.append(context)
        answer = self.answers.pop(0) if self.answers else RuleSelector().choose(context)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_bad_selector_answers_never_move_the_arm_and_the_fallback_decides():
    selector = Scripted(Decision("pick", 7), Decision("fly", 1))  # two invalid answers (CS-32)
    rig = Rig(A, selector=selector)
    report = rig.run(max_cycles=1)
    assert report.successes == 1 and len(selector.contexts) == 2
    assert any("fallback: pick fish 1" in str(s) for s in rig.operator.states)
    context = selector.contexts[0]
    assert context.image.shape == (180, 320, 3) and context.gripper_state == "vacuum off"
    assert [f.fish_id for f in context.candidates] == [1] and context.attempts == {1: 0}

    crashing = Scripted(
        TimeoutError("no answer in 10 s"), Decision("pick", 1, "after a retry", "vlm")
    )
    rig = Rig(A, selector=crashing)
    assert rig.run(max_cycles=1).successes == 1  # asked again once (Table CS-1)
    assert any("vlm: pick fish 1" in str(s) for s in rig.operator.states)

    waiting = Scripted(*[Decision("reobserve", None, "not sure")] * 3)
    rig = Rig(A, selector=waiting)
    assert rig.run(max_cycles=1).successes == 1 and rig.table.observations >= 3  # then the rule

    ending = Scripted(Decision("end", None, "done"))
    rig = Rig(A, selector=ending)
    assert rig.run().ended.startswith("stopped at: The selector ended the run")
    assert rig.table.fish and rig.gripper.events == []


def test_a_fish_that_moved_before_the_grasp_gets_a_new_decision():
    rig = Rig(A)
    rig.table.before_observation[2] = lambda: rig.table.fish.__setitem__(0, (0.33, -0.10, {}))
    report = rig.run(max_cycles=1)
    assert report.successes == 1 and report.cycles == 1
    assert any("isn't where it was" in str(s) for s in rig.operator.states)
    approach = next(pose for kind, pose in rig.arm.moves if kind == "lin")
    assert np.allclose(approach[:2, 3], (0.33, -0.10))  # grasped where it is now


def test_last_failed_fish_and_attempts_reach_the_selector():
    selector = Scripted()
    rig = Rig(A, B, selector=selector, answers=[True, True])
    rig.gripper.seals = False
    rig.run(max_cycles=3)
    first, second, third = selector.contexts[:3]
    assert first.last_result == "" and first.last_failed_id is None
    assert second.last_result == "E_NO_SEAL" and second.last_failed_id is not None
    assert max(third.attempts.values()) == 1 and sum(third.attempts.values()) == 2  # one each


def test_cycle_limit_operator_stop_and_dry_run():
    assert Rig(A, B).run(max_cycles=1).ended == "cycle limit of 1 reached"
    rig = Rig(A)
    rig.operator.at_step["carry"] = lambda: setattr(rig.operator, "stop", True)
    report = rig.run()
    assert report.ended == "stopped: operator stop" and rig.cycle.sequence.holding
    assert rig.gripper.state() == GripperState.HOLDING  # the vacuum stays on: nothing is dropped

    dry = Rig(A, gripper=NoGripper(), dry_run=True)
    report = dry.run(max_cycles=2)
    assert report.successes == 2 and len(dry.table.fish) == 1  # moved through it, picked nothing
    lowest = min(pose[2, 3] for _, pose in dry.arm.moves)
    approach = next(pose for kind, pose in dry.arm.moves if kind == "lin")
    assert lowest == pytest.approx(0.115) and approach[2, 3] == pytest.approx(0.15)  # 50 mm short
    assert not dry.cycle.vision_hold


def test_a_stale_fish_list_starts_no_grasp():
    rig = Rig(A, settings=ControlSettings(settle_s=0.0, fish_list_max_age_s=-1.0))  # always stale
    report = rig.run()
    assert report.cycles == 0 and "keeps moving" in rig.operator.alerts[0]


# ---------------------------------------------------------------------------
# Settings, console, script
# ---------------------------------------------------------------------------
def test_control_settings_file_matches_the_defaults(tmp_path):
    assert load_settings() == ControlSettings()  # ControlSys/configs/control.yaml
    custom = tmp_path / "control.yaml"
    custom.write_text("max_attempts_per_fish: 2\n")
    assert load_settings(custom).max_attempts_per_fish == 2
    custom.write_text("max_attempts: 2\n")
    with pytest.raises(ValueError, match="max_attempts"):
        load_settings(custom)
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")


def test_console_without_an_operator_stops_at_alerts(capsys):
    console = Console(window=False, interactive=False, step=True)
    assert console.alert("fish dropped") is False and console.confirm_move("lift") is False
    console.state("observing", "pose 'left'")
    console.state("observing", "pose 'left'")  # printed once
    console.show(np.zeros((10, 10, 3), np.uint8))  # no window: nothing happens
    out = capsys.readouterr().out
    assert out.count("[observing] pose 'left'") == 1 and "fish dropped" in out
    assert not console.stop_requested() and Console(window=False, interactive=False).confirm_move(
        "x"
    )


def test_script_refuses_the_real_arm_until_the_workcell_is_verified(capsys):
    assert set(pick_place.MOVING_MODES) < set(pick_place.REAL_MODES) < set(pick_place.MODES)
    with pytest.raises(SystemExit) as stop:
        pick_place.start_checks("dry-run", CELL, None, NoGripper())
    assert "not verified" in str(stop.value)

    import dataclasses

    verified = dataclasses.replace(CELL, verified=True)
    pick_place.start_checks("dry-run", verified, None, NoGripper())  # passes
    with pytest.raises(SystemExit, match="not calibrated"):
        pick_place.start_checks("run", verified, None, SimGripper())
    ready = dataclasses.replace(verified, calibrated=True)
    pick_place.start_checks("run", ready, None, SimGripper())
    on = SimGripper()
    on.vacuum_on()
    with pytest.raises(SystemExit, match="turn the vacuum off"):
        pick_place.start_checks("run", ready, None, on)
    faulty = type("IO", (), {"fault": lambda self: "emergency stop pressed"})()
    with pytest.raises(SystemExit, match="emergency stop pressed"):
        pick_place.start_checks("dry-run", ready, faulty, NoGripper())
    with pytest.raises(SystemExit, match="Not confirmed|Cancelled"):
        pick_place.confirm_real_motion("dry-run", yes=False)
    pick_place.confirm_real_motion("dry-run", yes=True)
    assert "THE REAL ARM WILL MOVE" in capsys.readouterr().out
