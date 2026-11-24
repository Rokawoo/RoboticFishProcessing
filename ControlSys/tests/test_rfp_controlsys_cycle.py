"""Tests for the pick-and-place cycle on a simulated arm, gripper and table of fish:
the selector rules, every failure response of Table CS-1, the log and the script's checks."""

import csv
import json
import math
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Arm" / "tests"))
from arm_helpers import make_workcell  # the tests' own layout
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
from rfp_vision.geometry import pose_down
from rfp_vision.settings import VisionSettings
from rfp_vision.vision_system import Observation

CELL = make_workcell()
K = VisionSettings(width=320, height=180).lens()  # the camera as set, small


def fish(fish_id, x, y, heading_deg=90.0, confidence=0.9, code="none", overlap=False, known=True,
         table_z=None):  # fmt: skip
    ok = code == "none"
    pose = pose_down((x, y, 0.10), math.radians(heading_deg)) if ok else None
    outline = np.array([[100, 60], [220, 60], [220, 120], [100, 120]], float)
    heading = math.radians(heading_deg) if ok else None
    bbox = (x - 0.1, x + 0.1, y - 0.3, y + 0.3, 0.0, 0.1)  # vision places failed fish too
    return Fish(fish_id, ok, code, confidence, pose, pose, heading, 0.60, 0.12, 0.10, bbox,
                overlap, outline, heading_known=known, table_z=table_z,
                table_measured=table_z is not None)  # fmt: skip


class Hold:
    """Stands in for the vision hold monitor."""

    def __init__(self):
        self.cues, self.lost_at, self.updates, self.active = ["depth"], None, 0, False

    def set_background(self, frame):
        pass

    def start(self, frame, table_distance=None):
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

    def observe(self, t_base_camera, on_frame=None, frames=None) -> Observation:
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
    def __init__(
        self, *fish_at, answers=(), gripper=None, selector=None, log=None, cell=CELL, **options
    ):
        self.table = Table(*fish_at)
        self.arm = SimArm(cell)
        self.gripper = gripper or TableGripper(self.table, self.arm)
        self.clock = Clock()
        self.operator = Operator(self.clock, answers)
        settings = options.pop("settings", ControlSettings(settle_s=0.0))
        options.setdefault("vision_hold", False)
        self.cycle = PickPlaceCycle(
            self.arm, self.gripper, self.table, selector or RuleSelector(), cell, settings,
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
    # a look to find them, one from straight above each, then SY-22: three passes over
    # the two observation poses without a fish
    assert rig.table.observations == 1 + 2 + 3 * 2
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


def test_a_fish_measured_just_past_the_pick_area_s_edge_is_still_picked():
    # vision put one lying at the edge 7 mm out, and the run asked the operator about it
    from rfp_arm.workcell import PICK_AREA_TOLERANCE

    edge = CELL.pick_area.x[1]
    rig = Rig((edge + 0.02, -0.10))
    assert rig.run(max_cycles=1).successes == 1
    rig = Rig((edge + PICK_AREA_TOLERANCE + 0.03, -0.10))  # further out: not this cell's fish
    report = rig.run()
    assert report.successes == 0 and rig.gripper.events == []
    assert CELL.pick_area_seen.x[1] == pytest.approx(edge + PICK_AREA_TOLERANCE)


def test_only_valid_fish_inside_the_pick_area_are_candidates():
    rig = Rig((0.30, -0.10, {"code": "low_confidence"}), (0.40, 0.25))  # the 2nd: outside
    report = rig.run()
    assert report.cycles == 0 and rig.gripper.events == []
    assert (
        "none can be picked: fish at x 300, y -100 mm: low_confidence" in (rig.operator.alerts[0])
    )
    assert report.ended.startswith("stopped at: Fish are on the table")


def test_fish_seen_outside_the_pick_area_do_not_keep_the_run_going():
    # a placed fish at the edge of the frame (out_of_frame) beside the pick area: clear (SY-22)
    rig = Rig((0.40, 0.45, {"code": "out_of_frame"}))
    assert rig.run().ended == "pick area clear" and rig.operator.alerts == []
    rig = Rig((0.40, -0.10, {"code": "out_of_frame"}))  # the same inside it: the operator
    assert "x 400, y -100 mm: out_of_frame" in rig.run().ended and rig.operator.alerts


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

    def protective_stop(pose, speed, accel, tick=None, until=None):
        moves.append(pose)
        raise ArmError(E_PROTECTIVE_STOP, "emergency stop pressed")

    rig.arm.move_lin = protective_stop
    report = rig.run()
    assert len(moves) == 1 and report.failures == {"E_PROTECTIVE_STOP": 1}
    assert "Reset the arm" in report.ended and rig.operator.alerts == []


def test_unreachable_fish_are_skipped_and_the_others_picked():
    rig = Rig(A, B)
    rig.arm.fail = lambda kind, target: (
        "no IK solution" if kind == "ptp" and np.hypot(*(target[:2, 3] - A)) < 0.01 else None
    )
    rig.table.before_observation[14] = lambda: rig.table.fish.clear()  # then end the run
    report = rig.run()
    assert report.failures == {"E_MOTION": 3} and report.successes == 1
    assert len(report.skipped) == 1 and B not in [f[:2] for f in rig.table.fish]


def test_a_fish_set_down_off_its_slot_is_taken_away_and_its_slot_used_next():
    # The move down into slot 1 fails twice: the fish is set down where the arm is, in the
    # placement zone. Vision doesn't pick fish there: the operator takes it away, as placed
    # fish are, and slot 1 is still free for the next.
    slot1 = CELL.slots[0]
    refused = []

    def fail(kind, target):
        low = target[2, 3] < 0.3 and math.dist(target[:2, 3], slot1) < 0.01
        if kind == "lin" and low and len(refused) < 2:
            refused.append(1)
            return "no IK solution"
        return None

    rig = Rig(A, B, answers=[True])
    rig.arm.fail = fail
    report = rig.run()
    assert len(refused) == 2 and report.successes == 1
    assert len(rig.operator.alerts) == 1 and "off its slot" in rig.operator.alerts[0]
    assert "Take it away" in rig.operator.alerts[0]
    lowered = [pose for kind, pose in rig.arm.moves if kind == "lin" and pose[2, 3] < 0.3]
    assert math.dist(lowered[-1][:2, 3], slot1) < 0.01  # the next fish: slot 1


def test_a_fish_put_back_where_it_lay_is_not_taken_for_one_set_down_off_its_slot():
    # Lying just past the pick area's edge (it counts as inside), a failed lift sets it
    # down again where it lay: not in the placement zone, so no slot is given up for it
    edge = CELL.pick_area.x[1]
    lifts = []

    def fail(kind, target):
        lifting = math.dist(target[:2, 3], (edge + 0.02, -0.10)) < 0.01 and target[2, 3] > 0.15
        if kind == "lin" and lifting and len(lifts) < 2:
            lifts.append(kind)
            return "no straight path"
        return None

    rig = Rig((edge + 0.02, -0.10))
    rig.arm.fail = fail
    report = rig.run(max_cycles=2)
    assert len(lifts) == 2 and not any("off its slot" in a for a in rig.operator.alerts)
    assert report.successes == 1 and rig.cycle._slot == 1  # placed in slot 1 after all


def test_a_detour_to_turn_the_fish_is_carried_over_what_lies_on_it():
    # Straight from the fish to the slot nothing is in the way; the detour to a turning
    # spot passes over a fish placed already
    rig = Rig()
    fish_ = fish(1, 0.40, -0.10)
    slot = CELL.slots[0]
    rig.cycle._placed.append(((1.20, -0.10), 0.60, 0.20))  # 0.8 m off the straight way
    straight = rig.cycle._clear_height(fish_, slot, None)
    detour = rig.cycle._clear_height(fish_, slot, None, way=[(1.00, -0.10)])
    assert straight == pytest.approx(CELL.table_z) and detour == pytest.approx(0.20)


def test_a_plan_refused_for_where_a_fish_lies_skips_it_without_pausing(monkeypatch):
    from rfp_arm.interfaces import E_MOTION
    from rfp_controlsys import cycle as cycle_module

    real = cycle_module.plan_grasp

    def refuse(workcell, grasp_pose, *args, **kwargs):
        if min(math.dist(grasp_pose[:2, 3], at) for at in (A, C)) < 0.01:
            raise ArmError(E_MOTION, "lift pose: fish end would enter the exclusion zone")
        return real(workcell, grasp_pose, *args, **kwargs)

    monkeypatch.setattr(cycle_module, "plan_grasp", refuse)
    rig = Rig(A, C, B)
    report = rig.run()
    # each retried (vision measures again), then skipped; nothing moved, so plans refused
    # in a row don't pause the run as failed cycles would: the third fish is still picked
    assert report.failures == {"E_MOTION": 6} and report.successes == 1
    assert len(report.skipped) == 2 and not any("in a row" in a for a in rig.operator.alerts)


def test_a_plan_the_arm_would_fold_into_itself_on_is_refused_before_anything_moves(monkeypatch):
    # grasp.plan_problem (tested in Arm/tests) solves each pose from the arm's joints; here
    # a stand-in for it refuses fish A's plans
    from rfp_arm.kinematics import FR5
    from rfp_controlsys import cycle as cycle_module

    checked = []

    def problem(plan, workcell, chain, joints):
        checked.append(len(joints))
        near_a = math.dist(plan.pregrasp[:2, 3], A) < 0.05
        return "the gripper would hit the forearm at 'lower'" if near_a else None

    class JointArm(SimArm):  # an arm that tells its joints: the plan is checked from them
        def joints(self):
            return dict(zip(FR5.joint_names, [0.0] * 6))

    monkeypatch.setattr(cycle_module, "plan_problem", problem)
    rig = Rig(A, B)
    rig.arm = rig.cycle.arm = JointArm(CELL)
    rig.gripper.arm = rig.arm
    report = rig.run()
    assert checked and all(n == 6 for n in checked)
    # A refused each time before anything moved toward it, and left; B picked; no pause
    assert report.successes == 1 and len(report.skipped) == 1
    assert not any("in a row" in a for a in rig.operator.alerts)
    assert A in [f[:2] for f in rig.table.fish]
    assert not any(math.dist(pose[:2, 3], A) < 0.05 for kind, pose in rig.arm.moves)


def test_the_grasp_is_measured_from_near_the_fish_not_from_the_search_height():
    # The search looks from observation.height (0.65 m on the robot PC: a wide view). The
    # grasp goes by the depth of the look over the fish, which is better the closer it is
    # (the D405's noise grows with the square of the distance): that look is just high
    # enough to see all of it, never above close_height, and never so low that the cups
    # come within pregrasp_distance of the tallest fish of its type
    from rfp_arm.poses import tool_down
    from rfp_controlsys.cycle import Sighting

    cell = make_workcell(lambda data: data["observation"].update(height=0.65))
    rig = Rig(A, cell=cell)
    rig.table.settings = VisionSettings()  # stand-ins: up to 0.22 m tall
    tallest = rig.table.settings.fish.thickness_range_m[1]

    def heights(length, thickness):
        target = Sighting(np.array(A), 0.0, length, thickness, True)
        return [pose.camera[2, 3] - cell.table_z for pose in rig.cycle._look_poses(target)]

    margin = ControlSettings().look_margin
    fits = margin * 0.60 / (2 * math.tan(math.radians(87.0) / 2)) + 0.10 / 2
    assert heights(0.60, 0.10)[0] == pytest.approx(fits) and fits < 0.65 - 0.2
    over_tcp = cell.camera_of_tcp(tool_down((0.0, 0.0, 0.0), 0.0))[2, 3]
    floor = tallest + cell.grasp.pregrasp_distance + over_tcp
    assert min(heights(0.60, 0.10)) == pytest.approx(floor)
    assert heights(0.15, 0.0) == pytest.approx([floor, floor])  # no height measured: not lower
    assert max(heights(1.5, 0.10)) == pytest.approx(cell.close_look_height)  # 0.62


def test_a_fish_cut_off_is_looked_at_from_higher_until_it_is_seen_whole():
    heights = []

    class Closer(Table):
        def observe(self, t_base_camera, on_frame=None, frames=None):
            observation = super().observe(t_base_camera, on_frame)
            heights.append(round(float(t_base_camera[2, 3]), 3))
            whole = t_base_camera[2, 3] > 0.50  # high enough: the whole fish, too wide
            for f in observation.fish_list.fish:
                code = "geometry_out_of_range" if whole else "out_of_frame"
                f.failures, f.failure_code, f.seen_at = (code,), code, (*A, 0.0)
            return observation

    rig = Rig((*A, {"code": "out_of_frame"}))
    rig.table.__class__ = Closer
    report = rig.run()
    # each look over it from higher than the last, as it didn't fit the one before, until
    # one sees all of it
    focus = heights[1:]
    whole_at = next(i for i, height in enumerate(focus) if height > 0.50)
    assert whole_at >= 1 and focus[: whole_at + 1] == sorted(focus[: whole_at + 1])
    assert "geometry_out_of_range" in report.ended  # why, as the look that saw all of it says


class Turned(Table):
    """A camera turned 90 deg on its mount that the workcell doesn't know of: each fish is
    seen turned about the camera."""

    def observe(self, t_base_camera, on_frame=None, frames=None):
        camera, true = t_base_camera[:2, 3], self.fish
        turn = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.fish = [(*(camera + turn @ (np.array(f[:2]) - camera)), f[2]) for f in true]
        try:
            return super().observe(t_base_camera, on_frame, frames)
        finally:
            self.fish = true


def test_a_camera_turned_on_its_mount_stops_the_run():
    rig = Rig(A)
    rig.table.__class__ = Turned
    report = rig.run(max_cycles=3)
    assert "as if the camera were turned about" in report.ended and "calibrate" in report.ended
    assert report.successes == 0


def test_a_table_seen_off_table_z_stops_the_run_before_the_cups_go_down():
    # The fish's height is measured from the table the camera sees next to it: 30 mm off
    # table.z, the camera's height on the gripper or table.z is wrong, and the cups' depth
    # with them (on the real arm they went through a fish into the table)
    rig = Rig((*A, {"table_z": CELL.table_z + 0.03}))
    report = rig.run(max_cycles=1)
    assert "table.z or the camera's mount" in report.ended and report.successes == 0
    assert all(kind == "ptp" for kind, _ in rig.arm.moves)  # looked; never went down
    assert any("30 mm above the workcell's table.z" in s for s in rig.operator.states)
    ok = Rig((*A, {"table_z": CELL.table_z - 0.01}))  # within table_tolerance_m: picked
    assert ok.run(max_cycles=1).successes == 1


def test_the_table_watched_on_the_way_down_is_placed_by_where_its_frame_was_taken():
    # A frame taken higher up, placed by where the camera is now, puts the table too low:
    # each frame is kept with the camera pose it was taken from
    from rfp_controlsys.cycle import table_seen

    rig = Rig(A)
    cycle = rig.cycle
    camera = pose_down((0.40, -0.10, 0.30), 0.0)  # straight down, 0.30 m over table.z
    depth = np.full((180, 320), 0.30, np.float32)  # all table, 0.30 m from the lens
    cycle._frame = Frame(np.zeros((180, 320, 3), np.uint8), depth, K, 0.0, 0.0, 1)
    cycle._frame_camera = camera
    assert cycle._table_below() == pytest.approx(CELL.table_z, abs=1e-6)
    assert table_seen(cycle._frame, camera, (0.07, 1.0)) == pytest.approx(CELL.table_z, abs=1e-6)
    lower = rig.arm.tcp_pose()
    lower[2, 3] -= 0.10  # the arm has gone 0.10 m further down since: the frame's pose counts
    rig.arm.tcp_pose = lambda: lower
    assert cycle._table_below() == pytest.approx(CELL.table_z, abs=1e-6)
    cycle._frame = Frame(np.zeros((180, 320, 3), np.uint8), None, K, 0.0, 0.0, 2)
    assert cycle._table_below() is None  # no depth: can't tell
    holes = np.zeros((180, 320), np.float32)  # nothing in range: can't tell
    assert table_seen(Frame(None, holes, K, 0.0, 0.0, 3), camera, (0.07, 1.0)) is None


def test_the_cups_rims_and_their_lowest_lip():
    from rfp_controlsys.cycle import cup_rims, lowest_lip

    settings = VisionSettings()
    rims = cup_rims(settings)
    long, wide = settings.cup_footprint_m
    assert rims.shape == (48, 3) and np.allclose(rims[:, 2], 0.0)  # six cups, eight points each
    assert np.ptp(rims[:, 0]) == pytest.approx(long) and np.ptp(rims[:, 1]) == pytest.approx(wide)
    level = pose_down((0.4, 0.0, 0.10), 0.0)
    assert lowest_lip(level, rims) == pytest.approx(0.10) == lowest_lip(level, None)
    tilted = level.copy()
    turn = math.radians(10)
    tilted[:3, :3] = level[:3, :3] @ np.array(
        [[math.cos(turn), 0, math.sin(turn)], [0, 1, 0], [-math.sin(turn), 0, math.cos(turn)]]
    )
    assert lowest_lip(tilted, rims) == pytest.approx(0.10 - long / 2 * math.sin(turn), abs=1e-3)
    assert cup_rims(None) is None


def test_two_fish_near_where_one_was_expected_are_no_sign_of_a_turned_camera():
    # Either may be taken for the other: a jump between them says nothing about the mount
    rig = Rig(A, (A[0] + 0.06, A[1]))
    rig.table.__class__ = Turned
    assert "as if the camera were turned" not in rig.run(max_cycles=2).ended


def test_what_lies_under_the_arm_is_the_arm_not_a_fish(monkeypatch):
    # The detector took the arm's shoulder, seen at the image's edge near the base, for a
    # fish: no look reached it, and the run ended asking the operator about it
    from rfp_arm.collision import CollisionModel
    from rfp_arm.kinematics import FR5
    from rfp_controlsys import cycle as cycle_module

    under = []

    def over(self, chain, joints, xy, below):
        under.append(below)
        return "upper arm" if math.dist(xy, B) < 0.01 else None

    class JointArm(SimArm):
        def joints(self):
            return dict(zip(FR5.joint_names, [0.0] * 6))

    monkeypatch.setattr(CollisionModel, "over", over)
    monkeypatch.setattr(cycle_module, "plan_problem", lambda *args: None)  # joints made up
    rig = Rig(A, B)
    rig.arm = rig.cycle.arm = JointArm(CELL)
    rig.gripper.arm = rig.arm
    report = rig.run()
    assert report.ended == "pick area clear" and report.successes == 1 and under
    assert all(not math.dist(s.xy, B) < 0.01 for s in rig.cycle.sightings)


def test_a_fish_cut_off_even_from_the_highest_look_gets_one_look_from_there():
    # The same look again, from as high as looks go and aimed at the same place, would see
    # the same: a move and a camera pass for nothing
    heights = []

    class AlwaysCut(Table):
        def observe(self, t_base_camera, on_frame=None, frames=None):
            observation = super().observe(t_base_camera, on_frame)
            heights.append(round(float(t_base_camera[2, 3]), 3))
            for f in observation.fish_list.fish:
                f.failures, f.failure_code, f.seen_at = ("out_of_frame",), "out_of_frame", (*A, 0.0)
            return observation

    rig = Rig((*A, {"code": "out_of_frame"}))
    rig.table.__class__ = AlwaysCut
    rig.run()  # set aside after its looks; the operator is asked, nobody answers: stop
    top = max(heights)
    assert heights.count(top) <= ControlSettings().max_looks_per_fish  # once each time


def test_every_mode_starts_by_saying_what_keeps_the_layout_from_working():
    lines = pick_place.layout_report(CELL, VisionSettings())  # the largest stand-ins
    assert lines and all(line.startswith(("❌", "⚠️")) for line in lines)
    assert not [line for line in lines if line.startswith("❌")]  # the placeholder layout works


def test_slots_are_used_in_turn_as_placed_fish_are_taken_away():
    def slots_used(rig):
        return [s.rsplit(" ", 1)[1] for s in rig.operator.states if s.startswith("✅ fish placed")]

    rig = Rig(A, B, C)  # three fish, two slots: no pause for a full placement zone
    assert rig.run().successes == 3 and not rig.operator.alerts
    assert slots_used(rig) == ["1", "2", "1"]
    one = make_workcell(lambda d: d["placement"].update(slots=[list(CELL.slots[0])]))
    single = Rig(A, B, cell=one)  # one slot is enough
    assert single.run().successes == 2 and slots_used(single) == ["1", "1"]
    assert len(single.cycle._placed) == 1  # the first was taken away: carries don't clear it


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
    slow = ControlSettings(settle_s=0.0, recheck_after_s=0.0)  # a decision that takes a while
    rig = Rig(A, settings=slow)
    assert rig.run(max_cycles=1).successes == 1 and rig.table.observations == 3  # with a recheck
    rig = Rig(A, settings=slow)
    rig.table.before_observation[3] = lambda: rig.table.fish.__setitem__(0, (0.33, -0.10, {}))
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
def test_control_settings_from_a_file_reject_unknown_names(tmp_path):
    assert load_settings(panel=None) == ControlSettings()
    custom = tmp_path / "control.yaml"
    custom.write_text("max_attempts_per_fish: 2\n")
    assert load_settings(custom, panel=None).max_attempts_per_fish == 2
    custom.write_text("max_attempts: 2\n")
    with pytest.raises(ValueError, match="max_attempts"):
        load_settings(custom, panel=None)
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
    pick_place.start_checks("run", verified, None, SimGripper())  # calibrated first, then run
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


def test_the_controller_is_made_ready_only_when_the_operator_says_so(monkeypatch, capsys):
    # In manual mode, or with an error, the driver doesn't move the arm: the operator is asked
    # to let the program clear the errors and switch to automatic (nothing moves); -y says yes
    class IO:
        stateless = False

        def __init__(self, fault):
            self.faults, self.readied = [fault], 0

        def fault(self):
            return self.faults[-1]

        def make_ready(self):  # None: nothing left in the way
            self.readied += 1
            self.faults.append(None)

    manual = IO("the robot is in manual mode: switch it to automatic")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)  # no terminal: no answer
    with pytest.raises(SystemExit, match="On the pendant"):
        pick_place.ready_controller(manual, yes=False)
    assert manual.readied == 0  # nothing sent without a yes
    pick_place.ready_controller(manual, yes=True)
    assert manual.readied == 1 and "errors cleared, automatic" in capsys.readouterr().out
    stopped = IO("emergency stop pressed")  # released by hand only
    with pytest.raises(SystemExit, match="Release the emergency stop"):
        pick_place.ready_controller(stopped, yes=True)
    assert stopped.readied == 0
    ready = IO(None)
    pick_place.ready_controller(ready, yes=False)  # nothing to do, nothing asked
    assert ready.readied == 0
    blind = IO(None)
    blind.stateless = True  # 3.7: no state to tell automatic from manual
    pick_place.ready_controller(blind, yes=False)  # not asked to: goes on as the pendant has it
    assert blind.readied == 0 and "Left as it is" in capsys.readouterr().out
    pick_place.ready_controller(blind, yes=True)
    assert blind.readied == 1


def test_a_moveit_driving_a_mock_arm_beside_the_real_one_is_refused():
    # FAIRINO's demo launch (mock hardware) left running publishes joint states too: a run
    # would plan from where the mock arm is, not the real one
    joints = [0.1, -1.0, 1.2, -0.5, 1.5, 0.3]

    class MoveIt:
        def joints(self):
            return dict(zip(CELL.joints, joints))

    class Controller:
        def __init__(self, positions):
            self.positions = positions

        def joint_positions(self):
            return self.positions

    near = [q + 0.01 for q in joints]  # within a degree: the same arm
    pick_place.same_arm(MoveIt(), Controller(near), CELL)
    pick_place.same_arm(MoveIt(), Controller(None), CELL)  # no state: the fault checks say why
    elsewhere = [0.0] * 6
    with pytest.raises(SystemExit, match="isn't driving this arm"):
        pick_place.same_arm(MoveIt(), Controller(elsewhere), CELL)


def test_a_driver_built_for_other_firmware_is_said_before_anything_moves(tmp_path, monkeypatch):
    # After a firmware update FAIRINO's driver must be built again for it
    group = pick_place.firmware_group
    assert group("V3.7.8") == (3, 7, 8) and group("3.7.2") == group("V3.7.5") == (3, 7, 5)
    assert group(None) is None and group("robot") is None
    workspace = tmp_path / "ws"
    (workspace / "install" / "fairino_hardware").mkdir(parents=True)
    (workspace / pick_place.FIRMWARE_STAMP).write_text("3.7.8\n", encoding="utf-8")
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(workspace / "install" / "fairino_hardware"))
    assert pick_place.driver_firmware() == "3.7.8"

    class Controller:
        def __init__(self, firmware):
            self.version = firmware

        def firmware(self):
            return self.version

    pick_place.check_driver(Controller("V3.7.8"))  # the same: nothing said
    pick_place.check_driver(Controller(None))  # the 3.7.5 driver can't tell: no check
    with pytest.raises(SystemExit, match="setup.sh --robot"):
        pick_place.check_driver(Controller("V3.9.5"))
    monkeypatch.setenv("AMENT_PREFIX_PATH", "")
    pick_place.check_driver(Controller("V3.9.5"))  # built by hand, no stamp: no check


def test_an_unexpected_error_still_says_the_fish_may_be_on_the_gripper(monkeypatch, capsys):
    # Anything the cycle doesn't know stopped the program without a word about the vacuum
    import types

    from rfp_controlsys import cycle as cycle_module

    def broken(self, max_cycles=None):
        self.sequence = types.SimpleNamespace(holding=True, let_go=lambda: None)
        raise RuntimeError("a bug")

    monkeypatch.setattr(cycle_module.PickPlaceCycle, "run", broken)
    monkeypatch.setattr(pick_place, "RunLog", lambda info: RunLog(info, root=None))
    vision = types.SimpleNamespace(
        camera=types.SimpleNamespace(name="test"), detector=types.SimpleNamespace(version="t")
    )
    arm = SimArm(CELL)
    with pytest.raises(RuntimeError, match="a bug"):
        pick_place.run_cycles(
            "run", arm, SimGripper(), vision, CELL, ControlSettings(),
            Console(window=False, interactive=False), 1,
        )  # fmt: skip
    out = capsys.readouterr().out
    assert "stopped by an error" in out and "vacuum is still on" in out


def test_the_run_says_how_fast_the_cell_worked():
    assert pick_place.pace(12, 300.0) == "5 min 00 s: 25 s a fish, 144 an hour"
    assert pick_place.pace(0, 65.0) == "1 min 05 s"  # nothing placed: the time alone


def test_each_setting_is_written_in_one_place():
    import yaml
    from rfp_arm.workcell import DEFAULT_FILE, PANEL_FILE

    def leaves(data, path=()):
        for key, value in data.items():
            if isinstance(value, dict):
                yield from leaves(value, (*path, key))
            else:
                yield (*path, key)

    def has(data, path):
        for key in path:
            if not isinstance(data, dict) or key not in data:
                return False
            data = data[key]
        return True

    panel = yaml.safe_load(PANEL_FILE.read_text(encoding="utf-8"))["workcell"]
    cell = yaml.safe_load(DEFAULT_FILE.read_text(encoding="utf-8"))
    twice = [".".join(p) for p in leaves(panel) if has(cell, p)]
    assert not twice, f"set in both control_panel.yaml and workcell.yaml: {twice}"
    for folder in ("Vision/configs", "ControlSys/configs", "Simulation"):  # no copies of
        root = PANEL_FILE.parent / folder  # the dataclasses' defaults
        assert (
            not list(root.glob("vision.yaml"))
            + list(root.glob("control.yaml"))
            + list(root.glob("sim.yaml"))
        )


def test_control_panel_overrides_the_defaults_everywhere(tmp_path):
    from rfp_arm.workcell import WorkcellError
    from rfp_controlsys.settings import run_defaults
    from rfp_sim.settings import load_settings as load_sim
    from rfp_vision.settings import load_settings as load_vision

    load_vision(), load_settings(), load_workcell(), load_sim()  # the shipped panel loads
    assert run_defaults(tmp_path / "no_panel.yaml")["window"] is True

    panel = tmp_path / "panel.yaml"
    panel.write_text(
        "run: {mode: sim, cycles: 2}\n"
        "workcell:\n  gripper: {serial_port: COM9}\n  grasp: {contact_force: 22.5}\n"
        "vision:\n  min_confidence: 0.7\n  hold: {lost_frames: 5}\n"
        "control: {max_attempts_per_fish: 1}\n"
    )
    cell = load_workcell(panel=panel)
    assert cell.gripper.serial_port == "COM9" and cell.gripper.driver == "serial"  # merged in
    assert cell.grasp.contact_force == 22.5 and cell.grasp.press_force_max == 40.0
    vision = load_vision(panel=panel)
    assert (
        vision.min_confidence == 0.7 and vision.hold.lost_frames == 5 and vision.hold.min_iou == 0.5
    )
    assert load_settings(panel=panel).max_attempts_per_fish == 1
    run = run_defaults(panel)
    assert (run["mode"], run["cycles"], run["step"]) == ("sim", 2, False)
    assert run_defaults(tmp_path / "no_panel.yaml")["mode"] is None  # no file: the defaults

    panel.write_text("workcell: {grasp: {contact_forse: 1}}\n")
    with pytest.raises(WorkcellError, match="contact_forse"):
        load_workcell(panel=panel)
    panel.write_text("visions: {min_confidence: 0.7}\n")
    with pytest.raises(ValueError, match="unknown section"):
        load_vision(panel=panel)
    panel.write_text("run: {modes: sim}\n")
    with pytest.raises(ValueError, match="unknown run setting"):
        run_defaults(panel)


# ---------------------------------------------------------------------------
# Taught demos
# ---------------------------------------------------------------------------
def test_the_taught_pickup_demo_plays_on_the_simulated_arm(tmp_path):
    from rfp_arm.kinematics import FR5
    from rfp_controlsys import demo as demos

    pickup = demos.load_demo(demos.demo_path("fish_pickup"))
    assert pickup.name == "fish_pickup" and len(pickup.steps) == 16 and len(pickup.poses) == 6
    assert pickup.steps[0] == ("vacuum", False) and pickup.steps[6] == ("vacuum", True)  # off, on
    cell = demos.demo_workcell(CELL)  # taught poses lie outside the placeholder zones
    arm, gripper, said, waits = SimArm(cell), SimGripper(), [], []
    assert demos.play(pickup, arm, gripper, cell, say=said.append, sleep=waits.append)
    assert [kind for kind, _ in arm.moves] == [
        "joints", "joints", "lin", "lin", "lin", "lin", "joints", "lin", "joints", "joints",
    ]  # fmt: skip
    assert gripper.events == ["release", "vacuum on", "release"] and waits == [1.0, 1.0, 1.0]
    grasp = cell.tcp_of_tip(FR5.fk(pickup.poses["grasp"]))
    assert np.allclose(arm.moves[3][1], grasp)  # the taught grasp pose, reached in a line
    assert np.allclose(arm.tcp_pose(), cell.tcp_of_tip(FR5.fk(pickup.poses["start"])))
    assert len(said) == 16 and "16/16  joints: start" in said[-1]
    with pytest.raises(ArmError, match="work zone"):  # with the cell's own zones it is refused
        demos.play(pickup, SimArm(CELL), SimGripper(), CELL, say=said.append, sleep=waits.append)

    stopped = SimArm(cell)
    asked = []
    assert not demos.play(pickup, stopped, SimGripper(), cell, say=said.append,
                          confirm=lambda step: asked.append(step) or len(asked) < 3)  # fmt: skip
    assert len(stopped.moves) == 1 and len(asked) == 3  # the operator said stop at step 3

    readings = iter([[0.1, -1.0, 1.2, -1.5, -1.5, 0.0], [0.2, -1.1, 1.3, -1.6, -1.5, 0.3]])
    typed = iter(["j home", "", "l over fish", "l fish", "on", "w 0.5", "w", "undo", "off", None])
    taught = demos.record(
        lambda: next(readings), lambda prompt: next(typed), "mine", say=said.append
    )
    assert taught.steps == [
        ("joints", "home"),
        ("line", "fish"),
        ("vacuum", "on"),
        ("vacuum", "off"),
    ]
    assert taught.poses["fish"][0] == 0.2
    file = tmp_path / "mine.yaml"
    demos.save_demo(file, taught)
    again = demos.load_demo(file)
    assert again.steps == taught.steps and again.poses["home"] == pytest.approx(
        taught.poses["home"], abs=1e-4
    )
    assert demos.record(list, lambda prompt: "done", "empty", say=said.append) is None
    assert demos.demo_path("mine").name == "mine.yaml" and demos.demo_path(str(file)) == file

    for text, problem in (
        ("poses: {a: [0, 0, 0, 0, 0, 0]}\nsteps: [{jump: a}]\n", "unknown step"),
        ("poses: {a: [0, 0, 0, 0, 0, 0]}\nsteps: [{joints: b}]\n", "names no pose"),
        ("poses: {a: [0, 0]}\nsteps: [{line: a}]\n", "needs 6 joints"),
        ("poses: {}\nsteps: []\n", "no steps"),
        ("just text", "not a demo file"),
    ):
        file.write_text(text)
        with pytest.raises(demos.DemoError, match=problem):
            demos.load_demo(file)
