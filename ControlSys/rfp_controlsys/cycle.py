"""
The pick-and-place cycle (SY-20, CS-10 to CS-64): find a fish, go over it and look from
straight above, let the selector choose, check its answer, run the grasp sequence, and
answer every failure as in Table CS-1. It talks to the arm, the gripper, the vision system
and the operator only through their interfaces, so the same cycle runs offline, in
simulation, as a dry run and with vacuum (CS-90).

How it moves, to be quick and sure of each fish:
- search: the observation poses one after another, a quick look from each, until one sees
  a fish; every fish seen is remembered (`Sighting`), so after a place the arm goes straight
  to the next one it knows of;
- focus: the camera over the fish's middle, level, the image's long side along it, as high
  as it takes to see all of it (from observation.height up to close_height), each look aimed
  at what the last one saw (a fish cut off at the image's edge: its middle half its expected
  length from the end that shows), up to focus_steps looks;
- a fish whose position jumps when the camera moves means the camera's mount is wrong:
  the run stops and says so (pick_place.py --calibrate measures it);
- the look before the grasp (CS-41) is taken again only if the decision took longer than
  recheck_after_s; the carry height follows what lies on the way.

Not done here yet: the payload update (CS-45) and the VLM (a FishSelector, selector.py).
"""

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from rfp_arm.grasp import (
    WRIST_MOVE,
    GraspSequence,
    OperatorStop,
    plan_grasp,
    plan_problem,
    wrist_back_pose,
)
from rfp_arm.interfaces import (
    E_DROP,
    E_MOTION,
    E_RELEASE,
    OPERATOR_RESET_CODES,
    Arm,
    ArmError,
    Gripper,
    MotionAborted,
)
from rfp_arm.kinematics import FR5
from rfp_arm.poses import tool_down, wrap, yaw_of
from rfp_arm.workcell import ObservationPose, Workcell
from rfp_vision.camera import CameraError, Frame
from rfp_vision.fish_list import Fish, FishList, heading_difference, match_fish
from rfp_vision.geometry import to_frame
from rfp_vision.overlay import draw_fish_list
from rfp_vision.vision_system import Observation

from rfp_controlsys.selector import (
    Decision,
    DecisionContext,
    FishSelector,
    fallback_choice,
    validate_decision,
)
from rfp_controlsys.settings import ControlSettings

# Table AS-1's steps as the system states of CS-72
STATE_OF_STEP = {
    "pre-grasp": "approaching",
    "vacuum on": "approaching",
    "approach": "approaching",
    "confirm hold": "holding",
    "lift": "holding",
    "carry": "carrying",
    "lower": "placing",
    "release": "placing",
    "retreat": "placing",
}
MAX_MOVED = 5  # times in a row the chosen fish isn't where it was at the re-check: pause
MAX_IDLE = 25  # cycles in a row without a grasp or a pause: the operator is told why
# m outside the pick area a fish's estimated middle may lie and still be looked at: the
# middle of one cut off at the image's edge is only an estimate
PICK_AREA_SLACK = 0.10
SAME_WHOLE = 0.05  # m: a fish seen whole this close to a sighting is that fish
SAME_CUT = 0.15  # m: the same, seen cut off (its middle is only estimated)
FOCUS_MATCH = 0.35  # m from where the fish was expected: the fish a focus look is about
CAMERA_MOVED = 0.03  # m the camera moves between two looks for the mount check to count
# A jump at least this share of the camera's move: the camera turned 60 deg or more on its
# mount (2 sin(30 deg); half a turn: twice the move). At half the move two long fish lying
# close together, one taken for the other, stopped a run in the simulator
MOUNT_JUMP_SHARE = 1.0
DEFAULT_LENGTH = 0.6  # m: a fish's expected length when vision's settings don't say
STILL_AIMED = 0.02  # m: a fish's estimated middle moved less, the next look aims the same


class RunStopped(Exception):
    """The run can't go on (operator stop at an alert, arm fault, camera failure)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class RunReport:
    """SY-23."""

    cycles: int = 0
    successes: int = 0
    failures: dict[str, int] = field(default_factory=dict)  # by failure code
    skipped: list[str] = field(default_factory=list)  # fish given up on, by position (CS-62)
    ended: str = ""

    def as_record(self) -> dict:
        return {
            "cycles": self.cycles,
            "successes": self.successes,
            "failures": self.failures,
            "skipped_fish": self.skipped,
            "ended_because": self.ended,
        }


class Attempts:
    """Grasp attempts per fish over a run (CS-62). A fish is known by where it lies."""

    def __init__(self, same_fish_m: float):
        self._same = same_fish_m
        self._fish: list[tuple[np.ndarray, int]] = []

    def _index(self, point: np.ndarray) -> int | None:
        for i, (known, _) in enumerate(self._fish):
            if np.linalg.norm(known[:2] - point[:2]) <= self._same:
                return i
        return None

    def count(self, point: np.ndarray) -> int:
        i = self._index(point)
        return 0 if i is None else self._fish[i][1]

    def add(self, point: np.ndarray) -> int:
        i = self._index(point)
        if i is None:
            self._fish.append((np.array(point, float), 1))
            return 1
        self._fish[i] = (self._fish[i][0], self._fish[i][1] + 1)
        return self._fish[i][1]


@dataclass(eq=False)  # each sighting is its own fish, whatever its numbers
class Sighting:
    """A fish seen on the table, as best known: where its middle is, along which axis, how
    long (seen cut off: at least that), and why it wasn't picked when last looked at."""

    xy: np.ndarray
    axis: float  # rad, body axis yaw in {B}
    length: float  # m
    thickness: float  # m, its height on the table
    whole: bool  # last seen all of it
    looks: int = 0  # focus looks that ended without a candidate
    why: str = ""  # its failure code and what was measured, when it couldn't be picked
    seen_whole: bool = False  # some look saw all of it: `why` is from such a look


def go_to_pose(arm: Arm, workcell: Workcell, pose: ObservationPose, tick=None) -> None:
    """To an observation pose or a look at a fish. Back from a place at one end of the
    wrist's range the turn there can be longer than WRIST_MOVE (vacuum tube; the joint
    travel limit): the wrist turns back to its middle where the tool is first."""
    tcp = None if pose.joints is not None else workcell.observation_tcp(pose)
    there = pose.joints[-1] if pose.joints is not None else arm.wrist_angle(tcp)
    now = arm.wrist_angle()
    if there is not None and now is not None and abs(there - now) > WRIST_MOVE:
        back = wrist_back_pose(arm, workcell)
        if back is not None and workcell.pose_problem(back) is None:
            arm.move_ptp(back, tick)
    if pose.joints is not None:
        arm.move_joints(pose.joints, tick)
    else:
        arm.move_ptp(tcp, tick)


class PickPlaceCycle:
    def __init__(
        self,
        arm: Arm,
        gripper: Gripper,
        vision,
        selector: FishSelector,
        workcell: Workcell,
        settings: ControlSettings,
        console,
        log=None,
        dry_run: bool = False,
        vision_hold: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        controller: Callable[[str], str] | None = None,
    ):
        """
        `vision`: a VisionSystem. `console`: a Console. `log`: a RunLog or None.
        `dry_run`: the arm moves, the vacuum stays off, approaches stop short (CS-93).
        `vision_hold`: watch a held fish with the camera (off where the camera isn't on
        the arm that moves: simulation, offline). `controller`: runs a command line on
        the arm's controller (FairinoIO.execute), for the grasp settings' press commands.
        """
        self.arm = arm
        self.gripper = gripper
        self.vision = vision
        self.selector = selector
        self.workcell = workcell
        self.settings = settings
        self.console = console
        self.log = log
        self.dry_run = dry_run
        self.vision_hold = vision_hold and settings.vision_hold_check and not dry_run
        self._clock = clock
        self._sleep = sleep
        self._controller = controller
        self.report = RunReport()
        self.sequence: GraspSequence | None = None  # the grasp in progress or last run
        self._attempts = Attempts(settings.same_fish_m)  # by the fish's middle
        self.sightings: list[Sighting] = []  # fish seen and not placed yet
        self._pose_index = 0
        self._slot = 0
        self._placed: list[tuple[tuple[float, float], float, float]] = []  # slot, length, height
        self._empty = 0  # search passes in a row without any fish
        self._idle = 0  # cycles in a row that started no grasp
        self._reobserves = 0
        self._moved = 0
        self._decision_s = 0.0  # how long the last decision took (the selector, not vision)
        self._failed_in_row = 0
        self._last_result = ""
        self._last_failed: np.ndarray | None = None  # where the fish that just failed lies
        self._tool_yaw = 0.0
        self._frame = None  # the newest camera frame seen by the tick
        self._hold_text = ""
        self._live_at = -math.inf  # when the tick last ran the detector
        self._table_said = False
        # Called with each fish list (and its frame) vision makes: the 3D view shows them
        self.on_fish_list: Callable[[FishList, Frame], None] | None = None

    # -- the run -----------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> RunReport:
        """Cycles until the pick area is clear, the limit is reached or the run is stopped."""
        report = self.report
        try:
            while max_cycles is None or report.cycles < max_cycles:
                if not self._cycle():
                    break
            else:
                report.ended = f"cycle limit of {max_cycles} reached"
        except OperatorStop as stop:
            report.ended = f"stopped: {stop.reason}"
        except RunStopped as stopped:
            report.ended = stopped.reason
        except ArmError as e:  # e.g. the arm's state stopped arriving between moves
            report.ended = str(e)
        self.console.state("idle", report.ended)
        return report

    def _cycle(self) -> bool:
        """Find a fish, focus on it and, if one can be chosen, grasp it. False ends the run."""
        grasps = self.report.cycles
        going = self._find_and_grasp()
        self._idle = 0 if self.report.cycles > grasps else self._idle + 1
        if going and self._idle >= MAX_IDLE:  # looking and looking, never grasping
            self._idle = 0
            self._unpickable()
        return going

    def _unpickable(self) -> None:
        """Fish there, but none to pick: tell the operator why, then look at each again."""
        self._pause("Fish are on the table but none can be picked: " + self._why())
        for sighting in self.sightings:  # someone may have moved them
            sighting.looks = 0

    def _find_and_grasp(self) -> bool:
        target = self._next_target()
        if target is None:
            if not self._search():  # a whole pass saw no fish
                self._empty += 1
                if self._empty >= self.settings.empty_passes_to_end:
                    self.report.ended = "pick area clear"  # SY-22
                    return False
                return True
            self._empty = 0
            target = self._next_target()
            if target is None:  # fish there, but each looked at already: tell the operator
                self._unpickable()
                return True
        self._empty = 0
        focused = self._focus(target)
        if focused is None:
            return True
        pose, observation, target = focused
        candidates = self._candidates(observation.fish_list)
        if not candidates:
            target.looks += 1
            return True
        if self._slot >= len(self.workcell.slots):
            self._pause("The placement zone is full (AS-56). Clear it, then resume.")
            self._slot = 0
            self._placed.clear()
            return True  # look again: someone was at the table

        decision = self._decide(observation, candidates)
        if decision.action == "reobserve":  # another look, not the fish's fault (max_reobserve)
            return True
        if decision.action == "end":
            self._pause("The selector ended the run, but candidate fish remain.")
            return True
        chosen = observation.fish_list.by_id(decision.fish_id)
        fish = self._recheck(chosen, pose, observation)
        if fish is None:
            self._moved += 1
            self.console.info(f"fish {chosen.fish_id} isn't where it was: new decision (CS-41)")
            if self._moved >= MAX_MOVED:
                self._moved = 0
                self._pause("The chosen fish keeps moving between two looks.")
            return True
        self._moved = 0
        self.report.cycles += 1
        self._grasp(fish, decision, pose, len(candidates), observation.fish_list)
        return True

    # -- finding fish -------------------------------------------------------------
    def _next_target(self) -> Sighting | None:
        """The fish seen and not given up on that is nearest to where the camera looks."""
        s = self.settings
        open_ = [
            sighting
            for sighting in self.sightings
            if sighting.looks < s.max_looks_per_fish
            and self._attempts.count(sighting.xy) < s.max_attempts_per_fish
        ]
        if not open_:
            return None
        here = self.workcell.camera_of_tcp(self.arm.tcp_pose())[:2, 3]
        return min(open_, key=lambda sighting: float(np.linalg.norm(sighting.xy - here)))

    def _search(self) -> bool:
        """The observation poses in turn, a quick look from each, until one sees a fish to
        go to. False if a whole pass saw no fish at all."""
        poses = self.workcell.observation_poses
        seen = False
        for _ in range(len(poses)):
            pose = poses[self._pose_index]
            self._observe(pose, frames=self.settings.search_frames)
            seen = seen or bool(self.sightings)
            if self._next_target() is not None:
                return True
            self._next_pose()
        return seen

    def _next_pose(self) -> None:
        self._pose_index = (self._pose_index + 1) % len(self.workcell.observation_poses)

    def _focus(self, target: Sighting) -> tuple[ObservationPose, Observation, Sighting] | None:
        """Look at the fish from straight above its middle, as high as it takes to see all
        of it, each look aimed at what the last saw (up to focus_steps). The pose, the
        observation that saw all of it (or the last one) and the fish's sighting; None if it
        can't be looked at or is gone."""
        s = self.settings
        found = None
        for _ in range(max(1, s.focus_steps)):
            before = (target.xy.copy(), target.whole, self._camera_xy())
            reached = self._look_at(target)
            if reached is None:
                return found
            pose, observation = reached
            fish = self._match(observation.fish_list, before[0])
            if fish is None:  # gone, or never was a fish
                if target in self.sightings:
                    self.sightings.remove(target)
                self.console.info("the fish isn't there any more")
                return None
            if self._alone(observation.fish_list, before[0]):  # not one fish for another
                self._check_mount(fish, *before, observation.fish_list.camera_pose[:2, 3])
            target = self._sighting_of(fish) or target
            found = (pose, observation, target)
            if "out_of_frame" not in fish.failures:
                break
            highest = pose.camera[2, 3] - self.workcell.table_z >= self._look_heights()[1] - 1e-3
            if highest and float(np.linalg.norm(target.xy - before[0])) < STILL_AIMED:
                break  # as high as a look goes, aimed where it was: the next would see the same
            # It didn't fit: it is at least as long as the view along it, so look from higher
            height = pose.camera[2, 3] - self.workcell.table_z - target.thickness * 0.5
            target.length = max(target.length, 2.0 * height * math.tan(self._hfov() / 2.0))
        return found

    def _check_mount(self, fish: Fish, xy, whole: bool, camera_before, camera_now) -> None:
        """A fish lying still stays where it is whichever way the camera looks at it. With
        the camera turned on its mount by an angle the arm doesn't know of, a fish seen whole
        from two places jumps by 2 sin(angle / 2) times the camera's move: then stop."""
        if not whole or "out_of_frame" in fish.failures:
            return
        moved = float(np.linalg.norm(np.asarray(camera_now) - camera_before))
        guess = fish.estimate(self._expected_length())
        if moved <= CAMERA_MOVED or guess is None:
            return
        jump = float(np.linalg.norm(guess[0] - xy))
        if jump > self.settings.max_jump_m and jump > MOUNT_JUMP_SHARE * moved:
            turned = math.degrees(2 * math.asin(min(1.0, jump / (2 * moved))))
            raise RunStopped(
                f"a fish lying still seemed to move {jump * 1000:.0f} mm when the camera moved "
                f"{moved * 1000:.0f} mm, as if the camera were turned about {turned:.0f} deg "
                "from where the workcell has it (camera.mount). Measure it: pick_place.py "
                "--mode dry-run --calibrate"
            )

    def _alone(self, fish_list: FishList, xy) -> bool:
        """Whether only one fish of a look lies near where a fish was expected."""
        near = 0
        for fish in fish_list.fish:
            guess = fish.estimate(self._expected_length())
            near += guess is not None and float(np.linalg.norm(guess[0] - xy)) <= FOCUS_MATCH
        return near <= 1

    def _camera_xy(self) -> np.ndarray:
        return self.workcell.camera_of_tcp(self.arm.tcp_pose())[:2, 3].copy()

    def _sighting_of(self, fish: Fish) -> Sighting | None:
        guess = fish.estimate(self._expected_length())
        if guess is None:
            return None
        near = [s for s in self.sightings if np.linalg.norm(s.xy - guess[0]) < 1e-9]
        return near[0] if near else None

    def _look_at(self, target: Sighting) -> tuple[ObservationPose, Observation] | None:
        """Go over a fish and look: the highest look the arm reaches up to the height that
        fits the fish, the way round nearest the wrist's middle. None if none is reachable."""
        for pose in self._look_poses(target):
            try:
                self._go_to(pose)
            except MotionAborted as aborted:
                raise OperatorStop(aborted.reason) from None
            except ArmError as e:
                if e.code != E_MOTION:
                    raise RunStopped(f"{pose.name}: {e}") from None
                continue
            return pose, self._observe(pose, move=False)
        target.looks = self.settings.max_looks_per_fish
        target.why = "no look at it within the arm's reach"
        self.console.info(f"no look at the fish at {_where(target.xy)} within reach")
        return None

    def _look_poses(self, target: Sighting) -> list[ObservationPose]:
        """Where to look at a fish from, best first: straight above its middle, the image's
        long side along it, high enough to see its expected length (and lower ones, in case
        the arm can't reach that high there), each the way round nearest the wrist's middle."""
        s, cell = self.settings, self.workcell
        low, high = self._look_heights()
        hfov = self._hfov()
        length = max(target.length, self._expected_length() if not target.whole else 0.0)
        reach = s.look_margin * length / (2.0 * math.tan(hfov / 2.0))
        height = min(max(reach + target.thickness * 0.5, low), high)
        heights = [height] + [h for h in np.arange(height - 0.05, low - 1e-9, -0.05)]
        if heights[-1] > low + 1e-9:
            heights.append(low)
        poses = []
        x, y = target.xy
        for h in heights:
            options = []
            for yaw in (target.axis, target.axis + math.pi):
                camera = tool_down((float(x), float(y), cell.table_z + float(h)), yaw)
                tcp = cell.tcp_of_camera(camera)
                if cell.pose_problem(tcp) is not None:
                    continue
                options.append((self._wrist_cost(tcp), camera))
            for _, camera in sorted(options, key=lambda option: option[0]):
                name = f"look at x {x * 1000:.0f}, y {y * 1000:.0f} mm from {h:.2f} m"
                poses.append(ObservationPose(name, camera=camera))
        return poses

    def _wrist_cost(self, tcp: np.ndarray) -> float:
        """How far the wrist would be from its middle at this TCP pose (rad); outside its
        range: infinite. Without the arm's prediction, the tool's turn from now."""
        low, high = self.workcell.wrist_range
        middle = (low + high) / 2
        wrist = self.arm.wrist_angle(tcp)
        if wrist is None:
            return abs(wrap(yaw_of(tcp) - yaw_of(self.arm.tcp_pose())))
        if not low <= wrist <= high:
            return math.inf
        return abs(wrist - middle)

    def _look_heights(self) -> tuple[float, float]:
        """(lowest, highest) camera height above the table for a look at one fish."""
        cell = self.workcell
        heights = [
            pose.camera[2, 3] - cell.table_z
            for pose in cell.observation_poses
            if pose.camera is not None
        ]
        low = min(heights) if heights else 0.45
        high = cell.close_look_height if cell.close_look_height is not None else low
        return low, max(low, high)

    def _hfov(self) -> float:
        frame = self._frame
        if frame is not None:
            return 2.0 * math.atan(frame.intrinsics.width / 2.0 / frame.intrinsics.fx)
        return math.radians(getattr(getattr(self.vision, "settings", None), "hfov_deg", 87.0))

    def _expected_length(self) -> float:
        """A fish's length as vision measures it: the middle of the fish type's range."""
        settings = getattr(self.vision, "settings", None)
        if settings is None:
            return DEFAULT_LENGTH
        low, high = settings.fish.length_range_m
        return (low + high) / 2

    def _match(self, fish_list: FishList, xy) -> Fish | None:
        """The fish of a look nearest to where a fish was expected."""
        best, nearest = None, FOCUS_MATCH
        for fish in fish_list.fish:
            guess = fish.estimate(self._expected_length())
            if guess is None:
                continue
            distance = float(np.linalg.norm(guess[0] - xy))
            if distance <= nearest:
                best, nearest = fish, distance
        return best

    # -- observing ---------------------------------------------------------------
    def _observe(
        self, pose: ObservationPose, move: bool = True, frames: int | None = None
    ) -> Observation:
        """A look from a pose (CS-14): there, the arm at rest, a fish list from `frames`
        frames (default: vision's stable_frames). Updates the fish seen."""
        name = pose.name if pose.name.startswith("look at") else f"pose '{pose.name}'"
        self.console.state("observing", name)
        if move:
            self._move(lambda: self._go_to(pose), f"can't reach observation pose '{pose.name}'")
        self._sleep(self.settings.settle_s)
        tcp = self.arm.tcp_pose()
        self._tool_yaw = yaw_of(tcp)
        try:
            if frames is None:
                observation = self.vision.observe(self.workcell.camera_of_tcp(tcp))
            else:
                observation = self.vision.observe(self.workcell.camera_of_tcp(tcp), frames=frames)
        except CameraError as e:
            raise RunStopped(f"camera failure: {e}") from None
        self._frame = observation.frame
        self.vision.hold.set_background(observation.frame)  # nothing is held here
        self._remember(observation)
        self._show(observation)
        if self.on_fish_list is not None:
            self.on_fish_list(observation.fish_list, observation.frame)
        return observation

    def _remember(self, observation: Observation) -> None:
        """The fish of a look in the sightings: each matched to one seen before (or new),
        and the ones this look should have seen but didn't, forgotten."""
        fish_list, frame = observation.fish_list, observation.frame
        camera = fish_list.camera_pose
        expected = self._expected_length()
        under_arm = self._under_arm(float(camera[2, 3]))
        seen, unmatched = [], []
        for fish in fish_list.fish:
            guess = fish.estimate(expected)
            if guess is None:
                continue
            if not self.workcell.pick_area.contains(guess[0], -PICK_AREA_SLACK):
                continue  # placed fish at the frame's edge, or one lying beyond (SY-22)
            if under_arm is not None and under_arm(guess[0]):
                continue  # the arm's own body (the detector took its shoulder for a fish)
            whole = "out_of_frame" not in fish.failures
            sighting = self._sighting_near(guess[0], whole, seen)
            if sighting is None:
                unmatched.append((fish, guess, whole))
                continue
            self._update(sighting, fish, guess, whole)
            seen.append(sighting)
        # A cut-off fish's middle is an estimate that moves from look to look: a fish not
        # matched above is the nearest sighting not seen yet, if there is one near enough
        for fish, guess, whole in unmatched:
            sighting = self._sighting_near(guess[0], whole, seen, FOCUS_MATCH)
            if sighting is None:
                sighting = Sighting(guess[0], guess[1], guess[2], fish.thickness, whole)
                self.sightings.append(sighting)
            self._update(sighting, fish, guess, whole)
            seen.append(sighting)
        for sighting in list(self.sightings):
            if sighting not in seen and _in_view(sighting.xy, self.workcell, frame, camera):
                self.sightings.remove(sighting)  # where it was is in view: it's gone

    def _plan_problem(self, plan) -> str | None:
        """What keeps the arm from going through a grasp plan from where it is (IK and the
        arm against itself, grasp.plan_problem); None if nothing does or it can't tell."""
        joints = getattr(self.arm, "joints", None)
        if joints is None:
            return None
        try:
            now = joints()
        except ArmError:
            return None
        return plan_problem(plan, self.workcell, FR5, [now[name] for name in FR5.joint_names])

    def _under_arm(self, camera_z: float):
        """A test of whether the arm's own body (now) lies between the camera and a point
        (x, y) of the table; None if the arm can't tell (no joints or collision model)."""
        collision, joints = self.workcell.collision, getattr(self.arm, "joints", None)
        if collision is None or joints is None:
            return None
        try:
            now = joints()
        except ArmError:
            return None
        q = [now[name] for name in FR5.joint_names]
        return lambda xy: collision.over(FR5, q, xy, camera_z) is not None

    def _update(self, sighting: Sighting, fish: Fish, guess, whole: bool) -> None:
        """A sighting, from what one look saw of its fish."""
        xy, axis, length = guess
        # seen cut off, it is at least as long as any look has shown (or a focus found)
        sighting.length = length if whole else max(sighting.length, length)
        sighting.xy, sighting.axis = xy, axis
        sighting.thickness, sighting.whole = max(fish.thickness, 0.0), whole
        if fish.vision_ok:
            sighting.why = ""
        elif whole or not sighting.seen_whole:  # a look from afar doesn't know better
            sighting.why = _reason(fish)
        sighting.seen_whole = sighting.seen_whole or whole
        self._say_table(fish)

    def _sighting_near(
        self, xy, whole: bool, taken: list, radius: float | None = None
    ) -> Sighting | None:
        """The sighting this fish is: the nearest one not taken by another fish of the same
        look, within `radius`, else SAME_WHOLE if both saw all of it, else SAME_CUT (an
        estimated middle)."""
        near = []
        for i, sighting in enumerate(self.sightings):
            distance = float(np.linalg.norm(sighting.xy - xy))
            if radius is not None:
                limit = radius
            else:
                limit = SAME_WHOLE if whole and sighting.whole else SAME_CUT
            if sighting not in taken and distance <= limit:
                near.append((distance, i))
        return self.sightings[min(near)[1]] if near else None

    def _say_table(self, fish: Fish) -> None:
        """Once a run: the table measured around a fish, if the workcell has it elsewhere."""
        if self._table_said or not fish.table_measured or fish.table_z is None:
            return
        off = fish.table_z - self.workcell.table_z
        if abs(off) > self.settings.table_warning_m:
            self._table_said = True
            self.console.info(
                f"⚠️  the camera measures the table at z = {fish.table_z * 1000:.0f} mm, "
                f"{abs(off) * 1000:.0f} mm {'below' if off < 0 else 'above'} the workcell's "
                "table.z: set it (vision measures fish from the table it sees)"
            )

    def _go_to(self, pose: ObservationPose) -> None:
        go_to_pose(self.arm, self.workcell, pose, self._tick)

    def _move(self, motion: Callable[[], None], what: str) -> None:
        """A move outside the grasp sequence: planned again once, then the run stops."""
        try:
            try:
                motion()
            except ArmError as e:
                if e.code != E_MOTION:
                    raise
                self.console.info(f"{e}: planning again")
                motion()
        except MotionAborted as aborted:
            raise OperatorStop(aborted.reason) from None
        except ArmError as e:
            raise RunStopped(f"{what}: {e}") from None

    def _candidates(self, fish_list: FishList) -> list[Fish]:
        """CS-11, CS-13, CS-62: valid, in the pick area, in no exclusion zone, attempts left."""
        cell, limit = self.workcell, self.settings.max_attempts_per_fish
        return [
            fish
            for fish in fish_list.valid
            if cell.in_pick_area(fish.middle)
            and not any(zone.contains(fish.grasp_point) for zone in cell.exclusion_zones)
            and self._attempts.count(fish.middle) < limit
        ]

    def _show(self, observation: Observation, chosen: int | None = None) -> np.ndarray:
        fish_list = observation.fish_list
        frame = observation.frame
        image = draw_fish_list(frame.color, fish_list, frame.intrinsics, chosen)
        self.console.show(image, [f"{len(fish_list.valid)} valid of {len(fish_list.fish)} fish"])
        return image

    def _why(self) -> str:
        """Why the fish seen can't be picked, for an alert (CS-73)."""
        reasons = []
        for sighting in self.sightings:
            if self._attempts.count(sighting.xy) >= self.settings.max_attempts_per_fish:
                why = "attempt limit reached (CS-62)"
            else:
                why = sighting.why or "not chosen"
            reasons.append(f"fish at {_where(sighting.xy)}: {why}")
        return "; ".join(reasons) or "attempt limit reached for each (CS-62)"

    # -- deciding ----------------------------------------------------------------
    def _decide(self, observation: Observation, candidates: list[Fish]) -> Decision:
        """The selector's answer, validated; asked again once, then the fallback rule."""
        s = self.settings
        self.console.state("deciding", f"{len(candidates)} candidate fish")
        fish_list = observation.fish_list
        failed_here = None
        if self._last_failed is not None:
            for fish in candidates:
                if np.linalg.norm(fish.middle[:2] - self._last_failed[:2]) <= s.same_fish_m:
                    failed_here = fish.fish_id
        context = DecisionContext(
            fish_list,
            candidates,
            self._show(observation),
            "ready",
            self.gripper.state().value,
            self._last_result,
            failed_here,
            {fish.fish_id: self._attempts.count(fish.middle) for fish in candidates},
        )
        decision = None
        started = time.monotonic()
        for _ in range(2):
            try:
                answer = self.selector.choose(context)
                problem = validate_decision(answer, candidates)
            # a selector (later a VLM client) failing in any way must not end the run
            except Exception as e:  # noqa: BLE001
                problem = f"selector failed: {e}"
            if problem is None:
                decision = answer
                break
            self.console.info(f"selector answer rejected: {problem}")
        if decision is None:
            decision = fallback_choice(candidates)
        self._decision_s = time.monotonic() - started
        if decision.action == "reobserve":
            self._reobserves += 1
            if self._reobserves >= s.max_reobserve:
                decision = fallback_choice(candidates)
        if decision.action != "reobserve":
            self._reobserves = 0
        fish = f" fish {decision.fish_id}" if decision.fish_id is not None else ""
        self.console.info(f"{decision.source}: {decision.action}{fish} ({decision.reason})")
        return decision

    def _recheck(
        self, chosen: Fish, pose: ObservationPose, observation: Observation
    ) -> Fish | None:
        """The chosen fish, checked before the grasp (CS-41): if the decision took
        recheck_after_s or longer (a slow selector), looked at again from the same pose and
        found at the same place; None if it moved. Timed by the decision alone, not the
        look's age: vision's own time would make the run depend on the computer's load."""
        s = self.settings
        fish = chosen
        if self._decision_s >= s.recheck_after_s:
            observation = self._observe(pose, move=False)
            fish = match_fish(chosen, observation.fish_list.valid, s.recheck_position_m)
            if fish is None or heading_difference(fish, chosen) > math.radians(
                s.recheck_heading_deg
            ):
                return None
        if time.monotonic() - observation.fish_list.stamp > s.fish_list_max_age_s:  # CS-53
            return None
        image = self._show(observation, fish.fish_id)
        if self.log is not None and s.save_images:
            self.log.image(f"cycle_{self.report.cycles + 1:03d}", image)
        return fish

    # -- grasping ----------------------------------------------------------------
    def _grasp(
        self,
        fish: Fish,
        decision: Decision,
        pose: ObservationPose,
        candidates: int,
        fish_list: FishList | None = None,
    ) -> None:
        cell = self.workcell
        started = self._clock()
        point = fish.grasp_point
        slot = cell.slots[self._slot]
        clear = self._clear_height(fish, slot, fish_list)
        row = {
            "cycle": self.report.cycles,
            "time": time.strftime("%H:%M:%S"),
            "observation_pose": pose.name,
            "fish_id": fish.fish_id,
            "decision_source": decision.source,
            "decision_reason": decision.reason,
            "candidates": candidates,
            "grasp_x_mm": round(point[0] * 1000, 1),
            "grasp_y_mm": round(point[1] * 1000, 1),
            "grasp_z_mm": round(point[2] * 1000, 1),
            "heading_deg": round(math.degrees(fish.heading), 1),
            "heading_known": fish.heading_known,
            "length_mm": round(fish.total_length * 1000),
            "thickness_mm": round(fish.thickness * 1000),
            "tilt_deg": round(math.degrees(fish.surface_tilt), 1),
            "confidence": round(fish.confidence, 2),
            "slot": self._slot + 1,
            "attempt": self._attempts.count(fish.middle) + 1,
            "result": "stopped",
        }
        self.sequence = GraspSequence(
            self.arm,
            self.gripper,
            cell,
            hold_check=_VisionHold(self) if self.vision_hold else None,
            tick=self._tick,
            on_step=lambda step: self.console.state(STATE_OF_STEP[step], step),
            require_hold_confirmation=self.settings.require_hold_confirmation,
            clock=self._clock,
            sleep=self._sleep,
            controller=self._controller,
        )
        try:

            def planned(clear_z: float | None):
                return plan_grasp(
                    cell, fish.grasp_pose, fish.heading_known, fish.total_length, fish.thickness,
                    slot, self._tool_yaw, self.dry_run, self.arm.wrist_angle, fish.middle, clear_z,
                )  # fmt: skip

            plan = planned(clear)
            if clear is not None and plan.carry_via:  # over what lies where it really goes
                way = [(p[:3, 3] + p[:3, 0] * plan.middle_x)[:2] for p in plan.carry_via]
                higher = self._clear_height(fish, slot, fish_list, way)
                if higher > clear + 1e-9:  # a detour to turn the fish passes over another
                    plan = planned(higher)
            if problem := self._plan_problem(plan):  # nothing moved: like a refused plan
                raise ArmError(E_MOTION, f"the plan can't run: {problem}")
            row["carry_mm"] = round((plan.carry[2, 3] - cell.table_z) * 1000)
            if plan.wrist is not None:
                path = [round(math.degrees(a)) for a in plan.wrist]
                row["wrist_deg"] = "/".join(str(a) for a in path)
                self.console.info(
                    f"wrist (j6) {path[0]} -> {path[1]} -> {path[2]} -> {path[0]} deg: "
                    f"sweeps {math.degrees(plan.wrist_sweep):.0f} deg; carried "
                    f"{(plan.carry[2, 3] - cell.table_z) * 1000:.0f} mm up"
                )
            self.sequence.run(plan)
        except ArmError as error:
            row.update(result="failed", failure_code=error.code, detail=error.message,
                       failed_step=self.sequence.step or "plan")  # fmt: skip
            self._log_row(row, started)
            self._failed(fish, error)
        except BaseException:
            self._log_row(row, started)
            raise
        else:
            row["result"] = "success"
            self._log_row(row, started)
            self.console.info(f"✅ fish placed in slot {self._slot + 1}")
            self._placed.append((tuple(slot), fish.total_length, max(fish.thickness, 0.0)))
            self._slot += 1
            self.report.successes += 1
            self._failed_in_row = 0
            self._last_result, self._last_failed = "success", None
            for sighting in self._sightings_at(fish.middle):  # it's in its slot now
                self.sightings.remove(sighting)

    def _sightings_at(self, middle) -> list[Sighting]:
        """The sightings of the fish whose middle is at `middle`."""
        xy = np.asarray(middle[:2], float)
        return [
            sighting
            for sighting in self.sightings
            if np.linalg.norm(sighting.xy - xy) <= max(SAME_CUT, sighting.length / 4)
        ]

    def _clear_height(self, fish: Fish, slot, fish_list: FishList | None, way=None) -> float | None:
        """The top (z in {B}) of what a fish carried from where it lies to the slot passes
        over: the other fish seen near the way (from vision's heights) and the fish placed
        already; the table if nothing. `way`: where the held fish's middle passes on the
        way (x, y), if not straight there (a detour to turn it). None to carry at the fixed
        height (carry_over_seen off)."""
        cell, s = self.workcell, self.settings
        if not s.carry_over_seen:
            return None
        start, end = np.asarray(fish.middle[:2], float), np.asarray(slot, float)
        path = [start, *(np.asarray(point, float)[:2] for point in way or ()), end]
        reach = fish.total_length / 2 + s.carry_corridor_m  # the held fish swings this far
        top = cell.table_z
        others = []
        if fish_list is not None:
            for other in fish_list.fish:
                if other.bbox is None or other is fish:
                    continue
                middle = np.array([(other.bbox[0] + other.bbox[1]) / 2,
                                   (other.bbox[2] + other.bbox[3]) / 2])  # fmt: skip
                if np.linalg.norm(middle - start) < 0.05:
                    continue  # the fish itself
                half = max(other.bbox[1] - other.bbox[0], other.bbox[3] - other.bbox[2]) / 2
                others.append((middle, half, other.bbox[5]))
        itself = self._sightings_at(fish.middle)
        for sighting in self.sightings:
            if sighting not in itself:
                others.append((sighting.xy, sighting.length / 2, cell.table_z + sighting.thickness))
        for placed, length, height in self._placed:
            others.append((np.asarray(placed, float), length / 2, cell.table_z + height))
        for middle, half, height in others:
            near = min(_segment_distance(middle, a, b) for a, b in itertools.pairwise(path))
            if near <= reach + half:
                top = max(top, height)
        return top

    def _log_row(self, row: dict, started: float) -> None:
        row["duration_s"] = round(self._clock() - started, 1)
        felt = self.sequence.contact_depth
        row["contact_mm"] = "" if felt is None else round(felt * 1000, 1)
        if self.log is not None:
            self.log.cycle(row)

    def _failed(self, fish: Fish, error: ArmError) -> None:
        """Table CS-1's response to a failed grasp."""
        s = self.settings
        sequence = self.sequence
        code = error.code
        self.report.failures[code] = self.report.failures.get(code, 0) + 1
        self._last_result, self._last_failed = code, fish.middle
        attempts = self._attempts.add(fish.middle)
        step = sequence.step or "planning"
        self.console.info(
            f"❌ {error} (step '{step}', attempt {attempts} of {s.max_attempts_per_fish})"
        )
        for sighting in self._sightings_at(fish.middle):  # for the operator, if it stays
            sighting.why = f"{code} at '{step}'"
        if code in OPERATOR_RESET_CODES:  # AS-73, SF-23
            raise RunStopped(f"{error}. Reset the arm, then start again.")
        try:
            if error.still_holding:
                self._pause(
                    f"{error}. The vacuum is still on and the fish may be on the gripper. "
                    "Support the fish: resuming turns the vacuum off."
                )
                sequence.let_go()
            elif code == E_RELEASE:
                self._pause("The gripper didn't release the fish (E_RELEASE). Check the vacuum.")
            elif (
                error.fish_put_back
                and self.workcell.placement_zone.contains(error.tcp_pose[:2, 3])
                and not self.workcell.in_pick_area(error.tcp_pose[:2, 3])
            ):  # off its slot, turned anyhow: the next fish would go on top of it
                slot = self.workcell.slots[self._slot]
                self._pause(
                    f"The fish was set down at {_where(error.tcp_pose[:2, 3])}, off its slot. "
                    f"Put it in slot {self._slot + 1} or take it away, then resume."
                )
                self._placed.append((tuple(slot), fish.total_length, max(fish.thickness, 0.0)))
                self._slot += 1
            elif sequence.step and not error.fish_put_back:  # the arm is somewhere in the grasp
                sequence.let_go()
                if code != E_DROP:
                    sequence.retreat_up()
                dropped_at = error.tcp_pose
                if (
                    code == E_DROP
                    and dropped_at is not None
                    and not self.workcell.in_pick_area(dropped_at[:2, 3])
                ):
                    self._pause("The fish was dropped outside the pick area. Put it back or away.")
        except MotionAborted as aborted:
            raise OperatorStop(aborted.reason) from None
        except ArmError as e:
            raise RunStopped(f"recovery after {code} failed: {e}") from None
        if attempts >= s.max_attempts_per_fish:  # CS-62
            where = f"x {fish.middle[0] * 1000:.0f}, y {fish.middle[1] * 1000:.0f} mm"
            self.report.skipped.append(where)
            self.console.info(f"fish at {where} is skipped for the rest of the run")
        if not sequence.step:  # refused while planning: nothing moved, nothing to fault
            return
        self._failed_in_row += 1
        if self._failed_in_row >= s.max_failed_cycles:  # CS-63, CS-64
            self._failed_in_row = 0
            self._pause(f"{s.max_failed_cycles} cycles failed in a row.")

    def _pause(self, message: str) -> None:
        """Alert the operator and wait (CS-64). Raises RunStopped unless they resume."""
        self.console.state("paused", message)
        if not self.console.alert(message):
            raise RunStopped(f"stopped at: {message}")

    def _tick(self) -> str | None:
        """Called while the arm moves: the newest camera frame to the window (with what vision
        sees in it, at most every live_view_s), the stop key."""
        try:
            self._frame = self.vision.frame()
        except CameraError as e:
            return f"camera failure: {e}"
        image = self._live(self._frame)
        self.console.show(image, [self._hold_text] if self._hold_text else None)
        return "operator stop" if self.console.stop_requested() else None

    def _live(self, frame: Frame) -> np.ndarray:
        """The frame with the fish vision finds in it, while the arm moves without a fish
        (a held one fills the view): at most every live_view_s, else the frame as it is."""
        every = self.settings.live_view_s
        holding = self.sequence is not None and self.sequence.holding
        detector = getattr(self.vision, "detector", None)
        locator = getattr(self.vision, "locator", None)
        if every <= 0 or holding or detector is None or locator is None:
            return frame.color
        now = time.monotonic()
        if now - self._live_at < every:
            return frame.color
        self._live_at = now
        camera = self.workcell.camera_of_tcp(self.arm.tcp_pose())
        fish_list = locator.locate(frame, detector.detect(frame.color), camera)
        if self.on_fish_list is not None:
            self.on_fish_list(fish_list, frame)
        return draw_fish_list(frame.color, fish_list, frame.intrinsics)


class _VisionHold:
    """The vision hold monitor as the grasp sequence's HoldCheck."""

    def __init__(self, cycle: PickPlaceCycle):
        self._cycle = cycle
        self._seen = -1

    def start(self) -> bool:
        cycle = self._cycle
        camera = cycle.workcell.camera_of_tcp(cycle.arm.tcp_pose())
        table = float(camera[2, 3] - cycle.workcell.table_z)  # the camera looks about down
        try:
            cues = cycle.vision.hold.start(cycle.vision.frame(), table_distance=table)
        except CameraError:
            cues = []
        cycle.console.info(
            f"vision hold check: {' + '.join(cues)}" if cues else "vision hold check: no cue here"
        )
        return bool(cues)

    def lost(self) -> bool:
        cycle = self._cycle
        frame = cycle._frame
        if frame is None or frame.frame_id == self._seen or not cycle.vision.hold.active:
            return False
        self._seen = frame.frame_id
        status = cycle.vision.hold.update(frame)
        cycle._hold_text = f"hold: {status}"
        if status.dropped:
            cycle.console.info(f"vision: the fish is gone ({status})")
        return status.dropped

    def stop(self) -> None:
        self._cycle.vision.hold.stop()
        self._cycle._hold_text = ""


def _where(xy) -> str:
    return f"x {xy[0] * 1000:.0f}, y {xy[1] * 1000:.0f} mm"


def _reason(fish: Fish) -> str:
    return f"{fish.failure_code} ({fish.detail})" if fish.detail else fish.failure_code


def _in_view(xy, cell: Workcell, frame: Frame, camera: np.ndarray, margin: float = 0.1) -> bool:
    """Whether a point on the table lies well inside the image of a look."""
    point = np.array([xy[0], xy[1], cell.table_z + 0.05])
    in_camera = to_frame(np.linalg.inv(camera), point)[0]
    if in_camera[2] <= 0:
        return False
    k = frame.intrinsics
    u, v = k.project(in_camera)[0]
    return margin * k.width <= u <= (1 - margin) * k.width and (
        margin * k.height <= v <= (1 - margin) * k.height
    )


def _segment_distance(point, start, end) -> float:
    """Distance from a point to the segment start-end, in the plane."""
    point, start, end = (np.asarray(v, float) for v in (point, start, end))
    along = end - start
    share = float(np.clip((point - start) @ along / max(along @ along, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + share * along)))
