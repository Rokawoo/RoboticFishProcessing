"""
The pick-and-place cycle (SY-20, CS-10 to CS-64): look at the table from the next
observation pose, let the selector choose a fish, check its answer, look again, run the
grasp sequence, and answer every failure as in Table CS-1. It talks to the arm, the
gripper, the vision system and the operator only through their interfaces, so the same
cycle runs offline, in simulation, as a dry run and with vacuum (CS-90).

Not done here yet: the payload update (CS-45) and the VLM (a FishSelector, selector.py).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from rfp_arm.carry import Footprint
from rfp_arm.grasp import (
    WRIST_MOVE,
    GraspSequence,
    OperatorStop,
    plan_grasp,
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
from rfp_arm.poses import wrap, yaw_of
from rfp_arm.workcell import ObservationPose, Workcell
from rfp_vision.camera import CameraError
from rfp_vision.fish_list import Fish, FishList, heading_difference, match_fish
from rfp_vision.geometry import inside_polygon, pixels_on_plane
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
# An outline this much (share of its points) inside a footprint grown this much (m) is the
# same fish: seen again, or in part at the image's edge
SAME_FISH_SHARE = 0.8
SAME_FISH_GROW = 0.03
# Steps after which a failed grasp may have left the fish off the pick area (a slot, the way)
AFTER_LIFT = ("lift", "carry", "lower", "release", "retreat")


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
        the arm that moves: simulation, offline). `controller`: sends a command line to
        the arm's controller (FairinoIO.send), for the grasp settings' press commands.
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
        self._pose_index = 0
        self._slot = 0
        self._empty = 0  # observations in a row without any fish
        self._unusable = 0  # observations in a row with fish but no candidate
        self._reobserves = 0
        self._moved = 0
        self._failed_in_row = 0
        self._last_result = ""
        self._last_failed: np.ndarray | None = None  # where the fish that just failed lies
        self._tool_yaw = 0.0
        self._frame = None  # the newest camera frame seen by the tick
        self._hold_text = ""
        # What lies on the table, for a low carry: this cycle's looks, the fish placed since
        # the placement zone was last cleared, and why the table isn't known ("": it is)
        self._looks: list[Observation] = []
        self._placed: list[Footprint] = []
        self._unknown = ""
        fish_type = getattr(getattr(vision, "settings", None), "fish", None)
        self._rigid = bool(getattr(fish_type, "rigid", False))  # unknown: it may droop

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
        """One observation and, if a fish is chosen, one grasp. False ends the run."""
        s = self.settings
        poses = self.workcell.observation_poses
        pose = poses[self._pose_index]
        self._looks = []
        observation = self._observe(pose)
        fish_list = observation.fish_list
        candidates = self._candidates(fish_list)
        # Fish seen outside the pick area (placed ones at the frame's edge) don't count (SY-22)
        in_area = [f for f in fish_list.fish if self._maybe_in_pick_area(f)]

        if not in_area:
            self._empty += 1
            self._next_pose()
            if self._empty >= s.empty_passes_to_end * len(poses):
                self.report.ended = "pick area clear"  # SY-22
                return False
            return True
        self._empty = 0
        if not candidates:  # cut off at the image's edge: look again, centered on one
            closer = self._look_closer(fish_list)
            if closer is not None:
                pose, observation = closer
                fish_list = observation.fish_list
                candidates = self._candidates(fish_list)
        if not candidates:
            self._unusable += 1
            self._next_pose()
            if self._unusable >= len(poses):  # a full pass with fish but none to pick
                self._unusable = 0
                # why, as the last look saw it (a closer look sees more than the first)
                seen = [f for f in fish_list.fish if self._maybe_in_pick_area(f)] or in_area
                self._pause("Fish are on the table but none can be picked: " + _why(seen))
            return True
        self._unusable = 0
        if self._slot >= len(self.workcell.slots):
            self._pause("The placement zone is full (AS-56). Clear it, then resume.")
            self._slot, self._placed, self._unknown = 0, [], ""
            return True  # look again: someone was at the table

        decision = self._decide(observation, candidates)
        if decision.action == "reobserve":
            self._next_pose()
            return True
        if decision.action == "end":
            self._pause("The selector ended the run, but candidate fish remain.")
            return True
        chosen = fish_list.by_id(decision.fish_id)
        fish = self._recheck(chosen, pose)
        if fish is None:
            self._moved += 1
            self.console.info(f"fish {chosen.fish_id} isn't where it was: new decision (CS-41)")
            if self._moved >= MAX_MOVED:
                self._moved = 0
                self._pause("The chosen fish keeps moving between two looks.")
            return True
        self._moved = 0
        self.report.cycles += 1
        self._grasp(fish, decision, pose, len(candidates))
        return True

    def _next_pose(self) -> None:
        self._pose_index = (self._pose_index + 1) % len(self.workcell.observation_poses)

    # -- observing ---------------------------------------------------------------
    def _observe(self, pose: ObservationPose, move: bool = True) -> Observation:
        """Go to an observation pose (CS-14), wait for the arm to rest, get a fish list."""
        self.console.state("observing", f"pose '{pose.name}'")
        if move:
            self._move(lambda: self._go_to(pose), f"can't reach observation pose '{pose.name}'")
            self._sleep(self.settings.settle_s)
        tcp = self.arm.tcp_pose()
        self._tool_yaw = yaw_of(tcp)
        try:
            observation = self.vision.observe(self.workcell.camera_of_tcp(tcp))
        except CameraError as e:
            raise RunStopped(f"camera failure: {e}") from None
        self._looks.append(observation)
        self._frame = observation.frame
        self.vision.hold.set_background(observation.frame)  # nothing is held here
        self._show(observation)
        return observation

    def _look_closer(self, fish_list: FishList) -> tuple[ObservationPose, Observation] | None:
        """
        The biggest fish in the pick area that vision saw cut off at the image's edge, looked
        at again from straight above its middle with the image's long side along it, and
        again (aimed at what that look saw) while it is still cut off, up to close_looks
        times. The pose and the observation from there, or None if no fish needs it or the
        arm can't get there.
        """
        s, cell = self.settings, self.workcell
        if cell.close_look_height is None or s.close_looks < 1:
            return None
        clipped = [
            fish
            for fish in fish_list.fish
            if "out_of_frame" in fish.failures
            and fish.seen_at is not None
            and cell.pick_area.contains(fish.seen_at[:2])
            and self._attempts.count(np.array(fish.seen_at[:2])) < s.max_attempts_per_fish
        ]
        if not clipped:
            return None
        seen_at = max(clipped, key=lambda fish: _area(fish.mask_polygon)).seen_at
        found = None
        for _ in range(s.close_looks):
            pose = self._close_pose(seen_at)
            if pose is None:
                break
            self.console.state("observing", f"pose '{pose.name}'")
            try:
                self.arm.move_ptp(cell.observation_tcp(pose), self._tick)
            except MotionAborted as aborted:
                raise OperatorStop(aborted.reason) from None
            except ArmError as e:
                if e.code != E_MOTION:
                    raise RunStopped(f"{pose.name}: {e}") from None
                self.console.info(f"no closer look: {e}")
                break
            self._sleep(s.settle_s)
            found = (pose, self._observe(pose, move=False))
            nearest = min(
                (f for f in found[1].fish_list.fish if f.seen_at is not None),
                key=lambda f: math.dist(f.seen_at[:2], seen_at[:2]),
                default=None,
            )
            if nearest is None or "out_of_frame" not in nearest.failures:
                break
            seen_at = nearest.seen_at
        return found

    def _close_pose(self, seen_at: tuple[float, float, float]) -> ObservationPose | None:
        """The look centered on a fish seen at (x, y, body axis): of the two ways to turn the
        camera along the fish, the one inside the limits with the wrist nearest its middle.
        There the camera is on the side away from the arm, and the grasp that follows takes
        the same way round (plan_grasp keeps the empty gripper near the middle too), so the
        wrist doesn't turn round to look and back to grab."""
        cell = self.workcell
        x, y, axis = seen_at
        low, high = cell.wrist_range
        middle = (low + high) / 2
        options = []
        for yaw in (axis, axis + math.pi):
            camera = cell.close_look_camera((x, y), yaw)
            tcp = cell.tcp_of_camera(camera)
            if cell.pose_problem(tcp) is not None:
                continue
            wrist = self.arm.wrist_angle(tcp)
            if wrist is not None and not low <= wrist <= high:
                continue
            if wrist is not None:
                away = abs(wrist - middle)
            else:  # the arm can't tell: the least turn of the tool from now
                away = abs(wrap(yaw_of(tcp) - yaw_of(self.arm.tcp_pose())))
            options.append((away, camera))
        if not options:
            return None
        camera = min(options, key=lambda option: option[0])[1]
        name = f"close look at x {x * 1000:.0f}, y {y * 1000:.0f} mm"
        return ObservationPose(name, camera=camera)

    def _go_to(self, pose: ObservationPose) -> None:
        """To an observation pose or a closer look. Back from a place at one end of the
        wrist's range the turn there can be longer than WRIST_MOVE (vacuum tube; the joint
        travel limit): the wrist turns back to its middle where the tool is first."""
        tcp = None if pose.joints is not None else self.workcell.observation_tcp(pose)
        there = pose.joints[-1] if pose.joints is not None else self.arm.wrist_angle(tcp)
        now = self.arm.wrist_angle()
        if there is not None and now is not None and abs(there - now) > WRIST_MOVE:
            back = wrist_back_pose(self.arm, self.workcell)
            if back is not None and self.workcell.pose_problem(back) is None:
                self.arm.move_ptp(back, self._tick)
        if pose.joints is not None:
            self.arm.move_joints(pose.joints, self._tick)
        else:
            self.arm.move_ptp(tcp, self._tick)

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
            if cell.pick_area.contains(fish.middle)
            and not any(zone.contains(fish.grasp_point) for zone in cell.exclusion_zones)
            and self._attempts.count(fish.middle) < limit
        ]

    def _maybe_in_pick_area(self, fish: Fish) -> bool:
        """Where vision put the fish (the middle of its outline in {B}, else of its bounding box);
        a fish it couldn't place at all counts as in the pick area."""
        if fish.middle is not None:
            point = fish.middle[:2]
        elif fish.bbox is not None:
            point = ((fish.bbox[0] + fish.bbox[1]) / 2, (fish.bbox[2] + fish.bbox[3]) / 2)
        else:
            return True
        return self.workcell.pick_area.contains(point)

    def _show(self, observation: Observation, chosen: int | None = None) -> np.ndarray:
        fish_list = observation.fish_list
        frame = observation.frame
        image = draw_fish_list(frame.color, fish_list, frame.intrinsics, chosen)
        self.console.show(image, [f"{len(fish_list.valid)} valid of {len(fish_list.fish)} fish"])
        return image

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
        if decision.action == "reobserve":
            self._reobserves += 1
            if self._reobserves >= s.max_reobserve:
                decision = fallback_choice(candidates)
        if decision.action != "reobserve":
            self._reobserves = 0
        fish = f" fish {decision.fish_id}" if decision.fish_id is not None else ""
        self.console.info(f"{decision.source}: {decision.action}{fish} ({decision.reason})")
        return decision

    def _recheck(self, chosen: Fish, pose: ObservationPose) -> Fish | None:
        """The chosen fish in a new fish list from the same pose, if it didn't move (CS-41)."""
        s = self.settings
        observation = self._observe(pose, move=False)
        fish = match_fish(chosen, observation.fish_list.valid, s.recheck_position_m)
        if fish is None or heading_difference(fish, chosen) > math.radians(s.recheck_heading_deg):
            return None
        if time.monotonic() - observation.fish_list.stamp > s.fish_list_max_age_s:  # CS-53
            return None
        image = self._show(observation, fish.fish_id)
        if self.log is not None and s.save_images:
            self.log.image(f"cycle_{self.report.cycles + 1:03d}", image)
        return fish

    # -- grasping ----------------------------------------------------------------
    def _on_table(self, fish: Fish) -> tuple[Footprint | None, list[Footprint] | None, str]:
        """For a low carry: the chosen fish's footprint, and everything else on the table as
        this cycle's looks and this run's placements show it; None and why where it isn't
        known. A fish cut off at the image's edge may reach anywhere past it, unless another
        look or a slot holds all of it."""
        if fish.outline is None or fish.seen_at is None:
            return None, None, "where it lies isn't known"
        carried = Footprint.around("the fish", fish.outline, fish.seen_at[2])
        if self._unknown:
            return carried, None, self._unknown
        if not any(self._sees_pick_area(look) for look in self._looks):
            return carried, None, "no look saw all of the pick area"
        seen, cut = [], []
        for look in self._looks:
            for other in look.fish_list.fish:
                if other.outline is None or other.seen_at is None:
                    return carried, None, f"where fish {other.fish_id} lies isn't known"
                if _mostly_in(other.outline, carried):  # the fish itself, again or in part
                    continue
                if "out_of_frame" in other.failures:
                    cut.append(other)
                else:
                    name = f"fish {other.fish_id}"
                    seen.append(Footprint.around(name, other.outline, other.seen_at[2]))
        known = seen + self._placed
        for other in cut:
            if not any(_mostly_in(other.outline, footprint) for footprint in known):
                return carried, None, f"fish {other.fish_id} is cut off at the image's edge"
        return carried, known, ""

    def _sees_pick_area(self, look: Observation) -> bool:
        """Whether a look's image holds all of the pick area (on the table): every fish
        lying there is in it, at least in part."""
        frame = look.frame
        w, h = frame.size
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        camera = look.fish_list.camera_pose
        view = pixels_on_plane(frame.intrinsics, camera, corners, self.workcell.table_z)[:, :2]
        area = self.workcell.pick_area
        inside = np.array([(x, y) for x in area.x for y in area.y])
        # millimeters: pointPolygonTest works in float32
        return bool(np.isfinite(view).all() and inside_polygon(view * 1e3, inside * 1e3).all())

    def _grasp(
        self, fish: Fish, decision: Decision, pose: ObservationPose, candidates: int
    ) -> None:
        cell = self.workcell
        started = self._clock()
        point = fish.grasp_point
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
        carried, on_table, unknown = self._on_table(fish)
        try:
            plan = plan_grasp(
                cell, fish.grasp_pose, fish.heading_known, fish.total_length, fish.thickness,
                cell.slots[self._slot], self._tool_yaw, self.dry_run, self.arm.wrist_angle,
                fish.middle, carried=carried, on_table=on_table, rigid=self._rigid,
            )  # fmt: skip
            if plan.wrist is not None:
                path = [round(math.degrees(a)) for a in plan.wrist]
                row["wrist_deg"] = "/".join(str(a) for a in path)
                self.console.info(
                    f"wrist (j6) {path[0]} -> {path[1]} -> {path[2]} -> {path[0]} deg: "
                    f"sweeps {math.degrees(plan.wrist_sweep):.0f} deg"
                )
            height = round((plan.carry[2, 3] - cell.table_z) * 1000)
            row["carry_mm"] = height
            why = (plan.carry_note or unknown) if cell.grasp.low_carry else ""
            self.console.info(f"carried {height} mm above the table" + (f": {why}" if why else ""))
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
            if carried is not None:
                name = f"the fish in slot {self._slot + 1}"
                self._placed.append(carried.moved(plan.lift, plan.lower, name))
            else:
                self._unknown = "where a placed fish lies isn't known"
            self._slot += 1
            self.report.successes += 1
            self._failed_in_row = 0
            self._last_result, self._last_failed = "success", None

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
        # The next looks see a fish left in the pick area; one dropped elsewhere the operator
        # is asked to clear. Set down or let go anywhere else (a slot, on the way): not known
        where = error.tcp_pose
        if (
            sequence.step in AFTER_LIFT
            and (code != E_DROP or error.still_holding)
            and (where is None or not self.workcell.pick_area.contains(where[:2, 3]))
        ):
            self._unknown = "a fish may lie outside the pick area since a failed grasp"
        self.console.info(
            f"❌ {error} (step '{step}', attempt {attempts} of {s.max_attempts_per_fish})"
        )
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
            elif sequence.step and not error.fish_put_back:  # the arm is somewhere in the grasp
                sequence.let_go()
                if code != E_DROP:
                    sequence.retreat_up()
                dropped_at = error.tcp_pose
                if (
                    code == E_DROP
                    and dropped_at is not None
                    and not self.workcell.pick_area.contains(dropped_at[:2, 3])
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
        """Called while the arm moves: newest camera frame to the window, the stop key."""
        try:
            self._frame = self.vision.frame()
        except CameraError as e:
            return f"camera failure: {e}"
        self.console.show(self._frame.color, [self._hold_text] if self._hold_text else None)
        return "operator stop" if self.console.stop_requested() else None


class _VisionHold:
    """The vision hold monitor as the grasp sequence's HoldCheck."""

    def __init__(self, cycle: PickPlaceCycle):
        self._cycle = cycle
        self._seen = -1

    def start(self) -> bool:
        cycle = self._cycle
        try:
            cues = cycle.vision.hold.start(cycle.vision.frame())
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


def _mostly_in(outline: np.ndarray, footprint: Footprint) -> bool:
    """Whether an outline lies within a footprint: the same fish."""
    return float(footprint.holds(outline, SAME_FISH_GROW).mean()) >= SAME_FISH_SHARE


def _area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _why(fish: list[Fish]) -> str:
    """The failure codes of the fish, for an alert (CS-73)."""
    codes = [f"fish {f.fish_id}: {f.failure_code}" for f in fish if not f.vision_ok]
    return "; ".join(codes) if codes else "attempt limit reached for each (CS-62)"
