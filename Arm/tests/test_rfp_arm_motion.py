"""Tests for the arm code that needs no ROS: poses, kinematics, the workcell and its
limits, the trajectory check, the gripper states and the grasp sequence on the simulated arm."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from arm_helpers import SEED, FakeClock, FakeIO, make_workcell, solve_ik, workcell_data
from rfp_arm import poses
from rfp_arm.fairino import command_text, io_bit
from rfp_arm.grasp import STEPS, GraspSequence, OperatorStop, plan_grasp
from rfp_arm.gripper import (
    NoGripper,
    SerialGripper,
    SimGripper,
    SuctionGripper,
    make_gripper,
    wait_for,
)
from rfp_arm.interfaces import (
    E_DROP,
    E_MOTION,
    E_NO_SEAL,
    E_RELEASE,
    ArmError,
    GripperState,
    MotionAborted,
)
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.safety import check_trajectory
from rfp_arm.sim import SimArm, interpolate
from rfp_arm.workcell import (
    GripperSettings,
    WorkcellError,
    load_workcell,
    workcell_from_mapping,
)

CELL = make_workcell()
SLOT = CELL.slots[0]


# ---------------------------------------------------------------------------
# Poses and kinematics
# ---------------------------------------------------------------------------
def test_pose_conversions_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        xyz, rpy = rng.uniform(-1, 1, 3), rng.uniform(-1.5, 1.5, 3)
        t = poses.transform(xyz, rpy)
        assert np.allclose(t @ poses.invert(t), np.eye(4), atol=1e-12)
        assert np.allclose(poses.matrix_rpy(t), rpy)
        q = poses.quaternion(t)
        assert q[3] >= 0 and np.allclose(poses.from_quaternion(xyz, q), t)
    half_turn = poses.transform((0, 0, 0), (math.pi, 0, 0))  # trace -1: the other branch
    assert np.allclose(poses.from_quaternion((0, 0, 0), poses.quaternion(half_turn)), half_turn)
    assert np.allclose(poses.axis_rotation((0, 0, 2), 0.3), poses.rot_z(0.3))
    assert poses.wrap(3 * math.pi) == pytest.approx(math.pi) and poses.wrap(-math.pi) == math.pi


def test_tool_down_points_the_tool_at_the_table():
    pose = poses.tool_down((0.4, 0.1, 0.3), math.radians(30))
    assert np.allclose(pose[:3, 2], (0, 0, -1)) and np.allclose(pose[:3, 3], (0.4, 0.1, 0.3))
    assert poses.yaw_of(pose) == pytest.approx(math.radians(30))
    assert poses.tilt_from_down(pose) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    turned = poses.tool_down((0.4, 0.1, 0.3), math.radians(120))
    assert poses.rotation_angle(pose, turned) == pytest.approx(math.radians(90))
    middle = interpolate(pose, turned, 0.5)
    assert poses.yaw_of(middle) == pytest.approx(math.radians(75))
    assert np.allclose(interpolate(pose, poses.tool_down((0.6, 0.1, 0.3), 0.5), 0.25)[:3, 3],
                       (0.45, 0.1, 0.3))  # fmt: skip
    back = interpolate(pose, poses.tool_down((0.4, 0.1, 0.3), math.radians(210)), 1.0)  # half turn
    assert poses.yaw_of(back) == pytest.approx(math.radians(-150))


URDF = """<robot name="two">
  <link name="base"/><link name="arm"/><link name="hand"/><link name="tool"/>
  <joint name="swing" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
  <joint name="slide" type="prismatic"><parent link="arm"/><child link="hand"/>
    <origin xyz="1 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.2"/></joint>
  <joint name="mount" type="fixed"><parent link="hand"/><child link="tool"/>
    <origin xyz="0 0 -0.1"/></joint>
</robot>"""


def test_forward_kinematics_from_a_urdf_and_of_the_fr5():
    chain = Chain.from_urdf(URDF, "base", "tool")
    assert chain.joint_names == ["swing", "slide"] and chain.links == ("arm", "hand", "tool")
    tip = chain.fk({"swing": math.pi / 2, "slide": 0.1})
    assert np.allclose(tip[:3, 3], (0, 1.1, 0.4)) and np.allclose(
        chain.fk([0, 0])[:3, 3], (1, 0, 0.4)
    )
    assert chain.limit_violation({"swing": 0.5, "slide": 0.1}) is None
    assert "swing" in chain.limit_violation({"swing": 1.5, "slide": 0.1})
    with pytest.raises(ValueError, match="no chain"):
        Chain.from_urdf(URDF, "tool", "base")
    # the FR5 at zero: straight out along -X (frcobot_ros2's URDF), tip 0.82 m from the base axis
    assert np.allclose(FR5.fk([0] * 6)[:3, 3], (-0.82001, -0.1021, 0.05), atol=1e-4)
    assert FR5.joint_names == list(CELL.joints) and FR5.links[-1] == CELL.tip_link
    target = CELL.tip_of_tcp(poses.tool_down((0.4, -0.1, 0.3), 1.0))
    assert np.allclose(FR5.fk(solve_ik(target)), target, atol=1e-6)


# ---------------------------------------------------------------------------
# Workcell
# ---------------------------------------------------------------------------
def test_default_workcell_loads_unverified_and_consistent():
    cell = load_workcell()
    assert not cell.verified and not cell.calibrated  # placeholders: the real arm refuses them
    assert cell.problems() == [] and cell.group == "fairino5_v6_group"
    assert len(cell.slots) >= 2  # AS-51
    assert cell.pick_area.x[1] - cell.pick_area.x[0] == pytest.approx(0.4)  # SY-15: 500 x 400 mm
    assert cell.pick_area.y[1] - cell.pick_area.y[0] == pytest.approx(0.5)
    assert cell.grasp.pregrasp_distance == 0.10 and cell.motion.tcp_speed_max == 0.25
    assert cell.gripper.driver == "serial" and cell.gripper.serial_port  # the cell's ESP32
    for pose in cell.observation_poses:  # AS-11: straight down from 450 mm
        tcp = cell.observation_tcp(pose)
        camera = cell.camera_of_tcp(tcp)
        assert np.allclose(camera, pose.camera) and camera[2, 3] == pytest.approx(0.45)
        assert np.allclose(cell.tcp_of_tip(cell.tip_of_tcp(tcp)), tcp)


def test_workcell_file_errors_say_what_is_wrong(tmp_path):
    def problems(change) -> str:
        data = workcell_data()
        change(data)
        return "; ".join(workcell_from_mapping(data).problems())

    assert "slot 1 is outside" in problems(lambda d: d["placement"]["slots"].__setitem__(0, [0, 0]))
    assert "at least 2 slots" in problems(lambda d: d["placement"]["slots"].pop())
    assert "overlaps the pick area" in problems(
        lambda d: d["zones"]["placement"].update(y=[0.10, 0.72])
    )
    assert "pick area is not inside" in problems(
        lambda d: d["zones"]["pick_area"].update(x=[0.2, 2])
    )
    assert "needs 6 joints" in problems(
        lambda d: d["observation"]["poses"].__setitem__(0, {"name": "a", "joints_deg": [0, 0]})
    )
    assert "approach_speed" in problems(lambda d: d["motion"].update(approach_speed=0.5))
    assert "serial_port is empty" in problems(lambda d: d["gripper"].update(serial_port=""))
    assert "must be 'serial' or 'controller'" in problems(lambda d: d["gripper"].update(driver="x"))
    assert "observation pose 'right'" in problems(lambda d: d["observation"].update(height=1.5))

    def load(text: str):
        file = tmp_path / "cell.yaml"
        file.write_text(text, encoding="utf-8")
        return load_workcell(file)

    import yaml

    data = workcell_data()
    data["motion"]["tcp_speed"] = 1.0
    with pytest.raises(WorkcellError, match="unknown entry tcp_speed"):
        load(yaml.safe_dump(data))
    data = workcell_data()
    del data["table"]
    with pytest.raises(WorkcellError, match="missing 'table'"):
        load(yaml.safe_dump(data))
    data = workcell_data()
    data["zones"]["work"]["x"] = [1.0, -1.0]
    with pytest.raises(WorkcellError, match="must be"):
        load(yaml.safe_dump(data))
    with pytest.raises(WorkcellError, match="not found"):
        load_workcell(tmp_path / "missing.yaml")
    taught = workcell_data()
    taught["observation"]["poses"] = [{"name": "taught", "joints_deg": [10, -20, 30, 0, -90, 0]}]
    cell = workcell_from_mapping(taught)
    assert cell.observation_poses[0].joints[1] == pytest.approx(math.radians(-20))
    assert cell.observation_tcp(cell.observation_poses[0]) is None


def test_tool_camera_and_held_fish_must_stay_in_the_zones():
    def problem(x, y, z, yaw=0.0, fish=0.0, **options):
        return CELL.pose_problem(poses.tool_down((x, y, z), yaw), fish, **options) or ""

    assert problem(0.4, 0.0, 0.3) == ""
    assert "leave the work zone" in problem(1.4, 0.0, 0.3)
    assert "leave the work zone" in problem(0.4, 0.0, 1.05)  # above it
    assert "exclusion zone 'operator side'" in problem(1.05, 0.0, 0.3)
    assert "TCP would be 2.0 mm above the table" in problem(0.4, 0.0, 0.002)
    assert problem(0.4, 0.0, 0.006) == ""  # 5 mm clearance
    # a fish hangs along the tool's x axis: both of its ends count
    assert problem(0.7, 0.0, 0.3, yaw=math.pi / 2, fish=0.7) == ""
    assert "fish end would enter the exclusion zone" in problem(0.7, 0.0, 0.3, yaw=0.0, fish=0.7)
    assert "fish end would leave the work zone" in problem(0.4, -0.6, 0.3, math.pi / 2, fish=0.7)
    # beside the table the floor doesn't apply, and a path may start there
    assert problem(-0.6, 0.0, -0.2, work_zone=False) == ""
    assert "exclusion" in problem(1.05, 0.0, 0.3, work_zone=False)
    low_camera = make_workcell(lambda d: d["camera"]["mount"].update(xyz=[0.0, -0.09, 0.25]))
    assert "camera would be" in low_camera.pose_problem(poses.tool_down((0.4, 0.0, 0.02), 0.0))


# ---------------------------------------------------------------------------
# Trajectory check
# ---------------------------------------------------------------------------
def joints_for(x, y, z, yaw=1.0, seed=SEED):
    q = solve_ik(CELL.tip_of_tcp(poses.tool_down((x, y, z), yaw)), seed)
    assert q is not None
    return q


def joint_path(start, end, seconds, points=40):
    shares = np.linspace(0, 1, points)[:, None]
    return start + (end - start) * shares, np.linspace(0, seconds, points)


def test_planned_trajectories_are_checked_for_speed_zones_and_posture():
    names = FR5.joint_names
    a, b = joints_for(0.35, -0.2, 0.36), joints_for(0.50, 0.1, 0.20)
    ok = check_trajectory(CELL, FR5, names, *joint_path(a, b, 6.0))
    assert ok.ok and 0.03 < ok.peak_tcp_speed < 0.25 and ok.duration == 6.0
    assert np.allclose(ok.end_tcp[:3, 3], (0.50, 0.1, 0.20), atol=1e-5) and "mm/s" in ok.summary()
    fast = check_trajectory(CELL, FR5, names, *joint_path(a, b, 0.8))
    assert not fast.ok and fast.too_fast and "above 250 mm/s" in fast.problem
    assert check_trajectory(CELL, FR5, names, *joint_path(a, b, 0.8), tcp_speed_max=5.0).ok

    shuffled = [names[i] for i in (3, 0, 5, 1, 4, 2)]  # joint order is the message's, not ours
    positions, times = joint_path(a, b, 6.0)
    assert check_trajectory(CELL, FR5, shuffled, positions[:, [3, 0, 5, 1, 4, 2]], times).ok

    low = joints_for(0.50, 0.1, 0.003, seed=b)
    assert "above the table" in check_trajectory(CELL, FR5, names, *joint_path(b, low, 6.0)).problem
    front = joints_for(0.55, 0.0, 0.3)
    outside = front.copy()  # the base turned 170 degrees: the tool is behind the table's edge
    outside[0] += math.radians(-170 if front[0] > 0 else 170)
    leaving = check_trajectory(CELL, FR5, names, *joint_path(front, outside, 30.0))
    assert not leaving.too_fast and "leave the work zone" in leaving.problem
    entering = check_trajectory(CELL, FR5, names, *joint_path(outside, front, 30.0))
    assert entering.ok  # the arm may start where it was left, outside
    held = check_trajectory(CELL, FR5, names, *joint_path(a, b, 6.0), fish_length=2.5)
    assert "fish end would leave the work zone" in held.problem

    spun = a.copy()  # the wrist turning 210 degrees on the spot: not a move this cell needs
    spun[5] += math.radians(210 if a[5] < 0 else -210)
    spin = check_trajectory(CELL, FR5, names, *joint_path(a, spun, 60.0))
    assert not spin.too_fast and "turn 210 deg" in spin.problem
    beyond = a.copy()
    beyond[2] = 3.0
    assert "joint limit" in check_trajectory(CELL, FR5, names, *joint_path(a, beyond, 9.0)).problem

    assert "no points" in check_trajectory(CELL, FR5, names, np.zeros((0, 6)), np.zeros(0)).problem
    assert (
        "no joint j6"
        in check_trajectory(CELL, FR5, names[:5], np.zeros((2, 5)), np.array([0.0, 1.0])).problem
    )
    positions, times = joint_path(a, b, 6.0)
    assert "backward" in check_trajectory(CELL, FR5, names, positions, times[::-1]).problem


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------
def test_gripper_states_follow_table_as4():
    io, clock = FakeIO(), FakeClock()
    gripper = SuctionGripper(io, GripperSettings(), 1.0, clock, clock.sleep)
    assert gripper.has_switch and gripper.state() == GripperState.VACUUM_OFF
    gripper.vacuum_on()
    assert io.outputs == {1: False, 0: True} and gripper.state() == GripperState.NO_SEAL
    io.inputs = {0: True, 1: True}  # the switch reaches the holding threshold
    assert gripper.state() == GripperState.HOLDING
    assert wait_for(gripper, GripperState.HOLDING, 1.0, clock, clock.sleep)
    gripper.release()  # vacuum off, blow-off pulse of 0.2 s
    assert io.calls[-3:] == [(0, False), (1, True), (1, False)]
    io.inputs = {0: False, 1: True}  # vacuum not gone yet
    assert gripper.state() == GripperState.RELEASING
    clock.advance(1.0)
    assert gripper.state() == GripperState.ERROR  # still there 1 s after release (AS-33)
    io.inputs = {}
    assert gripper.state() == GripperState.VACUUM_OFF
    io.seconds_since_reading = 0.6  # no I/O reading for 500 ms
    assert gripper.state() == GripperState.ERROR
    assert not wait_for(gripper, GripperState.VACUUM_OFF, 0.3, clock, clock.sleep)

    blind = SuctionGripper(FakeIO(), GripperSettings(vacuum_switch=False, blowoff_time=0), 1.0)
    assert not blind.has_switch and blind.state() == GripperState.VACUUM_OFF
    blind.vacuum_on()
    assert blind.state() == GripperState.NO_SEAL  # on, but nothing tells whether it holds
    blind.release()
    assert blind._io.calls[-1] == (0, False)  # no blow-off
    assert NoGripper().state() == GripperState.VACUUM_OFF


class Port:
    """A serial port that remembers what was written to it."""

    def __init__(self, port, baud):
        self.port, self.baud, self.sent, self.broken, self.closed = port, baud, [], False, False

    def write(self, data: bytes) -> None:
        if self.broken:
            raise OSError("device disconnected")
        self.sent.append(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_serial_gripper_sends_one_line_per_switch():
    settings = GripperSettings(
        driver="serial", serial_port="COM7", serial_on="ON", serial_off="OFF"
    )
    naps, ports = [], []

    def open_port(port, baud):
        ports.append(Port(port, baud))
        return ports[-1]

    gripper = SerialGripper(settings, open_port, naps.append)
    port = ports[0]
    assert (port.port, port.baud) == ("COM7", 115200) and naps == [2.0]  # the board restarts
    assert port.sent == [b"OFF\n"] and gripper.state() == GripperState.VACUUM_OFF  # a known start
    assert not gripper.has_switch  # nothing tells whether the cups hold: the camera has to
    gripper.vacuum_on()
    assert port.sent[-1] == b"ON\n" and gripper.state() == GripperState.NO_SEAL
    gripper.release()
    assert port.sent[-1] == b"OFF\n" and gripper.state() == GripperState.VACUUM_OFF
    port.broken = True
    with pytest.raises(ArmError, match="stopped answering") as error:
        gripper.vacuum_on()
    assert error.value.code == "E_COMM" and gripper.state() == GripperState.VACUUM_OFF
    gripper.close()
    assert port.closed

    def no_port(port, baud):
        raise OSError("could not open port 'COM7'")

    with pytest.raises(ArmError, match="can't open the gripper's port"):
        SerialGripper(settings, no_port, naps.append)
    assert isinstance(make_gripper(GripperSettings(), FakeIO()), SuctionGripper)
    with pytest.raises(ArmError, match="command server"):
        make_gripper(GripperSettings(), None)


def test_grasp_with_the_serial_gripper_relies_on_the_camera():
    settings = GripperSettings(driver="serial", serial_port="COM7")
    gripper = SerialGripper(settings, Port, lambda seconds: None)
    hold = Hold()
    rig = Rig(gripper=gripper, hold_check=hold)
    rig.run()
    assert tuple(rig.steps) == STEPS and hold.events == ["start", "stop"]
    assert [bytes(line) for line in gripper._port.sent] == [b"0\n", b"1\n", b"0\n"]
    blind = Rig(gripper=SerialGripper(settings, Port, lambda seconds: None))
    assert blind.fails().code == E_NO_SEAL  # no camera cue either: it doesn't lift


def test_controller_commands_and_io_bits():
    assert (
        command_text("SetDO", 0, 1) == "SetDO(0,1)" and command_text("StopMotion") == "StopMotion()"
    )
    assert io_bit(0b00000101, 0, 0) and not io_bit(0b00000101, 0, 1) and io_bit(0b00000101, 0, 2)
    assert io_bit(0, 0b00000010, 9) and not io_bit(0xFF, 0, 8)
    with pytest.raises(ValueError):
        io_bit(0, 0, 16)


# ---------------------------------------------------------------------------
# Grasp plan
# ---------------------------------------------------------------------------
def fish_pose(x=0.40, y=-0.10, heading_deg=100.0, thickness=0.10):
    return poses.tool_down((x, y, CELL.table_z + thickness), math.radians(heading_deg))


def plan(heading_deg=100.0, known=True, tool_yaw_deg=90.0, dry_run=False, length=0.60, **fish):
    thickness = fish.get("thickness", 0.10)
    return plan_grasp(CELL, fish_pose(heading_deg=heading_deg, **fish), known, length, thickness,
                      SLOT, math.radians(tool_yaw_deg), dry_run)  # fmt: skip


def test_grasp_plan_follows_table_as1():
    p = plan()
    heights = {name: round(pose[2, 3], 4) for name, pose in p.poses().items()}
    assert heights == {
        "pre-grasp": 0.20,  # 100 mm above the grasp point
        "approach": 0.09,  # 10 mm past it
        "lift": 0.41,  # 60% of the length + 50 mm above the table
        "carry": 0.41,
        "lower": 0.115,  # thickness + 15 mm: at most 20 mm above where the fish rests (AS-24)
        "retreat": 0.215,
    }
    for name in ("pre-grasp", "approach", "lift"):
        assert np.allclose(p.poses()[name][:2, 3], (0.40, -0.10))  # straight up and down (AS-23)
    for name in ("carry", "lower", "retreat"):
        assert np.allclose(p.poses()[name][:2, 3], SLOT)
    assert all(poses.tilt_from_down(pose) < 1e-9 for pose in p.poses().values())
    assert plan(dry_run=True).approach_end[2, 3] == pytest.approx(0.15)  # 50 mm short (CS-93)
    assert plan(length=0.75).lift[2, 3] == pytest.approx(0.50)  # capped at lift_height_max


def test_grasp_plan_turns_the_fish_to_the_placement_heading():
    def yaws(**options):
        p = plan(**options)
        return [
            round(math.degrees(v))
            for v in (poses.yaw_of(p.pregrasp), poses.yaw_of(p.lower), p.turn)
        ]

    assert yaws(heading_deg=100) == [100, 0, -100]  # cups along the body, head ends toward +X
    # head pointing away from the tool's x axis: grasp the other way round, same result
    assert yaws(heading_deg=-80) == [100, 180, 80]
    assert yaws(heading_deg=100, tool_yaw_deg=-90) == [-80, 180, -100]
    # head end unknown: the body ends along X by the shorter turn
    assert yaws(heading_deg=100, known=False) == [100, 180, 80]
    assert yaws(heading_deg=60, known=False) == [60, 0, -60]


def test_grasp_plan_refuses_what_the_limits_forbid():
    def refused(**options) -> str:
        with pytest.raises(ArmError) as error:
            plan(**options)
        assert error.value.code == E_MOTION
        return error.value.message

    assert "outside the pick area" in refused(x=0.70)
    assert "above the table" in refused(thickness=0.0)  # the cups would be pressed into the table
    assert "fish end would enter the exclusion zone" in refused(length=1.3)
    tilted = fish_pose()
    tilted[:3, :3] = tilted[:3, :3] @ poses.rot_x(math.radians(30))
    with pytest.raises(ArmError, match="off vertical"):
        plan_grasp(CELL, tilted, True, 0.6, 0.1, SLOT, 0.0)
    with pytest.raises(ArmError, match="slot is outside"):
        plan_grasp(CELL, fish_pose(), True, 0.6, 0.1, (0.4, 0.0), 0.0)


# ---------------------------------------------------------------------------
# Grasp sequence on the simulated arm
# ---------------------------------------------------------------------------
class Rig:
    """A simulated arm, a gripper and a GraspSequence wired together."""

    def __init__(self, gripper=None, **options):
        self.clock = FakeClock()
        self.arm = SimArm(CELL)
        self.gripper = gripper or SimGripper()
        self.steps: list[str] = []
        self.ticks = 0
        self.at_tick = {}  # tick number -> action
        self.sequence = GraspSequence(
            self.arm, self.gripper, CELL, tick=self.tick, on_step=self.steps.append,
            clock=self.clock, sleep=self.clock.sleep, **options,
        )  # fmt: skip

    def tick(self):
        self.ticks += 1
        self.clock.advance(0.05)
        action = self.at_tick.get(self.ticks)
        return action() if action else None

    def run(self, **options):
        self.sequence.run(plan(**options))

    def fails(self, **options) -> ArmError:
        with pytest.raises(ArmError) as error:
            self.run(**options)
        return error.value


def test_grasp_sequence_runs_the_nine_steps_in_order():
    rig = Rig()
    rig.run()
    assert tuple(rig.steps) == STEPS and rig.sequence.step == "" and not rig.sequence.holding
    assert rig.gripper.events == ["vacuum on", "release"]
    assert [kind for kind, _ in rig.arm.moves] == ["ptp", "lin", "lin", "lin", "lin", "lin"]
    assert np.allclose(rig.arm.tcp_pose()[:3, 3], (*SLOT, 0.215)) and rig.arm._fish_length == 0.0


def test_dry_run_moves_without_vacuum_and_stops_short():
    rig = Rig(gripper=NoGripper())
    rig.run(dry_run=True)
    assert rig.steps == ["pre-grasp", "approach", "lift", "carry", "lower", "retreat"]
    assert rig.arm.moves[1][1][2, 3] == pytest.approx(0.15) and not rig.sequence.holding


def test_no_seal_stops_at_confirm_hold_and_can_be_recovered():
    rig = Rig(gripper=SimGripper(seals=False))
    error = rig.fails()
    assert error.code == E_NO_SEAL and rig.sequence.step == "confirm hold"
    assert error.gripper_state == GripperState.NO_SEAL and error.tcp_pose[2, 3] == pytest.approx(
        0.09
    )
    assert rig.steps[-1] == "confirm hold" and not rig.sequence.holding
    rig.sequence.let_go()
    rig.sequence.retreat_up()
    assert rig.gripper.state() == GripperState.VACUUM_OFF
    assert rig.arm.tcp_pose()[2, 3] == pytest.approx(0.19)


def test_a_fish_lost_on_the_way_is_a_drop_and_the_vacuum_goes_off():
    rig = Rig()
    rig.at_tick[30] = rig.gripper.drop  # during the lift or carry
    error = rig.fails()
    assert error.code == E_DROP and rig.sequence.step in ("lift", "carry")
    assert not error.still_holding and not rig.sequence.holding
    assert rig.gripper.events == ["vacuum on", "release"] and rig.arm._fish_length == 0.0
    assert 3 <= rig.ticks - 30 <= 4  # holding not reported for more than 100 ms (AS-2)


class Hold:
    """A HoldCheck that reports the fish gone at its nth question."""

    def __init__(self, lost_at=None, sees=True):
        self.lost_at, self.sees, self.asked, self.events = lost_at, sees, 0, []

    def start(self) -> bool:
        self.events.append("start")
        return self.sees

    def lost(self) -> bool:
        self.asked += 1
        return self.lost_at is not None and self.asked >= self.lost_at

    def stop(self) -> None:
        self.events.append("stop")


def test_vision_drop_with_the_switch_still_holding_keeps_the_vacuum_on():
    hold = Hold(lost_at=12)
    rig = Rig(hold_check=hold)
    error = rig.fails()
    assert error.code == E_DROP and error.still_holding and "vacuum left on" in error.message
    assert rig.gripper.state() == GripperState.HOLDING and rig.sequence.holding
    assert hold.events == ["start"]
    rig.sequence.let_go()  # the operator's decision
    assert rig.gripper.state() == GripperState.VACUUM_OFF and hold.events == ["start", "stop"]


class BlindGripper(SimGripper):
    has_switch = False  # no vacuum switch: the state can't confirm a hold


def test_without_a_switch_vision_confirms_the_hold_and_reports_the_drop():
    rig = Rig(gripper=BlindGripper(), hold_check=Hold(sees=False))
    assert "can't be confirmed" in rig.fails().message and rig.sequence.step == "confirm hold"
    Rig(gripper=BlindGripper(), require_hold_confirmation=False).run()  # lift blind, if allowed
    hold = Hold()
    Rig(gripper=BlindGripper(), hold_check=hold).run()
    assert hold.events == ["start", "stop"] and hold.asked > 20  # asked all the way
    rig = Rig(gripper=BlindGripper(), hold_check=Hold(lost_at=5))
    error = rig.fails()
    assert error.code == E_DROP and not error.still_holding  # nothing contradicts vision
    assert rig.gripper.state() == GripperState.VACUUM_OFF


class StuckGripper(SimGripper):
    def release(self) -> None:
        self.events.append("release")  # the valve doesn't open: the vacuum stays


def test_a_fish_that_does_not_come_off_is_a_release_error():
    rig = Rig(gripper=StuckGripper())
    error = rig.fails()
    assert error.code == E_RELEASE and rig.sequence.step == "release" and rig.sequence.holding
    assert rig.steps[-1] == "release" and error.gripper_state == GripperState.HOLDING


def test_a_failed_move_is_planned_again_once_then_the_fish_is_put_back():
    def block_carry(times):
        calls = []

        def fail(kind, target):
            if np.allclose(target[:2, 3], SLOT) and target[2, 3] > 0.3:
                calls.append(kind)
                return "no straight path" if len(calls) <= times else None
            return None

        return fail

    rig = Rig()
    rig.arm.fail = block_carry(1)
    rig.run()  # the second plan works
    assert tuple(rig.steps) == STEPS

    rig = Rig()
    rig.arm.fail = block_carry(2)
    error = rig.fails()
    assert error.code == E_MOTION and rig.sequence.step == "carry"
    assert error.fish_put_back and not error.still_holding and not rig.sequence.holding
    assert rig.gripper.events == ["vacuum on", "release"]
    assert np.allclose(rig.arm.moves[-2][1][:3, 3], (0.40, -0.10, 0.115))  # set down where it was
    assert rig.arm.tcp_pose()[2, 3] == pytest.approx(0.215)

    rig = Rig()
    rig.arm.fail = lambda kind, target: "unreachable" if kind == "ptp" else None
    error = rig.fails()  # before the fish is held: nothing to put back
    assert rig.sequence.step == "pre-grasp" and not error.fish_put_back and not error.still_holding


def test_the_operator_can_stop_at_any_time():
    rig = Rig()
    rig.at_tick[40] = lambda: "operator stop"
    with pytest.raises(OperatorStop, match="operator stop"):
        rig.run()
    assert rig.sequence.holding and rig.gripper.state() == GripperState.HOLDING  # nothing dropped
    rig = Rig()
    rig.at_tick[40] = rig.arm.stop
    with pytest.raises(OperatorStop, match="stopped"):
        rig.run()


def test_simulated_arm_applies_the_limits():
    arm = SimArm(CELL)
    start = arm.tcp_pose()
    assert np.allclose(start, CELL.observation_tcp(CELL.observation_poses[0]))
    with pytest.raises(ArmError, match="work zone"):
        arm.move_ptp(poses.tool_down((2.0, 0, 0.3), 0.0))
    with pytest.raises(ArmError, match="speed limit"):
        arm.move_lin(poses.tool_down((0.4, 0, 0.3), 0.0), 0.5, 0.5)
    with pytest.raises(ArmError, match="exclusion zone"):  # a straight line through a zone
        blocked = make_workcell(
            lambda d: d["zones"]["exclusion"].append(
                {"name": "post", "x": [0.38, 0.42], "y": [0.10, 0.14], "z": [0.0, 1.0]}
            )
        )
        SimArm(blocked).move_lin(poses.tool_down((0.40, 0.40, 0.36), 0.0), 0.1, 0.5)
    assert np.allclose(arm.tcp_pose(), start) and arm.moves == []
    with pytest.raises(MotionAborted, match="enough"):
        arm.move_lin(poses.tool_down((0.4, 0.2, 0.3), 0.0), 0.1, 0.5, tick=lambda: "enough")
    assert not np.allclose(arm.tcp_pose(), start)  # it stopped on the way
    arm.move_joints(solve_ik(CELL.tip_of_tcp(poses.tool_down((0.4, 0.0, 0.3), 0.5))))
    assert np.allclose(arm.tcp_pose()[:3, 3], (0.4, 0.0, 0.3), atol=1e-6) and arm.fault() is None
    naps = []
    paced = SimArm(CELL, realtime=True, sleep=naps.append)
    target = paced.tcp_pose()
    target[:3, 3] += (0.0, 0.2, -0.1)
    paced.move_lin(target, 0.1, 0.5)
    assert sum(naps) == pytest.approx(math.hypot(0.2, 0.1) / 0.1)  # the distance at 0.1 m/s
