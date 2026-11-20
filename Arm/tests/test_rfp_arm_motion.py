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
    SimGripper,
    SuctionGripper,
    ValveGripper,
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
from rfp_arm.poses import interpolate
from rfp_arm.safety import check_trajectory
from rfp_arm.sim import SimArm
from rfp_arm.valve import HttpValve, SerialValve, ValveError
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


def test_inverse_kinematics_keeps_the_posture_and_link_poses_chain_up():
    seed = SEED.copy()
    for xyz, yaw in (
        ((0.40, -0.10, 0.30), 1.0),
        ((0.55, 0.10, 0.20), 0.4),
        ((0.30, -0.30, 0.40), 2.0),
    ):
        target = CELL.tip_of_tcp(poses.tool_down(xyz, yaw))
        q = FR5.ik(target, seed)
        assert q is not None and np.allclose(FR5.fk(q), target, atol=1e-6)
        assert np.abs(q - seed)[:5].max() < math.pi  # the same posture (the wrist turns)
        seed = q
    assert FR5.ik(CELL.tip_of_tcp(poses.tool_down((2.0, 0.0, 0.3), 0.0)), SEED) is None  # too far
    links = FR5.link_poses(seed)
    assert (
        len(links) == 7
        and np.allclose(links[0], np.eye(4))
        and np.allclose(links[-1], FR5.fk(seed))
    )


def test_inverse_kinematics_reaches_far_swings_in_the_same_posture():
    # From the lift over a fish to the far slot the base swings 83 deg: one solve overshoots
    # into another posture and gets nowhere. In steps it gets there as the wrist predicted.
    from rfp_arm.kinematics import wrist_angle_at

    lift = np.radians([-17, -71, -102, -97, 90, 1])
    target = CELL.tip_of_tcp(poses.tool_down((0.38, 0.60, 0.38), 0.0))
    q = FR5.ik(target, lift)
    assert q is not None and np.allclose(FR5.fk(q), target, atol=1e-6)
    assert q[4] == pytest.approx(lift[4], abs=0.05)  # the wrist didn't flip over
    predicted = wrist_angle_at(FR5, dict(zip(FR5.joint_names, lift)), target)
    assert q[5] == pytest.approx(predicted, abs=1e-4)


def test_a_plan_is_checked_pose_by_pose_from_the_arm_s_joints_before_it_runs():
    import dataclasses

    from rfp_arm.grasp import plan_problem
    from rfp_arm.kinematics import wrist_angle_at

    start = joints_for(0.40, -0.10, 0.36, yaw=1.2)

    def wrist(pose):  # as the arm predicts it (Arm.wrist_angle)
        if pose is None:
            return float(start[-1])
        return wrist_angle_at(FR5, dict(zip(FR5.joint_names, start)), CELL.tip_of_tcp(pose))

    for x, y, heading in ((0.30, -0.25, 0.0), (0.45, 0.05, 2.0), (0.55, -0.20, -1.0)):
        fish = poses.tool_down((x, y, 0.10), heading)
        plan = plan_grasp(CELL, fish, True, 0.6, 0.1, tuple(CELL.slots[0]), 0.0, wrist=wrist)
        assert plan_problem(plan, CELL, FR5, start) is None
    # In the posture folded back over the base (the IK took it once, near the base) the
    # gripper comes against the forearm: such a plan is refused before anything moves
    folded = np.radians([93, -34, -58, -178, 90, 9])
    into = CELL.tcp_of_tip(FR5.fk(np.radians([100, -34, -56, -180, 90, -10])))
    bad = dataclasses.replace(plan, pregrasp=into)
    assert (
        plan_problem(bad, CELL, FR5, folded) == "the gripper would hit the forearm at 'pre-grasp'"
    )
    far = dataclasses.replace(plan, lift=poses.tool_down((1.5, 0.0, 0.3), 0.0))
    assert "no joint solution at 'lift'" in plan_problem(far, CELL, FR5, start)


def test_near_the_base_the_arm_swings_round_rather_than_fold_back_over_it():
    # Close to the base the posture folded back over it is nearer in joint angles: a solve
    # from here took it (the base turned 4 deg, not 107), and in that posture the gripper
    # later ran into the forearm over fish near the base. The wrist's prediction agrees.
    from rfp_arm.kinematics import wrist_angle_at

    here = np.radians([-19, -36, -105, -129, 90, 56])  # TCP 0.12 m from the base's axis
    target = FR5.fk(np.radians([-23, -27, -85, -158, 90, 38]))  # reached folded back
    assert FR5.ahead(here) > 0 > FR5.ahead(np.radians([-23, -27, -85, -158, 90, 38]))
    q = FR5.ik(target, here)
    assert q is not None and np.allclose(FR5.fk(q), target, atol=1e-6)
    assert FR5.ahead(q) > 0 and math.degrees(q[0]) == pytest.approx(-126, abs=2)
    predicted = wrist_angle_at(FR5, dict(zip(FR5.joint_names, here)), target)
    assert q[5] == pytest.approx(predicted, abs=1e-4)


# ---------------------------------------------------------------------------
# Workcell
# ---------------------------------------------------------------------------
def test_default_workcell_loads_unverified_and_consistent():
    cell = load_workcell()
    assert not cell.verified and not cell.calibrated  # placeholders: the real arm refuses them
    assert cell.problems() == [] and cell.group == "fairino5_v6_group"
    assert len(cell.slots) >= 2  # AS-51
    spans = sorted(high - low for low, high in (cell.pick_area.x, cell.pick_area.y))
    assert spans == pytest.approx([0.4, 0.5])  # SY-15: 500 x 400 mm
    assert cell.grasp.pregrasp_distance == 0.10 and cell.motion.tcp_speed_max == 0.25
    assert cell.gripper.driver == "serial" and cell.gripper.serial_port == "/dev/ttyACM0"
    for pose in cell.observation_poses:  # AS-11: straight down from 450 mm
        tcp = cell.observation_tcp(pose)
        camera = cell.camera_of_tcp(tcp)
        assert np.allclose(camera, pose.camera) and camera[2, 3] == pytest.approx(0.45)
        assert np.allclose(cell.tcp_of_tip(cell.tip_of_tcp(tcp)), tcp)
    # a look centered on one fish: straight down from close_height, the image along `yaw`
    close = cell.close_look_camera((0.3, -0.1), 0.5)
    assert cell.close_look_height == pytest.approx(0.62)
    assert np.allclose(close[:3, 3], (0.3, -0.1, cell.table_z + 0.62))
    assert np.allclose(close[:3, 2], (0, 0, -1)) and poses.yaw_of(close) == pytest.approx(0.5)


def test_without_close_height_there_are_no_close_looks():
    data = workcell_data()
    del data["observation"]["close_height"]
    assert workcell_from_mapping(data).close_look_height is None
    data["observation"]["close_height"] = 0.05
    assert "close_height" in "; ".join(workcell_from_mapping(data).problems())


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
    assert "must be 'serial', 'http' or" in problems(lambda d: d["gripper"].update(driver="x"))
    assert "http_host is empty" in problems(lambda d: d["gripper"].update(driver="http"))
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
    assert "exclusion zone 'operator side'" in problem(1.10, 0.0, 0.3)
    assert "TCP would be 2.0 mm above the table" in problem(0.4, 0.0, 0.002)
    assert problem(0.4, 0.0, 0.006) == ""  # 5 mm clearance
    # a fish hangs along the tool's x axis: both of its ends count
    assert problem(0.75, 0.0, 0.3, yaw=math.pi / 2, fish=0.7) == ""
    assert "fish end would enter the exclusion zone" in problem(0.75, 0.0, 0.3, yaw=0.0, fish=0.7)
    assert "fish end would leave the work zone" in problem(0.4, -0.6, 0.3, math.pi / 2, fish=0.7)
    # beside the table the floor doesn't apply, and a path may start there
    assert problem(-0.6, 0.0, -0.2, work_zone=False) == ""
    assert "exclusion" in problem(1.05, 0.0, 0.3, work_zone=False)
    low_camera = make_workcell(lambda d: d["camera"]["mount"].update(xyz=[0.0, -0.09, 0.27]))
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


def test_a_plan_that_moves_the_arm_in_no_time_is_refused():
    # A step of no time was left out of the speed check: a jump passed any speed limit
    from rfp_arm.safety import check_trajectory

    start = joints_for(0.40, -0.10, 0.36, yaw=1.2)
    moved = start.copy()
    moved[0] += 0.2
    positions = np.array([start, moved, moved])
    still = check_trajectory(CELL, FR5, FR5.joint_names, positions[[0, 0, 1]],
                             np.array([0.0, 0.0, 2.0]), 0.0)  # fmt: skip
    assert still.ok  # the same point twice at one time: fine
    jump = check_trajectory(CELL, FR5, FR5.joint_names, positions, np.array([0.0, 0.0, 2.0]), 0.0)
    assert not jump.ok and "no time" in jump.problem


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


class Board:
    """The valve board's serial port: answers like Arm/firmware/valve_control."""

    def __init__(self, port="COM7", baud=115200, timeout=2.0):
        self.port, self.baud = port, baud
        self.valve, self.lines, self.received = False, [], []
        self.unplugged = self.closed = False
        self.dtr = self.rts = None

    def reset_input_buffer(self) -> None:
        self.lines.clear()

    def write(self, data: bytes) -> None:
        if self.unplugged:
            raise OSError("device disconnected")
        command = data.decode().strip().upper()
        self.received.append(command)
        if command in ("ON", "OFF"):
            self.valve = command == "ON"
        elif command.startswith("PULSE"):
            if int(command[5:] or 0) <= 0:
                self.lines.append("ERR PULSE needs ms > 0")
                return
            self.valve = True
        elif command != "STATUS":
            self.lines.append(f"ERR unknown command: {command}")
            return
        self.lines.append("OK ON" if self.valve else "OK OFF")

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        return (self.lines.pop(0) + "\r\n").encode() if self.lines else b""

    def close(self) -> None:
        self.closed = True


def test_valve_board_is_switched_and_answers_are_checked():
    board = Board()
    board.lines = ["Connecting to WiFi....", "READY"]  # what a starting board prints
    valve = SerialValve("COM7", open_port=lambda port, baud, timeout: board)
    assert board.received == ["STATUS"] and not valve.is_on()
    assert valve.on() is True and board.valve and valve.is_on()
    assert valve.pulse(500) is True and board.received[-1] == "PULSE 500"
    assert valve.off() is False and not board.valve
    with pytest.raises(ValveError, match="ERR PULSE needs ms"):
        valve.pulse(0)
    board.unplugged = True
    with pytest.raises(ValveError, match="stopped answering"):
        valve.on()
    valve.close()
    assert board.closed

    def no_port(port, baud, timeout):
        raise OSError("could not open port")

    with pytest.raises(ValveError, match="can't open COM7"):
        SerialValve("COM7", open_port=no_port)
    clock = FakeClock()
    silent = Board()
    silent.write = lambda data: None  # a board that never answers
    silent.readline = lambda: clock.advance(0.5) or b""  # each read times out
    with pytest.raises(ValveError, match="doesn't answer"):
        SerialValve("COM7", open_port=lambda *a: silent, clock=clock)

    pages = {"/status": "OK OFF", "/on": "OK ON", "/off": "OK OFF", "/pulse?ms=250": "OK ON"}
    asked = []

    def fetch(url):
        asked.append(url)
        return pages[url.removeprefix("http://192.168.4.1")]

    wifi = HttpValve("192.168.4.1", fetch=fetch)
    assert wifi.on() and not wifi.off() and wifi.pulse(250)
    assert asked[-1] == "http://192.168.4.1/pulse?ms=250"

    def down(url):
        raise OSError("timed out")

    with pytest.raises(ValveError, match="can't be reached"):
        HttpValve("192.168.4.1", fetch=down)


def test_valve_gripper_turns_the_vacuum_off_first_and_cannot_feel_a_hold():
    board = Board()
    board.valve = True  # left on by whoever used it last
    gripper = ValveGripper(SerialValve("COM7", open_port=lambda *a: board))
    assert not board.valve and gripper.state() == GripperState.VACUUM_OFF  # a known start
    assert not gripper.has_switch  # nothing tells whether the cups hold: the camera has to
    gripper.vacuum_on()
    assert board.valve and gripper.state() == GripperState.NO_SEAL
    gripper.release()
    assert not board.valve and gripper.state() == GripperState.VACUUM_OFF
    gripper.pulse(300)
    assert board.received[-1] == "PULSE 300"
    board.unplugged = True
    with pytest.raises(ArmError, match="stopped answering") as error:
        gripper.vacuum_on()
    assert error.value.code == "E_COMM" and gripper.state() == GripperState.VACUUM_OFF

    inverted = Board()
    other_way = ValveGripper(SerialValve("COM7", open_port=lambda *a: inverted), inverted=True)
    assert inverted.valve and other_way.state() == GripperState.VACUUM_OFF
    other_way.vacuum_on()
    assert not inverted.valve and other_way.state() == GripperState.NO_SEAL

    assert isinstance(make_gripper(GripperSettings(), FakeIO()), SuctionGripper)
    with pytest.raises(ArmError, match="command server"):
        make_gripper(GripperSettings(), None)
    with pytest.raises(ArmError, match="can't be reached|can't open|pyserial"):
        make_gripper(GripperSettings(driver="serial", serial_port="/dev/no-such-port"), None)


def test_grasp_with_the_valve_gripper_relies_on_the_camera():
    board = Board()
    gripper = ValveGripper(SerialValve("COM7", open_port=lambda *a: board))
    hold = Hold()
    rig = Rig(gripper=gripper, hold_check=hold)
    before = rig.clock.now
    rig.run()
    assert tuple(rig.steps) == STEPS and hold.events == ["start", "stop"]
    assert [c for c in board.received if c != "STATUS"] == ["OFF", "ON", "OFF"]
    assert rig.clock.now - before >= 2.0  # 1 s for the vacuum to build, 1 s to let go
    blind = Rig(gripper=ValveGripper(SerialValve("COM7", open_port=lambda *a: Board())))
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
    # head end unclear: into the slot as vision guessed it (still better than a coin toss;
    # the shorter turn, 80 deg the other way round, placed it backwards half the time)
    assert yaws(heading_deg=100, known=False) == [100, 0, -100]
    assert yaws(heading_deg=60, known=False) == [60, 0, -60]


def test_grasp_plan_refuses_what_the_limits_forbid():
    def refused(**options) -> str:
        with pytest.raises(ArmError) as error:
            plan(**options)
        assert error.value.code == E_MOTION
        return error.value.message

    assert "outside the pick area" in refused(x=0.70)
    assert "above the table" in refused(thickness=0.0)  # the cups would be pressed into the table
    # lying along X near the far edge: lifted, its end reaches the operator's side
    assert "fish end would enter the exclusion zone" in refused(x=0.50, heading_deg=0, length=1.2)
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

    def __init__(self, gripper=None, cell=CELL, **options):
        self.clock = FakeClock()
        self.arm = SimArm(cell)
        self.gripper = gripper or SimGripper()
        self.steps: list[str] = []
        self.ticks = 0
        self.at_tick = {}  # tick number -> action
        self.sequence = GraspSequence(
            self.arm, self.gripper, cell, tick=self.tick, on_step=self.steps.append,
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
    rig.run()  # no straight carry: point-to-point (carry_ptp_fallback)
    assert tuple(rig.steps) == STEPS
    to_slot = next(kind for kind, target in rig.arm.moves if np.allclose(target[:2, 3], SLOT))
    assert to_slot == "ptp"

    rig = Rig()
    rig.arm.fail = block_carry(2)
    rig.run()  # neither way, then the second plan works
    assert tuple(rig.steps) == STEPS

    rig = Rig()
    rig.arm.fail = block_carry(4)  # straight and point-to-point, planned twice
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


def test_a_carry_planned_again_goes_on_from_the_stop_it_was_heading_for():
    # Refused straight and point-to-point, the carry is planned again: back to the first
    # stop, the tool would turn the short way, which the stops keep the wrist from
    import dataclasses

    base = plan()
    via = [poses.interpolate(base.lift, base.carry, share) for share in (1 / 3, 2 / 3)]
    refused = []

    def fail(kind, target):
        to_stop = kind == "lin" and np.allclose(target, via[1])
        to_carry = kind == "ptp" and np.allclose(target, base.carry)
        if (to_stop or to_carry) and len(refused) < 2:
            refused.append(kind)
            return "no path"
        return None

    rig = Rig()
    rig.arm.fail = fail
    rig.sequence.run(dataclasses.replace(base, carry_via=via))
    stops = [pose for kind, pose in rig.arm.moves if any(np.allclose(pose, v) for v in via)]
    assert refused == ["lin", "ptp"] and len(stops) == 2 and tuple(rig.steps) == STEPS
    assert np.allclose(stops[0], via[0]) and np.allclose(stops[1], via[1])


def test_a_fish_whose_cups_were_past_the_pick_area_s_edge_is_set_down_where_it_lay():
    # The cups hold a fish ahead of its middle, so with its middle inside the pick area they
    # can be outside it: a failed lift left such a fish hanging on the cups
    middle = np.array([0.58, -0.10, CELL.table_z + 0.10])
    p = plan_grasp(CELL, fish_pose(x=0.66), True, 0.60, 0.10, SLOT, math.radians(90.0), False,
                   middle=middle)  # fmt: skip
    assert not CELL.in_pick_area(p.surface[:2, 3])

    def fail(kind, target):
        lifting = np.allclose(target[:2, 3], p.surface[:2, 3]) and target[2, 3] > 0.3
        return "no straight path" if kind == "lin" and lifting else None

    rig = Rig()
    rig.arm.fail = fail
    with pytest.raises(ArmError) as error:
        rig.sequence.run(p)
    assert rig.sequence.step == "lift" and error.value.fish_put_back
    assert not error.value.still_holding and rig.gripper.events[-1] == "release"


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
        x, y = (start[:2, 3] + (0.40, 0.40)) / 2  # half way there
        post = {"name": "post", "x": [x - 0.02, x + 0.02], "y": [y - 0.02, y + 0.02]}
        blocked = make_workcell(lambda d: d["zones"]["exclusion"].append({**post, "z": [0, 1]}))
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


# ---------------------------------------------------------------------------
# The wrist and its vacuum tube
# ---------------------------------------------------------------------------
def test_wrist_angle_is_predicted_without_inverse_kinematics():
    from rfp_arm.kinematics import wrist_angle_at

    here = joints_for(0.40, -0.10, 0.36, yaw=1.2)
    joints = dict(zip(FR5.joint_names, here))
    for x, y, z, yaw in (
        (0.30, -0.25, 0.20, 1.0),
        (0.55, 0.10, 0.45, 1.6),
        (0.38, 0.45, 0.20, 0.6),
    ):
        tip = CELL.tip_of_tcp(poses.tool_down((x, y, z), yaw))
        there = solve_ik(tip, here)  # the same posture, by inverse kinematics
        assert wrist_angle_at(FR5, joints, tip) == pytest.approx(there[5], abs=1e-4)
    assert (
        wrist_angle_at(FR5, joints, CELL.tip_of_tcp(poses.tool_down((0.0, 0.05, 0.3), 0.0))) is None
    )
    tilted = dict(zip(FR5.joint_names, [0.0] * 6))  # the tool isn't vertical here
    assert (
        wrist_angle_at(FR5, tilted, CELL.tip_of_tcp(poses.tool_down((0.4, 0.0, 0.3), 0.0))) is None
    )


def test_plans_that_wind_the_wrist_are_refused_but_unwinding_is_allowed():
    names = FR5.joint_names
    assert CELL.wrist_range == pytest.approx((math.radians(-170), math.radians(170)))
    start = joints_for(0.40, -0.10, 0.30)
    start[5] = 0.0
    wound = start.copy()
    wound[5] = math.radians(172)
    winding = check_trajectory(CELL, FR5, names, *joint_path(start, wound, 30.0))
    assert "the wrist (j6) would turn to 172 deg" in winding.problem and "tube" in winding.problem
    assert check_trajectory(CELL, FR5, names, *joint_path(wound, start, 30.0)).ok  # coming back
    tight = make_workcell(lambda d: d["tool"].update(wrist_range_deg=[-90, 90]))
    half = start.copy()
    half[5] = math.radians(100)
    assert (
        "would turn to 100 deg"
        in check_trajectory(tight, FR5, names, *joint_path(start, half, 30.0)).problem
    )
    assert check_trajectory(CELL, FR5, names, *joint_path(start, half, 30.0)).ok
    taught = workcell_data()
    taught["observation"]["poses"] = [
        {"name": "twisted", "joints_deg": [0, -90, -100, -80, 90, 175]}
    ]
    assert "turns the wrist past its range" in "; ".join(workcell_from_mapping(taught).problems())


def fake_wrist(now_deg: float, offset_deg: float):
    """An arm whose wrist is at `now_deg` and would be at the tool's yaw + `offset_deg`."""

    def wrist(pose=None):
        if pose is None:
            return math.radians(now_deg)
        return poses.wrap(poses.yaw_of(pose) + math.radians(offset_deg))

    return wrist


def test_the_slot_gets_the_fish_middle_when_the_cups_grasp_it_nearer_the_head():
    fish = fish_pose(heading_deg=100.0)  # cups at (0.40, -0.10), head toward 100 deg
    toward_head = np.array([math.cos(math.radians(100)), math.sin(math.radians(100))])
    middle = np.array([*(fish[:2, 3] - 0.07 * toward_head), fish[2, 3]])  # 70 mm behind them
    p = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(90), middle=middle)
    for pose in (p.carry, p.lower, p.retreat):
        # the fish turns with the cups: its middle, 70 mm behind them along the placed
        # fish's heading, lands on the slot
        heading = CELL.placement_heading
        placed_middle = pose[:2, 3] - 0.07 * np.array([math.cos(heading), math.sin(heading)])
        assert np.allclose(placed_middle, SLOT, atol=1e-6)
    plain = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(90))
    assert np.allclose(plain.lower[:2, 3], SLOT)  # no middle given: the cups go to the slot
    # the held fish's middle, for the arm's checks: along the TCP's x from the cups
    assert np.allclose(p.surface[:2, 3] + p.surface[:2, 0] * p.middle_x, middle[:2], atol=1e-3)
    assert plain.middle_x == 0.0


def test_a_held_fish_is_checked_from_its_middle_not_from_the_cups():
    at = (0.40, CELL.work_zone.y[1] - 0.45, 0.40)  # 0.45 m from the work zone's +Y edge
    # The cups hold a 0.80 m fish 0.11 m nearer its head: its middle is 0.11 m behind them
    head_out = poses.tool_down(at, math.radians(90))  # head toward the edge, tail inward
    assert CELL.pose_problem(head_out, 0.80, True, -0.11) is None  # head end 0.29 m out: in
    tail_out = poses.tool_down(at, math.radians(-90))  # tail toward the edge
    assert CELL.pose_problem(tail_out, 0.80) is None  # taken as centered: 0.40 m each way
    problem = CELL.pose_problem(tail_out, 0.80, True, -0.11)  # but the tail reaches 0.51 m
    assert problem is not None and "fish end" in problem


def test_grasp_takes_the_way_round_that_turns_the_wrist_least():
    def plan_with(wrist, heading_deg=100.0, known=True, cell=CELL):
        fish = fish_pose(heading_deg=heading_deg)
        return plan_grasp(cell, fish, known, 0.60, 0.10, SLOT, math.radians(90), wrist=wrist)

    def path(plan):
        return [round(math.degrees(a)) for a in plan.wrist]

    # Wrist at -14 deg. Gripped one way it goes to 21, then 156 at the slot; the other way
    # to -159, then -24 turns it less in all, but the empty gripper 159 deg out has the
    # camera toward the arm: within 90 deg of the middle where a way allows
    p = plan_with(fake_wrist(-14, 156), heading_deg=-135)
    assert path(p) == [-14, 21, 156] and not p.reset_first
    assert round(math.degrees(poses.yaw_of(p.pregrasp))) == -135  # the cups' x to the head
    assert round(math.degrees(poses.yaw_of(p.lower))) == 0  # its head toward +X
    # sweeps of 89 and 91 deg count as equal: the way that stays nearer the middle
    assert path(plan_with(fake_wrist(3, 92), heading_deg=0)) == [3, -88, -88]
    # clear of the range's ends (+-170) if a way allows: 159 deg rather than 151 to -165
    assert path(plan_with(fake_wrist(-14, -165), heading_deg=130)) == [-14, 145, 15]

    # Head end unclear: the fish goes in as vision guessed where the wrist allows (here the
    # longer turn of the fish, -100 deg, not +80, which also turns the wrist less)
    unknown = plan_with(fake_wrist(0, -40), known=False)
    assert path(unknown) == [0, 60, -40] and round(math.degrees(unknown.turn)) == -100

    # an arm that can't tell: the wrist is taken to turn with the tool's yaw
    assert round(math.degrees(poses.yaw_of(plan_with(None).pregrasp))) == 100
    blind = plan_with(lambda pose=None: None)
    assert round(math.degrees(poses.yaw_of(blind.pregrasp))) == 100 and blind.wrist is None
    tight = make_workcell(lambda d: d["tool"].update(wrist_range_deg=[-30, 30]))
    with pytest.raises(ArmError, match="wrist would turn to .* vacuum tube"):
        plan_with(None, cell=tight)
    arm = SimArm(CELL)
    assert arm.wrist_angle() == pytest.approx(0.0)  # where it starts
    assert arm.wrist_angle(
        poses.tool_down((0.4, 0.0, 0.3), poses.yaw_of(arm.tcp_pose()) + 0.5)
    ) == (pytest.approx(0.5))


def fr5_wrist():
    """The FR5's wrist (j6) as the arm predicts it (Arm.wrist_angle), at the first
    observation pose in the posture the layout check assumes."""
    from rfp_arm.kinematics import wrist_angle_at
    from rfp_arm.layout import reach_joints

    joints = reach_joints(CELL, CELL.observation_tcp(CELL.observation_poses[0]))
    by_name = dict(zip(FR5.joint_names, joints))

    def wrist(pose=None):
        if pose is None:
            return float(joints[-1])
        return wrist_angle_at(FR5, by_name, CELL.tip_of_tcp(pose))

    return wrist


def test_a_carry_turns_the_wrist_evenly_on_its_straight_way():
    # A straight move turns the tool evenly while the base swings unevenly, which can drag
    # the wrist past its stop half way: the carry goes through poses where it is on its way
    wrist = fr5_wrist()
    fish = fish_pose(x=0.55, y=-0.25, heading_deg=-60.0)
    p = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(-90), wrist=wrist)
    assert p.carry_via  # 0.6 m and more: split every 0.15 m
    _, at_grasp, at_slot = p.wrist
    start, end = p.lift[:3, 3], p.carry[:3, 3]
    shares = []
    for pose in p.carry_via:
        along = (pose[:3, 3] - start) @ (end - start) / ((end - start) @ (end - start))
        assert np.allclose(pose[:3, 3], start + along * (end - start), atol=1e-9)  # on the way
        expected = at_grasp + (at_slot - at_grasp) * along
        assert wrist(pose) == pytest.approx(expected, abs=1e-3)  # the wrist on its way
        assert CELL.pose_problem(pose, 0.60, True, p.middle_x) is None
        shares.append(along)
    assert shares == sorted(shares)


def test_a_fish_that_would_sweep_off_the_table_is_turned_where_it_has_room(monkeypatch):
    from rfp_arm import grasp

    wrist = fr5_wrist()
    fish = fish_pose(x=0.55, y=-0.25, heading_deg=-60.0)
    monkeypatch.setattr(grasp, "_even_wrist_carry", lambda *args: None)  # its way sweeps out
    p = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(-90), wrist=wrist)
    spot = p.carry_via[0][:3, 3]
    assert all(np.allclose(pose[:3, 3], spot) for pose in p.carry_via)  # turned in one place
    assert poses.yaw_of(p.carry_via[0]) == pytest.approx(poses.yaw_of(p.lift))  # there as lifted
    assert poses.yaw_of(p.carry_via[-1]) == pytest.approx(poses.yaw_of(p.carry))  # on as placed
    turns = [wrist(pose) for pose in p.carry_via]
    steps = np.diff(turns)
    assert (steps * np.sign(steps.sum()) > 0).all()  # the wrist one way, evenly
    for pose in p.carry_via:  # its ends on the table and out of the keep-out zones meanwhile
        assert CELL.pose_problem(pose, 0.60, True, p.middle_x) is None


def test_the_wrist_turns_a_held_fish_at_most_half_a_turn_and_no_move_further():
    def plan_with(wrist, heading_deg):
        fish = fish_pose(heading_deg=heading_deg)
        return plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, 0.0, wrist=wrist)

    def path(plan):
        return [round(math.degrees(a)) for a in plan.wrist]

    # Placed, the wrist ends at -19 or 161 (the cups either way round). Held the near way
    # (-84 at the grasp) the fish would turn 245 deg, past half a turn: so the far way, the
    # empty gripper at 96, past its 90 deg band, the fish turned 115
    assert path(plan_with(fake_wrist(0, -19), heading_deg=115)) == [0, 96, -19]
    # The grasp 200 deg from where the wrist is (as after a closer look on the far side):
    # it turns back to its middle where it is first, then to the grasp
    far = plan_with(fake_wrist(120, 0), heading_deg=-80)
    assert path(far) == [120, -80, 0] and far.reset_first
    # whatever the fish and the wrist: a held fish turns at most 180 deg, the wrist stays in
    # its range, and no move turns it more than 180
    rng = np.random.default_rng(3)
    planned = 0
    for _ in range(300):
        now, offset, heading = rng.uniform(-90, 90), rng.uniform(-180, 180), rng.uniform(-180, 180)
        try:
            p = plan_with(fake_wrist(now, offset), heading)
        except ArmError:
            continue  # refused: nothing runs
        planned += 1
        start, grasp, slot = (math.degrees(a) for a in p.wrist)
        assert abs(slot - grasp) <= 180 + 1e-6 and max(abs(grasp), abs(slot)) <= 170
        assert abs(grasp - start) <= 180 or p.reset_first
    assert planned > 250


def test_a_long_turn_to_the_grasp_goes_back_to_the_middle_first():
    rig = Rig()
    here = rig.arm.tcp_pose()  # the wrist at 0 here
    rig.arm.move_ptp(poses.tool_down(here[:3, 3], poses.yaw_of(here) + math.radians(120)))
    assert math.degrees(rig.arm.wrist_angle()) == pytest.approx(120)
    fish = fish_pose(heading_deg=-170)  # the grasp at -80: 200 deg away
    p = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, 0.0, wrist=rig.arm.wrist_angle)
    assert p.reset_first
    rig.sequence.run(p)
    turned_back = rig.arm.moves[1][1]
    assert np.allclose(turned_back[:3, 3], here[:3, 3])  # where it was, the wrist at 0
    assert rig.arm.wrist_angle(turned_back) == pytest.approx(0.0, abs=1e-6)
    assert math.degrees(rig.arm.wrist_angle(rig.arm.moves[2][1])) == pytest.approx(-80, abs=1)


# ---------------------------------------------------------------------------
# Onto the fish: along its surface's slope, and feeling for it
# ---------------------------------------------------------------------------
def sloping_fish(tilt_deg: float, heading_deg: float = 100.0):
    """A grasp pose as Vision gives it for a surface climbing `tilt_deg` toward the head."""
    pose = fish_pose(heading_deg=heading_deg)
    pose[:3, :3] = pose[:3, :3] @ poses.rot_y(math.radians(tilt_deg))
    return pose


def test_the_approach_follows_a_sloping_surface():
    fish = sloping_fish(10.0)
    p = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(90))
    axis = fish[:3, 2]  # into the surface: 10 degrees off straight down
    assert poses.tilt_from_down(p.surface) == pytest.approx(math.radians(10))
    assert np.allclose(p.at_depth(0.0), p.surface) and np.allclose(p.surface[:3, 3], fish[:3, 3])
    assert np.allclose(p.pregrasp[:3, 3], fish[:3, 3] - 0.10 * axis)  # back along the axis
    assert np.allclose(p.approach_end[:3, 3], fish[:3, 3] + 0.010 * axis)
    assert np.allclose(p.pregrasp[:3, :3], p.approach_end[:3, :3])  # a straight push
    assert np.allclose(p.lift[:2, 3], p.approach_end[:2, 3])  # then straight up, tool level again
    assert poses.tilt_from_down(p.lift) == pytest.approx(0.0, abs=1e-9) and p.lift[2, 3] == 0.41
    dry = plan_grasp(CELL, fish, True, 0.60, 0.10, SLOT, math.radians(90), dry_run=True)
    assert np.allclose(dry.approach_end[:3, 3], fish[:3, 3] - 0.05 * axis)
    # head the other way: the cups turn about the same axis, still square on the surface
    other = plan_grasp(CELL, sloping_fish(10.0, -80.0), True, 0.60, 0.10, SLOT, math.radians(90))
    assert round(math.degrees(poses.yaw_of(other.surface))) == 100
    assert poses.tilt_from_down(other.surface) == pytest.approx(math.radians(10))
    assert other.surface[2, 0] < 0 < fish[2, 0]  # their x axes run opposite ways along the slope
    with pytest.raises(ArmError, match="20 deg off vertical"):
        plan_grasp(CELL, sloping_fish(20.0), True, 0.60, 0.10, SLOT, 0.0)
    rig = Rig()
    rig.sequence.run(p)
    assert tuple(rig.steps) == STEPS and np.allclose(rig.arm.moves[1][1], p.approach_end)


def lowest(rig) -> float:
    return min(pose[2, 3] for _, pose in rig.arm.moves)


def test_the_cups_are_lowered_until_the_fish_pushes_back():
    rig = Rig()
    rig.arm.contact_surface = 0.095  # the fish is 5 mm lower than the camera said (0.100)
    rig.run()
    assert tuple(rig.steps) == STEPS
    # 15 N on a 3 N/mm surface: 5 mm into it, so 10 mm past the expected height, in 3 mm steps
    felt = rig.sequence.contact_depth
    assert felt == pytest.approx(0.012, abs=1e-6)
    assert 0.100 - felt - 0.010 - 1e-9 <= lowest(rig) < 0.100 - felt  # pressed on, not through
    start = next(pose for kind, pose in rig.arm.moves if kind == "lin")
    assert start[2, 3] == pytest.approx(0.115)  # the slow part starts 15 mm above

    hard = Rig()
    hard.arm.contact_surface, hard.arm.contact_stiffness = 0.100, 20000.0
    hard.run()  # stiff: the force limit is reached before the press distance
    assert hard.sequence.contact_depth == pytest.approx(0.003) and lowest(hard) == pytest.approx(
        0.097
    )

    nothing = Rig()
    nothing.arm.contact_surface = 0.05  # no fish under the cups
    error = nothing.fails()
    assert error.code == E_NO_SEAL and "nothing felt 25 mm past" in error.message
    assert nothing.sequence.step == "approach" and lowest(nothing) == pytest.approx(0.075)
    assert nothing.gripper.state() != GripperState.VACUUM_OFF  # still on: Control lets go

    blind = Rig(cell=make_workcell(lambda d: d["grasp"].update(feel_contact=False)))
    blind.arm.contact_surface = 0.095
    blind.run()  # switched off: to the planned depth, as an arm that can't feel does
    assert blind.sequence.contact_depth is None and lowest(blind) == pytest.approx(0.090)
    dry = Rig(gripper=NoGripper())
    dry.arm.contact_surface = 0.095
    dry.run(dry_run=True)
    assert dry.sequence.contact_depth is None and lowest(dry) == pytest.approx(0.115)


def test_controller_commands_bracket_the_press():
    soft = make_workcell(
        lambda d: d["grasp"].update(
            press_start_commands=["Soft(1)"], press_end_commands=["Soft(0)"]
        )
    )
    sent = []

    def controller(text):
        sent.append((text, rig.sequence.step))
        return "0"

    rig = Rig(cell=soft, controller=controller)
    rig.run()
    assert sent == [("Soft(1)", "approach"), ("Soft(0)", "confirm hold")]
    sent.clear()
    rig = Rig(cell=soft, gripper=SimGripper(seals=False), controller=controller)
    assert rig.fails().code == E_NO_SEAL and [text for text, _ in sent] == ["Soft(1)", "Soft(0)"]

    def refuses(text):
        raise ArmError("E_COMM", f"the controller answered '-1' to {text} (driver V3)")

    rig = Rig(cell=soft, controller=refuses)
    error = rig.fails()
    assert error.code == "E_COMM" and "answered '-1' to Soft(1)" in error.message
    assert [kind for kind, _ in rig.arm.moves] == ["ptp"]  # it didn't go down
    Rig(cell=soft).run()  # no controller to send them to (simulation): skipped


def test_contact_force_is_read_from_joint_torques():
    from rfp_arm.contact import ContactEstimator

    joints = dict(zip(FR5.joint_names, joints_for(0.40, -0.10, 0.20)))
    estimator = ContactEstimator(FR5, CELL.tcp)
    pattern = estimator.pattern(joints)  # N m per N along the tool axis
    assert abs(pattern[1]) > 0.2 and abs(pattern[5]) < 1e-6  # the shoulder feels it, j6 doesn't
    holding = 3.0 * np.cos(list(joints.values()))  # what the motors give with nothing touching
    assert not estimator.zeroed
    estimator.zero(holding)
    assert estimator.force(joints, holding) == 0.0
    assert estimator.force(joints, holding - 25.0 * pattern) == pytest.approx(25.0)
    assert estimator.force(joints, holding + 25.0 * pattern) == pytest.approx(25.0)  # either sign
    sideways = np.cross(pattern[:3], [0, 0, 1])  # a change that isn't a push on the tool
    assert estimator.force(joints, holding + np.r_[sideways, 0, 0, 0]) < 1.0
    arm = SimArm(CELL)
    assert arm.contact_force() is None  # nothing to feel unless a surface is set
    arm.contact_surface = 0.50
    assert arm.contact_force(zero=True) == 0.0 and arm.contact_force() == 0.0


def test_looking_at_the_valve_does_not_switch_it_and_late_answers_are_skipped():
    board = Board()
    board.valve = True  # a fish may be on the cups
    looking = ValveGripper(SerialValve("COM7", open_port=lambda *a: board), start_off=False)
    assert board.valve and looking.state() == GripperState.NO_SEAL  # read, not switched
    assert "OFF" not in board.received
    inverted = Board()  # inverted wiring: valve off = vacuum on
    other = ValveGripper(SerialValve("COM7", open_port=lambda *a: inverted), True, start_off=False)
    assert other.state() == GripperState.NO_SEAL
    stale = Board()
    valve = SerialValve("COM7", open_port=lambda *a: stale)
    stale.lines.append("OK ON")  # a late answer to some earlier command
    assert valve.off() is False and not stale.valve  # this command's own answer is read
