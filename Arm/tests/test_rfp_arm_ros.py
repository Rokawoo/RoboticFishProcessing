"""Tests for the ROS side of the arm (moveit_arm.py, fairino.py, ros_link.py) against
fake_ros.py: a fake move_group, FR5 and FAIRINO command server with the real packages'
message fields. No ROS needed; what it can and can't show is in fake_ros.py."""

import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from arm_helpers import make_workcell, solve_ik
from fake_ros import FakeRobot, fr5_urdf, install, message
from rfp_arm import poses
from rfp_arm.fairino import FairinoIO
from rfp_arm.grasp import STEPS, GraspSequence, plan_grasp
from rfp_arm.gripper import SuctionGripper
from rfp_arm.interfaces import (
    E_COMM,
    E_DROP,
    E_MOTION,
    E_NO_SEAL,
    E_PROTECTIVE_STOP,
    ArmError,
    GripperState,
    MotionAborted,
)
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.moveit_arm import MoveItArm
from rfp_arm.ros_link import RosLink

CELL = make_workcell()
# The arm over the pick area as at an observation pose, wrist (j6) in the middle of its range
HOME = solve_ik(CELL.tip_of_tcp(CELL.observation_tcp(CELL.observation_poses[0])))
HOME[5] = 0.0
OBSERVE = CELL.tcp_of_tip(FR5.fk(HOME))
YAW = poses.yaw_of(OBSERVE)


def down(x: float, y: float, z: float, turn: float = 0.4) -> np.ndarray:
    """A TCP pose pointing down, turned `turn` rad from the observation pose's yaw."""
    return poses.tool_down((x, y, z), YAW + turn)


@pytest.fixture
def ros(monkeypatch):
    """start(...) -> (robot, link): a fake robot with the fake rclpy installed around it."""
    started = []

    def start(joints=HOME, **options):
        robot = FakeRobot(joints, **options)
        install(monkeypatch, robot)
        link = RosLink()
        started.append((robot, link))
        return robot, link

    yield start
    for robot, link in started:
        link.close()
        robot.close()


def connected(ros, cell=CELL, robot_options=None, **arm_options):
    robot, link = ros(**(robot_options or {}))
    arm = MoveItArm(link, cell, log=lambda text: None, **arm_options)
    arm.connect(timeout=5.0)
    return robot, arm


def tcp(robot) -> np.ndarray:
    return CELL.tcp_of_tip(robot.tip())


# ---------------------------------------------------------------------------
# The fake itself
# ---------------------------------------------------------------------------
def test_fake_messages_are_as_strict_as_the_real_ones():
    pose = message("geometry_msgs/msg/Pose")()
    assert pose.orientation.w == 1.0 and pose.position.x == 0.0  # the .msg defaults
    pose.position.x = 0.5
    with pytest.raises(TypeError, match="must be float"):
        pose.position.x = 1  # rclpy refuses an int for a float64
    with pytest.raises(AttributeError, match="no field 'xx'"):
        pose.position.xx = 1.0
    with pytest.raises(AttributeError):
        message("moveit_msgs/msg/JointConstraint")(joint="j1")
    box = message("shape_msgs/msg/SolidPrimitive")
    assert box.BOX == 1 and message("moveit_msgs/msg/CollisionObject").ADD == b"\x00"
    with pytest.raises(ValueError, match="takes"):
        box(dimensions=[1.0, 2.0, 3.0, 4.0])  # float64[<=3]
    with pytest.raises(TypeError):
        box(dimensions=[1, 2, 3])
    state = message("fairino_msgs/msg/RobotNonrtState")()
    assert state.dgt_input_l == 0 and len(state.safetyboxsig) == 6
    assert Chain.from_urdf(fr5_urdf(), "base_link", "wrist3_link").links == FR5.links


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------
def test_connect_reads_the_model_and_limits_and_loads_the_scene(ros):
    robot, link = ros()
    robot.max_trans_vel = 0.25  # what the Requirement Spec wants in pilz_cartesian_limits.yaml
    arm = MoveItArm(link, CELL, log=lambda text: None)
    arm.connect(timeout=5.0)
    assert np.allclose(arm.tcp_pose(), OBSERVE, atol=1e-5)  # from /joint_states, any joint order
    assert "0.25 m/s" in arm.limits() and arm.fault() is None
    assert robot.requests == [] and robot.executed == []  # connecting moves nothing

    objects = {item.id: item for item in robot.scene.world.collision_objects}
    assert set(objects) == {"table", "zone operator side"} and robot.scene.is_diff
    table = objects["table"]
    assert table.header.frame_id == "base_link" and table.operation == b"\x00"
    assert table.primitives[0].type == 1  # a box as large as the work zone
    assert table.primitives[0].dimensions == pytest.approx([1.7, 1.7, 0.05])
    top = table.primitive_poses[0].position.z + 0.025
    assert top == pytest.approx(CELL.table_z - 0.01)  # just under the surface: touching is no hit
    tool = robot.scene.robot_state.attached_collision_objects[0]
    assert tool.link_name == "wrist3_link" and tool.object.header.frame_id == "wrist3_link"
    assert tool.touch_links == ["wrist2_link", "wrist3_link"] and robot.scene.robot_state.is_diff
    assert tool.object.primitives[0].dimensions == pytest.approx([0.16, 0.19, 0.14])


def test_connect_says_what_is_missing(ros):
    robot, link = ros()
    del robot.actions["move_action"]
    with pytest.raises(ArmError, match="is move_group running") as error:
        MoveItArm(link, CELL).connect(timeout=0.3)
    assert error.value.code == E_COMM
    robot, link = ros()
    robot.publish_joints = False
    with pytest.raises(ArmError, match="no /joint_states"):
        MoveItArm(link, CELL).connect(timeout=0.3)
    robot, link = ros()
    other = make_workcell(lambda d: d["robot"].update(tip_link="tool0"))
    with pytest.raises(ArmError, match="robot model"):
        MoveItArm(link, other).connect(timeout=2.0)


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------
def test_joint_and_point_to_point_moves_are_planned_checked_and_run(ros):
    robot, arm = connected(ros)
    target = down(0.40, -0.10, 0.20)
    arm.move_ptp(target)
    assert np.allclose(tcp(robot), target, atol=1e-4) and np.allclose(
        arm.tcp_pose(), target, atol=1e-4
    )
    request = robot.requests[0]
    assert request.pipeline_id == "pilz_industrial_motion_planner" and request.planner_id == "PTP"
    assert request.group_name == "fairino5_v6_group" and request.start_state.is_diff
    assert request.max_velocity_scaling_factor == CELL.motion.ptp_velocity_scale
    goal = request.goal_constraints[0]
    assert goal.position_constraints[0].link_name == "wrist3_link"  # the tip link, not the TCP
    tip = goal.position_constraints[0].constraint_region.primitive_poses[0].position
    assert tip.z == pytest.approx(0.20 + 0.22)  # the TCP is 220 mm past the tip link
    assert len(robot.executed) == 1

    arm.move_ptp(target)  # already there: nothing is planned
    assert len(robot.requests) == 1
    arm.move_joints(HOME)
    assert np.allclose(robot.joints, HOME, atol=1e-6) and robot.requests[-1].planner_id == "PTP"
    joints = robot.requests[-1].goal_constraints[0].joint_constraints
    assert [j.joint_name for j in joints] == list(CELL.joints) and joints[0].position == HOME[0]
    with pytest.raises(ArmError, match="6 joint positions"):
        arm.move_joints([0.0, 0.0])


def test_straight_moves_are_straight_and_as_slow_as_asked(ros):
    robot, arm = connected(ros)
    start = tcp(robot)[:3, 3].copy()
    target = down(0.45, 0.10, 0.25, turn=0.8)
    arm.move_lin(target, speed=0.05, accel=0.5)
    assert np.allclose(tcp(robot), target, atol=1e-4)
    request = robot.requests[0]
    assert request.planner_id == "LIN"
    assert request.max_velocity_scaling_factor == pytest.approx(0.05 / 1.0)  # of Pilz's 1.0 m/s
    assert request.max_acceleration_scaling_factor == pytest.approx(0.5 / 2.25)
    line = target[:3, 3] - start
    for joints in robot.path:
        offset = CELL.tcp_of_tip(FR5.fk(joints))[:3, 3] - start
        along = offset @ line / (line @ line)
        assert np.linalg.norm(offset - along * line) < 1e-3 and -0.01 < along < 1.01
    arm.move_lin(down(0.45, 0.10, 0.30), speed=5.0, accel=0.5)  # asked too fast
    assert robot.requests[-1].max_velocity_scaling_factor == pytest.approx(0.25)  # capped (SF-21)


def test_a_too_fast_point_to_point_plan_is_planned_again_slower(ros):
    robot, arm = connected(ros, robot_options={"speedup": 400.0})
    robot.joint_speed = 40.0  # as if joint_limits.yaml allowed far more than the TCP limit
    arm.move_ptp(down(0.40, 0.10, 0.20))
    scales = [r.max_velocity_scaling_factor for r in robot.requests]
    assert len(scales) >= 2 and scales == sorted(scales, reverse=True) and len(robot.executed) == 1
    points = robot.executed[0].joint_trajectory.points
    seconds = points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
    assert seconds > 0.35 / 0.25  # 0.35 m at no more than 250 mm/s

    robot.joint_speed = 4000.0  # never slow enough within the tries: refused, not run
    with pytest.raises(ArmError, match="above 250 mm/s") as error:
        arm.move_ptp(down(0.40, -0.20, 0.30))
    assert error.value.code == E_MOTION and len(robot.executed) == 1


def test_plans_outside_the_limits_are_refused_before_anything_moves(ros):
    robot, arm = connected(ros)
    before = robot.joints.copy()
    with pytest.raises(ArmError, match="refused: TCP would be .* mm above the table") as error:
        arm.move_ptp(down(0.40, -0.10, 0.002))
    assert error.value.code == E_MOTION and len(robot.requests) == 1 and robot.executed == []
    with pytest.raises(ArmError, match="planning failed: no IK solution"):
        arm.move_ptp(down(0.40, -0.10, 0.99))  # out of reach
    arm.set_held_fish(2.5)  # a held fish that long would reach out of the work zone
    with pytest.raises(ArmError, match="fish end would leave the work zone"):
        arm.move_lin(down(0.40, -0.10, 0.30), 0.1, 0.5)
    arm.set_held_fish(0.0)
    assert np.array_equal(robot.joints, before) and robot.executed == []
    wrong_group = make_workcell(lambda d: d["robot"].update(group="arm"))
    stranger = MoveItArm(arm.link, wrong_group, log=lambda text: None)
    stranger.connect(timeout=5.0)
    with pytest.raises(ArmError, match="invalid group name"):
        stranger.move_ptp(down(0.40, -0.10, 0.30))


def test_a_tick_stops_the_arm_where_it_is(ros):
    robot, arm = connected(ros, robot_options={"speedup": 4.0})
    target = down(0.40, 0.10, 0.20)
    calls = []

    def tick():
        calls.append(1)
        return "operator stop" if len(calls) == 5 else None

    with pytest.raises(MotionAborted, match="operator stop"):
        arm.move_ptp(target, tick)
    assert robot.cancels == 1
    stopped = robot.joints.copy()
    assert not np.allclose(stopped, HOME, atol=1e-3)  # it had left
    assert np.linalg.norm(tcp(robot)[:3, 3] - target[:3, 3]) > 0.05  # and didn't arrive
    time.sleep(0.1)
    assert np.array_equal(robot.joints, stopped)  # MoveIt holds it there
    arm.move_ptp(target)  # the next move starts from where it stopped
    assert np.allclose(tcp(robot), target, atol=1e-4)


def test_a_controller_fault_stops_the_move_and_blocks_the_next(ros):
    fault = {"text": None}
    robot, arm = connected(ros, robot_options={"speedup": 4.0}, fault=lambda: fault["text"])

    def press_stop():
        fault["text"] = "emergency stop pressed"

    with pytest.raises(ArmError, match="emergency stop pressed") as error:
        arm.move_ptp(down(0.40, 0.10, 0.20), press_stop)
    assert error.value.code == E_PROTECTIVE_STOP and robot.cancels == 1
    assert error.value.tcp_pose is not None
    planned = len(robot.requests)
    with pytest.raises(ArmError, match="emergency stop pressed"):
        arm.move_ptp(down(0.40, 0.10, 0.20))
    assert len(robot.requests) == planned  # not even planned (SF-23)


def test_step_mode_asks_before_each_move_and_a_lost_arm_is_reported(ros):
    asked = []
    robot, arm = connected(ros, confirm=lambda text: asked.append(text) or len(asked) > 1)
    target = down(0.40, -0.10, 0.25)
    with pytest.raises(MotionAborted, match="not confirmed"):
        arm.move_ptp(target)
    assert robot.executed == [] and "point-to-point move" in asked[0] and "mm/s" in asked[0]
    arm.move_ptp(target)  # confirmed this time
    assert len(robot.executed) == 1 and np.allclose(tcp(robot), target, atol=1e-4)

    robot.publish_joints = False  # the driver died
    time.sleep(0.6)
    with pytest.raises(ArmError, match="no joint state") as error:
        arm.tcp_pose()
    assert error.value.code == E_COMM
    arm.stop()  # nothing running: no error


# ---------------------------------------------------------------------------
# FAIRINO I/O and the gripper on it
# ---------------------------------------------------------------------------
def test_fairino_io_switches_outputs_and_reads_the_vacuum_switch(ros):
    robot, link = ros(fairino=True)
    assert FairinoIO.present(link)
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    assert io.fault() is None and io.age() < 0.2 and io.output(0) is False
    gripper = SuctionGripper(io, CELL.gripper, CELL.grasp.release_timeout)
    assert gripper.state() == GripperState.VACUUM_OFF

    robot.sealed = lambda: False
    gripper.vacuum_on()
    assert robot.commands == ["SetDO(1,0)", "SetDO(0,1)"] and robot.outputs[0]
    time.sleep(0.05)
    assert io.output(0) is True and io.input(1) is True and io.input(0) is False
    assert gripper.state() == GripperState.NO_SEAL
    robot.sealed = lambda: True
    time.sleep(0.05)
    assert gripper.state() == GripperState.HOLDING
    gripper.release()  # with its blow-off pulse on DO1
    assert robot.commands[-3:] == ["SetDO(0,0)", "SetDO(1,1)", "SetDO(1,0)"]
    time.sleep(0.05)
    assert gripper.state() == GripperState.VACUUM_OFF

    report = "\n".join(io.report())
    assert "DO15..DO0: 0000000000000000" in report and "emergency stop: released" in report
    flange = io.flange_position()
    assert np.allclose(flange, robot.tip()[:3, 3], atol=1e-6)
    robot.emergency_stop = True
    time.sleep(0.05)
    assert io.fault() == "emergency stop pressed" and "PRESSED" in "\n".join(io.report())
    with pytest.raises(ArmError, match="answered '-1'") as error:
        io.set_output(99, True)  # the fake knows SetDO(0-15) only as far as its pattern goes
    assert error.value.code == E_COMM


def test_without_a_command_server_the_real_arm_modes_stop(ros):
    robot, link = ros()
    assert not FairinoIO.present(link, wait=0.2)
    with pytest.raises(ArmError, match="ros2_cmd_server"):
        FairinoIO(link).connect(timeout=0.2)
    robot, link = ros(fairino=True)
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    robot.topics.discard("nonrt_state_data")  # the server stops publishing
    time.sleep(1.1)
    assert "no robot state" in io.fault() and io.input(0) is None
    assert SuctionGripper(io, CELL.gripper).state() == GripperState.ERROR  # Table AS-4


# ---------------------------------------------------------------------------
# The grasp sequence on the fake arm and gripper
# ---------------------------------------------------------------------------
def grasp_rig(ros, cell=CELL, speedup=40.0):
    robot, link = ros(fairino=True, speedup=speedup)
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    arm = MoveItArm(link, cell, fault=io.fault, log=lambda text: None)
    arm.connect(timeout=5.0)
    gripper = SuctionGripper(io, cell.gripper, cell.grasp.release_timeout)
    steps = []
    sequence = GraspSequence(arm, gripper, cell, on_step=steps.append)
    fish = down(0.40, -0.10, 0.10)
    plan = plan_grasp(cell, fish, True, 0.60, 0.10, cell.slots[0], YAW)
    held = {"fish": False}

    def sealed():  # the cups seal once they are pressed onto the fish, and then keep it
        if tcp(robot)[2, 3] < 0.095:
            held["fish"] = True
        return held["fish"]

    robot.sealed = sealed
    return robot, sequence, plan, steps, held


def test_grasp_sequence_runs_through_moveit_and_the_controller_io(ros):
    robot, sequence, plan, steps, _ = grasp_rig(ros)
    sequence.run(plan)
    assert tuple(steps) == STEPS and not sequence.holding
    assert np.allclose(tcp(robot), plan.retreat, atol=1e-4)
    assert [r.planner_id for r in robot.requests] == ["PTP", "LIN", "LIN", "LIN", "LIN", "LIN"]
    approach, lift, carry = robot.requests[1:4]
    assert approach.max_velocity_scaling_factor == pytest.approx(0.05)  # 50 mm/s (AS-22)
    assert lift.max_velocity_scaling_factor == pytest.approx(0.15)
    assert carry.max_acceleration_scaling_factor == pytest.approx(0.5 / 2.25)  # AS-43
    assert robot.commands == ["SetDO(1,0)", "SetDO(0,1)", "SetDO(0,0)", "SetDO(1,1)", "SetDO(1,0)"]
    assert len(robot.executed) == 6 and robot.cancels == 0


def test_no_seal_and_a_drop_are_reported_from_the_controller_inputs(ros):
    quick = make_workcell(lambda d: d["grasp"].update(seal_timeout=0.2))
    robot, sequence, plan, steps, held = grasp_rig(ros, quick)
    robot.sealed = lambda: False
    with pytest.raises(ArmError) as error:
        sequence.run(plan)
    assert error.value.code == E_NO_SEAL and steps[-1] == "confirm hold"
    assert error.value.gripper_state == GripperState.NO_SEAL and robot.outputs[0]  # vacuum still on
    sequence.let_go()
    sequence.retreat_up()
    assert not robot.outputs[0] and tcp(robot)[2, 3] == pytest.approx(0.19, abs=1e-3)

    robot, sequence, plan, steps, held = grasp_rig(ros, speedup=3.0)

    def lose_the_fish(step):
        steps.append(step)
        if step == "carry":
            held["fish"] = False
            robot.sealed = lambda: False

    sequence._on_step = lose_the_fish
    with pytest.raises(ArmError) as error:
        sequence.run(plan)
    assert error.value.code == E_DROP and sequence.step == "carry" and robot.cancels == 1
    assert not robot.outputs[0] and not sequence.holding  # vacuum off (AS-72)
    assert np.linalg.norm(tcp(robot)[:2, 3] - plan.carry[:2, 3]) > 0.05  # stopped on the way
