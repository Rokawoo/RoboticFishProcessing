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


def down(x: float, y: float, z: float, turn: float = 0.4):
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


def tcp(robot):
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
    body = CELL.tool_box
    assert tool.object.primitives[0].dimensions == pytest.approx(body.size)


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
    goal = request.goal_constraints[0]  # joints from the robot model, in the present posture
    names = [j.joint_name for j in goal.joint_constraints]
    planned = np.array([j.position for j in goal.joint_constraints])
    assert names == list(CELL.joints) and not goal.position_constraints
    assert np.allclose(CELL.tcp_of_tip(FR5.fk(planned)), target, atol=1e-6)
    assert np.abs(planned - HOME)[:5].max() < 1.0  # no flip to another posture
    assert len(robot.executed) == 1

    class NoSolution(type(arm._chain)):  # the model's IK finds nothing: MoveIt's (a pose goal)
        def ik(self, target, seed, iterations=200):
            return None

    model = arm._chain
    arm._chain = NoSolution(model.joints, model.base_link, model.tip_link, model.links)
    arm.move_ptp(down(0.40, -0.05, 0.20))
    goal = robot.requests[-1].goal_constraints[0]
    assert goal.position_constraints[0].link_name == "wrist3_link"  # the tip link, not the TCP
    tip = goal.position_constraints[0].constraint_region.primitive_poses[0].position
    assert tip.z == pytest.approx(0.20 + CELL.tcp[2, 3])  # the TCP is past the tip link
    arm._chain = model
    arm.move_ptp(target)

    requests = len(robot.requests)
    arm.move_ptp(target)  # already there: nothing is planned
    assert len(robot.requests) == requests
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


@pytest.mark.parametrize("firmware", ["3.7.5", "3.7.8", "3.9.5"])
def test_fairino_io_works_with_each_firmware_driver(ros, firmware):
    # Up to 3.8.0 the state has one error_code, and the 3.7.5 driver answers "1" for success
    robot, link = ros(fairino=True, firmware=firmware)
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    assert io.driver == ("V2" if firmware == "3.7.5" else "V3") and io.fault() is None
    io.set_output(0, True)  # confirmed from the state, whatever the answer means
    assert robot.outputs[0] and io.output(0) is True
    assert io.execute("SetDO(0,0)") == ("1" if firmware == "3.7.5" else "0")
    with pytest.raises(ArmError, match="answered") as error:
        io.execute("NoSuchCommand()")  # "0" means an error to the 3.7.5 driver
    assert error.value.code == E_COMM
    robot.error = (7, 3)
    time.sleep(0.05)
    sub = 3 if firmware == "3.9.5" else 0  # a sub code from 3.8.2 on
    assert io.fault() == f"controller error 7-{sub}"
    assert f"driver: {io.driver}" in "\n".join(io.report())
    assert io.firmware() == (None if firmware == "3.7.5" else f"V{firmware}")
    assert np.allclose(io.joint_positions(), robot.joints, atol=1e-6)  # its state: degrees


@pytest.mark.parametrize("firmware", ["3.7.5", "3.9.5"])
def test_a_command_server_that_sends_no_state_is_named_and_worked_without(ros, firmware):
    # On the real 3.7 controller the command server sent no state: everything stopped at
    # "no robot state". Say why; with require_state off, go on with what the answers tell
    robot, link = ros(fairino=True, firmware=firmware)
    robot.publish_state = False
    with pytest.raises(ArmError, match="firmware") as error:
        FairinoIO(link).connect(timeout=0.3)
    assert "setup.sh --robot" in error.value.message
    io = FairinoIO(link)
    io.connect(timeout=0.3, require_state=False)
    assert io.stateless and io.driver == ("V2" if firmware == "3.7.5" else "V3")
    io.set_output(0, True)  # judged by the answer ("1" from the 3.7.5 driver, "0" after)
    assert robot.outputs[0]
    assert io.fault() is None and io.joint_positions() is None and io.joint_torques() is None
    robot.publish_state = True  # a state after all: read as usual from then on
    time.sleep(0.1)
    assert np.allclose(io.joint_positions(), robot.joints, atol=1e-6)


def test_the_empty_states_of_a_dispatcher_without_a_driver_read_as_nothing(ros):
    # FAIRINO's newest server on a controller it has no driver for: states of zeros, which
    # the check that MoveIt drives this arm took for the arm's joints
    robot, link = ros(fairino=True)
    robot.reconnect_flag = 2
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    assert io.joint_positions() is None and io.joint_torques() is None
    assert "setup.sh --robot" in io.fault()


def test_joined_trajectories_keep_each_time_a_valid_duration():
    # A time just under a whole second rounded its fraction up to 1e9 ns: no Duration holds
    # that, and the blended trajectory would have been refused
    from fake_ros import message
    from rfp_arm.moveit_arm import _joined

    def trajectory(*times):
        made = message("moveit_msgs/msg/RobotTrajectory")()
        for sec, nanosec in times:
            point = message("trajectory_msgs/msg/JointTrajectoryPoint")(positions=[0.0] * 6)
            point.time_from_start.sec, point.time_from_start.nanosec = sec, nanosec
            made.joint_trajectory.points.append(point)
        return made

    joined = _joined([trajectory((0, 0), (1, 999_999_999)), trajectory((0, 0), (0, 1))])
    last = joined.joint_trajectory.points[-1].time_from_start
    assert (last.sec, last.nanosec) == (2, 0)
    nanoseconds = [p.time_from_start.nanosec for p in joined.joint_trajectory.points]
    assert all(0 <= ns < 1_000_000_000 for ns in nanoseconds)


def test_a_command_server_not_connected_to_the_controller_is_a_fault(ros):
    robot, link = ros(fairino=True)
    robot.reconnect_flag = 1  # the newest driver's dispatcher, still looking for the controller
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    assert "isn't connected to the controller" in io.fault()


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
    planners = [r.planner_id for r in robot.requests]
    # the move over the fish (planned again slower where a point of the arm was too fast),
    # the approach and the retreat; lift, carry and lower as one blended sequence
    assert planners[-2:] == ["LIN", "LIN"] and set(planners[:-2]) == {"PTP"}
    approach = robot.requests[-2]
    assert approach.max_velocity_scaling_factor == pytest.approx(0.05)  # 50 mm/s (AS-22)
    [sequence] = robot.sequences
    lift, carry, lower = (item.req for item in sequence.items)
    assert [item.blend_radius > 0 for item in sequence.items] == [True, True, False]
    assert [r.start_state.is_diff for r in (lift, carry, lower)] == [True, False, False]
    assert {r.planner_id for r in (lift, carry, lower)} == {"LIN"}
    assert lift.max_velocity_scaling_factor == pytest.approx(0.20)  # carry_speed
    assert carry.max_acceleration_scaling_factor == pytest.approx(0.5 / 2.25)  # AS-43
    assert robot.commands == ["SetDO(1,0)", "SetDO(0,1)", "SetDO(0,0)", "SetDO(1,1)", "SetDO(1,0)"]
    assert len(robot.executed) == len(planners) - planners.count("PTP") + 2  # no stop between
    assert robot.cancels == 0


def test_without_pilz_s_sequence_service_lift_carry_and_lower_stop_at_each_corner(ros):
    robot, link = ros(fairino=True)
    del robot.services["plan_sequence_path"]
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    said = []
    arm = MoveItArm(link, CELL, fault=io.fault, log=said.append)
    arm.connect(timeout=5.0)
    assert not arm.can_blend and any("plan_sequence_path" in line for line in said)
    gripper = SuctionGripper(io, CELL.gripper, CELL.grasp.release_timeout)
    steps = []
    sequence = GraspSequence(arm, gripper, CELL, on_step=steps.append)
    plan = plan_grasp(CELL, down(0.40, -0.10, 0.10), True, 0.60, 0.10, CELL.slots[0], YAW)
    held = []
    robot.sealed = lambda: bool(held) or (tcp(robot)[2, 3] < 0.095 and not held.append(1))
    sequence.run(plan)
    assert tuple(steps) == STEPS and robot.sequences == []
    assert [r.planner_id for r in robot.requests][-5:] == ["LIN"] * 5  # approach to retreat


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


def test_the_arm_knows_its_wrist_angle_and_where_it_would_be(ros):
    robot, arm = connected(ros)
    assert arm.wrist_angle() == pytest.approx(0.0, abs=1e-6)  # j6 of HOME
    target = down(0.50, 0.10, 0.20, turn=1.0)
    predicted = arm.wrist_angle(target)
    arm.move_ptp(target)
    assert arm.wrist_angle() == pytest.approx(predicted, abs=1e-3)  # what the planner assumed
    assert robot.joints[5] == pytest.approx(predicted, abs=1e-3)
    tight = make_workcell(lambda d: d["tool"].update(wrist_range_deg=[-20, 20]))
    careful = MoveItArm(arm.link, tight, log=lambda text: None)
    careful.connect(timeout=5.0)
    with pytest.raises(ArmError, match="the wrist .j6. would turn to"):
        careful.move_ptp(down(0.40, -0.10, 0.25, turn=-1.2))
    careful.move_joints(HOME)  # back toward the range: allowed
    assert robot.joints[5] == pytest.approx(0.0, abs=1e-6)


def test_the_arm_feels_a_surface_through_the_controller_torques(ros):
    robot, link = ros(fairino=True)
    robot.tcp = CELL.tcp
    io = FairinoIO(link)
    io.connect(timeout=2.0)
    assert len(io.joint_torques()) == 6
    arm = MoveItArm(link, CELL, fault=io.fault, torques=io.joint_torques, log=lambda text: None)
    arm.connect(timeout=5.0)
    above = down(0.40, -0.10, 0.20, turn=0.0)
    arm.move_ptp(above)
    time.sleep(0.05)
    assert arm.contact_force(zero=True) == pytest.approx(0.0, abs=0.5)
    robot.contact_z = 0.195  # something springy 5 mm below the cups: 3 N per mm
    pressed = above.copy()
    pressed[2, 3] = 0.185
    arm.move_lin(pressed, 0.02, 0.5)
    time.sleep(0.05)
    assert arm.contact_force() == pytest.approx(30.0, abs=3.0)  # 10 mm into it
    assert MoveItArm(link, CELL, log=lambda text: None).contact_force() is None  # no torques


def test_grasp_feels_for_the_fish_on_the_fake_arm(ros):
    quick = make_workcell(lambda d: d["grasp"].update(feel_settle=0.03))
    robot, sequence, plan, steps, _ = grasp_rig(ros, quick)
    robot.tcp = quick.tcp
    robot.contact_z = 0.095  # 5 mm lower than the camera said
    sequence.arm._torques = sequence.arm._fault.__self__.joint_torques  # the same FairinoIO
    sequence.run(plan)
    assert tuple(steps) == STEPS
    assert sequence.contact_depth == pytest.approx(0.012, abs=0.0035)  # felt, a step either way
    assert sum(r.planner_id == "LIN" for r in robot.requests) > 8  # the small steps down
