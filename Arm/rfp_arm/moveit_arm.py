"""
The FR5 through MoveIt 2 (AS-10): every move is planned by the Pilz industrial motion
planner (PTP or LIN) with MoveIt's /move_action in plan-only mode, checked here in fixed
code (safety.check_trajectory), then run with /execute_trajectory. The same code drives
the real arm (frcobot_ros2's fairino_hardware) and MoveIt's mock hardware in RViz.

THIS MOVES THE ARM. Nothing moves on import or on connect(): only move_joints, move_ptp and
move_lin do. A Tick, a controller fault, Ctrl+C and stop() cancel the running trajectory
(MoveIt then holds the arm where it is); if this process is killed instead, the move in
progress runs to its end: use the emergency stop.

Needs ROS 2 Humble with MoveIt 2 and move_group running (fairino5_v6_moveit2_config).
"""

import math
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np

from rfp_arm.interfaces import (
    E_COMM,
    E_MOTION,
    E_PROTECTIVE_STOP,
    ArmError,
    MotionAborted,
    Tick,
)
from rfp_arm.kinematics import Chain
from rfp_arm.poses import quaternion, rotation_angle
from rfp_arm.ros_link import RosLink
from rfp_arm.safety import TrajectoryCheck, check_trajectory
from rfp_arm.workcell import Box, Workcell

PIPELINE = "pilz_industrial_motion_planner"
MOVE_ACTION = "move_action"
EXECUTE_ACTION = "execute_trajectory"
SCENE_SERVICE = "apply_planning_scene"
MOVE_GROUP_NODE = "move_group"
STATE_TIMEOUT = 0.5  # s without /joint_states: the link to the arm is lost (SF-42)
TABLE_GAP = 0.01  # the table's collision box ends this far below the surface (m)
TABLE_THICKNESS = 0.05
SLOWER_TRIES = 4  # times a too-fast point-to-point plan is planned again, slower

# moveit_msgs/MoveItErrorCodes
ERROR_NAMES = {
    1: "success",
    99999: "failure",
    -1: "planning failed",
    -2: "invalid motion plan",
    -3: "plan invalidated by an environment change",
    -4: "control failed",
    -5: "unable to acquire sensor data",
    -6: "timed out",
    -7: "preempted",
    -10: "start state in collision",
    -11: "start state violates path constraints",
    -12: "goal in collision",
    -13: "goal violates path constraints",
    -14: "goal constraints violated",
    -15: "invalid group name",
    -16: "invalid goal constraints",
    -17: "invalid robot state",
    -18: "invalid link name",
    -19: "invalid object name",
    -21: "frame transform failure",
    -22: "collision checking unavailable",
    -23: "robot state stale",
    -24: "sensor info stale",
    -25: "communication failure",
    -31: "no IK solution",
}


class MoveItArm:
    def __init__(
        self,
        link: RosLink,
        workcell: Workcell,
        confirm: Callable[[str], bool] | None = None,
        fault: Callable[[], str | None] | None = None,
        log: Callable[[str], None] = print,
    ):
        """
        `confirm`: asked with a description of each planned move before it runs; False
        cancels it (step mode). `fault`: the controller's fault state (fairino.py), polled
        during every move; None on mock hardware.
        """
        from moveit_msgs.action import ExecuteTrajectory, MoveGroup
        from moveit_msgs.srv import ApplyPlanningScene
        from rcl_interfaces.srv import GetParameters
        from rclpy.action import ActionClient
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.link = link
        self.workcell = workcell
        self._confirm = confirm
        self._fault = fault
        self._log = log
        self._lock = threading.Lock()
        self._joints: dict[str, float] = {}
        self._joints_stamp = 0.0
        self._urdf: str | None = None
        self._chain: Chain | None = None
        self._fish_length = 0.0
        self._active = None  # goal handle of the running trajectory
        self._max_trans_vel = workcell.motion.pilz_max_trans_vel
        self._max_trans_acc = workcell.motion.pilz_max_trans_acc

        node = link.node
        self._move = ActionClient(node, MoveGroup, MOVE_ACTION)
        self._execute = ActionClient(node, ExecuteTrajectory, EXECUTE_ACTION)
        self._scene = node.create_client(ApplyPlanningScene, SCENE_SERVICE)
        self._parameters = node.create_client(GetParameters, f"{MOVE_GROUP_NODE}/get_parameters")
        node.create_subscription(JointState, "joint_states", self._on_joints, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "robot_description", self._on_urdf, latched)

    # -- setup -------------------------------------------------------------
    def connect(self, timeout: float = 15.0) -> None:
        """Wait for MoveIt and the arm's state, read the robot model, load the planning scene.
        Raises ArmError(E_COMM) with what is missing. Doesn't move the arm."""
        deadline = time.monotonic() + timeout

        def left() -> float:
            return max(deadline - time.monotonic(), 0.1)

        for client, name in ((self._move, MOVE_ACTION), (self._execute, EXECUTE_ACTION)):
            if not client.wait_for_server(timeout_sec=left()):
                raise ArmError(E_COMM, f"MoveIt's /{name} isn't there: is move_group running?")
        if not self._scene.wait_for_service(timeout_sec=left()):
            raise ArmError(E_COMM, f"MoveIt's /{SCENE_SERVICE} service isn't there")
        while not self._have_joints():
            if time.monotonic() > deadline:
                raise ArmError(E_COMM, "no /joint_states with the arm's joints: is the driver up?")
            time.sleep(0.05)
        self._read_parameters(left())
        while self._urdf is None:
            if time.monotonic() > deadline:
                raise ArmError(E_COMM, "no robot model on /robot_description or from move_group")
            time.sleep(0.05)
        try:
            cell = self.workcell
            self._chain = Chain.from_urdf(self._urdf, cell.base_frame, cell.tip_link)
        except ValueError as e:
            raise ArmError(E_COMM, f"robot model: {e}") from None
        missing = set(self._chain.joint_names) - set(self.workcell.joints)
        if missing:
            raise ArmError(E_COMM, f"workcell robot.joints lacks {sorted(missing)}")
        self._apply_scene(left())

    def _have_joints(self) -> bool:
        with self._lock:
            return all(name in self._joints for name in self.workcell.joints)

    def _on_joints(self, msg) -> None:
        with self._lock:
            self._joints.update(zip(msg.name, msg.position))  # message order isn't guaranteed
            self._joints_stamp = time.monotonic()

    def _on_urdf(self, msg) -> None:
        self._urdf = msg.data

    def _read_parameters(self, timeout: float) -> None:
        """Pilz's Cartesian limits (to turn m/s into its scaling factors) and, if the
        /robot_description topic is silent, the robot model, from move_group's parameters."""
        from rcl_interfaces.srv import GetParameters

        names = [
            "robot_description_planning.cartesian_limits.max_trans_vel",
            "robot_description_planning.cartesian_limits.max_trans_acc",
            "robot_description",
        ]
        try:
            if not self._parameters.wait_for_service(timeout_sec=min(timeout, 3.0)):
                raise TimeoutError
            values = self.link.call(self._parameters, GetParameters.Request(names=names), 3.0)
            velocity, acceleration, urdf = values.values
        except (TimeoutError, ValueError):
            self._log(
                "   (move_group's parameters can't be read: using the workcell's "
                f"pilz_max_trans_vel = {self._max_trans_vel:g} m/s)"
            )
            return
        if velocity.type == 3 and velocity.double_value > 0:  # PARAMETER_DOUBLE
            self._max_trans_vel = velocity.double_value
        if acceleration.type == 3 and acceleration.double_value > 0:
            self._max_trans_acc = acceleration.double_value
        if urdf.type == 4 and urdf.string_value and self._urdf is None:  # PARAMETER_STRING
            self._urdf = urdf.string_value

    def _apply_scene(self, timeout: float) -> None:
        """The table, the exclusion zones and the tool as collision objects (CS-42, C.6)."""
        from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene

        cell = self.workcell
        zone = cell.work_zone
        top = cell.table_z - TABLE_GAP
        table = Box("table", zone.x, zone.y, (top - TABLE_THICKNESS, top))
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for box in [table, *cell.exclusion_zones]:
            name = "zone " + box.name if box is not table else box.name
            scene.world.collision_objects.append(self._box(name, box, cell.base_frame))
        if cell.tool_box is not None:
            tool = AttachedCollisionObject()
            tool.link_name = cell.tip_link
            tool.object = self._box("tool", cell.tool_box, cell.tip_link)
            tool.touch_links = list(self._chain.links[-2:])
            scene.robot_state.attached_collision_objects.append(tool)
        try:
            response = self.link.call(self._scene, ApplyPlanningScene.Request(scene=scene), timeout)
        except TimeoutError:
            raise ArmError(E_COMM, "MoveIt didn't answer the planning scene request") from None
        if not response.success:
            raise ArmError(E_COMM, "MoveIt refused the planning scene (table, zones, tool)")

    @staticmethod
    def _box(name: str, box: Box, frame: str):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        item = CollisionObject()
        item.header.frame_id = frame
        item.id = name
        item.pose.orientation.w = 1.0
        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [float(v) for v in box.size]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in box.center)
        pose.orientation.w = 1.0
        item.primitives.append(shape)
        item.primitive_poses.append(pose)
        item.operation = CollisionObject.ADD
        return item

    # -- state ---------------------------------------------------------------
    def joints(self) -> dict[str, float]:
        """Joint positions (rad) by name. Raises ArmError(E_COMM) if they are stale."""
        with self._lock:
            age = time.monotonic() - self._joints_stamp
            joints = dict(self._joints)
        if age > STATE_TIMEOUT:
            raise ArmError(E_COMM, f"no joint state for {age:.1f} s")
        return joints

    def tcp_pose(self) -> np.ndarray:
        return self.workcell.tcp_of_tip(self._chain.fk(self.joints()))

    def set_held_fish(self, length: float) -> None:
        self._fish_length = length

    def fault(self) -> str | None:
        return self._fault() if self._fault else None

    def limits(self) -> str:
        return (
            f"Pilz straight-line limits {self._max_trans_vel:g} m/s, {self._max_trans_acc:g} m/s2; "
            f"this program's TCP limit {self.workcell.motion.tcp_speed_max * 1000:.0f} mm/s"
        )

    # -- moves ---------------------------------------------------------------
    def move_joints(self, joints: Sequence[float], tick: Tick | None = None) -> None:
        from moveit_msgs.msg import Constraints, JointConstraint

        names = self.workcell.joints
        if len(joints) != len(names):
            raise ArmError(E_MOTION, f"{len(names)} joint positions needed, got {len(joints)}")
        now = self.joints()
        if all(abs(now[n] - q) < 2e-3 for n, q in zip(names, joints)):
            return
        goal = Constraints()
        for name, position in zip(names, joints):
            item = JointConstraint()
            item.joint_name = name
            item.position = float(position)
            item.tolerance_above = item.tolerance_below = 1e-3
            item.weight = 1.0
            goal.joint_constraints.append(item)
        target = self.workcell.tcp_of_tip(self._chain.fk(dict(zip(names, joints))))
        self._ptp("joint move", goal, target, tick)

    def move_ptp(self, tcp_pose: np.ndarray, tick: Tick | None = None) -> None:
        if self._there(tcp_pose):
            return
        self._ptp("point-to-point move", self._pose_goal(tcp_pose), tcp_pose, tick)

    def move_lin(
        self, tcp_pose: np.ndarray, speed: float, accel: float, tick: Tick | None = None
    ) -> None:
        if self._there(tcp_pose):
            return
        limit = self.workcell.motion.tcp_speed_max
        speed = min(speed, limit)
        velocity_scale = min(1.0, speed / self._max_trans_vel)
        acceleration_scale = min(1.0, accel / self._max_trans_acc)
        goal = self._pose_goal(tcp_pose)
        for attempt in range(3):  # near a singularity a straight line needs fast joints: slower
            try:
                plan = self._plan("LIN", goal, velocity_scale, acceleration_scale)
                break
            except ArmError as e:
                if e.code != E_MOTION or attempt == 2:
                    raise
                velocity_scale /= 2
        check = self._check(plan, tcp_pose, limit)
        if not check.ok:
            raise ArmError(E_MOTION, f"straight move refused: {check.problem}", self.tcp_pose())
        self._run(plan, f"straight move, {check.summary()}", tcp_pose, tick)

    def stop(self) -> None:
        handle = self._active
        if handle is not None:
            handle.cancel_goal_async()

    def _ptp(self, what: str, goal, target: np.ndarray, tick: Tick | None) -> None:
        m = self.workcell.motion
        scale = m.ptp_velocity_scale
        for _ in range(SLOWER_TRIES):
            plan = self._plan("PTP", goal, scale, m.ptp_acceleration_scale)
            check = self._check(plan, target, m.tcp_speed_max)
            if not check.too_fast:
                break
            scale *= 0.9 * m.tcp_speed_max / check.peak_tcp_speed
        if not check.ok:
            raise ArmError(E_MOTION, f"{what} refused: {check.problem}", self.tcp_pose())
        self._run(plan, f"{what}, {check.summary()}", target, tick)

    def _there(self, tcp_pose: np.ndarray) -> bool:
        now = self.tcp_pose()
        return np.linalg.norm(now[:3, 3] - tcp_pose[:3, 3]) < 5e-4 and rotation_angle(
            now, tcp_pose
        ) < math.radians(0.2)

    def _pose_goal(self, tcp_pose: np.ndarray):
        """Goal constraints putting the tip link where the TCP pose needs it."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
        from shape_msgs.msg import SolidPrimitive

        cell = self.workcell
        tip = cell.tip_of_tcp(np.asarray(tcp_pose, float))
        position = PositionConstraint()
        position.header.frame_id = cell.base_frame
        position.link_name = cell.tip_link
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [1e-4]
        center = Pose()
        center.position.x, center.position.y, center.position.z = (float(v) for v in tip[:3, 3])
        center.orientation.w = 1.0
        position.constraint_region.primitives.append(region)
        position.constraint_region.primitive_poses.append(center)
        position.weight = 1.0
        orientation = OrientationConstraint()
        orientation.header.frame_id = cell.base_frame
        orientation.link_name = cell.tip_link
        x, y, z, w = quaternion(tip)
        orientation.orientation.x, orientation.orientation.y = x, y
        orientation.orientation.z, orientation.orientation.w = z, w
        orientation.absolute_x_axis_tolerance = 1e-3
        orientation.absolute_y_axis_tolerance = 1e-3
        orientation.absolute_z_axis_tolerance = 1e-3
        orientation.weight = 1.0
        goal = Constraints()
        goal.position_constraints.append(position)
        goal.orientation_constraints.append(orientation)
        return goal

    def _plan(self, planner: str, goal, velocity_scale: float, acceleration_scale: float):
        """A planned moveit_msgs/RobotTrajectory from where the arm is now. Nothing moves."""
        from moveit_msgs.action import MoveGroup

        if fault := self.fault():
            raise ArmError(E_PROTECTIVE_STOP, fault, self.tcp_pose())
        m = self.workcell.motion
        request = MoveGroup.Goal()
        request.request.group_name = self.workcell.group
        request.request.pipeline_id = PIPELINE
        request.request.planner_id = planner
        request.request.num_planning_attempts = 1
        request.request.allowed_planning_time = float(m.planning_time)
        request.request.max_velocity_scaling_factor = float(velocity_scale)
        request.request.max_acceleration_scaling_factor = float(acceleration_scale)
        request.request.start_state.is_diff = True  # = the arm's state now
        request.request.goal_constraints.append(goal)
        request.planning_options.plan_only = True
        request.planning_options.planning_scene_diff.is_diff = True
        request.planning_options.planning_scene_diff.robot_state.is_diff = True
        try:
            handle = self.link.wait(self._move.send_goal_async(request), 5.0)
            if not handle.accepted:
                raise ArmError(E_MOTION, "MoveIt rejected the planning request", self.tcp_pose())
            result = self.link.wait(handle.get_result_async(), m.planning_time + 5.0).result
        except TimeoutError:
            raise ArmError(E_COMM, "MoveIt didn't answer the planning request") from None
        code = result.error_code.val
        if code != 1:
            name = ERROR_NAMES.get(code, f"error {code}")
            raise ArmError(E_MOTION, f"{planner} planning failed: {name}", self.tcp_pose())
        return result.planned_trajectory

    def _check(self, plan, target: np.ndarray, tcp_speed_max: float) -> TrajectoryCheck:
        """The fixed-code check of a plan, including that it starts here and ends there."""
        trajectory = plan.joint_trajectory
        names = list(trajectory.joint_names)
        if not trajectory.points:
            return TrajectoryCheck("the plan has no points", False, 0.0, 0.0, 0.0, None)
        positions = np.array([list(point.positions) for point in trajectory.points], float)
        times = np.array(
            [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points]
        )
        check = check_trajectory(
            self.workcell, self._chain, names, positions.reshape(len(times), -1), times,
            self._fish_length, tcp_speed_max,
        )  # fmt: skip
        if check.end_tcp is None:
            return check
        now = self.joints()
        if any(abs(now[name] - q) > 0.02 for name, q in zip(names, positions[0])):
            check.problem, check.too_fast = "the plan doesn't start where the arm is", False
        elif np.linalg.norm(check.end_tcp[:3, 3] - target[:3, 3]) > 2e-3 or rotation_angle(
            check.end_tcp, target
        ) > math.radians(1.0):
            check.problem, check.too_fast = "the plan doesn't end at the target", False
        return check

    def _run(self, plan, what: str, target: np.ndarray, tick: Tick | None) -> None:
        """Execute a checked plan, watching the Tick and the controller. THE ARM MOVES."""
        from moveit_msgs.action import ExecuteTrajectory

        if self._confirm is not None and not self._confirm(what):
            raise MotionAborted("move not confirmed")
        points = plan.joint_trajectory.points
        duration = points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
        try:
            handle = self.link.wait(
                self._execute.send_goal_async(ExecuteTrajectory.Goal(trajectory=plan)), 5.0
            )
        except TimeoutError:
            raise ArmError(E_COMM, "MoveIt didn't answer the execution request") from None
        if not handle.accepted:
            raise ArmError(E_MOTION, "MoveIt rejected the trajectory", self.tcp_pose())
        self._active = handle
        finished = threading.Event()
        result = handle.get_result_async()
        result.add_done_callback(lambda _: finished.set())
        deadline = time.monotonic() + duration * 1.5 + 5.0
        try:
            while not finished.wait(0.02):
                reason = tick() if tick else None
                if reason:
                    raise MotionAborted(reason)
                if fault := self.fault():
                    raise ArmError(E_PROTECTIVE_STOP, fault)
                if time.monotonic() > deadline:
                    raise ArmError(E_MOTION, "the move didn't finish in time")
        except BaseException as stopped:  # also Ctrl+C: never leave a trajectory running
            self._cancel(handle, finished)
            if isinstance(stopped, ArmError) and stopped.tcp_pose is None:
                stopped.tcp_pose = self._tcp_or_none()
            raise
        finally:
            self._active = None
        code = result.result().result.error_code.val
        if code != 1:
            name = ERROR_NAMES.get(code, f"error {code}")
            raise ArmError(E_MOTION, f"execution failed: {name}", self._tcp_or_none())
        self._settle()
        error = float(np.linalg.norm(self.tcp_pose()[:3, 3] - target[:3, 3]))
        if error > self.workcell.motion.goal_tolerance:
            raise ArmError(
                E_MOTION, f"the move ended {error * 1000:.1f} mm from its target", self.tcp_pose()
            )

    def _cancel(self, handle, finished: threading.Event) -> None:
        try:
            self.link.wait(handle.cancel_goal_async(), 2.0)
        except TimeoutError:
            self._log("⚠️  MoveIt didn't confirm the stop: use the emergency stop if the arm moves")
        finished.wait(3.0)

    def _settle(self, timeout: float = 1.0) -> None:
        """Wait until the joints stop changing (the state lags the end of the trajectory)."""
        deadline = time.monotonic() + timeout
        last = self.joints()
        while time.monotonic() < deadline:
            time.sleep(0.05)
            now = self.joints()
            if all(abs(now[name] - last[name]) < 1e-4 for name in self.workcell.joints):
                return
            last = now

    def _tcp_or_none(self) -> np.ndarray | None:
        try:
            return self.tcp_pose()
        except ArmError:
            return None
