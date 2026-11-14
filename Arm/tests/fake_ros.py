"""
A stand-in for ROS 2, MoveIt and the FAIRINO driver, so moveit_arm.py and fairino.py run
in tests on any machine:

- message classes built from ros_messages.json (the real packages' fields, see
  make_ros_messages.py), as strict as rclpy's: an unknown field or a value of the wrong
  type (an int where a float belongs) raises;
- `FakeRobot`: an FR5 that plans PTP and LIN moves like move_group with the Pilz planner
  does, runs trajectories faster than real time, publishes joint states, keeps the
  planning scene, and answers the FAIRINO command server's SetDO with a vacuum switch;
- a minimal rclpy (node, action and service clients, executor) wired to that robot.

`install(monkeypatch, robot)` puts them in sys.modules for one test.

It checks what this code sends and how it reacts; it can't tell whether the real MoveIt
accepts a request. That is what `pick_place.py --mode sim` on the robot PC is for.
"""

import json
import math
import re
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
from arm_helpers import solve_ik
from rfp_arm.kinematics import FR5
from rfp_arm.poses import from_quaternion, matrix_rpy
from rfp_arm.sim import interpolate

DEFINITIONS = json.loads((Path(__file__).parent / "ros_messages.json").read_text(encoding="utf-8"))
INTEGERS = {"char", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
PYTHON_TYPE = {"float32": float, "float64": float, "bool": bool, "string": str, "byte": bytes}
PYTHON_TYPE.update(dict.fromkeys(INTEGERS, int))
_classes: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
class Message:
    """Base of the generated classes: fixed fields, type-checked like rclpy's messages."""

    _fields: dict[str, tuple[str, str, str]]  # name -> (type, array suffix, default)

    def __init__(self, **values):
        for name, (kind, array, default) in self._fields.items():
            object.__setattr__(self, name, _default(kind, array, default))
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name not in self._fields:
            raise AttributeError(f"{type(self).__name__} has no field '{name}'")
        kind, array, _ = self._fields[name]
        if array:
            if isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
                raise TypeError(f"{type(self).__name__}.{name} must be a sequence")
            value = list(value)
            limit = array.strip("[]<=")
            if limit and (
                len(value) > int(limit) or ("<" not in array and len(value) != int(limit))
            ):
                raise ValueError(f"{type(self).__name__}.{name} takes {array} values")
            for item in value:
                _check(type(self).__name__, name, kind, item)
        else:
            _check(type(self).__name__, name, kind, value)
        object.__setattr__(self, name, value)

    def __repr__(self):
        return (
            f"{type(self).__name__}({', '.join(f'{n}={getattr(self, n)!r}' for n in self._fields)})"
        )


class Anything:
    """A message of a package that isn't in ros_messages.json: takes any field."""


def _check(owner: str, name: str, kind: str, value) -> None:
    expected = PYTHON_TYPE.get(kind) or message(kind)
    wrong = not isinstance(value, expected) or (expected is int and isinstance(value, bool))
    if wrong:
        raise TypeError(
            f"{owner}.{name} must be {expected.__name__} ({kind}), got {type(value).__name__}"
        )


def _default(kind: str, array: str, default: str):
    if array:
        size = array.strip("[]")
        return [_default(kind, "", "") for _ in range(int(size))] if size.isdigit() else []
    if kind in ("float32", "float64"):
        return float(default or 0)
    if kind in INTEGERS:
        return int(default or 0)
    if kind == "bool":
        return default.lower() == "true"
    if kind == "string":
        return default.strip("\"'")
    if kind == "byte":
        return bytes([int(default or 0)])
    return message(kind)()


def message(name: str) -> type:
    """The class of 'package/msg/Name' (or '..._Goal', '..._Request', ...)."""
    if name not in _classes:
        definition = DEFINITIONS[name]
        if definition["fields"] is None:
            _classes[name] = type(name.split("/")[-1], (Anything,), {})
        else:
            fields = {f[0]: (f[1], f[2], f[3]) for f in definition["fields"]}
            constants = {
                key: _default(kind, "", value)
                for key, (kind, value) in definition["constants"].items()
            }
            _classes[name] = type(name.split("/")[-1], (Message,), {"_fields": fields, **constants})
    return _classes[name]


def message_modules() -> dict[str, types.ModuleType]:
    """'moveit_msgs.msg', 'moveit_msgs.action', ... with their classes, as rclpy lays them out."""
    modules: dict[str, types.ModuleType] = {}
    for name in DEFINITIONS:
        package, kind, short = name.split("/")
        modules.setdefault(package, types.ModuleType(package))
        module = modules.setdefault(f"{package}.{kind}", types.ModuleType(f"{package}.{kind}"))
        setattr(modules[package], kind, module)
        if kind == "msg":
            setattr(module, short, message(name))
        else:  # MoveGroup.Goal, GetParameters.Request, ...
            base, part = short.rsplit("_", 1)
            holder = getattr(module, base, None) or type(base, (), {})
            setattr(holder, part, message(name))
            setattr(module, base, holder)
    return modules


def fr5_urdf() -> str:
    """The FR5's chain as URDF text, as robot_state_publisher would publish it."""
    links = ["base_link", *FR5.links]
    parts = ['<robot name="fairino5_v6_robot">'] + [f'<link name="{name}"/>' for name in links]
    for joint, parent, child in zip(FR5.joints, links, links[1:]):
        xyz = " ".join(repr(float(v)) for v in joint.origin[:3, 3])
        rpy = " ".join(repr(float(v)) for v in matrix_rpy(joint.origin))
        parts.append(
            f'<joint name="{joint.name}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/>'
            f'<limit lower="{joint.lower}" upper="{joint.upper}" effort="1" velocity="3.15"/>'
            "</joint>"
        )
    return "\n".join([*parts, "</robot>"])


# ---------------------------------------------------------------------------
# rclpy
# ---------------------------------------------------------------------------
class Future:
    def __init__(self):
        self._result = None
        self._done = False
        self._callbacks = []
        self._lock = threading.Lock()

    def set_result(self, result) -> None:
        with self._lock:
            self._result, self._done = result, True
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(self)

    def add_done_callback(self, callback) -> None:
        with self._lock:
            if not self._done:
                self._callbacks.append(callback)
                return
        callback(self)

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._result


class GoalHandle:
    def __init__(self, robot, accepted: bool):
        self.accepted = accepted
        self._robot = robot
        self.result_future = Future()
        self.cancelled = False

    def get_result_async(self) -> Future:
        return self.result_future

    def cancel_goal_async(self) -> Future:
        self.cancelled = True
        self._robot.cancels += 1
        future = Future()
        future.set_result(True)
        return future


class ActionClient:
    def __init__(self, node, action_type, name: str):
        self._robot, self._name = node.robot, name

    def wait_for_server(self, timeout_sec=None) -> bool:
        return self._name in self._robot.actions

    def send_goal_async(self, goal) -> Future:
        future = Future()
        handle = GoalHandle(self._robot, True)
        self._robot.jobs.append(lambda: self._robot.actions[self._name](goal, handle))
        future.set_result(handle)
        return future


class ServiceClient:
    def __init__(self, robot, name: str):
        self._robot, self._name = robot, name

    def wait_for_service(self, timeout_sec=None) -> bool:
        return self._name in self._robot.services

    def call_async(self, request) -> Future:
        future = Future()
        self._robot.jobs.append(
            lambda: future.set_result(self._robot.services[self._name](request))
        )
        return future


class Node:
    def __init__(self, name: str, robot):
        self.name, self.robot = name, robot

    def create_subscription(self, message_type, topic: str, callback, qos):
        self.robot.subscribe(topic, callback, getattr(qos, "durability", None) == "transient_local")

    def create_client(self, service_type, name: str) -> ServiceClient:
        return ServiceClient(self.robot, name)

    def count_publishers(self, topic: str) -> int:
        return int(topic in self.robot.topics)

    def destroy_node(self) -> None:
        self.robot.nodes_destroyed += 1


class Executor:
    def __init__(self):
        self._stop = threading.Event()

    def add_node(self, node) -> None:
        pass

    def spin(self) -> None:  # the robot's own thread delivers messages and answers
        self._stop.wait()

    def shutdown(self) -> None:
        self._stop.set()


def install(monkeypatch, robot: "FakeRobot") -> None:
    """Put the fake rclpy and message packages in sys.modules for this test."""
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda: None
    rclpy.shutdown = lambda: None
    rclpy.create_node = lambda name: Node(name, robot)
    action = types.ModuleType("rclpy.action")
    action.ActionClient = ActionClient
    executors = types.ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = Executor
    qos = types.ModuleType("rclpy.qos")
    qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient_local")
    qos.QoSProfile = lambda depth=10, durability=None: types.SimpleNamespace(
        depth=depth, durability=durability
    )
    rclpy.action, rclpy.executors, rclpy.qos = action, executors, qos
    modules = {"rclpy": rclpy, "rclpy.action": action, "rclpy.executors": executors,
               "rclpy.qos": qos, **message_modules()}  # fmt: skip
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------
class FakeRobot:
    """An FR5 behind move_group and (with `fairino=True`) the FAIRINO command server."""

    GROUP, TIP, BASE = "fairino5_v6_group", "wrist3_link", "base_link"

    def __init__(self, joints, fairino: bool = False, speedup: float = 40.0):
        self.joints = np.array(joints, float)
        self.speedup = speedup  # trajectories run this much faster than their timing
        self.max_trans_vel, self.max_trans_acc = 1.0, 2.25  # Pilz's Cartesian limits
        self.joint_speed = 3.15  # rad/s at scaling factor 1
        self.requests = []  # every planning request (moveit_msgs MotionPlanRequest)
        self.executed = []  # every trajectory that was run
        self.scene = None
        self.cancels = 0
        self.nodes_destroyed = 0
        self.publish_joints = True
        self.path = []  # joint positions while trajectories ran
        self.outputs = [False] * 16
        self.inputs = [False] * 16
        self.emergency_stop = False
        self.sealed = lambda: True  # whether the cups seal when the vacuum is on
        self.commands = []
        self.jobs = []
        self._subscribers: dict[str, list] = {}
        self._running = None
        self._vacuum_since = self._off_since = None
        self.actions = {"move_action": self._plan, "execute_trajectory": self._execute}
        self.services = {
            "apply_planning_scene": self._apply_scene,
            "move_group/get_parameters": self._parameters,
        }
        self.topics = {"joint_states", "robot_description"}
        if fairino:
            self.services["fairino_remote_command_service"] = self._command
            self.topics.add("nonrt_state_data")
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._alive = False
        self._thread.join(2.0)

    def tip(self) -> np.ndarray:
        return FR5.fk(self.joints)

    # -- topics ----------------------------------------------------------------
    def subscribe(self, topic: str, callback, latched: bool) -> None:
        self._subscribers.setdefault(topic, []).append(callback)
        if topic == "robot_description" and latched:
            self.jobs.append(lambda: callback(message("std_msgs/msg/String")(data=fr5_urdf())))

    def _publish(self, topic: str, msg) -> None:
        for callback in self._subscribers.get(topic, []):
            callback(msg)

    def _loop(self) -> None:
        while self._alive:
            while self.jobs:
                self.jobs.pop(0)()
            self._advance()
            if self.publish_joints:
                order = [3, 0, 5, 1, 4, 2]  # like the real driver: not in j1..j6 order
                self._publish("joint_states", message("sensor_msgs/msg/JointState")(
                    name=[f"j{i + 1}" for i in order],
                    position=[float(self.joints[i]) for i in order],
                ))  # fmt: skip
            if "nonrt_state_data" in self.topics:
                self._publish("nonrt_state_data", self._state())
            time.sleep(0.002)

    # -- planning (move_group with the Pilz planner) -------------------------------
    def _plan(self, goal, handle) -> None:
        request = goal.request
        self.requests.append(request)
        result = message("moveit_msgs/action/MoveGroup_Result")()
        result.error_code.val = self._plan_into(goal, result)
        handle.result_future.set_result(types.SimpleNamespace(result=result, status=4))

    def _plan_into(self, goal, result) -> int:
        request = goal.request
        if not goal.planning_options.plan_only:
            return 99999  # this code must check a plan before it runs
        if request.group_name != self.GROUP:
            return -15
        if request.pipeline_id != "pilz_industrial_motion_planner":
            return 99999
        scales = (request.max_velocity_scaling_factor, request.max_acceleration_scaling_factor)
        if not all(0 < s <= 1 for s in scales) or not request.start_state.is_diff:
            return -17
        if len(request.goal_constraints) != 1:
            return -16
        constraints = request.goal_constraints[0]
        start = self.joints.copy()
        if constraints.joint_constraints:
            by_name = {c.joint_name: c.position for c in constraints.joint_constraints}
            if set(by_name) != set(FR5.joint_names) or request.planner_id != "PTP":
                return -16
            end = np.array([by_name[name] for name in FR5.joint_names])
            positions, times = self._ptp(start, end, *scales)
        else:
            if len(constraints.position_constraints) != 1 or (
                len(constraints.orientation_constraints) != 1
            ):
                return -16
            where, how = constraints.position_constraints[0], constraints.orientation_constraints[0]
            if where.link_name != self.TIP or how.link_name != self.TIP:
                return -18
            if (
                where.header.frame_id != self.BASE
                or len(where.constraint_region.primitive_poses) != 1
            ):
                return -16
            point = where.constraint_region.primitive_poses[0].position
            q = how.orientation
            if abs(math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2) - 1) > 1e-6:
                return -16
            target = from_quaternion((point.x, point.y, point.z), (q.x, q.y, q.z, q.w))
            if request.planner_id == "PTP":
                end = solve_ik(target, start)
                if end is None:
                    return -31
                positions, times = self._ptp(start, end, *scales)
            elif request.planner_id == "LIN":
                line = self._lin(start, target, scales[0])
                if line is None:
                    return -1
                positions, times = line
            else:
                return -1
        trajectory = result.planned_trajectory.joint_trajectory
        trajectory.joint_names = list(FR5.joint_names)
        point_type = message("trajectory_msgs/msg/JointTrajectoryPoint")
        for row, t in zip(positions, times):
            point = point_type(positions=[float(v) for v in row])
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = round(float(t - int(t)) * 1e9)
            trajectory.points.append(point)
        return 1

    def _ptp(self, start, end, velocity_scale, acceleration_scale):
        travel = float(np.abs(end - start).max())
        duration = max(travel / (velocity_scale * self.joint_speed), 0.2)
        times = np.arange(0.0, duration + 0.1, 0.1)
        times[-1] = duration
        return start + (end - start) * (times / duration)[:, None], times

    def _lin(self, start, target, velocity_scale):
        begin = FR5.fk(start)
        distance = float(np.linalg.norm(target[:3, 3] - begin[:3, 3]))
        duration = max(distance / (velocity_scale * self.max_trans_vel), 0.2)
        times = np.arange(0.0, duration + 0.1, 0.1)
        times[-1] = duration
        rows, q = [], start
        for t in times:
            q = solve_ik(interpolate(begin, target, t / duration), q)
            if q is None:
                return None
            rows.append(q)
        return np.array(rows), times

    # -- execution ---------------------------------------------------------------
    def _execute(self, goal, handle) -> None:
        trajectory = goal.trajectory.joint_trajectory
        positions = np.array([p.positions for p in trajectory.points])
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                          for p in trajectory.points])  # fmt: skip
        self.executed.append(goal.trajectory)
        if len(positions) == 0 or np.abs(positions[0] - self.joints).max() > 0.01:
            self._finish(handle, -17)  # doesn't start where the arm is
            return
        self._running = (positions, times, time.monotonic(), handle)

    def _advance(self) -> None:
        if self._running is None:
            return
        positions, times, started, handle = self._running
        if handle.cancelled:
            self._running = None
            self._finish(handle, -7)
            return
        t = (time.monotonic() - started) * self.speedup
        if t >= times[-1]:
            self.joints = positions[-1].copy()
            self._running = None
            self.path.append(self.joints.copy())
            self._finish(handle, 1)
            return
        self.joints = np.array([np.interp(t, times, positions[:, i]) for i in range(6)])
        self.path.append(self.joints.copy())

    @staticmethod
    def _finish(handle, code: int) -> None:
        result = message("moveit_msgs/action/ExecuteTrajectory_Result")()
        result.error_code.val = code
        handle.result_future.set_result(types.SimpleNamespace(result=result, status=4))

    # -- services ----------------------------------------------------------------
    def _apply_scene(self, request):
        self.scene = request.scene
        return message("moveit_msgs/srv/ApplyPlanningScene_Response")(success=True)

    def _parameters(self, request):
        value = message("rcl_interfaces/msg/ParameterValue")
        known = {
            "robot_description_planning.cartesian_limits.max_trans_vel": self.max_trans_vel,
            "robot_description_planning.cartesian_limits.max_trans_acc": self.max_trans_acc,
            "robot_description": fr5_urdf(),
        }
        values = []
        for name in request.names:
            found = known.get(name)
            if isinstance(found, float):
                values.append(value(type=3, double_value=found))
            elif isinstance(found, str):
                values.append(value(type=4, string_value=found))
            else:
                values.append(value())
        return message("rcl_interfaces/srv/GetParameters_Response")(values=values)

    # -- the FAIRINO command server ------------------------------------------------
    def _command(self, request):
        self.commands.append(request.cmd_str)
        response = message("fairino_msgs/srv/RemoteCmdInterface_Response")
        match = re.fullmatch(r"SetDO\((\d+),([01])\)", request.cmd_str)
        if match is None or int(match[1]) > 15:
            return response(cmd_res="-1")
        index, on = int(match[1]), match[2] == "1"
        if index == 0 and on != self.outputs[0]:
            self._vacuum_since, self._off_since = (
                (time.monotonic(), None) if on else (None, time.monotonic())
            )
        self.outputs[index] = on
        return response(cmd_res="0")

    def _state(self):
        """The vacuum switch: DI1 while there is vacuum, DI0 once a sealed fish is held."""
        now = time.monotonic()
        vacuum = self.outputs[0] and now - self._vacuum_since > 0.01
        fading = self._off_since is not None and now - self._off_since < 0.01
        self.inputs[1] = vacuum or fading
        self.inputs[0] = vacuum and self.sealed()
        word = lambda bits: sum(1 << i for i, on in enumerate(bits) if on)
        flange = self.tip()[:3, 3] * 1000
        state = message("fairino_msgs/msg/RobotNonrtState")(
            dgt_output_l=word(self.outputs) & 0xFF, dgt_output_h=word(self.outputs) >> 8,
            dgt_input_l=word(self.inputs) & 0xFF, dgt_input_h=word(self.inputs) >> 8,
            emg=int(self.emergency_stop),
            flange_x_cur_pos=float(flange[0]), flange_y_cur_pos=float(flange[1]),
            flange_z_cur_pos=float(flange[2]),
        )  # fmt: skip
        return state
