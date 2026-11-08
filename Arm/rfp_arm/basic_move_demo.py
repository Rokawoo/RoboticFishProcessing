#!/usr/bin/env python3
"""
FR5 movement demo: from the current pose, nod up/down (shoulder j2) CYCLES
times, then swing left/right (base j1) CYCLES times, returning home in between.

This moves the arm. Run it in simulation (mock hardware) first; on the real
arm, at low speed with someone on the emergency stop (see Arm/README.md).
Ctrl+C does not stop a trajectory the controller has accepted (ROS 2 actions
keep running when the client exits): use the emergency stop.

Needs the FR5 driver and trajectory controller running:
    python3 basic_move_demo.py
Angles are in radians (0.05 rad ~= 3 deg).
"""

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]  # j1 base (left/right), j2 shoulder (up/down)
CONTROLLER = "/fairino5_controller/follow_joint_trajectory"
LIFT = 0.05  # rad added to j2 for "up"; make negative if the arm goes down first
SWING = 0.05  # rad added to / subtracted from j1; make negative if it goes right first
CYCLES = 3
STEP = 2  # seconds per move; larger = slower


def build_goal(home: list[float]) -> FollowJointTrajectory.Goal:
    """Up, home (x CYCLES), then left, home, right, home (x CYCLES)."""
    up = home.copy()
    up[1] += LIFT
    left = home.copy()
    left[0] += SWING
    right = home.copy()
    right[0] -= SWING
    sequence = [up, home] * CYCLES + [left, home, right, home] * CYCLES

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINTS
    # Each point's time_from_start is when it must be reached. The controller
    # starts from the current pose, so home isn't the first point.
    for i, pose in enumerate(sequence):
        goal.trajectory.points.append(
            JointTrajectoryPoint(positions=pose, time_from_start=Duration(sec=(i + 1) * STEP))
        )
    return goal


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("basic_move_demo")
    try:
        ok, msg = wait_for_message(JointState, node, "/joint_states")
        if not ok:
            raise SystemExit("No /joint_states message received")
        position = dict(zip(msg.name, msg.position))  # message order isn't guaranteed
        home = [position[j] for j in JOINTS]

        client = ActionClient(node, FollowJointTrajectory, CONTROLLER)
        client.wait_for_server()
        future = client.send_goal_async(build_goal(home))
        rclpy.spin_until_future_complete(node, future)
        handle = future.result()
        if not handle.accepted:
            raise SystemExit("Goal rejected by the controller")
        future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, future)
        print("error_code:", future.result().result.error_code)  # 0 = success
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
