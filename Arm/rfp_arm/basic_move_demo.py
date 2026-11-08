#!/usr/bin/env python3
"""
Basic movement demo for the Fairino FR5 arm.

What it does:
    Starting from wherever the arm is right now, it:
      1. Nods up and down   (CYCLES times), using the shoulder joint (j2)
      2. Swings left and right (CYCLES times), using the base joint (j1)
    and always comes back to where it started.

How it works (the big picture):
    - ROS 2 programs talk to each other over "topics" (streams of messages)
      and "actions" (long-running requests with a result at the end).
    - We READ the arm's current joint angles from the /joint_states topic.
    - We build a TRAJECTORY: a list of target poses, each with a time stamp.
    - We SEND that trajectory to the arm's controller through an action,
      then wait for it to report back that the motion is finished.

Units: all joint angles are in radians (1 rad ~= 57.3 deg).
"""

import rclpy                                         # core ROS 2 Python library
from rclpy.action import ActionClient                # lets us send action goals
from rclpy.wait_for_message import wait_for_message  # grab a single message from a topic
from builtin_interfaces.msg import Duration          # ROS time span (sec + nanosec)
from sensor_msgs.msg import JointState               # message type for joint angles
from trajectory_msgs.msg import JointTrajectoryPoint # one pose + when to reach it
from control_msgs.action import FollowJointTrajectory  # the "move along this path" action


# ---------------------------------------------------------------------------
# Settings: tweak these to change the motion
# ---------------------------------------------------------------------------

# The FR5 has 6 joints, numbered from the base (j1) up to the wrist (j6).
#   j1 = base rotation  -> turns the whole arm left/right
#   j2 = shoulder       -> tilts the arm up/down
#   j3 = elbow, j4-j6 = wrist joints (not used in this demo)
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

LIFT = 0.05    # How far to tilt the shoulder (j2) for "up". 0.05 rad ~= 3 deg.
               # If the arm goes DOWN first, make this negative.
SWING = 0.05   # How far to turn the base (j1) for "left"/"right". 0.05 rad ~= 3 deg.
               # If the arm goes RIGHT first, make this negative.
CYCLES = 3     # How many times to repeat each motion (up/down, then left/right).
STEP = 2       # Seconds to spend on each single move. Bigger = slower and safer.


# ---------------------------------------------------------------------------
# Setup: start ROS 2 and create a "node" (our program's identity in ROS)
# ---------------------------------------------------------------------------
rclpy.init()
node = rclpy.create_node("basic_move_demo")


# ---------------------------------------------------------------------------
# Step 1: Find out where the arm is right now
# ---------------------------------------------------------------------------
# /joint_states publishes the name and angle of every joint, many times a second.
# We wait for just one message.
_, msg = wait_for_message(JointState, node, "/joint_states")

# The joints in the message are NOT guaranteed to be in j1..j6 order,
# so we pair each name with its angle in a dictionary, e.g. {"j1": 0.12, ...},
# and then read them back out in the order we want.
pos = dict(zip(msg.name, msg.position))
home = [pos[j] for j in JOINTS]   # the starting pose; we'll return here between moves

# Each pose below is a list of 6 angles (one per joint), in JOINTS order.
# We copy "home" and change just one joint, so everything else stays still.
# Remember Python lists start at index 0:  index 0 = j1,  index 1 = j2.

up = home.copy()
up[1] += LIFT        # shoulder (j2) tilted up

left = home.copy()
left[0] += SWING     # base (j1) turned left

right = home.copy()
right[0] -= SWING    # base (j1) turned right


# ---------------------------------------------------------------------------
# Step 2: Build the trajectory (the list of poses to visit, in order)
# ---------------------------------------------------------------------------
# Sequence, with CYCLES = 3:
#   up, home, up, home, up, home,                         <- nodding
#   left, home, right, home, left, home, right, home, ... <- swinging
#
# Multiplying a list repeats it:  [a, b] * 3  ->  [a, b, a, b, a, b]
sequence = [up, home] * CYCLES + [left, home, right, home] * CYCLES

goal = FollowJointTrajectory.Goal()
goal.trajectory.joint_names = JOINTS   # tells the controller which angle is which joint

# Every point needs a "time_from_start": WHEN the arm should arrive there,
# counted from the moment the motion begins. Point 0 at 2 s, point 1 at 4 s, etc.
# The controller smoothly moves between points on its own.
# (We don't add "home" as the first point: the controller already starts
#  from the arm's current pose.)
for i, pose in enumerate(sequence):
    arrive_at = Duration(sec=(i + 1) * STEP)
    goal.trajectory.points.append(
        JointTrajectoryPoint(positions=pose, time_from_start=arrive_at))


# ---------------------------------------------------------------------------
# Step 3: Send the trajectory to the arm and wait for it to finish
# ---------------------------------------------------------------------------
# An action has two stages:
#   a) the controller ACCEPTS or REJECTS the goal (e.g. rejects bad joint names)
#   b) once accepted, the arm moves, and we later get the final RESULT
client = ActionClient(node, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
client.wait_for_server()   # block until the arm's controller is up and listening

# (a) Send the goal. "async" returns a "future": a placeholder for an answer
#     that hasn't arrived yet. spin_until_future_complete waits for it.
future = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, future)
handle = future.result()
if not handle.accepted:
    raise SystemExit("Goal rejected by the controller")

# (b) Wait for the motion to finish and print the outcome.
future = handle.get_result_async()
rclpy.spin_until_future_complete(node, future)
print("error_code:", future.result().result.error_code)  # 0 means success


# ---------------------------------------------------------------------------
# Cleanup: shut ROS 2 down cleanly
# ---------------------------------------------------------------------------
rclpy.shutdown()
