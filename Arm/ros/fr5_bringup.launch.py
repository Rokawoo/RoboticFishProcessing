"""
The FR5 for the pick-and-place pipeline in one launch: FAIRINO's MoveIt 2 config
(fairino5_v6_moveit2_config) with the real driver or MoveIt's mock hardware, without editing
FAIRINO's files, and FAIRINO's command server with the real arm.

    ros2 launch Arm/ros/fr5_bringup.launch.py                   # the real FR5
    ros2 launch Arm/ros/fr5_bringup.launch.py hardware:=mock    # mock hardware (--mode sim)
    ros2 launch Arm/ros/fr5_bringup.launch.py rviz:=true        # with RViz

`./robot.sh ros [arguments]` runs it with ROS and the workspace sourced, and pick_place.py
starts it by itself when a mode needs MoveIt and none is running (control_panel.yaml:
run.bringup).

What it does that FAIRINO's demo.launch.py doesn't:
- hardware:=real gives ros2_control_node fairino_hardware/FairinoHardwareInterface instead of
  mock_components/GenericSystem (in the robot description it is given; the xacro file stays);
- joint_limits.yaml next to this file: FAIRINO's limits with real accelerations (theirs are
  0.7 rad/s2: a 30 degree point-to-point move took over 5 s at the pipeline's scaling);
- Pilz's sequence capability in move_group (blended moves: MoveGroupSequenceService);
- FAIRINO's command server (I/O, controller state) with the real arm: cmd_server:=false to
  leave it out, e.g. when it runs in another terminal.
Needs ROS 2 Humble with MoveIt 2 and the workspace with frcobot_ros2 built (./setup.sh --robot).
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

HERE = Path(__file__).resolve().parent
ROBOT = "fairino5_v6_robot"
PACKAGE = "fairino5_v6_moveit2_config"
MOCK = "mock_components/GenericSystem"
REAL = "fairino_hardware/FairinoHardwareInterface"
CAPABILITIES = (
    "pilz_industrial_motion_planner/MoveGroupSequenceAction "
    "pilz_industrial_motion_planner/MoveGroupSequenceService"
)


def robot_description(xacro_file: Path, hardware: str) -> str:
    """The robot description of FAIRINO's xacro file, with the hardware plugin asked for."""
    import xacro

    urdf = xacro.process_file(str(xacro_file)).toxml()
    if hardware == "real":
        if MOCK not in urdf:
            raise RuntimeError(f"{xacro_file.name} has no {MOCK} to replace with {REAL}")
        urdf = urdf.replace(MOCK, REAL)
    return urdf


def _actions(context):
    from ament_index_python.packages import get_package_share_directory
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder
    from moveit_configs_utils.launches import (
        generate_move_group_launch,
        generate_moveit_rviz_launch,
        generate_rsp_launch,
        generate_spawn_controllers_launch,
        generate_static_virtual_joint_tfs_launch,
    )

    hardware = LaunchConfiguration("hardware").perform(context)
    if hardware not in ("real", "mock"):
        raise RuntimeError(f"hardware:={hardware}: real or mock")
    rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    server = LaunchConfiguration("cmd_server").perform(context).lower()
    server = hardware == "real" if server == "auto" else server == "true"

    share = Path(get_package_share_directory(PACKAGE))
    config = (
        MoveItConfigsBuilder(ROBOT, package_name=PACKAGE)
        .joint_limits(file_path=str(HERE / "joint_limits.yaml"))
        .to_moveit_configs()
    )
    description = robot_description(share / "config" / f"{ROBOT}.urdf.xacro", hardware)
    config.robot_description = {"robot_description": description}
    config.move_group_capabilities = {"capabilities": CAPABILITIES}

    parts = [
        generate_static_virtual_joint_tfs_launch,
        generate_rsp_launch,
        generate_move_group_launch,
        generate_spawn_controllers_launch,
    ]
    if rviz:
        parts.append(generate_moveit_rviz_launch)
    actions = [entity for make in parts for entity in make(config).entities]
    actions.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[config.robot_description, str(share / "config" / "ros2_controllers.yaml")],
            output="screen",
        )
    )
    if server:
        actions.append(
            Node(package="fairino_hardware", executable="ros2_cmd_server", output="screen")
        )
    print(
        f"[fr5_bringup] hardware: {REAL if hardware == 'real' else MOCK}; command server: "
        f"{'yes' if server else 'no'}; RViz: {'yes' if rviz else 'no'}"
    )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hardware", default_value="real", description="real: the FR5; mock: no arm"
            ),
            DeclareLaunchArgument("rviz", default_value="false", description="true: RViz too"),
            DeclareLaunchArgument(
                "cmd_server",
                default_value="auto",
                description="FAIRINO's command server: auto (with the real arm), true, false",
            ),
            # Declared before move_group's own (empty) default, so this one counts
            DeclareLaunchArgument("capabilities", default_value=CAPABILITIES),
            OpaqueFunction(function=_actions),
        ]
    )
