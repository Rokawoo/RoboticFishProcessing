"""Tests for starting MoveIt and FAIRINO's driver for a run (rfp_arm/bringup.py, the launch
file, robot.sh) without ROS: the commands, the child process, and when a mode starts it.
Whether the launch itself comes up needs the robot PC: ./robot.sh ros hardware:=mock."""

import ast
import sys
import types
from pathlib import Path

import pytest
from rfp_arm.bringup import LAUNCH_FILE, Bringup, launch_command

REPO = Path(__file__).resolve().parents[2]


def test_the_launch_command_says_which_hardware():
    command = launch_command()
    assert command[:3] == ["ros2", "launch", str(LAUNCH_FILE)]
    assert "hardware:=real" in command and "rviz:=false" in command
    assert not any(part.startswith("cmd_server") for part in command)  # the launch decides
    mock = launch_command("mock", rviz=True, cmd_server=False)
    assert {"hardware:=mock", "rviz:=true", "cmd_server:=false"} <= set(mock)
    with pytest.raises(ValueError, match="real' or 'mock"):
        launch_command("sim")


def test_the_launch_file_swaps_in_the_real_driver_without_touching_fairino_s_files():
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)  # valid Python; it imports ROS only when launched
    names = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)}
    assert ast.literal_eval(names["MOCK"]) == "mock_components/GenericSystem"
    assert ast.literal_eval(names["REAL"]) == "fairino_hardware/FairinoHardwareInterface"
    assert "MoveGroupSequenceService" in ast.literal_eval(names["CAPABILITIES"])
    assert (LAUNCH_FILE.parent / "joint_limits.yaml").is_file()
    assert "ros2_cmd_server" in source and "generate_launch_description" in source


def _launch_module(monkeypatch):
    """The launch file as a module, with stand-ins for ROS's launch package."""
    import importlib.util

    launch = types.ModuleType("launch")
    launch.LaunchDescription = list
    actions = types.ModuleType("launch.actions")
    for name in ("DeclareLaunchArgument", "OpaqueFunction", "Shutdown"):
        setattr(actions, name, lambda *args, **kwargs: kwargs)
    actions.RegisterEventHandler = lambda handler: handler
    handlers = types.ModuleType("launch.event_handlers")
    handlers.OnProcessExit = lambda target_action, on_exit: types.SimpleNamespace(
        target_action=target_action, on_exit=on_exit
    )
    substitutions = types.ModuleType("launch.substitutions")
    substitutions.LaunchConfiguration = str
    stand_ins = {
        "launch": launch,
        "launch.actions": actions,
        "launch.event_handlers": handlers,
        "launch.substitutions": substitutions,
    }
    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("fr5_bringup_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launch_swaps_either_plugin_and_starts_the_server_once_the_driver_is_up(monkeypatch):
    launch = _launch_module(monkeypatch)
    mock, real = f"<plugin>{launch.MOCK}</plugin>", f"<plugin>{launch.REAL}</plugin>"
    assert launch.swap_plugin(mock, "real") == real == launch.swap_plugin(real, "real")
    assert launch.swap_plugin(real, "mock") == mock == launch.swap_plugin(mock, "mock")
    with pytest.raises(RuntimeError, match="neither"):
        launch.swap_plugin("<robot/>", "real")
    server = object()  # a 3.7 controller takes one SDK connection: the driver's first
    waiting = launch.after_all(["spawner a", "spawner b"], server)
    assert [handler.target_action for handler in waiting] == ["spawner a", "spawner b"]
    assert waiting[0].on_exit(None, None) is None  # one spawner still running
    assert waiting[1].on_exit(None, None) == [server]  # the last one done: the server starts
    assert launch.after_all([], server) == [server]


def test_a_launch_that_dies_on_the_way_up_is_said_at_once(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "ControlSys"))
    from rfp_controlsys import pick_place

    link = types.SimpleNamespace(node=types.SimpleNamespace(count_publishers=lambda topic: 0))
    log = tmp_path / "launch.log"
    died = "print('[ERROR] [ros2_control_node-5]: process has died [pid 42, exit code 1]')"
    bringup = Bringup([sys.executable, "-c", died], log)
    bringup.start()
    bringup.process.wait(timeout=30)
    with pytest.raises(SystemExit) as stopped:
        pick_place.wait_for_launch(link, bringup, timeout=5.0)
    bringup.stop()
    said = str(stopped.value)
    assert "stopped" in said and "[ERROR] [ros2_control_node-5]" in said
    assert pick_place.DRIVER_STOPPED in said and str(log) in said
    log.write_text("[ERROR] [launch]: package 'fairino5_v6_moveit2_config' not found\n")
    assert pick_place.MISSING_PACKAGE in _exit_text(pick_place, link, bringup)


def _exit_text(pick_place, link, bringup) -> str:
    with pytest.raises(SystemExit) as stopped:
        pick_place.wait_for_launch(link, bringup, timeout=1.0)
    return str(stopped.value)


def test_the_joint_limits_keep_fairino_s_speeds_and_allow_real_accelerations():
    import yaml

    data = yaml.safe_load((LAUNCH_FILE.parent / "joint_limits.yaml").read_text(encoding="utf-8"))
    joints = data["joint_limits"]
    assert sorted(joints) == [f"j{i}" for i in range(1, 7)]
    for limits in joints.values():  # Pilz needs acceleration limits for every joint
        assert limits["has_acceleration_limits"] and 0.7 < limits["max_acceleration"] <= 5.0
        assert limits["max_velocity"] <= 3.2  # FAIRINO's


def test_robot_sh_sources_ros_the_workspace_and_the_environment():
    script = (REPO / "robot.sh").read_text(encoding="utf-8")
    assert "\r\n" not in script  # bash refuses CRLF (.gitattributes keeps LF)
    assert "source /opt/ros/humble/setup.bash" in script
    assert 'source "$workspace/install/setup.bash"' in script and ".venv/bin/activate" in script
    assert "Arm/ros/fr5_bringup.launch.py" in script and 'main.py" "$@"' in script


def test_a_bringup_runs_as_a_child_process_until_stopped(tmp_path):
    log = tmp_path / "launch.log"
    command = [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"]
    bringup = Bringup(command, log)
    bringup.start()
    try:
        assert bringup.running()
    finally:
        bringup.stop(timeout=5.0)
    assert not bringup.running()
    bringup.stop()  # twice: nothing left to stop
    assert log.is_file()


def test_without_ros_on_the_path_it_says_to_use_robot_sh():
    with pytest.raises(RuntimeError, match="robot.sh"):
        Bringup(["no-such-ros2-command"]).start()


def test_a_mode_starts_moveit_only_if_nothing_runs_and_the_panel_allows_it(monkeypatch, capsys):
    from rfp_controlsys import pick_place

    started = []
    monkeypatch.setattr(Bringup, "start", lambda self: started.append(self.command))

    def link(publishers: int):
        node = types.SimpleNamespace(count_publishers=lambda topic: publishers)
        return types.SimpleNamespace(node=node)

    assert pick_place.bring_up(link(1), "run", True) is None and started == []  # running
    assert pick_place.bring_up(link(0), "run", False, wait=0.2) is None and started == []
    assert "./robot.sh ros" in capsys.readouterr().out
    real = pick_place.bring_up(link(0), "dry-run", True, wait=0.2)
    assert real is not None and "hardware:=real" in started[-1]
    assert "hardware:=mock" in pick_place.bring_up(link(0), "sim", True, wait=0.2).command

    # FAIRINO's command server started by hand, MoveIt not: no second server beside it
    # (both would compete for the controller), even one that sends no state
    def server_only(topic):
        return 0

    node = types.SimpleNamespace(
        count_publishers=server_only,
        get_service_names_and_types=lambda: [("/fairino_remote_command_service", [])],
    )
    pick_place.bring_up(types.SimpleNamespace(node=node), "run", True, wait=0.2)
    assert "cmd_server:=false" in started[-1]
    # found a little later, as a new node discovers what runs: not started twice
    seen = iter([0, 0, 1])
    late = types.SimpleNamespace(node=types.SimpleNamespace(count_publishers=lambda t: next(seen)))
    assert pick_place.bring_up(late, "run", True) is None and len(started) == 3


def _bash() -> str | None:
    """A bash that runs the repository's scripts with this machine's paths (on Windows Git
    Bash, not WSL's), or None."""
    import os
    import shutil
    import subprocess

    git = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "bin" / "bash.exe"
    for bash in (shutil.which("bash"), str(git) if git.is_file() else None):
        if bash is None:
            continue
        try:
            kind = subprocess.run(
                [bash, "-c", "uname -s"], check=False, capture_output=True, text=True, timeout=30
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if os.name != "nt" or kind.startswith(("MINGW", "MSYS", "CYGWIN")):
            return bash
    return None


def test_robot_sh_sources_what_bashrc_did_and_finds_fairino_s_workspace(tmp_path):
    # On the arm's PC two setup files had to be sourced by hand before anything ran: the
    # script sources the ones ~/.bashrc names and finds the workspace with FAIRINO's driver
    import os
    import subprocess

    bash = _bash()
    if bash is None:
        pytest.skip("no bash that runs the script here")
    home = tmp_path / "home"
    for folder in ("aaa/install", "ws_moveit/install", "fr_ws/install/fairino_hardware"):
        (home / folder).mkdir(parents=True)
    for workspace in ("aaa", "ws_moveit", "fr_ws"):
        (home / workspace / "install" / "setup.bash").write_text("", encoding="utf-8")
    bashrc = [
        "case $- in *i*) ;; *) return;; esac",
        "source /opt/ros/humble/setup.bash",
        "  source ~/ws_moveit/install/setup.bash",
        '. "$HOME/fr_ws/install/setup.bash"',
        "# source ~/old/install/setup.bash",
        "alias ll='ls -l'",
    ]
    (home / ".bashrc").write_bytes(("\n".join(bashrc) + "\n").encode())
    script = Path(__file__).resolve().parents[2] / "robot.sh"
    assert b"\r" not in script.read_bytes()  # bash refuses CRLF (.gitattributes keeps LF)
    found = subprocess.run(
        [
            bash,
            "-c",
            'source "$0"; echo "$HOME"; echo ---; bashrc_setup_files; echo ---; driver_workspace',
            str(script),
        ],
        check=False,
        env={"HOME": home.as_posix(), "PATH": os.environ.get("PATH", "")},  # nothing else leaks in
        capture_output=True,
        text=True,
        timeout=60,
    )
    seen_home, files, workspace = found.stdout.split("---")
    home_text = seen_home.strip()  # as bash writes it (Git Bash: /tmp/... for ...Temp/...)
    assert files.split() == [
        "/opt/ros/humble/setup.bash",
        f"{home_text}/ws_moveit/install/setup.bash",
        f"{home_text}/fr_ws/install/setup.bash",
    ]
    assert workspace.strip() == f"{home_text}/fr_ws"  # the one with the driver built
