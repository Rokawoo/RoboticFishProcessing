"""Tests for tools/setup_rfp.py: the step runner and the parts that need no network, conda
or sudo (the installs themselves are tried by running the setup on each kind of computer)."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "setup_rfp.py"


@pytest.fixture(scope="module")
def setup_rfp():
    spec = importlib.util.spec_from_file_location("setup_rfp", SETUP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["setup_rfp"] = module  # @dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


def step(setup_rfp, done: dict, name: str, works: bool = True, by_hand: bool = False):
    def install():
        if not works:
            raise RuntimeError("no network")
        done[name] = True

    return setup_rfp.Step(name, lambda: (done[name], "state"), None if by_hand else install, "do")


def test_steps_are_checked_first_and_only_the_missing_ones_installed(setup_rfp):
    done = {"there": True, "missing": False, "broken": False}
    setup = setup_rfp.Setup(yes=True)
    steps = [step(setup_rfp, done, n, works=n != "broken") for n in done]
    assert not setup.run(steps)  # one failed
    assert setup.results == [("ok", "there"), ("installed", "missing"), ("failed", "broken")]
    assert setup_rfp.Setup(yes=True).run(steps[:2])  # run again: all there now


def test_check_installs_nothing_and_says_what_it_would_do(setup_rfp, capsys):
    done = {"a": False, "b": False}
    steps = [step(setup_rfp, done, "a"), step(setup_rfp, done, "b", by_hand=True)]
    setup = setup_rfp.Setup(check_only=True)
    assert not setup.run(steps) and done == {"a": False, "b": False}
    out = capsys.readouterr().out
    assert "setup would: do" in out and "by hand: do" in out
    assert setup.results == [("missing", "a"), ("by hand", "b")]


def test_without_a_terminal_nothing_is_installed_unless_told(setup_rfp):
    done = {"a": False}
    setup = setup_rfp.Setup()  # pytest's stdin is no terminal
    assert not setup.run([step(setup_rfp, done, "a")])
    assert setup.results == [("skipped", "a")] and not done["a"]


def test_a_check_that_breaks_is_reported_not_raised(setup_rfp):
    def broken():
        raise OSError("no such file")

    assert setup_rfp.safely(broken) == (False, "can't tell (no such file)")


def test_the_environment_is_judged_by_what_pip_would_still_install(setup_rfp, monkeypatch):
    def pip_says(text):
        monkeypatch.setattr(setup_rfp, "output", lambda cmd, cwd=None: text)

    pip_says("Collecting pyserial\nWould install pyserial-3.5 robotic-fish-processing-0.0.1")
    assert setup_rfp.would_install(Path("python")) == ["pyserial-3.5"]
    pip_says("Requirement already satisfied: numpy\nWould install robotic-fish-processing-0.0.1")
    assert setup_rfp.would_install(Path("python")) == []  # the project itself always is


def test_a_change_to_the_requirements_means_another_look(setup_rfp, tmp_path, monkeypatch):
    for name in ("requirements.txt", "environment.yml", "pyproject.toml"):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(setup_rfp, "REPO", tmp_path)
    before = setup_rfp.requirements_hash()
    (tmp_path / "requirements.txt").write_text("numpy\npyserial\n")
    assert setup_rfp.requirements_hash() != before


def test_vs_code_is_pointed_at_the_environment(setup_rfp, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_rfp, "REPO", tmp_path)
    env = type(
        "Env",
        (),
        {
            "label": "test",
            "python": tmp_path / "env" / "python.exe",
            "check": None,
            "install": None,
        },
    )()
    step = next(s for s in setup_rfp.project_steps(env) if s.name == "VS Code's Python")
    settings = tmp_path / ".vscode" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"editor.rulers": [100]}')
    assert not step.check()[0]
    step.install()
    assert step.check()[0]
    assert '"editor.rulers"' in settings.read_text()  # the other settings stay
    settings.write_text('{\n  // a comment\n  "a": 1\n}')  # JSON with comments: by hand
    assert setup_rfp.safely(step.check)[0] is False
    with pytest.raises(RuntimeError, match="by hand"):
        step.install()


def test_the_robot_pc_steps(setup_rfp):
    steps = {s.name: s for s in setup_rfp.robot_steps("3.7.8")}
    assert next(iter(steps)) == "Ubuntu 22.04" and steps["Ubuntu 22.04"].install is None
    for name in (
        "ROS 2 apt source",
        "ROS 2 Humble, MoveIt 2, ros2_control",
        "librealsense (the D405)",
    ):
        assert steps[name].sudo and steps[name].install is not None
    assert not steps["frcobot_ros2 (FAIRINO's driver)"].sudo  # cloned as the user
    # the driver at the commit the simulation's FR5 meshes come from
    assert re.fullmatch("[0-9a-f]{40}", setup_rfp.frcobot_commit())
    # built by the setup once the firmware is known; by hand (asked for) when it isn't
    assert steps["FAIRINO's driver built"].install is not None
    unknown = {s.name: s for s in setup_rfp.robot_steps(None)}
    assert unknown["FAIRINO's driver built"].install is None
    assert "--firmware" in unknown["FAIRINO's driver built"].how


def test_fairino_s_driver_is_the_one_for_the_controller_firmware(setup_rfp, tmp_path):
    tags = ["V3.0.0_RobotV3.7.8", "v3.0.0_RobotV3.9.8", "v3.0.0_RobotV3.9.8_v1.1",
            "V3.0.0_RobotV4.0.1.1"]  # fmt: skip
    ref = setup_rfp.frcobot_ref
    assert ref("3.7.8", tags) == ref("V3.7.8", tags) == "V3.0.0_RobotV3.7.8"
    assert ref("3.9.8", tags) == "v3.0.0_RobotV3.9.8_v1.1"  # its latest fix
    assert ref("3.7.5", tags) == ref("3.7.3", tags) == setup_rfp.FRCOBOT_COMMITS["3.7.5"]
    assert ref("3.9.5", tags) == setup_rfp.FRCOBOT_COMMITS["3.9.5"]  # never tagged
    for unsupported in ("3.7.1", "3.8.1", "seven"):
        with pytest.raises(ValueError):
            ref(unsupported, tags)
    # one folder per firmware (3.9.6 on): only the matching one is built
    for name in ("fairino_hardware", "fairino_hardware_master", "fairino_hardware_v3_9_8",
                 "fairino_hardware_v4_0_1"):  # fmt: skip
        (tmp_path / "new" / name).mkdir(parents=True)
    keep, others = setup_rfp.driver_folders(tmp_path / "new", "4.0.1.1")
    assert keep.name == "fairino_hardware_v4_0_1"  # no four-part folder: the first three
    assert sorted(p.name for p in others) == ["fairino_hardware_master", "fairino_hardware_v3_9_8"]
    assert setup_rfp.driver_folders(tmp_path / "new", "3.9.7")[0] is None
    (tmp_path / "old" / "fairino_hardware").mkdir(parents=True)  # the driver itself
    single = tmp_path / "old" / "fairino_hardware"
    assert setup_rfp.driver_folders(tmp_path / "old", "3.7.8") == (single, [])


def test_the_controller_is_asked_for_its_firmware(setup_rfp):
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    asked = []

    def answer():
        connection, _ = server.accept()
        with connection:
            asked.append(connection.recv(1024))
            connection.sendall(b"/f/bIII4III905III27IIIFR5,v3.7.8-release,V3.2III/b/f")

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    port = server.getsockname()[1]
    assert setup_rfp.controller_firmware("127.0.0.1", 2.0, port) == "3.7.8"
    thread.join(2.0)
    server.close()
    assert b"GetSoftwareVersion()" in asked[0]
    assert setup_rfp.controller_firmware("127.0.0.1", 0.5, port) is None  # nobody there
