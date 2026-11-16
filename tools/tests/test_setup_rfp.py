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


def test_the_robot_pc_steps(setup_rfp):
    steps = {s.name: s for s in setup_rfp.robot_steps()}
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
