#!/usr/bin/env python3
"""
Sets up RoboticFishProcessing on this computer: every step is checked first and only done
if it is missing, so it is safe to run again (after a pull, say). Windows and Linux. Uses
only Python's standard library: it runs before any of the project's packages exist.

Usage (from the repository root):
    .\\setup.ps1                   Windows (PowerShell): the development PC
    ./setup.sh                    Linux: a development PC
    ./setup.sh --robot            the robot PC (Ubuntu 22.04): also ROS 2 Humble, MoveIt 2,
                                  ros2_control, librealsense, serial access, frcobot_ros2
    ... --check                   only report what is there and what is missing
    ... --yes                     install without asking (sudo still asks for its password)
    ... --test                    run the test suite at the end

It can't, and says what to do instead: install the NVIDIA driver, choose FAIRINO's hardware
package for your controller's software version, flash the valve board, measure the cell.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_NAME = "rfp"
STAMP = "rfp_setup.sha256"  # in the environment: what requirements it was set up from
WINDOWS = os.name == "nt"
ROS_PACKAGES = [
    "ros-humble-desktop",
    "ros-dev-tools",
    "ros-humble-moveit",
    "ros-humble-ros2-control",
    "ros-humble-ros2-controllers",
]
REALSENSE_PACKAGES = ["librealsense2-utils", "librealsense2-dev", "librealsense2-udev-rules"]
REALSENSE_DKMS_KERNELS = ("5.15", "5.19", "6.5")  # what librealsense2-dkms supports
REALSENSE_KEY = "https://librealsense.realsenseai.com/Debian/librealsenseai.asc"
REALSENSE_REPO = "https://librealsense.realsenseai.com/Debian/apt-repo"
ROS_APT_SOURCE = "https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest"
ROS_APT_DEB = "https://github.com/ros-infrastructure/ros-apt-source/releases/download"
FRCOBOT_URL = "https://github.com/FAIR-INNOVATION/frcobot_ros2"
WORKSPACE = Path.home() / "rfp_ws"
IMPORTS = "import rfp_arm, rfp_vision, rfp_controlsys, rfp_sim, cv2, torch, ultralytics, pyvista"
Check = Callable[[], tuple[bool, str]]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    check: Check  # (done, what is there or missing)
    install: Callable[[], None] | None  # None: can't be done here
    how: str  # what installing does, or what to do by hand
    sudo: bool = False


class Setup:
    def __init__(self, check_only: bool = False, yes: bool = False):
        self.check_only = check_only
        self.yes = yes
        self.results: list[tuple[str, str]] = []  # (status, step name)

    def run(self, steps: list[Step]) -> bool:
        """Every step in order; True if nothing is left missing."""
        for step in steps:
            self.results.append((self.do(step), step.name))
        print("\n" + "=" * 60)
        for status, name in self.results:
            print(f"  [{status}] {name}")
        print("=" * 60)
        return all(status in ("ok", "installed") for status, _ in self.results)

    def do(self, step: Step) -> str:
        done, detail = safely(step.check)
        if done:
            print(f"[ok] {step.name}: {detail}")
            return "ok"
        print(f"[missing] {step.name}: {detail}")
        if step.install is None:
            print(f"    by hand: {step.how}")
            return "by hand"
        if self.check_only:
            print(f"    setup would: {step.how}")
            return "missing"
        if not self.ask(f"    {step.how}{' (sudo)' if step.sudo else ''}. Go ahead? [Y/n] "):
            return "skipped"
        try:
            step.install()
        except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            print(f"    failed: {e}")
            return "failed"
        done, detail = safely(step.check)
        print(f"[{'installed' if done else 'failed'}] {step.name}: {detail}")
        return "installed" if done else "failed"

    def ask(self, question: str) -> bool:
        if self.yes:
            return True
        if not sys.stdin.isatty():
            print(f"{question}no (nobody at the terminal: run with --yes)")
            return False
        return input(question).strip().lower() in ("", "y", "yes")


def safely(check: Check) -> tuple[bool, str]:
    try:
        return check()
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        return False, f"can't tell ({e})"


def sh(cmd: list, sudo: bool = False, data: bytes | None = None, cwd: Path | None = None) -> None:
    """Run a command, shown first; raises CalledProcessError if it fails."""
    if sudo and not WINDOWS and os.geteuid() != 0:
        cmd = ["sudo", *cmd]
    print("    $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], input=data, cwd=cwd, check=True)


def output(cmd: list, cwd: Path | None = None) -> str:
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, cwd=cwd, check=False
    )
    if result.returncode != 0:
        raise subprocess.SubprocessError((result.stderr or result.stdout).strip()[-300:])
    return result.stdout.strip()


def requirements_hash() -> str:
    """Of what the Python environment is built from: a change means it needs an update."""
    digest = hashlib.sha256()
    for name in ("requirements.txt", "environment.yml", "pyproject.toml"):
        digest.update((REPO / name).read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The Python environment
# ---------------------------------------------------------------------------
class CondaEnv:
    """The `rfp` conda environment (development PCs)."""

    label = "conda"

    def __init__(self):
        self.conda = find_conda()

    @property
    def prefix(self) -> Path | None:
        if self.conda is None:
            return None
        envs = json.loads(output([self.conda, "env", "list", "--json"]))["envs"]
        return next((Path(e) for e in envs if Path(e).name == ENV_NAME), None)

    @property
    def python(self) -> Path | None:
        prefix = self.prefix
        if prefix is None:
            return None
        python = prefix / ("python.exe" if WINDOWS else "bin/python")
        return python if python.is_file() else None

    def check(self) -> tuple[bool, str]:
        if self.conda is None:
            return False, "no conda: install Miniconda (https://docs.conda.io), then run again"
        if self.python is None:
            return False, f"no '{ENV_NAME}' environment"
        return env_current(self.python, self.prefix)

    def install(self) -> None:
        """A missing package: pip adds it. No environment, or not Python 3.10: conda."""
        if self.conda is None:
            raise RuntimeError("install Miniconda or Anaconda first, then run setup again")
        python = self.python
        version = output([python, "-c", "import sys; print(sys.version[:5])"]) if python else ""
        if python is not None and version.startswith("3.10"):
            sh([python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO)
        else:
            verb = ["env", "update"] if self.prefix else ["env", "create"]
            sh([self.conda, *verb, "-n", ENV_NAME, "-f", REPO / "environment.yml"], cwd=REPO)
        (self.prefix / STAMP).write_text(requirements_hash())


class VenvEnv:
    """A venv on the system Python (the robot PC): ROS 2's rclpy is built for that Python,
    and the venv sees it (--system-site-packages)."""

    label = "venv"
    prefix = REPO / ".venv"

    @property
    def python(self) -> Path | None:
        python = self.prefix / "bin" / "python"
        return python if python.is_file() else None

    def check(self) -> tuple[bool, str]:
        if self.python is None:
            return False, f"no {self.prefix}"
        return env_current(self.python, self.prefix)

    def install(self) -> None:
        if self.python is None:
            sh(["/usr/bin/python3", "-m", "venv", "--system-site-packages", self.prefix])
        sh([self.python, "-m", "pip", "install", "--upgrade", "pip"])
        sh([self.python, "-m", "pip", "install", "-r", REPO / "requirements.txt"], cwd=REPO)
        (self.prefix / STAMP).write_text(requirements_hash())


def env_current(python: Path, prefix: Path) -> tuple[bool, str]:
    """Python 3.10 with every requirement installed: asks pip (a dry run) unless the
    environment was already found complete for these exact requirement files."""
    version = output([python, "-c", "import sys; print(sys.version.split()[0])"])
    if not version.startswith("3.10."):
        return False, f"{prefix}: Python {version}, not 3.10 (ROS 2 Humble's)"
    stamp = prefix / STAMP
    if not stamp.is_file() or stamp.read_text().strip() != requirements_hash():
        missing = would_install(python)
        if missing:
            return False, f"{prefix}: needs {', '.join(missing)}"
        stamp.write_text(requirements_hash())  # complete: no need to ask pip next time
    result = subprocess.run(
        [str(python), "-c", IMPORTS], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return False, f"{prefix}: {result.stderr.strip().splitlines()[-1]}"
    return True, f"{prefix}, Python {version}"


def would_install(python: Path) -> list[str]:
    """What `pip install -r requirements.txt` would still install (not counting the project
    itself, which an editable install always lists)."""
    dry = output([python, "-m", "pip", "install", "--dry-run", "-r", "requirements.txt"], REPO)
    lines = [line for line in dry.splitlines() if line.startswith("Would install ")]
    names = lines[-1].split()[2:] if lines else []
    return [name for name in names if not name.startswith("robotic-fish-processing")]


def find_conda() -> Path | None:
    """conda from the environment, PATH, this Python (conda's base) or the usual places."""
    base = Path(sys.base_prefix)
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    candidates += [base / "Scripts" / "conda.exe", base / "bin" / "conda"]
    home = Path.home()
    for root in ("miniconda3", "anaconda3", "Miniconda3", "Anaconda3", "miniforge3"):
        for parent in (home, home / "AppData" / "Local", Path("C:/ProgramData"), Path("/opt")):
            candidates += [parent / root / "Scripts" / "conda.exe", parent / root / "bin" / "conda"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


# ---------------------------------------------------------------------------
# The steps of each computer
# ---------------------------------------------------------------------------
def project_steps(env) -> list[Step]:
    """The project itself, on any computer, once there is a Python environment."""

    def python() -> Path:
        if env.python is None:
            raise ValueError("the Python environment isn't set up")
        return env.python

    def hooks() -> tuple[bool, str]:
        hook = REPO / ".git" / "hooks" / "pre-commit"
        done = hook.is_file() and "pre-commit" in hook.read_text(errors="replace")
        return done, "ruff checks each commit" if done else "not installed"

    def models() -> tuple[bool, str]:
        weights = REPO.glob("Vision/trained_cv_models/*/weights/best.pt")
        found = sorted(p.parent.parent.name for p in weights)
        return bool(found), ", ".join(found) if found else "no model (git pull, or train one)"

    meshes = REPO / "Simulation" / "models" / "fr5"

    def fr5_meshes() -> tuple[bool, str]:
        count = len(list(meshes.glob("*.STL")))
        return count >= 7, f"{count} of 7 link meshes in {meshes.relative_to(REPO)}"

    def fetch_meshes() -> None:
        code = (
            "from rfp_sim.assets import download_fr5_meshes\n"
            "from rfp_sim.settings import MESH_CACHE\n"
            "problem = download_fr5_meshes(MESH_CACHE)\n"
            "print(problem or 'downloaded and checked')\n"
            "raise SystemExit(1 if problem else 0)\n"
        )
        sh([python(), "-c", code])

    def gpu() -> tuple[bool, str]:
        code = "import torch; print(torch.cuda.is_available(), torch.version.cuda or '-')"
        available, cuda = output([python(), "-c", code]).split()[-2:]
        if available == "True":
            return True, f"CUDA {cuda}"
        return False, "no CUDA GPU in PyTorch: the model runs on the CPU, slowly"

    def project() -> tuple[bool, str]:
        code = (
            "from rfp_arm.workcell import load_workcell\n"
            "from rfp_vision.settings import load_settings\n"
            "cell = load_workcell()\n"
            "load_settings()\n"
            "print('verified' if cell.verified else 'placeholder values (verified: false)')\n"
        )
        return True, f"settings load; workcell: {output([python(), '-c', code], cwd=REPO)}"

    def hook() -> None:
        sh([python(), "-m", "pre_commit", "install"], cwd=REPO)

    return [
        Step(
            f"Python environment ({env.label})",
            env.check,
            env.install,
            "create or update it from environment.yml / requirements.txt",
        ),
        Step("Git hooks", hooks, hook, "pre-commit install (ruff on every commit)"),
        Step("Fish model", models, None, "git pull, or train one (Vision/README.md)"),
        Step(
            "FR5 meshes for the simulation",
            fr5_meshes,
            fetch_meshes,
            "download FAIRINO's meshes at the pinned commit, checked by SHA-256",
        ),
        Step("GPU", gpu, None, "install the NVIDIA driver (580+ for CUDA 13.0), then run again"),
        Step("Project settings", project, None, "see the error above"),
    ]


def robot_steps() -> list[Step]:
    """The robot PC (Ubuntu 22.04), before its Python environment."""

    def jammy() -> tuple[bool, str]:
        info = os_release()
        done = info.get("VERSION_CODENAME") == "jammy"
        return done, f"{info.get('PRETTY_NAME', 'unknown')} (ROS 2 Humble needs Ubuntu 22.04)"

    def packages_check(names: list[str]) -> Check:
        def check() -> tuple[bool, str]:
            missing = missing_packages(names)
            return not missing, "installed" if not missing else "missing " + " ".join(missing)

        return check

    def apt_install(names: list[str]) -> Callable[[], None]:
        return lambda: sh(["apt-get", "install", "-y", *missing_packages(names)], sudo=True)

    def ros_source() -> tuple[bool, str]:
        lists = Path("/etc/apt/sources.list.d")
        found = [p.name for p in lists.glob("ros2*")] if lists.is_dir() else []
        return bool(found), ", ".join(found) or "no ROS 2 apt source"

    def add_ros_source() -> None:
        sh(["apt-get", "install", "-y", "software-properties-common"], sudo=True)
        sh(["add-apt-repository", "-y", "universe"], sudo=True)
        with urllib.request.urlopen(ROS_APT_SOURCE, timeout=30) as response:
            tag = json.load(response)["tag_name"]
        url = f"{ROS_APT_DEB}/{tag}/ros2-apt-source_{tag}.jammy_all.deb"
        deb = Path(tempfile.gettempdir()) / "ros2-apt-source.deb"
        print(f"    downloading {url}")
        urllib.request.urlretrieve(url, deb)
        sh(["dpkg", "-i", deb], sudo=True)
        sh(["apt-get", "update"], sudo=True)

    def rosdep() -> tuple[bool, str]:
        done = Path("/etc/ros/rosdep/sources.list.d/20-default.list").is_file()
        return done, "initialized" if done else "not initialized"

    def init_rosdep() -> None:
        sh(["rosdep", "init"], sudo=True)
        sh(["rosdep", "update"])

    def install_realsense() -> None:
        key = Path("/etc/apt/keyrings/librealsenseai.gpg")
        if not key.is_file():
            sh(["mkdir", "-p", key.parent], sudo=True)
            with urllib.request.urlopen(REALSENSE_KEY, timeout=30) as response:
                armored = response.read()
            dearmor = subprocess.run(
                ["gpg", "--dearmor"], input=armored, capture_output=True, check=True
            )
            sh(["tee", key], sudo=True, data=dearmor.stdout)
        codename = os_release().get("VERSION_CODENAME", "jammy")
        line = f"deb [signed-by={key}] {REALSENSE_REPO} {codename} main\n"
        sh(["tee", "/etc/apt/sources.list.d/librealsense.list"], sudo=True, data=line.encode())
        sh(["apt-get", "update"], sudo=True)
        packages = list(REALSENSE_PACKAGES)
        kernel = ".".join(platform.release().split(".")[:2])
        if kernel in REALSENSE_DKMS_KERNELS:  # its kernel modules: only for these kernels
            packages.append("librealsense2-dkms")
        else:
            supported = ", ".join(REALSENSE_DKMS_KERNELS)
            print(f"    kernel {kernel}: no librealsense2-dkms (it supports {supported})")
        sh(["apt-get", "install", "-y", *packages], sudo=True)

    def dialout() -> tuple[bool, str]:
        import grp  # Linux only

        user = getpass.getuser()
        group = grp.getgrnam("dialout")
        done = user in group.gr_mem or group.gr_gid in os.getgroups()
        return done, f"{user} {'is' if done else 'is not'} in dialout (the valve board's port)"

    def add_dialout() -> None:
        sh(["usermod", "-aG", "dialout", getpass.getuser()], sudo=True)

    def frcobot() -> tuple[bool, str]:
        source = WORKSPACE / "src" / "frcobot_ros2"
        if not (source / ".git").is_dir():
            return False, f"not in {source}"
        built = (WORKSPACE / "install" / "fairino5_v6_moveit2_config").is_dir()
        head = output(["git", "-C", source, "rev-parse", "HEAD"])
        return True, f"{source} at {head[:8]}, {'built' if built else 'not built yet'}"

    def clone_frcobot() -> None:
        source = WORKSPACE / "src" / "frcobot_ros2"
        source.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", FRCOBOT_URL, source])
        sh(["git", "-C", source, "checkout", frcobot_commit()])

    venv = ["python3-venv", "python3-pip"]
    return [
        Step("Ubuntu 22.04", jammy, None, "ROS 2 Humble's packages are for Ubuntu 22.04"),
        Step(
            "Python venv support",
            packages_check(venv),
            apt_install(venv),
            "apt install " + " ".join(venv),
            sudo=True,
        ),
        Step(
            "ROS 2 apt source",
            ros_source,
            add_ros_source,
            "add packages.ros.org (the ros2-apt-source package)",
            sudo=True,
        ),
        Step(
            "ROS 2 Humble, MoveIt 2, ros2_control",
            packages_check(ROS_PACKAGES),
            apt_install(ROS_PACKAGES),
            "apt install " + " ".join(ROS_PACKAGES),
            sudo=True,
        ),
        Step("rosdep", rosdep, init_rosdep, "rosdep init, rosdep update", sudo=True),
        Step(
            "librealsense (the D405)",
            packages_check(REALSENSE_PACKAGES),
            install_realsense,
            "add librealsense's apt source and install it",
            sudo=True,
        ),
        Step(
            "Serial port access",
            dialout,
            add_dialout,
            "add you to dialout (log out and in again after)",
            sudo=True,
        ),
        Step(
            "frcobot_ros2 (FAIRINO's driver)",
            frcobot,
            clone_frcobot,
            f"clone it into {WORKSPACE / 'src'} at the pinned commit",
        ),
    ]


def missing_packages(names: list[str]) -> list[str]:
    installed = (
        subprocess.run(["dpkg", "-s", name], capture_output=True, check=False).returncode == 0
        for name in names
    )
    return [name for name, ok in zip(names, installed) if not ok]


def frcobot_commit() -> str:
    """The frcobot_ros2 commit the project is pinned to (the simulation's meshes too)."""
    text = (REPO / "Simulation" / "rfp_sim" / "assets.py").read_text(encoding="utf-8")
    found = re.search(r'FR5_COMMIT = "([0-9a-f]{40})"', text)
    if not found:
        raise ValueError("FR5_COMMIT not found in Simulation/rfp_sim/assets.py")
    return found.group(1)


def os_release() -> dict:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key: value.strip('"') for key, value in pairs}


ROBOT_NEXT = f"""
Still by hand on the robot PC:
  1. Build FAIRINO's driver. frcobot_ros2 has a hardware package per controller software
     version (fairino_hardware_v3_*, _v4_*): keep the one matching your controller (its
     software version is on the teach pendant), as FAIRINO's ROS 2 guide says, then:
         source /opt/ros/humble/setup.bash
         cd {WORKSPACE} && rosdep install --from-paths src -y --ignore-src && colcon build
  2. For the real arm, fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro
     must load fairino_hardware/FairinoHardwareInterface (mock_components/GenericSystem for
     --mode sim).
  3. Flash the valve board (Arm/firmware/valve_control, Arduino IDE with the ESP32 boards),
     with secrets.h copied from secrets.example.h.
  4. Measure the workcell (README.md, "Set up the workcell").
Then: source {REPO / ".venv" / "bin" / "activate"}"""


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # in order with the commands' own output
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--robot", action="store_true", help="also set up the robot PC (Linux)")
    parser.add_argument("--check", action="store_true", help="only report, install nothing")
    parser.add_argument("-y", "--yes", action="store_true", help="install without asking")
    parser.add_argument("--test", action="store_true", help="run the tests at the end")
    args = parser.parse_args(argv)
    if args.robot and WINDOWS:
        parser.error("--robot is for the robot PC (Ubuntu 22.04)")

    print("=" * 60)
    print(f"SETUP: {platform.system()} {platform.release()}, repository {REPO}")
    print("=" * 60)
    env = VenvEnv() if args.robot else CondaEnv()
    steps = (robot_steps() if args.robot else []) + project_steps(env)
    complete = Setup(check_only=args.check, yes=args.yes).run(steps)
    if args.test and env.python is not None and not args.check:
        print("\nTests:")
        tests = subprocess.run([str(env.python), "-m", "pytest", "-q"], cwd=REPO, check=False)
        complete = complete and tests.returncode == 0
    if args.robot:
        print(ROBOT_NEXT)
    elif env.python is not None:
        run = "python ControlSys/rfp_controlsys/pick_place.py --mode sim3d"
        print(f"\nNext: conda activate {ENV_NAME}, then {run}")
    print("\n" + ("Set up." if complete else "Not everything is set up: see [missing] above."))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
