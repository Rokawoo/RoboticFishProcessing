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
                                  for the controller's firmware, built
    ./setup.sh --robot --firmware 3.7.8   the same, for this controller software version (if
                                  it can't be read from the controller, 192.168.58.2)
    ... --check                   only report what is there and what is missing
    ... --yes                     install without asking (sudo still asks for its password)
    ... --test                    run the test suite at the end

It can't, and says what to do instead: install the NVIDIA driver, flash the valve board,
measure the cell.
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
WORKSPACE = Path(os.environ.get("RFP_WS", Path.home() / "rfp_ws"))
CONTROLLER_IP = "192.168.58.2"  # the FR5 controller's address (FAIRINO's drivers use this one)
# frcobot_ros2 has a driver per controller firmware: a tag for most (V3.0.0_RobotV<firmware>,
# found in the clone), these commits for the rest. 3.7.5 and older: the last V2 driver.
FRCOBOT_COMMITS = {
    "3.7.5": "e8a982767ce8459a7d859523734562344276372a",
    "3.8.2": "b2b072192dea21cc0d2108810c43c5355504ef01",
    "3.9.0": "f7a06f35c744dc042f61bc6359c78c52965c9eeb",
    "3.9.2": "3e7d7ee2ed556fd9320cbd3c475efb8654533143",
    "3.9.5": "6ee4abc1b832ced027e2b79c7be9b21b7b429d8d",
}
OLDEST_FIRMWARE = (3, 7, 2)  # FAIRINO: the ROS 2 driver needs firmware newer than 3.7.1
BUILT_STAMP = ".rfp_built"  # in the workspace: the frcobot_ros2 commit it was built from
# ... and the controller firmware its driver is for (pick_place.py compares it with the
# controller's, so a firmware update without a new driver is said, not puzzled over)
FIRMWARE_STAMP = ".rfp_firmware"
IMPORTS = "import rfp_arm, rfp_vision, rfp_controlsys, rfp_sim, cv2, torch, ultralytics, pyvista"
INTERPRETER = "python.defaultInterpreterPath"  # VS Code's (and Pylance's) Python for the folder
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
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, RuntimeError) as e:
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

    vscode = REPO / ".vscode" / "settings.json"  # this machine's: not in Git

    def editor_settings() -> dict:
        if not vscode.is_file():
            return {}
        try:
            return json.loads(vscode.read_text(encoding="utf-8"))
        except ValueError:
            raise RuntimeError(f"{vscode} has comments: set {INTERPRETER} by hand") from None

    def editor() -> tuple[bool, str]:
        current = editor_settings().get(INTERPRETER)
        if not current:
            return False, "VS Code has no Python environment set for this folder"
        done = Path(current) == python()
        return done, f"VS Code uses {current}"

    def set_editor() -> None:
        settings = editor_settings()
        settings[INTERPRETER] = str(python())
        vscode.parent.mkdir(exist_ok=True)
        vscode.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    return [
        Step(
            f"Python environment ({env.label})",
            env.check,
            env.install,
            "create or update it from environment.yml / requirements.txt",
        ),
        Step("Git hooks", hooks, hook, "pre-commit install (ruff on every commit)"),
        Step(
            "VS Code's Python",
            editor,
            set_editor,
            f"set {INTERPRETER} in .vscode/settings.json (if VS Code still shows another one: "
            "Python: Select Interpreter)",
        ),
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


def robot_steps(firmware: str | None = None) -> list[Step]:
    """The robot PC (Ubuntu 22.04), before its Python environment. `firmware`: the
    controller's software version, for FAIRINO's driver (None: unknown)."""
    source = WORKSPACE / "src" / "frcobot_ros2"

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

    def wanted() -> str:
        """The ref to check out: the driver for the firmware, else the pinned commit."""
        if firmware is None:
            return frcobot_commit()
        return frcobot_ref(firmware, output(["git", "-C", source, "tag", "-l"]).split())

    def frcobot() -> tuple[bool, str]:
        if not (source / ".git").is_dir():
            return False, f"not in {source}"
        head = output(["git", "-C", source, "rev-parse", "HEAD"])
        if firmware is None:
            return True, f"{source} at {head[:8]} (the controller's firmware unknown: --firmware)"
        ref = wanted()
        at = output(["git", "-C", source, "rev-parse", f"{ref}^{{commit}}"])
        keep, others = driver_folders(source, firmware)
        ignored = all((folder / "COLCON_IGNORE").is_file() for folder in others)
        if head == at and keep is not None and ignored:
            return True, f"{source} at {ref}: the driver for firmware {firmware} ({keep.name})"
        return False, f"{source} at {head[:8]}; firmware {firmware} needs {ref}"

    def get_frcobot() -> None:
        if not (source / ".git").is_dir():
            source.parent.mkdir(parents=True, exist_ok=True)
            sh(["git", "clone", FRCOBOT_URL, source])
        sh(["git", "-C", source, "fetch", "--tags", "origin"])
        sh(["git", "-C", source, "checkout", wanted()])
        if firmware is None:
            return
        keep, others = driver_folders(source, firmware)
        if keep is None:
            raise RuntimeError(f"no driver for firmware {firmware} in {source}")
        (keep / "COLCON_IGNORE").unlink(missing_ok=True)
        for folder in others:  # the same ros2_control plugin: only one driver may be built
            (folder / "COLCON_IGNORE").touch()
        print(f"    driver for firmware {firmware}: {keep.name}")

    def built() -> tuple[bool, str]:
        if not (WORKSPACE / "install" / "fairino5_v6_moveit2_config").is_dir():
            return False, f"{WORKSPACE} isn't built"
        head = output(["git", "-C", source, "rev-parse", "HEAD"])
        stamp = WORKSPACE / BUILT_STAMP
        if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == head:
            return True, f"{WORKSPACE} built from {head[:8]}"
        return False, f"{WORKSPACE} wasn't built from {head[:8]} (or was built by hand)"

    def build() -> None:
        script = (
            "source /opt/ros/humble/setup.bash && "
            "rosdep install --from-paths src -y --ignore-src && colcon build"
        )
        sh(["bash", "-c", script], cwd=WORKSPACE)
        head = output(["git", "-C", source, "rev-parse", "HEAD"])
        (WORKSPACE / BUILT_STAMP).write_text(head + "\n", encoding="utf-8")
        stamp = WORKSPACE / FIRMWARE_STAMP
        if firmware:
            stamp.write_text(firmware + "\n", encoding="utf-8")
        else:
            stamp.unlink(missing_ok=True)

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
            get_frcobot,
            f"clone it into {WORKSPACE / 'src'}, at "
            + (f"the driver for firmware {firmware}" if firmware else "the pinned commit"),
        ),
        Step(
            "FAIRINO's driver built",
            built,
            build if firmware else None,
            f"rosdep install and colcon build in {WORKSPACE}"
            if firmware
            else "pass --firmware (the controller's software version), then run this again",
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


def firmware_parts(firmware: str) -> tuple[int, ...]:
    """(3, 7, 8) of '3.7.8' or 'V3.7.8'; ValueError if it isn't a version."""
    text = firmware.strip().lstrip("vV")
    if not re.fullmatch(r"\d+(\.\d+){1,3}", text):
        raise ValueError(f"'{firmware}' is not a controller software version like 3.7.8")
    return tuple(int(part) for part in text.split("."))


def frcobot_ref(firmware: str, tags: list[str]) -> str:
    """The frcobot_ros2 tag or commit whose driver is for this controller firmware (FAIRINO:
    the driver must match it). ValueError if there is none: then the controller needs a
    firmware update to one there is a driver for."""
    parts = firmware_parts(firmware)
    version = ".".join(str(p) for p in parts)
    if parts < OLDEST_FIRMWARE:
        raise ValueError(f"firmware {version}: FAIRINO's ROS 2 driver needs 3.7.2 or newer")
    if parts <= (3, 7, 5):
        return FRCOBOT_COMMITS["3.7.5"]
    tagged = re.compile(rf"v3\.0\.0_robotv{re.escape(version)}(_v[\d.]+)?", re.IGNORECASE)
    matching = sorted(tag for tag in tags if tagged.fullmatch(tag))
    if matching:
        return matching[-1]  # its latest fix (_v1.1 after the plain tag)
    if version in FRCOBOT_COMMITS:
        return FRCOBOT_COMMITS[version]
    raise ValueError(
        f"frcobot_ros2 has no driver for firmware {version}: update the controller to one it "
        "has (FAIRINO's ROS 2 guide), then run this again with --firmware"
    )


def driver_folders(source: Path, firmware: str) -> tuple[Path | None, list[Path]]:
    """(The driver folder for this firmware, the other drivers' folders to leave out of the
    build). From firmware 3.9.6 on frcobot_ros2 holds a folder per firmware
    (fairino_hardware_v3_9_8, ...), each with the same ros2_control plugin: only the matching
    one may be built. Older checkouts have one fairino_hardware folder, the driver itself."""
    folders = sorted(p for p in source.glob("fairino_hardware_*") if p.is_dir())
    if not folders:
        single = source / "fairino_hardware"
        return (single if single.is_dir() else None), []
    parts = [str(p) for p in firmware_parts(firmware)]
    names = ["fairino_hardware_v" + "_".join(parts), "fairino_hardware_v" + "_".join(parts[:3])]
    keep = next((source / name for name in names if (source / name).is_dir()), None)
    return keep, [p for p in folders if p != keep]


def controller_firmware(
    ip: str = CONTROLLER_IP, timeout: float = 2.0, port: int = 8080
) -> str | None:
    """The controller's software version, asked on its command port (8080) as FAIRINO's own
    driver does; None if it doesn't answer. Reads only."""
    import socket

    request = b"/f/bIII4III905III20IIIGetSoftwareVersion()III/b/f"
    try:
        with socket.create_connection((ip, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(request)
            reply = connection.recv(1024).decode("ascii", errors="replace")
    except OSError:
        return None
    found = re.search(r"v(\d+(?:\.\d+){1,3})[^,]*,V", reply)
    return found.group(1) if found else None


def os_release() -> dict:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key: value.strip('"') for key, value in pairs}


ROBOT_NEXT = """
Still by hand on the robot PC:
  1. Flash the valve board (Arm/firmware/valve_control, Arduino IDE with the ESP32 boards),
     with secrets.h copied from secrets.example.h.
  2. Measure the workcell (README.md, "Set up the workcell").
Then everything runs through ./robot.sh (it sources ROS, the workspace and .venv, and starts
MoveIt with the FR5's driver and FAIRINO's command server when a mode needs them):
  ./robot.sh --mode check"""


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # in order with the commands' own output
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--robot", action="store_true", help="also set up the robot PC (Linux)")
    parser.add_argument("--check", action="store_true", help="only report, install nothing")
    parser.add_argument("-y", "--yes", action="store_true", help="install without asking")
    parser.add_argument("--test", action="store_true", help="run the tests at the end")
    parser.add_argument(
        "--firmware",
        help="--robot: the controller's software version, e.g. 3.7.8 (default: ask it)",
    )
    args = parser.parse_args(argv)
    if args.robot and WINDOWS:
        parser.error("--robot is for the robot PC (Ubuntu 22.04)")
    firmware = None
    if args.robot:
        try:
            firmware = args.firmware and ".".join(map(str, firmware_parts(args.firmware)))
        except ValueError as e:
            parser.error(str(e))
        firmware = firmware or controller_firmware()
        print(
            f"Controller firmware: {firmware}"
            + ("" if args.firmware else f" (read from {CONTROLLER_IP})")
            if firmware
            else f"Controller firmware: unknown ({CONTROLLER_IP} didn't say): pass --firmware"
        )

    print("=" * 60)
    print(f"SETUP: {platform.system()} {platform.release()}, repository {REPO}")
    print("=" * 60)
    env = VenvEnv() if args.robot else CondaEnv()
    steps = (robot_steps(firmware) if args.robot else []) + project_steps(env)
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
