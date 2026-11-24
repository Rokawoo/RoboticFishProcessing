"""
MoveIt and FAIRINO's driver for one run (Arm/ros/fr5_bringup.launch.py), started as a child
process when a mode needs them and they aren't running, and stopped when the run ends. It
runs `ros2 launch` in this program's environment: ./robot.sh sources ROS 2, the workspace
and the Python environment first. No ROS imports.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

LAUNCH_FILE = Path(__file__).resolve().parents[1] / "ros" / "fr5_bringup.launch.py"
STOP_TIMEOUT = 10.0  # s for ros2 launch to stop its nodes after Ctrl+C


def launch_command(
    hardware: str = "real", rviz: bool = False, cmd_server: bool | None = None
) -> list[str]:
    """`ros2 launch` of fr5_bringup.launch.py: the real FR5 or mock hardware, with RViz or
    not, FAIRINO's command server or not (None: with the real arm)."""
    if hardware not in ("real", "mock"):
        raise ValueError(f"hardware must be 'real' or 'mock', not '{hardware}'")
    command = ["ros2", "launch", str(LAUNCH_FILE), f"hardware:={hardware}"]
    command.append(f"rviz:={'true' if rviz else 'false'}")
    if cmd_server is not None:
        command.append(f"cmd_server:={'true' if cmd_server else 'false'}")
    return command


class Bringup:
    """The launch as a child process: start(), then stop() (also safe twice)."""

    def __init__(self, command: list[str], log_file: Path | None = None):
        self.command = command
        self.log_file = log_file
        self.process: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        if shutil.which(self.command[0]) is None:
            raise RuntimeError(
                f"{self.command[0]} isn't on the PATH: run this through ./robot.sh, which "
                "sources ROS 2 and the workspace with FAIRINO's driver"
            )
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_file, "w", encoding="utf-8")  # noqa: SIM115 (kept open)
        posix = os.name == "posix"
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=posix,  # its own process group: stopped together
        )

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def first_error(self) -> str | None:
        """The first line of the launch's log that starts with [ERROR], or None."""
        if self.log_file is None or not self.log_file.is_file():
            return None
        for line in self.log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lstrip().startswith("[ERROR]"):
                return line.strip()
        return None

    def log_text(self) -> str:
        if self.log_file is None or not self.log_file.is_file():
            return ""
        return self.log_file.read_text(encoding="utf-8", errors="replace")

    def stop(self, timeout: float = STOP_TIMEOUT) -> None:
        """Ctrl+C to the launch and its nodes, as in a terminal; killed if they hang."""
        process = self.process
        if process is not None and process.poll() is None:
            for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGKILL", None)):
                if sig is None:
                    break
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, sig)
                    elif sig == signal.SIGINT:
                        process.terminate()
                    else:
                        process.kill()
                except (ProcessLookupError, PermissionError, OSError):
                    break
                deadline = time.monotonic() + timeout
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if process.poll() is not None:
                    break
        if self._log is not None:
            self._log.close()
            self._log = None
