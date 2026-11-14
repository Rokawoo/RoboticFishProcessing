"""
The FR5 controller through frcobot_ros2's ros2_cmd_server: digital outputs with the
`SetDO(id,status)` string command on /fairino_remote_command_service, and digital inputs,
outputs, emergency stop and error codes from the state it publishes on /nonrt_state_data
(100 Hz). FairinoIO is the GripperIO of the real gripper (gripper.py).

Read from the driver's source (fairino_hardware V3.9.5), not yet checked on the arm:
a command answers "0" for success, and bit n of dgt_output_l / dgt_input_l is DO n / DI n
(n = 0-7; the _h bytes hold 8-15). `pick_place.py --mode check` prints what it reads.

Start the server with: ros2 run fairino_hardware ros2_cmd_server
"""

import threading
import time

from rfp_arm.interfaces import E_COMM, ArmError
from rfp_arm.ros_link import RosLink

SERVICE = "fairino_remote_command_service"
TOPIC = "nonrt_state_data"
SUCCESS = "0"
STATE_TIMEOUT = 1.0  # s without a state message: the controller is gone


def io_bit(low: int, high: int, index: int) -> bool:
    """Bit `index` (0-15) of a 16-bit I/O word given as its low and high bytes."""
    if not 0 <= index <= 15:
        raise ValueError(f"I/O index {index} is not 0-15")
    return bool(((high << 8) | low) >> index & 1)


def command_text(name: str, *arguments) -> str:
    """'SetDO(0,1)': the server takes no spaces, and numbers only."""
    return f"{name}({','.join(str(a) for a in arguments)})"


class FairinoIO:
    def __init__(self, link: RosLink):
        from fairino_msgs.msg import RobotNonrtState
        from fairino_msgs.srv import RemoteCmdInterface

        self.link = link
        self._request = RemoteCmdInterface.Request
        self._client = link.node.create_client(RemoteCmdInterface, SERVICE)
        self._lock = threading.Lock()
        self._state = None
        self._stamp = 0.0
        link.node.create_subscription(RobotNonrtState, TOPIC, self._on_state, 1)

    @staticmethod
    def present(link: RosLink, wait: float = 1.5) -> bool:
        """Whether a command server is publishing the controller's state: a real arm.
        Waits up to `wait` seconds: a new node needs a moment to discover publishers."""
        deadline = time.monotonic() + wait
        while link.node.count_publishers(TOPIC) == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def connect(self, timeout: float = 5.0) -> None:
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise ArmError(
                E_COMM,
                f"/{SERVICE} isn't there: start 'ros2 run fairino_hardware ros2_cmd_server'",
            )
        deadline = time.monotonic() + timeout
        while self._state is None:
            if time.monotonic() > deadline:
                raise ArmError(E_COMM, f"no robot state on /{TOPIC}")
            time.sleep(0.02)

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg
            self._stamp = time.monotonic()

    def command(self, name: str, *arguments, timeout: float = 2.0) -> str:
        """Send a command; the server's answer text."""
        request = self._request()
        request.cmd_str = command_text(name, *arguments)
        try:
            return self.link.call(self._client, request, timeout).cmd_res
        except TimeoutError:
            raise ArmError(E_COMM, f"no answer to {request.cmd_str}") from None

    # -- GripperIO -----------------------------------------------------------
    def set_output(self, index: int, on: bool) -> None:
        answer = self.command("SetDO", index, int(on))
        if answer != SUCCESS:
            raise ArmError(E_COMM, f"SetDO({index},{int(on)}) answered '{answer}'")

    def output(self, index: int) -> bool | None:
        state = self._fresh()
        return None if state is None else io_bit(state.dgt_output_l, state.dgt_output_h, index)

    def input(self, index: int) -> bool | None:
        state = self._fresh()
        return None if state is None else io_bit(state.dgt_input_l, state.dgt_input_h, index)

    def age(self) -> float:
        with self._lock:
            return time.monotonic() - self._stamp

    # -- controller state ------------------------------------------------------
    def fault(self) -> str | None:
        """Why the controller won't move the arm, or None."""
        state = self._fresh()
        if state is None:
            return f"no robot state for {self.age():.1f} s"
        if state.emg:
            return "emergency stop pressed"
        if state.main_error_code:
            return f"controller error {state.main_error_code}-{state.sub_error_code}"
        return None

    def report(self) -> list[str]:
        """What the controller reports now, for the operator (check mode)."""
        state = self._fresh()
        if state is None:
            return [f"no robot state on /{TOPIC}"]
        flange = [getattr(state, f"flange_{axis}_cur_pos") for axis in "xyzabc"]
        outputs = (state.dgt_output_h << 8) | state.dgt_output_l
        inputs = (state.dgt_input_h << 8) | state.dgt_input_l
        return [
            "controller flange pose (mm, deg): " + ", ".join(f"{v:.1f}" for v in flange),
            f"digital outputs DO15..DO0: {outputs:016b}",
            f"digital inputs  DI15..DI0: {inputs:016b}",
            (
                f"emergency stop: {'PRESSED' if state.emg else 'released'}; error code "
                f"{state.main_error_code}-{state.sub_error_code}; robot mode {state.robot_mode}"
            ),
        ]

    def flange_position(self) -> tuple[float, float, float] | None:
        """The controller's flange position in its base frame (m), or None."""
        state = self._fresh()
        if state is None:
            return None
        return tuple(
            v / 1000
            for v in (state.flange_x_cur_pos, state.flange_y_cur_pos, state.flange_z_cur_pos)
        )

    def _fresh(self):
        with self._lock:
            stale = time.monotonic() - self._stamp > STATE_TIMEOUT
            return None if stale else self._state
