"""
The FR5 controller through frcobot_ros2's ros2_cmd_server: digital outputs with the
`SetDO(id,status)` string command on /fairino_remote_command_service, and digital inputs,
outputs, emergency stop and error codes from the state it publishes on /nonrt_state_data
(10-100 Hz). FairinoIO is the GripperIO of the real gripper (gripper.py).

Firmware 3.7 and newer, read from each driver's source (frcobot_ros2's tags), not all
checked on an arm:
- The state message: up to firmware 3.8.0 one `error_code`; from 3.8.2 `main_error_code` and
  `sub_error_code`. The newest driver's version dispatcher publishes empty states with
  `reconnect_flag` 1 (or 2) while it isn't connected to the controller.
- The command's answer: the driver for firmware 3.7.5 and older (V2, no `abnormal_stop` in
  its state) answers "1" for success and "0" for an error; from 3.7.6 (V3, the SDK's
  error code) "0" for success, "-1" for a command it can't read.
Outputs are confirmed from the state as well, whatever the answer says.
`pick_place.py --mode check` prints which driver it found and what it reads.

A driver that publishes no state at all (with a 3.7 controller: one built for another
firmware; FAIRINO's newest server publishes only empty ones there) is named as the cause;
`connect(require_state=False)` goes on without it ("stateless"): the generation from whether
it answers GetRobotVersion(), outputs judged by the answer, no faults or torques read here
(the controller still stops the arm itself).

Start the server with: ros2 run fairino_hardware ros2_cmd_server (./robot.sh starts it).
"""

import math
import threading
import time

from rfp_arm.interfaces import E_COMM, ArmError
from rfp_arm.ros_link import RosLink

SERVICE = "fairino_remote_command_service"
TOPIC = "nonrt_state_data"
STATE_TIMEOUT = 1.0  # s without a state message: the controller is gone
OUTPUT_CONFIRM = 0.5  # s for a switched output to show in the state
# The command's answer for success, by driver generation (see the module's docstring)
SUCCESS = {"V2": "1", "V3": "0"}
ROBOT_MODES = {0: "automatic", 1: "manual"}


def io_bit(low: int, high: int, index: int) -> bool:
    """Bit `index` (0-15) of a 16-bit I/O word given as its low and high bytes."""
    if not 0 <= index <= 15:
        raise ValueError(f"I/O index {index} is not 0-15")
    return bool(((high << 8) | low) >> index & 1)


def command_text(name: str, *arguments) -> str:
    """'SetDO(0,1)': the server takes no spaces, and numbers only."""
    return f"{name}({','.join(str(a) for a in arguments)})"


def server_running(node) -> bool:
    """Whether FAIRINO's command server runs: its state topic has a publisher or its service
    is offered (a server that sends no state still offers the service)."""
    if node.count_publishers(TOPIC):
        return True
    names = getattr(node, "get_service_names_and_types", None)
    return names is not None and any(name.lstrip("/") == SERVICE for name, _ in names())


def driver_generation(state) -> str:
    """'V3' (firmware 3.7.6 and newer: the SDK's answers) or 'V2' (3.7.5 and older), from
    the fields of the state message the driver publishes."""
    return "V3" if hasattr(state, "abnormal_stop") else "V2"


def error_code(state) -> tuple[int, int]:
    """(main, sub) error code of a state message of any firmware (sub 0 before 3.8.2)."""
    if hasattr(state, "main_error_code"):
        return int(state.main_error_code), int(state.sub_error_code)
    return int(getattr(state, "error_code", 0)), 0


class FairinoIO:
    def __init__(self, link: RosLink, clock=time.monotonic, sleep=time.sleep):
        from fairino_msgs.msg import RobotNonrtState
        from fairino_msgs.srv import RemoteCmdInterface

        self.link = link
        self._request = RemoteCmdInterface.Request
        self._client = link.node.create_client(RemoteCmdInterface, SERVICE)
        self._lock = threading.Lock()
        self._state = None
        self._stamp = 0.0
        self._clock = clock
        self._sleep = sleep
        self.stateless = False  # connected without a state ever arriving
        self._generation: str | None = None  # the driver's, while there is no state
        link.node.create_subscription(RobotNonrtState, TOPIC, self._on_state, 1)

    @staticmethod
    def present(link: RosLink, wait: float = 1.5) -> bool:
        """Whether FAIRINO's command server runs (server_running): a real arm. Waits up to
        `wait` seconds: a new node needs a moment to discover what runs."""
        deadline = time.monotonic() + wait
        while not server_running(link.node):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def connect(self, timeout: float = 5.0, require_state: bool = True) -> None:
        """The command server answering and its state arriving. No state within `timeout`:
        ArmError saying why, or with `require_state` False `stateless` from then on."""
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise ArmError(
                E_COMM,
                f"/{SERVICE} isn't there: start 'ros2 run fairino_hardware ros2_cmd_server'",
            )
        deadline = time.monotonic() + timeout
        while self._state is None:
            if time.monotonic() > deadline:
                if require_state:
                    raise ArmError(E_COMM, self.no_state_reason())
                self.stateless = True
                # the drivers for 3.7.6 on answer GetRobotVersion(); 3.7.5's ("V2") doesn't
                self._generation = "V3" if self.firmware() else "V2"
                return
            time.sleep(0.02)

    def no_state_reason(self) -> str:
        """Why no state arrives, for the operator."""
        what = "no state topic" if self.link.node.count_publishers(TOPIC) == 0 else "no state"
        return (
            f"FAIRINO's command server publishes {what} (/{TOPIC}): its driver isn't the one "
            "for the controller's firmware (each 3.7.x has its own; FAIRINO's newest server "
            "sends a 3.7 controller nothing: ./setup.sh --robot reads the firmware and builds "
            "the right one), it can't reach the controller (192.168.58.2), or a second "
            "command server competes for it (run one: this program starts one with MoveIt)"
        )

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg
            self._stamp = self._clock()

    @property
    def driver(self) -> str:
        """The driver's generation ('V2' or 'V3'): from its state; without one, from its
        answers (stateless) or 'V3'."""
        with self._lock:
            state = self._state
        return (self._generation or "V3") if state is None else driver_generation(state)

    def succeeded(self, answer: str) -> bool:
        """Whether a command's answer means success for this driver."""
        return answer.strip() == SUCCESS[self.driver]

    def command(self, name: str, *arguments, timeout: float = 2.0) -> str:
        """Send a command; the server's answer text."""
        return self.send(command_text(name, *arguments), timeout)

    def send(self, text: str, timeout: float = 2.0) -> str:
        """Send a command as written, e.g. 'SetDO(0,1)'; the server's answer text."""
        request = self._request()
        request.cmd_str = text
        try:
            return self.link.call(self._client, request, timeout).cmd_res
        except TimeoutError:
            raise ArmError(E_COMM, f"no answer to {text}") from None

    def execute(self, text: str, timeout: float = 2.0) -> str:
        """Send a command as written; its answer. Raises ArmError(E_COMM) unless the answer
        means success for this driver (the grasp settings' press commands)."""
        answer = self.send(text, timeout)
        if not self.succeeded(answer):
            raise ArmError(
                E_COMM, f"the controller answered '{answer}' to {text} (driver {self.driver})"
            )
        return answer

    def firmware(self) -> str | None:
        """The controller's software version as the command server reports it (drivers for
        firmware 3.7.6 and newer: 'GetRobotVersion()'), or None. Reads only."""
        try:
            answer = self.send("GetRobotVersion()")
        except ArmError:
            return None
        code, _, text = answer.partition(",")
        if code.strip() != "0" or not text:
            return None
        return text.split(":", 1)[-1].strip() or None

    def joint_positions(self) -> list[float] | None:
        """The joints (rad, j1 to j6) as the controller reports them (its state: degrees),
        or None."""
        state = self._measured()
        if state is None:
            return None
        return [math.radians(float(getattr(state, f"j{i}_cur_pos"))) for i in range(1, 7)]

    def joint_torques(self) -> list[float] | None:
        """The joints' torques (N m, j1 to j6) as the controller reports them, or None."""
        state = self._measured()
        if state is None:
            return None
        return [float(getattr(state, f"j{i}_cur_tor")) for i in range(1, 7)]

    # -- GripperIO -----------------------------------------------------------
    def set_output(self, index: int, on: bool) -> None:
        """Switch a digital output (0-15); done once the controller's state shows it (the
        answer's meaning differs between drivers, the state doesn't)."""
        text = command_text("SetDO", index, int(on))
        answer = self.send(text)
        refused = f"{text} answered '{answer}' (driver {self.driver})"
        if not 0 <= index <= 15 or self._never_stated():  # no state to show it: the answer
            if not self.succeeded(answer):
                raise ArmError(E_COMM, refused)
            return
        deadline = self._clock() + OUTPUT_CONFIRM
        while self.output(index) != on:
            if self._clock() > deadline:
                if self.succeeded(answer):
                    refused = f"{text} answered '{answer}', but DO{index} didn't switch"
                raise ArmError(E_COMM, refused)
            self._sleep(0.01)

    def _never_stated(self) -> bool:
        with self._lock:
            return self.stateless and self._state is None

    def output(self, index: int) -> bool | None:
        state = self._fresh()
        return None if state is None else io_bit(state.dgt_output_l, state.dgt_output_h, index)

    def input(self, index: int) -> bool | None:
        state = self._fresh()
        return None if state is None else io_bit(state.dgt_input_l, state.dgt_input_h, index)

    def age(self) -> float:
        with self._lock:
            return self._clock() - self._stamp

    # -- controller state ------------------------------------------------------
    def fault(self) -> str | None:
        """Why the controller won't move the arm, or None."""
        state = self._fresh()
        if state is None:
            if self._never_stated():
                return None  # can't tell: said at the start; the controller stops the arm
            return f"no robot state for {self.age():.1f} s"
        reconnect = int(getattr(state, "reconnect_flag", 0))
        if reconnect:  # the newest driver's dispatcher, not connected to a matching driver
            return (
                f"the command server isn't connected to the controller (reconnect_flag "
                f"{reconnect}{': firmware without a matching driver' if reconnect == 2 else ''})"
                "; ./setup.sh --robot builds the driver for the controller's firmware"
            )
        if state.emg:
            return "emergency stop pressed"
        main, sub = error_code(state)
        if main:
            return f"controller error {main}-{sub}"
        return None

    def report(self) -> list[str]:
        """What the controller reports now, for the operator (check mode)."""
        state = self._fresh()
        if state is None:
            return [f"no robot state on /{TOPIC}"]
        flange = [getattr(state, f"flange_{axis}_cur_pos") for axis in "xyzabc"]
        outputs = (state.dgt_output_h << 8) | state.dgt_output_l
        inputs = (state.dgt_input_h << 8) | state.dgt_input_l
        main, sub = error_code(state)
        mode = int(state.robot_mode)
        generation = driver_generation(state)
        firmware = "3.7.6 or newer" if generation == "V3" else "3.7.5 or older"
        lines = [
            (
                f"driver: {generation} (firmware {firmware}; a command succeeded when it "
                f"answers '{SUCCESS[generation]}')"
            ),
            "controller flange pose (mm, deg): " + ", ".join(f"{v:.1f}" for v in flange),
            f"digital outputs DO15..DO0: {outputs:016b}",
            f"digital inputs  DI15..DI0: {inputs:016b}",
            (
                f"emergency stop: {'PRESSED' if state.emg else 'released'}; error code "
                f"{main}-{sub}; robot mode {mode} ({ROBOT_MODES.get(mode, 'unknown')})"
            ),
        ]
        if mode == 1:
            lines.append("   the robot is in manual mode: FAIRINO's driver moves it in automatic")
        return lines

    def flange_position(self) -> tuple[float, float, float] | None:
        """The controller's flange position in its base frame (m), or None."""
        state = self._measured()
        if state is None:
            return None
        return tuple(
            v / 1000
            for v in (state.flange_x_cur_pos, state.flange_y_cur_pos, state.flange_z_cur_pos)
        )

    def _measured(self):
        """The fresh state if it carries the controller's readings: not the empty ones the
        newest driver's dispatcher publishes while it has no driver for the controller
        (all zeros: taken for the arm's joints, they made MoveIt look like a mock arm)."""
        state = self._fresh()
        return None if state is None or int(getattr(state, "reconnect_flag", 0)) else state

    def _fresh(self):
        with self._lock:
            stale = self._clock() - self._stamp > STATE_TIMEOUT
            return None if stale else self._state
