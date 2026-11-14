"""
The suction gripper (AS-30), six cups under one vacuum, in the forms it can take:

    SerialGripper   the vacuum switched by a microcontroller board on a serial port
    SuctionGripper  vacuum and blow-off valves on control box outputs, the vacuum switch
                    on its inputs: the gripper states of Table AS-4. The I/O comes from a
                    GripperIO (fairino.py on the real arm)
    SimGripper      no hardware, for simulation and tests
    NoGripper       dry run: the vacuum stays off

All have vacuum_on(), release() and state() (interfaces.Gripper). No ROS.
"""

import time
from collections.abc import Callable
from typing import Protocol

from rfp_arm.interfaces import E_COMM, ArmError, Gripper, GripperState
from rfp_arm.workcell import GripperSettings


class GripperIO(Protocol):
    """Digital I/O of the control box."""

    def set_output(self, index: int, on: bool) -> None:
        """Switch a digital output. Raises ArmError(E_COMM) if the command fails."""

    def output(self, index: int) -> bool | None:
        """State of a digital output; None if unknown."""

    def input(self, index: int) -> bool | None:
        """State of a digital input; None if unknown."""

    def age(self) -> float:
        """Seconds since the last I/O reading."""


class SuctionGripper:
    def __init__(
        self,
        io: GripperIO,
        settings: GripperSettings,
        release_timeout: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._io = io
        self._settings = settings
        self._release_timeout = release_timeout
        self._clock = clock
        self._sleep = sleep
        self._commanded = False  # what this program last asked for
        self._off_since = clock()
        self.has_switch = settings.vacuum_switch

    def vacuum_on(self) -> None:
        self._io.set_output(self._settings.blowoff_do, False)
        self._io.set_output(self._settings.vacuum_do, True)
        self._commanded = True

    def release(self) -> None:
        s = self._settings
        self._io.set_output(s.vacuum_do, False)
        self._commanded = False
        self._off_since = self._clock()
        if s.blowoff_time > 0:
            self._io.set_output(s.blowoff_do, True)
            try:
                self._sleep(s.blowoff_time)
            finally:
                self._io.set_output(s.blowoff_do, False)

    def state(self) -> GripperState:
        s = self._settings
        if self._io.age() > s.state_timeout:
            return GripperState.ERROR
        vacuum_on = self._io.output(s.vacuum_do)
        if vacuum_on is None:
            vacuum_on = self._commanded
        if not self.has_switch:
            return GripperState.NO_SEAL if vacuum_on else GripperState.VACUUM_OFF
        holding, vacuum = self._io.input(s.holding_di), self._io.input(s.vacuum_di)
        if holding is None or vacuum is None:
            return GripperState.ERROR
        if vacuum_on:
            return GripperState.HOLDING if holding else GripperState.NO_SEAL
        if vacuum:
            late = self._clock() - self._off_since > self._release_timeout
            return GripperState.ERROR if late else GripperState.RELEASING
        return GripperState.VACUUM_OFF


def open_serial(port: str, baud: int):
    """An open pyserial port. DTR and RTS stay low: raising them restarts an ESP32."""
    try:
        import serial
    except ImportError:
        raise ArmError(E_COMM, "the serial gripper needs pyserial: pip install pyserial") from None
    link = serial.Serial()
    link.port, link.baudrate = port, baud
    link.dtr = link.rts = False
    link.timeout, link.write_timeout = 0.2, 1.0
    link.open()
    return link


class SerialGripper:
    """
    Vacuum switched by a microcontroller board on a serial port (an ESP32): one text line
    turns it on, another off (GripperSettings.serial_on / serial_off). The board reports
    nothing back, so the state is what was last sent, and a hold can only be confirmed by
    the camera (`has_switch` is False). Opening the port turns the vacuum off.
    """

    has_switch = False

    def __init__(
        self,
        settings: GripperSettings,
        open_port: Callable = open_serial,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._settings = settings
        try:
            self._port = open_port(settings.serial_port, settings.serial_baud)
        except OSError as e:  # pyserial's SerialException is an OSError
            raise ArmError(E_COMM, f"can't open the gripper's port: {e}") from None
        sleep(settings.serial_boot_s)
        self._on = False
        self._send(settings.serial_off)

    def vacuum_on(self) -> None:
        self._send(self._settings.serial_on)
        self._on = True

    def release(self) -> None:
        self._send(self._settings.serial_off)
        self._on = False

    def state(self) -> GripperState:
        return GripperState.NO_SEAL if self._on else GripperState.VACUUM_OFF

    def close(self) -> None:
        """Close the port. The vacuum stays as it is."""
        self._port.close()

    def _send(self, line: str) -> None:
        try:
            self._port.write((line + "\n").encode("ascii"))
            self._port.flush()
        except OSError as e:
            raise ArmError(E_COMM, f"the gripper's port stopped answering: {e}") from None


def make_gripper(settings: GripperSettings, io: "GripperIO | None", release_timeout: float = 1.0):
    """The real gripper the settings describe. `io`: the control box I/O, for 'controller'."""
    if settings.driver == "serial":
        return SerialGripper(settings)
    if io is None:
        raise ArmError(E_COMM, "the controller gripper needs the FAIRINO command server")
    return SuctionGripper(io, settings, release_timeout)


class SimGripper:
    """
    A gripper with no hardware: reports holding once the vacuum is on. For tests,
    `seals=False` never seals, and `drop()` loses the fish.
    """

    has_switch = True

    def __init__(self, seals: bool = True):
        self.seals = seals
        self.vacuum = False
        self.sealed = False
        self.events: list[str] = []

    def vacuum_on(self) -> None:
        self.vacuum, self.sealed = True, self.seals
        self.events.append("vacuum on")

    def release(self) -> None:
        self.vacuum = self.sealed = False
        self.events.append("release")

    def drop(self) -> None:
        self.sealed = False

    def state(self) -> GripperState:
        if not self.vacuum:
            return GripperState.VACUUM_OFF
        return GripperState.HOLDING if self.sealed else GripperState.NO_SEAL


class NoGripper:
    """Dry run (CS-93): the vacuum stays off whatever is asked."""

    has_switch = False

    def vacuum_on(self) -> None:
        pass

    def release(self) -> None:
        pass

    def state(self) -> GripperState:
        return GripperState.VACUUM_OFF


def wait_for(
    gripper: Gripper,
    wanted: GripperState,
    timeout: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Whether the gripper reports `wanted` within `timeout` seconds."""
    deadline = clock() + timeout
    while True:
        if gripper.state() == wanted:
            return True
        if clock() >= deadline:
            return False
        sleep(0.02)
