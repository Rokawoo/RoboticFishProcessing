"""
The vacuum valve's ESP32 board (firmware: Arm/firmware/valve_control), from the PC:
over USB serial or over Wi-Fi HTTP. The board answers every command with one line,
"OK ON", "OK OFF" or "ERR ...", so a command that didn't take effect is noticed.

    valve = SerialValve("/dev/ttyACM0")      # or HttpValve("192.168.4.1")
    valve.on(); valve.off(); valve.pulse(500); valve.is_on()

Serial needs pyserial. No ROS.
"""

import time
import urllib.error
import urllib.request
from collections.abc import Callable

BAUD = 115200
BOOT_TIMEOUT = 15.0  # s: a board that just restarted tries Wi-Fi for 10 s before it answers


class ValveError(RuntimeError):
    """The valve board can't be reached, or refused a command."""


def _state(reply: str, command: str) -> bool:
    """True/False for the board's 'OK ON' / 'OK OFF'; anything else is an error."""
    if reply == "OK ON":
        return True
    if reply == "OK OFF":
        return False
    raise ValveError(f"the valve board answered '{reply}' to {command}")


def open_serial(port: str, baud: int, timeout: float):
    """An open pyserial port. DTR and RTS stay low, or opening it would restart the ESP32
    (and a restart closes the valve: a held fish drops)."""
    try:
        import serial
    except ImportError:
        raise ValveError("the serial valve needs pyserial: pip install pyserial") from None
    link = serial.Serial()
    link.port, link.baudrate = port, baud
    link.timeout = link.write_timeout = timeout
    link.dtr = link.rts = False
    link.open()
    return link


class SerialValve:
    def __init__(
        self,
        port: str,
        baud: int = BAUD,
        timeout: float = 2.0,
        open_port: Callable = open_serial,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = f"valve board on {port}"
        self._timeout = timeout
        self._clock = clock
        try:
            self._link = open_port(port, baud, timeout)
        except OSError as e:  # pyserial's SerialException is an OSError
            raise ValveError(f"can't open {port}: {e}") from None
        self._link.reset_input_buffer()
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Ask for the status until the board answers (it may still be starting up)."""
        deadline = self._clock() + BOOT_TIMEOUT
        while True:
            try:
                self.is_on()
                return
            except ValveError:
                if self._clock() > deadline:
                    raise ValveError(f"the {self.name} doesn't answer") from None

    def send(self, command: str) -> str:
        """Send one command line; the board's reply line ('OK ...' or 'ERR ...')."""
        try:
            self._link.reset_input_buffer()  # a late answer to an earlier command isn't this one's
            self._link.write((command + "\n").encode("ascii"))
            self._link.flush()
            deadline = self._clock() + self._timeout
            while self._clock() <= deadline:
                line = self._link.readline().decode("ascii", "replace").strip()
                if line.startswith(("OK", "ERR")):  # skips what the board prints while starting
                    return line
        except OSError as e:
            raise ValveError(f"the {self.name} stopped answering: {e}") from None
        raise ValveError(f"no answer from the {self.name} to {command}")

    def on(self) -> bool:
        return _state(self.send("ON"), "ON")

    def off(self) -> bool:
        return _state(self.send("OFF"), "OFF")

    def pulse(self, milliseconds: int) -> bool:
        """Open the valve for a time; the board closes it by itself."""
        return _state(self.send(f"PULSE {int(milliseconds)}"), "PULSE")

    def is_on(self) -> bool:
        return _state(self.send("STATUS"), "STATUS")

    def close(self) -> None:
        """Close the port. The valve stays as it is."""
        self._link.close()


class HttpValve:
    def __init__(self, host: str, timeout: float = 2.0, fetch: Callable | None = None):
        """`host`: the board's address (192.168.4.1 when it runs its own access point)."""
        self.name = f"valve board at {host}"
        self._base = f"http://{host}"
        self._timeout = timeout
        self._fetch = fetch or self._get
        self.is_on()  # reachable?

    def _get(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=self._timeout) as response:
            return response.read().decode("ascii", "replace").strip()

    def send(self, path: str) -> str:
        try:
            return self._fetch(self._base + path)
        except (urllib.error.URLError, OSError) as e:
            raise ValveError(f"the {self.name} can't be reached: {e}") from None

    def on(self) -> bool:
        return _state(self.send("/on"), "/on")

    def off(self) -> bool:
        return _state(self.send("/off"), "/off")

    def pulse(self, milliseconds: int) -> bool:
        return _state(self.send(f"/pulse?ms={int(milliseconds)}"), "/pulse")

    def is_on(self) -> bool:
        return _state(self.send("/status"), "/status")

    def close(self) -> None:
        pass
