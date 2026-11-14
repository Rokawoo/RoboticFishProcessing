"""
What the Control System sees of the arm and the gripper (IF-30, IF-40): the Arm and
Gripper protocols, the gripper states of Table AS-4 and the error codes of Table AS-2.
The real arm (moveit_arm.py), the real gripper (gripper.py) and the simulated ones
(sim.py) all implement these, so the code above them is the same in every mode.
"""

import enum
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

# Table AS-2
E_NO_SEAL = "E_NO_SEAL"
E_DROP = "E_DROP"
E_RELEASE = "E_RELEASE"
E_COLLISION = "E_COLLISION"
E_PROTECTIVE_STOP = "E_PROTECTIVE_STOP"
E_MOTION = "E_MOTION"
E_COMM = "E_COMM"
OPERATOR_RESET_CODES = frozenset({E_COLLISION, E_PROTECTIVE_STOP, E_COMM})  # AS-73

# Called many times a second while the arm moves. Returning a reason stops the motion.
Tick = Callable[[], str | None]


class ArmError(Exception):
    """A failed arm or gripper step, with its Table AS-2 code (AS-25, AS-71)."""

    def __init__(self, code: str, message: str = "", tcp_pose: np.ndarray | None = None):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.tcp_pose = tcp_pose
        self.gripper_state: GripperState | None = None
        self.fish_put_back = False  # E_MOTION while holding: the fish was set down again
        self.still_holding = False  # the vacuum was left on: the fish may still be on the cups


class MotionAborted(Exception):
    """A Tick stopped a motion. `reason` is what the Tick returned."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class GripperState(enum.Enum):
    """Table AS-4."""

    VACUUM_OFF = "vacuum off"
    NO_SEAL = "vacuum on (no seal)"
    HOLDING = "holding"
    RELEASING = "releasing"
    ERROR = "error"


class Arm(Protocol):
    def tcp_pose(self) -> np.ndarray:
        """4x4 pose of the TCP in the base frame {B}, now."""

    def move_joints(self, joints: Sequence[float], tick: Tick | None = None) -> None:
        """Point-to-point move to joint positions (rad)."""

    def move_ptp(self, tcp_pose: np.ndarray, tick: Tick | None = None) -> None:
        """Point-to-point move of the TCP to a pose in {B}; the path between is not straight."""

    def move_lin(
        self, tcp_pose: np.ndarray, speed: float, accel: float, tick: Tick | None = None
    ) -> None:
        """Straight-line move of the TCP to a pose in {B} at up to `speed` m/s, `accel` m/s2."""

    def set_held_fish(self, length: float) -> None:
        """Tell the arm a fish of this length (m) hangs on the cups; 0 = nothing. From then
        on its checks keep both ends of the fish inside the work zone too (SF-10)."""

    def stop(self) -> None:
        """Controlled stop of any motion (SF-32)."""

    def fault(self) -> str | None:
        """Why the arm can't move (emergency stop, controller error), or None."""


class Gripper(Protocol):
    has_switch: bool  # False: no vacuum switch, the state can't tell holding from no seal

    def vacuum_on(self) -> None: ...

    def release(self) -> None:
        """Vacuum off, then a blow-off pulse."""

    def state(self) -> GripperState: ...


class HoldCheck(Protocol):
    """A second opinion on whether the fish is still on the cups (vision)."""

    def start(self) -> bool:
        """The cups are on the fish. False if this check can't tell from here."""

    def lost(self) -> bool:
        """Called while the fish is carried. True once the fish is gone."""

    def stop(self) -> None: ...
