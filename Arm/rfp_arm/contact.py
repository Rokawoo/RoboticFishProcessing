"""
Feeling contact without a force sensor: the force pressing back along the tool's axis,
from how the joint torques changed since the estimate was zeroed (a force F along the axis a
changes them by J^T a F). Its size is reported (the sign depends on the controller). Read it
as it was zeroed, at rest or moving steadily, near where it was zeroed (gravity's share
drifts as the arm moves). Thresholds need tuning on the arm (--mode check shows it). No ROS.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from rfp_arm.kinematics import Chain


class ContactEstimator:
    def __init__(self, chain: Chain, tcp: np.ndarray):
        """`tcp`: 4x4 pose of the TCP in the chain's tip link."""
        self._chain = chain
        self._tcp = tcp
        self._zero: np.ndarray | None = None

    @property
    def zeroed(self) -> bool:
        return self._zero is not None

    def zero(self, torques: Sequence[float]) -> None:
        """Take the present joint torques (N m, in the chain's joint order) as no contact."""
        self._zero = np.array(torques, float)

    def force(self, joints: Mapping[str, float], torques: Sequence[float]) -> float:
        """Newtons pressing back along the tool axis since zero()."""
        pattern = self.pattern(joints)
        change = np.array(torques, float) - self._zero
        return float(abs(change @ pattern) / max(pattern @ pattern, 1e-12))

    def pattern(self, joints: Mapping[str, float]) -> np.ndarray:
        """Joint torque (N m) per newton along the tool axis, for each joint."""
        names = self._chain.joint_names
        here = self._chain.fk(joints) @ self._tcp
        axis = here[:3, 2]
        moves = []
        for name in names:
            nudged = self._chain.fk({**joints, name: joints[name] + 1e-6}) @ self._tcp
            moves.append((nudged[:3, 3] - here[:3, 3]) / 1e-6 @ axis)
        return np.array(moves)
