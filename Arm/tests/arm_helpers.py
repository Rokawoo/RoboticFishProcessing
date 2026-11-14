"""Shared by the arm and control tests: the default workcell with changes, a clock that
only moves when told, a fake control box I/O and a small inverse kinematics solver."""

import copy

import numpy as np
import yaml
from rfp_arm.kinematics import FR5, Chain
from rfp_arm.workcell import DEFAULT_FILE, Workcell, workcell_from_mapping

# The FR5 reaching forward over the table with the tool pointing down, well inside its
# joint limits: where solve_ik starts from
SEED = np.radians([20.0, -100.0, -100.0, -70.0, 90.0, -60.0])


def workcell_data() -> dict:
    """The mapping of Arm/configs/workcell.yaml, to change and load."""
    return copy.deepcopy(yaml.safe_load(DEFAULT_FILE.read_text(encoding="utf-8")))


def make_workcell(change=None) -> Workcell:
    """The default workcell; `change(data)` edits its mapping first."""
    data = workcell_data()
    if change is not None:
        change(data)
    workcell = workcell_from_mapping(data)
    assert workcell.problems() == []
    return workcell


class FakeClock:
    """clock() and sleep() for code that waits: time passes only in sleep() and advance()."""

    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeIO:
    """Control box I/O (gripper.GripperIO) with outputs that just remember their state."""

    def __init__(self):
        self.outputs: dict[int, bool] = {}
        self.inputs: dict[int, bool] = {}
        self.calls: list[tuple[int, bool]] = []
        self.seconds_since_reading = 0.0

    def set_output(self, index: int, on: bool) -> None:
        self.outputs[index] = on
        self.calls.append((index, on))

    def output(self, index: int) -> bool | None:
        return self.outputs.get(index, False)

    def input(self, index: int) -> bool | None:
        return self.inputs.get(index, False)

    def age(self) -> float:
        return self.seconds_since_reading


def _twist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """6-vector taking pose a to pose b: translation, then rotation vector (base frame)."""
    r = b[:3, :3] @ a[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    rotation = axis / 2 if angle < 1e-6 else axis / (2 * np.sin(angle)) * angle
    return np.concatenate([b[:3, 3] - a[:3, 3], rotation])


def solve_ik(target_tip: np.ndarray, seed=SEED, chain: Chain = FR5, steps: int = 300):
    """Joint positions putting the tip link at `target_tip` (damped least squares from
    `seed`), or None. Enough for tests; the real arm's IK is MoveIt's."""
    seed = np.array(seed, float)
    q = seed.copy()
    lower = np.array([j.lower for j in chain.joints])
    upper = np.array([j.upper for j in chain.joints])
    for _ in range(steps):
        now = chain.fk(q)
        error = _twist(now, target_tip)
        if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-6:
            # the same angles, as near to the seed as the joint limits allow
            q = q - 2 * np.pi * np.round((q - seed) / (2 * np.pi))
            q = np.where(q > upper, q - 2 * np.pi, np.where(q < lower, q + 2 * np.pi, q))
            return q if ((q >= lower) & (q <= upper)).all() else None
        jacobian = np.zeros((6, len(q)))
        for i in range(len(q)):
            bumped = q.copy()
            bumped[i] += 1e-6
            jacobian[:, i] = _twist(now, chain.fk(bumped)) / 1e-6
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
        q = q + np.clip(step, -0.3, 0.3)
    return None
