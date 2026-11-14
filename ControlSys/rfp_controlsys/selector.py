"""
Which fish next (CS-20, CS-30). A FishSelector looks at the candidates and answers with a
Decision in the shape of the VLM answer of Table IF-5: pick a fish, re-observe, or end.
Every answer goes through `validate_decision` before anything moves (CS-31, CS-32), and
`fallback_choice` is the fixed rule used when a selector fails (CS-61).

RuleSelector follows the goals of Table CS-2 with fixed rules. A VLM selector is another
class with the same `choose`: nothing else in the Control System changes (CS-26).
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from rfp_vision.fish_list import Fish, FishList

ACTIONS = ("pick", "reobserve", "end")


@dataclass
class Decision:
    """Table IF-5's answer."""

    action: str  # "pick", "reobserve" or "end"
    fish_id: int | None = None
    reason: str = ""
    source: str = "rule"  # who decided: "rule", "fallback", later "vlm"


@dataclass
class DecisionContext:
    """Table IF-5's request: what a selector may base its answer on."""

    fish_list: FishList
    candidates: list[Fish]  # CS-13: valid, in the pick area, not skipped
    image: np.ndarray | None = None  # the observation with outlines and fish IDs drawn on it
    arm_state: str = "ready"
    gripper_state: str = "vacuum off"
    last_result: str = ""  # "", "success" or the last cycle's failure code
    last_failed_id: int | None = None  # this list's ID of the fish that just failed, if in it
    attempts: dict[int, int] = field(default_factory=dict)  # fish ID -> grasp attempts so far


class FishSelector(Protocol):
    def choose(self, context: DecisionContext) -> Decision: ...


def validate_decision(decision, candidates: list[Fish]) -> str | None:
    """Why an answer can't be used, or None (CS-31)."""
    if not isinstance(decision, Decision):
        return "not a decision"
    if decision.action not in ACTIONS:
        return f"unknown action '{decision.action}'"
    if decision.action == "pick":
        if not isinstance(decision.fish_id, int) or isinstance(decision.fish_id, bool):
            return "pick without a fish ID"
        if all(fish.fish_id != decision.fish_id for fish in candidates):
            return f"fish {decision.fish_id} is not a candidate"
    return None


def fallback_choice(candidates: list[Fish]) -> Decision:
    """CS-61: the most confident candidate that touches no other fish, else the most confident."""
    if not candidates:
        return Decision("reobserve", None, "no candidate fish", "fallback")
    free = [fish for fish in candidates if not fish.overlap]
    best = max(free or candidates, key=lambda fish: fish.confidence)
    return Decision("pick", best.fish_id, "most confident candidate", "fallback")


def clearance(fish: Fish, others: list[Fish]) -> float:
    """Distance from the fish's grasp point to the nearest other fish's (m); inf if alone."""
    distances = [
        float(np.linalg.norm(other.grasp_point[:2] - fish.grasp_point[:2]))
        for other in others
        if other is not fish and other.grasp_point is not None
    ]
    return min(distances, default=float("inf"))


class RuleSelector:
    """Table CS-2 by fixed rules: a fish touching no other first (goal 3), a fish that
    didn't just fail before the one that did (goal 5), then the one farthest from the
    others (goal 4), then the most confident. Goal 2 (a fish on top first) needs the
    `covered` check, which Vision doesn't have yet."""

    def choose(self, context: DecisionContext) -> Decision:
        candidates = context.candidates
        if not candidates:
            return Decision("reobserve", None, "no candidate fish")

        def rank(fish: Fish):
            return (
                fish.overlap,
                fish.fish_id == context.last_failed_id,
                -clearance(fish, context.fish_list.valid),
                -fish.confidence,
            )

        best = min(candidates, key=rank)
        why = "touches no other fish" if not best.overlap else "every candidate touches another"
        return Decision("pick", best.fish_id, f"{why}; confidence {best.confidence:.2f}")
