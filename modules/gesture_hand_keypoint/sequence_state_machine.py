"""
OPEN -> CLOSED -> OPEN -> CLOSED sequence state machine (redesign, replaces all prior Method 2
wave-motion logic). One instance per hand side ("Left"/"Right"/"unknown"), per track_id.

    WAITING_OPEN -> (open) -> WAITING_CLOSE_1 -> (closed) -> WAITING_OPEN_2 -> (open) ->
    WAITING_CLOSE_2 -> (closed) -> CONFIRMED

Must start from OPEN — a sequence starting at CLOSED does not count until an OPEN is first
observed (WAITING_OPEN only advances on OPEN, never on CLOSED).

Timeout (confirmed via spec): if the gap since the LAST transition exceeds
`max_transition_gap_seconds`, reset to WAITING_OPEN — no partial credit. No timeout pressure
while still sitting in WAITING_OPEN before the first transition (nothing to time from yet).

Palm height gate failure (confirmed with the user): treated as an IMMEDIATE reset to
WAITING_OPEN, stricter than a mere non-advancing frame — consistent with the state machine's
"no partial credit" philosophy applied to this gate too, not just the timeout.

A NEITHER (ambiguous) hand shape, or no hand-shape reading at all (low confidence / no hand),
simply does NOT advance the sequence that frame — it is not itself an immediate reset (only the
timeout and the height-gate failure are).
"""
from dataclasses import dataclass
from typing import Optional

from .config import GestureHandKeypointConfig
from .hand_shape import CLOSED, HandShape, OPEN

WAITING_OPEN = "WAITING_OPEN"
WAITING_CLOSE_1 = "WAITING_CLOSE_1"
WAITING_OPEN_2 = "WAITING_OPEN_2"
WAITING_CLOSE_2 = "WAITING_CLOSE_2"
CONFIRMED = "CONFIRMED"

# stage -> (hand shape required to advance, next stage)
_TRANSITIONS = {
    WAITING_OPEN: (OPEN, WAITING_CLOSE_1),
    WAITING_CLOSE_1: (CLOSED, WAITING_OPEN_2),
    WAITING_OPEN_2: (OPEN, WAITING_CLOSE_2),
    WAITING_CLOSE_2: (CLOSED, CONFIRMED),
}

# stage -> (opens consumed so far, closes consumed so far) in the CURRENT attempt — a pure
# lookup, not separate counter state, since the fixed OPEN->CLOSE->OPEN->CLOSE progression already
# determines these counts from the stage name alone. Logging/debug use only (e.g. distinguishing
# "stuck restarting the first open" from "stuck on the second close").
STAGE_COUNTS = {
    WAITING_OPEN: (0, 0),
    WAITING_CLOSE_1: (1, 0),
    WAITING_OPEN_2: (1, 1),
    WAITING_CLOSE_2: (2, 1),
    CONFIRMED: (2, 2),
}


@dataclass
class SequenceStateMachine:
    stage: str = WAITING_OPEN
    last_transition_time: Optional[float] = None

    def reset(self) -> None:
        self.stage = WAITING_OPEN
        self.last_transition_time = None

    def update(self, hand_shape: Optional[HandShape], height_gate_pass: bool,
               timestamp: float, config: GestureHandKeypointConfig) -> str:
        """Returns the stage AFTER processing this frame. If the returned stage is CONFIRMED,
        the caller (pipeline.py) is responsible for consuming that pulse (feeding the shared
        RED/YELLOW/GREEN confirmation tracker) and then resetting this machine — CONFIRMED is
        momentary, not a state this machine sits in across frames."""
        if not height_gate_pass:
            self.reset()
            return self.stage

        if self.stage == CONFIRMED:
            # Defensive: the caller should have already reset us after consuming CONFIRMED.
            self.reset()

        if self.last_transition_time is not None:
            if timestamp - self.last_transition_time > config.max_transition_gap_seconds:
                self.reset()

        expected_shape, next_stage = _TRANSITIONS[self.stage]
        if hand_shape == expected_shape:
            self.stage = next_stage
            self.last_transition_time = timestamp

        return self.stage
