"""
Unit tests for STAGE_COUNTS and the SequenceStateMachine progression it's derived from — pure
logic, no MediaPipe model or video file needed (mirrors test_run_logging.py's approach at the repo
root: real CV-model-driven verification stays in this module's existing
test_gesture_hand_keypoint.py video-based smoke test; this file covers the open_count/close_count
lookup added for decision logging, see plans/10_debug_logging_observability.md chunk 2).

Run with:
    python -m pytest project_tests/gesture_hand_keypoint/test_sequence_counts.py -v
"""
from dataclasses import dataclass
from typing import Optional

import pytest

from modules.gesture_hand_keypoint.hand_shape import CLOSED, OPEN
from modules.gesture_hand_keypoint.sequence_state_machine import (
    CONFIRMED, STAGE_COUNTS, SequenceStateMachine, WAITING_CLOSE_1, WAITING_CLOSE_2,
    WAITING_OPEN, WAITING_OPEN_2,
)


@dataclass
class _FakeConfig:
    """Only the one field SequenceStateMachine.update() actually reads."""
    max_transition_gap_seconds: Optional[float] = 5.0


def test_stage_counts_covers_every_stage():
    for stage in (WAITING_OPEN, WAITING_CLOSE_1, WAITING_OPEN_2, WAITING_CLOSE_2, CONFIRMED):
        assert stage in STAGE_COUNTS


def test_stage_counts_values():
    assert STAGE_COUNTS[WAITING_OPEN] == (0, 0)
    assert STAGE_COUNTS[WAITING_CLOSE_1] == (1, 0)
    assert STAGE_COUNTS[WAITING_OPEN_2] == (1, 1)
    assert STAGE_COUNTS[WAITING_CLOSE_2] == (2, 1)
    assert STAGE_COUNTS[CONFIRMED] == (2, 2)


def test_full_sequence_progression_matches_counts():
    """Drives a real SequenceStateMachine through a full OPEN->CLOSE->OPEN->CLOSE gesture and
    checks the counts derived at each step match what a human would expect from watching it."""
    machine = SequenceStateMachine()
    config = _FakeConfig()
    t = 1000.0

    stage = machine.update(OPEN, True, t, config)
    assert stage == WAITING_CLOSE_1
    assert STAGE_COUNTS[stage] == (1, 0)  # one open done, waiting for the first close

    stage = machine.update(CLOSED, True, t + 1, config)
    assert stage == WAITING_OPEN_2
    assert STAGE_COUNTS[stage] == (1, 1)  # one open, one close

    stage = machine.update(OPEN, True, t + 2, config)
    assert stage == WAITING_CLOSE_2
    assert STAGE_COUNTS[stage] == (2, 1)  # two opens, one close

    stage = machine.update(CLOSED, True, t + 3, config)
    assert stage == CONFIRMED
    assert STAGE_COUNTS[stage] == (2, 2)  # full sequence


def test_wrong_shape_does_not_advance_or_change_counts():
    machine = SequenceStateMachine()
    config = _FakeConfig()

    stage = machine.update(CLOSED, True, 1000.0, config)  # sequence must start with OPEN
    assert stage == WAITING_OPEN
    assert STAGE_COUNTS[stage] == (0, 0)


def test_height_gate_failure_resets_counts_to_zero():
    machine = SequenceStateMachine()
    config = _FakeConfig()

    machine.update(OPEN, True, 1000.0, config)
    assert STAGE_COUNTS[machine.stage] == (1, 0)

    stage = machine.update(CLOSED, False, 1001.0, config)  # height gate fails mid-sequence
    assert stage == WAITING_OPEN
    assert STAGE_COUNTS[stage] == (0, 0)


def test_timeout_resets_counts_to_zero():
    machine = SequenceStateMachine()
    config = _FakeConfig(max_transition_gap_seconds=1.0)

    machine.update(OPEN, True, 1000.0, config)
    assert STAGE_COUNTS[machine.stage] == (1, 0)

    stage = machine.update(CLOSED, True, 1005.0, config)  # gap of 5s > 1s timeout
    assert stage == WAITING_OPEN
    assert STAGE_COUNTS[stage] == (0, 0)
