"""
Unit tests for register_person._capture_phase_cli's RunLogger integration (plans/11_registration_
interactive_console.md chunk 7) — a fake cv2.VideoCapture and a fake person-count detector, no
real camera/model needed (registration_data.LiveSubjectDetector's real YOLO-pose model is never
constructed here). registration_data.CAPTURES_DIR is monkeypatched to a temp dir so this never
touches the real registration_captures/ folder.

All capture-phase timing constants are monkeypatched to 0 so every read is immediately eligible to
save (deterministic frame-count-driven test, not wall-clock-driven) — see each test's own comment.

Run with:
    python -m pytest test_register_person_logging.py -v
"""
import json
import os

import numpy as np
import pytest

import register_person
import registration_data as data
from run_logging import RunLogger


class _FakeCap:
    """Returns `frame_count` real frames, then ret=False forever after — mirrors cv2.VideoCapture
    running out of a finite video, bounding the test deterministically."""

    def __init__(self, frame_count: int):
        self._remaining = frame_count

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeDetector:
    """Always reports exactly 1 person in the ROI — the "accept every frame" case."""

    def count_in_roi(self, frame, roi_percent):
        return 1


@pytest.fixture
def isolated_captures_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CAPTURES_DIR", str(tmp_path / "registration_captures"))
    return str(tmp_path)


@pytest.fixture
def zero_timing(monkeypatch):
    """Every check-interval/save-interval gate becomes '>= 0', which is always true — so every
    single frame with person_count==1 gets saved immediately, no wall-clock dependency at all."""
    monkeypatch.setattr(register_person, "_COUNTDOWN_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_PERSON_CHECK_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_CAPTURE_INTERVAL_SECONDS", 0.0)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_capture_phase_saves_exactly_samples_needed(isolated_captures_dir, zero_timing, tmp_path):
    name = "testperson"
    data.reset_captures(name)
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["testperson"], "camera:0", "config/thresholds.yaml")

    cap = _FakeCap(frame_count=5)  # more than enough to reach samples_needed=3
    saved, frame_idx = register_person._capture_phase_cli(
        cap, _FakeDetector(), [0.2, 0.2, 0.8, 0.8], "FACE THE CAMERA", name, "front",
        samples_needed=3, show=False, stream=None, frame_idx=0, logger=logger,
    )
    logger.close(frame_count=frame_idx, exit_reason="completed")

    assert saved == 3
    assert frame_idx == 3  # countdown skipped (0s), one logged frame per capture-loop iteration

    records = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    assert len(records) == 3
    for i, record in enumerate(records):
        assert record["frame"] == i
        assert record["mode"] == "register"
        assert record["stage"] == "capture"
        assert record["person_name"] == name
        assert record["phase"] == "front"
        assert record["saved"] == i + 1
        assert record["samples_needed"] == 3
        assert record["person_count"] == 1


def test_frame_idx_threads_across_two_phase_calls(isolated_captures_dir, zero_timing, tmp_path):
    """Mirrors run()'s own front-then-back sequencing — frame_idx from the front phase's return
    value must be passed as the back phase's starting frame_idx, producing globally-unique,
    monotonic frame numbers across the whole registration, not reset per phase."""
    name = "testperson2"
    data.reset_captures(name)
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["testperson2"], "camera:0", "config/thresholds.yaml")

    front_saved, frame_idx = register_person._capture_phase_cli(
        _FakeCap(frame_count=3), _FakeDetector(), [0.2, 0.2, 0.8, 0.8], "FACE THE CAMERA", name,
        "front", samples_needed=2, show=False, stream=None, frame_idx=0, logger=logger,
    )
    back_saved, frame_idx = register_person._capture_phase_cli(
        _FakeCap(frame_count=3), _FakeDetector(), [0.15, 0.0, 0.85, 1.0], "TURN AROUND", name,
        "back", samples_needed=2, show=False, stream=None, frame_idx=frame_idx, logger=logger,
    )
    logger.close(frame_count=frame_idx, exit_reason="completed")

    assert front_saved == 2 and back_saved == 2
    assert frame_idx == 4

    records = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    assert [r["frame"] for r in records] == [0, 1, 2, 3]  # globally monotonic, not reset per phase
    assert [r["phase"] for r in records] == ["front", "front", "back", "back"]


def test_no_logger_is_a_safe_no_op(isolated_captures_dir, zero_timing):
    """register_person.run()'s logger parameter is optional — _capture_phase_cli must work
    identically (just without writing anything) when logger=None."""
    name = "testperson3"
    data.reset_captures(name)

    saved, frame_idx = register_person._capture_phase_cli(
        _FakeCap(frame_count=3), _FakeDetector(), [0.2, 0.2, 0.8, 0.8], "FACE THE CAMERA", name,
        "front", samples_needed=2, show=False, stream=None, frame_idx=0, logger=None,
    )
    assert saved == 2
    assert frame_idx == 2


def test_camera_failure_returns_zero_saved_without_crashing(isolated_captures_dir, zero_timing, tmp_path):
    """A camera that never produces a frame (ret=False immediately) must not crash the logger
    integration — countdown returns (0, frame_idx) immediately per _capture_phase_cli's own
    early-return branch."""
    name = "testperson4"
    data.reset_captures(name)
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    logger.start("register", ["testperson4"], "camera:0", "config/thresholds.yaml")

    saved, frame_idx = register_person._capture_phase_cli(
        _FakeCap(frame_count=0), _FakeDetector(), [0.2, 0.2, 0.8, 0.8], "FACE THE CAMERA", name,
        "front", samples_needed=2, show=False, stream=None, frame_idx=0, logger=logger,
    )
    logger.close(frame_count=frame_idx, exit_reason="error")

    assert saved == 0
    assert frame_idx == 0
