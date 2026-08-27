"""
Unit tests for register_person.run()'s KeyboardInterrupt handling — Ctrl+C during a headless
--stream-only capture (no --show, so no 'q'-key path exists) must cancel cleanly, not raise a raw
traceback, and must still release the camera and close the logger. No real camera/model needed:
cv2.VideoCapture and registration_data.LiveSubjectDetector are both monkeypatched out.

Run with:
    python -m pytest project_tests/test_register_person_keyboard_interrupt.py -v
"""
import json
import os

import numpy as np
import pytest

from scripts import register_person
from scripts import registration_data as data
from scripts.run_logging import RunLogger


class _FakeCapInterruptsOnNthRead:
    """Behaves like a normal camera for `raise_after` reads, then raises KeyboardInterrupt on the
    next .read() call — simulating Ctrl+C arriving mid-capture. isOpened()/release() included
    since run() (unlike _capture_phase_cli's existing tests) calls those directly on the cap."""

    def __init__(self, raise_after: int):
        self._remaining = raise_after
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self._remaining <= 0:
            raise KeyboardInterrupt
        self._remaining -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeDetector:
    def count_in_roi(self, frame, roi_percent):
        return 1


@pytest.fixture
def isolated_captures_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CAPTURES_DIR", str(tmp_path / "registration_captures"))
    return str(tmp_path)


@pytest.fixture
def zero_timing(monkeypatch):
    monkeypatch.setattr(register_person, "_COUNTDOWN_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_PERSON_CHECK_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_CAPTURE_INTERVAL_SECONDS", 0.0)


def test_ctrl_c_during_capture_returns_1_not_a_raised_exception(
        isolated_captures_dir, zero_timing, monkeypatch, tmp_path, capsys):
    fake_cap = _FakeCapInterruptsOnNthRead(raise_after=2)
    monkeypatch.setattr(register_person.cv2, "VideoCapture", lambda index: fake_cap)
    monkeypatch.setattr(data, "LiveSubjectDetector", lambda: _FakeDetector())

    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["testctrlc"], "camera:0", "config/thresholds.yaml")

    # must NOT raise — this is the entire point of the fix
    result = register_person.run("testctrlc", camera_index=0, front_samples=10, back_samples=10,
                                  show=False, stream=None, logger=logger)

    assert result == 1
    assert fake_cap.released  # camera cleanup still ran despite the interrupt
    assert "cancelled" in capsys.readouterr().out.lower()

    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["exit_reason"] == "stopped_early"
    assert meta["end_ts"] is not None  # logger.close() definitely ran, not left dangling


def test_ctrl_c_still_closes_logger_when_close_logger_on_exit_true(
        isolated_captures_dir, zero_timing, monkeypatch, tmp_path):
    """Sanity check for the default (one-shot CLI) path specifically — the REPL path
    (close_logger_on_exit=False) is covered by test_register_person_interactive.py instead."""
    fake_cap = _FakeCapInterruptsOnNthRead(raise_after=0)
    monkeypatch.setattr(register_person.cv2, "VideoCapture", lambda index: fake_cap)
    monkeypatch.setattr(data, "LiveSubjectDetector", lambda: _FakeDetector())

    logger = RunLogger(log_root=str(tmp_path / "runs"))
    logger.start("register", ["testctrlc2"], "camera:0", "config/thresholds.yaml")

    register_person.run("testctrlc2", camera_index=0, show=False, stream=None, logger=logger)

    assert logger._decisions_fh is None  # closed
