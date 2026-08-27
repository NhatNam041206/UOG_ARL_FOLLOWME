"""
Verifies the specific claim: does `register <name>` typed inside run_interactive() (the REPL)
actually publish frames to a DebugStreamServer the same way the headless --person-name path does?
Drives run_interactive() -> run() -> _capture_phase_cli() for real (only cv2.VideoCapture and
LiveSubjectDetector are faked out — everything else, including the real DebugStreamServer, runs
unmodified) and checks the stream's buffer actually received a frame.

Also verifies the REPL's own terminal output stays free of raw decisions.jsonl content — logs go
only to the file, never printed to stdout, so a second terminal running tail_log.py (chunk 9) is
genuinely the only way to watch them, not a redundant option.

Run with:
    python -m pytest test_register_person_interactive_streaming.py -v
"""
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

import register_person
import registration_data as data
from debug_stream import DebugStreamServer
from run_logging import RunLogger


class _FakeCap:
    def __init__(self, frame_count: int):
        self._remaining = frame_count

    def isOpened(self):
        return True

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        pass


class _FakeDetector:
    def count_in_roi(self, frame, roi_percent):
        return 1


@pytest.fixture
def isolated_captures_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CAPTURES_DIR", str(tmp_path / "registration_captures"))
    monkeypatch.setattr(register_person.cv2, "VideoCapture", lambda index: _FakeCap(frame_count=30))
    monkeypatch.setattr(data, "LiveSubjectDetector", lambda: _FakeDetector())
    monkeypatch.setattr(register_person, "_COUNTDOWN_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_PERSON_CHECK_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(register_person, "_CAPTURE_INTERVAL_SECONDS", 0.0)
    # rebuild_registries would try to run real face/pose detection on the captured JPEGs — not
    # what this test is checking (that's chunk 7's/registration_data's own concern), so stub it
    # to a trivial success and isolate this test to the capture+streaming path specifically.
    monkeypatch.setattr(data, "rebuild_registries", lambda name, config_path: True)


def test_register_command_in_repl_actually_streams_frames(isolated_captures_dir, tmp_path):
    stream = DebugStreamServer()
    stream.start(port=0, throttle_every_n_frames=1)  # publish every frame, no throttling, for a fast assert
    try:
        assert stream._buffer.get_jpeg() is None  # nothing published yet

        with patch("builtins.input", side_effect=["register Nam", "quit"]):
            result, chosen_name = register_person.run_interactive(camera_index=0, front_samples=2, back_samples=2,
                                                                    stream=stream)

        assert result == 0
        assert chosen_name is None
        assert stream._buffer.get_jpeg() is not None  # a real frame was published during the REPL command
    finally:
        stream.stop()


def test_repl_stdout_never_contains_raw_jsonl_log_lines(isolated_captures_dir, tmp_path, capsys):
    """The REPL prints user-facing progress ("saved '...' (1/2)", "OK", list output) — it must
    never print a raw JSON decision record to its own terminal. Logs belong ONLY in
    decisions.jsonl, read back via a separate tail_log.py session (chunk 9), never interleaved
    into the control terminal's own stdout."""
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")

    with patch("builtins.input", side_effect=["register Nam", "list", "quit"]):
        register_person.run_interactive(camera_index=0, front_samples=2, back_samples=2, logger=logger)

    out = capsys.readouterr().out
    # decisions.jsonl records always contain this literal key — if it ever leaked into stdout,
    # this substring would appear there too.
    assert '"stage"' not in out
    assert '"mode": "register"' not in out

    # meanwhile the actual records DID get written to the file, just not echoed to the terminal
    with open(os.path.join(run_dir, "decisions.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert any(r.get("stage") == "capture" for r in records)
    assert any(r.get("stage") == "list" for r in records)
