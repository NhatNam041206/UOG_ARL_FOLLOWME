"""
Unit tests for scripts.run_logging.RunLogger. Pure filesystem/JSON logic, no CV models or video
files needed — run with:

    python -m pytest project_tests/test_run_logging.py -v
"""
import json
import os

import pytest

from scripts.run_logging import RunLogger


@pytest.fixture
def thresholds_file(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text("steering:\n  kp: 1.0\n  servo_center_degrees: 90.0\n", encoding="utf-8")
    return str(path)


def test_start_creates_run_dir_and_meta(tmp_path, thresholds_file):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", ["--modules", "followme"], "camera:0", thresholds_file)

    assert os.path.isdir(run_dir)
    assert os.path.basename(run_dir).endswith("_followme")
    meta_path = os.path.join(run_dir, "meta.json")
    assert os.path.exists(meta_path)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["mode"] == "followme"
    assert meta["source"] == "camera:0"
    assert meta["argv"] == ["--modules", "followme"]
    assert meta["thresholds_snapshot"]["steering"]["kp"] == 1.0
    assert meta["end_ts"] is None
    assert meta["frame_count"] == 0
    assert meta["video_saved"] is False
    assert meta["stream_enabled"] is False
    assert "git_commit" in meta  # None is fine outside a repo, key must still exist


def test_log_frame_appends_jsonl(tmp_path, thresholds_file):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", [], "video:test.mp4", thresholds_file)

    logger.log_frame(frame=0, debug_state="WAITING_FOR_TRIGGER")
    logger.log_frame(frame=1, debug_state="TRACKING_STARTED")

    decisions_path = os.path.join(run_dir, "decisions.jsonl")
    with open(decisions_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0] == {"frame": 0, "debug_state": "WAITING_FOR_TRIGGER"}
    assert lines[1] == {"frame": 1, "debug_state": "TRACKING_STARTED"}


def test_log_frame_before_start_raises(tmp_path):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    with pytest.raises(RuntimeError):
        logger.log_frame(frame=0)


def test_set_video_info_updates_meta(tmp_path, thresholds_file):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", [], "camera:0", thresholds_file)

    logger.set_video_info(os.path.join(run_dir, "debug.avi"), fps=24.1, resolution=(640, 480))

    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["video_saved"] is True
    assert meta["video_path"] == "debug.avi"
    assert meta["video_fps"] == 24.1
    assert meta["video_resolution"] == [640, 480]


def test_set_stream_info_updates_meta(tmp_path, thresholds_file):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", [], "camera:0", thresholds_file)

    logger.set_stream_info("http://127.0.0.1:8080/stream")

    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["stream_enabled"] is True
    assert meta["stream_url"] == "http://127.0.0.1:8080/stream"


def test_close_finalizes_meta_and_closes_file(tmp_path, thresholds_file):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", [], "camera:0", thresholds_file)
    logger.log_frame(frame=0)

    logger.close(frame_count=42, exit_reason="user_quit")

    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["frame_count"] == 42
    assert meta["exit_reason"] == "user_quit"
    assert meta["end_ts"] is not None
    assert logger._decisions_fh is None  # file handle released, no leak across runs


def test_missing_thresholds_file_yields_empty_snapshot(tmp_path):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("followme", [], "camera:0", str(tmp_path / "does_not_exist.yaml"))

    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["thresholds_snapshot"] == {}
