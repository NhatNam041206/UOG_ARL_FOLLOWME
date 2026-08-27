"""
Unit tests for tail_log.py (plans/11_registration_interactive_console.md chunk 9) — pure
filesystem/string logic, no camera/model/network dependency at all. Run with:

    python -m pytest test_tail_log.py -v
"""
import json
import os
import threading
import time as real_time

import pytest

import tail_log


def test_format_record_flat_fields_with_timestamp():
    record = {"ts": 1735200000.123, "frame": 5, "debug_state": "TRACKING", "should_move": True}
    out = tail_log.format_record(record)
    assert out.startswith("[")  # HH:MM:SS.mmm prefix present
    assert "frame=5" in out
    assert "debug_state=TRACKING" in out
    assert "should_move=True" in out
    assert "ts=" not in out  # consumed into the prefix, not also printed as a field


def test_format_record_without_timestamp_has_no_prefix():
    record = {"frame": 1, "mode": "register"}
    out = tail_log.format_record(record)
    assert not out.startswith("[")
    assert out == "frame=1 mode=register"


def test_format_record_nested_dict_compacted_to_json_not_python_repr():
    """Schema-agnostic requirement (plans/11 §3.3.6): a followme-style nested block (gesture/
    tracking/face_identity) must render without this file knowing those field names specifically."""
    record = {"ts": 1.0, "tracking": {"state": "TRACKING", "horizontal_offset": 0.12}}
    out = tail_log.format_record(record)
    assert 'tracking={"state":"TRACKING","horizontal_offset":0.12}' in out


def test_format_record_list_value_pretrigger_style():
    record = {"ts": 1.0, "mode": "pretrigger", "people": [{"matched_person_name": "alice"}]}
    out = tail_log.format_record(record)
    assert 'people=[{"matched_person_name":"alice"}]' in out


def test_find_latest_run_ranks_by_decisions_jsonl_mtime(tmp_path):
    old_run = tmp_path / "20260101T000000Z_followme"
    new_run = tmp_path / "20260102T000000Z_followme"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / "decisions.jsonl").write_text("{}\n", encoding="utf-8")
    (new_run / "decisions.jsonl").write_text("{}\n", encoding="utf-8")

    old_mtime = real_time.time() - 100
    os.utime(old_run / "decisions.jsonl", (old_mtime, old_mtime))

    result = tail_log.find_latest_run(str(tmp_path))
    assert result == str(new_run / "decisions.jsonl")


def test_find_latest_run_ignores_run_dirs_without_decisions_jsonl(tmp_path):
    incomplete_run = tmp_path / "20260101T000000Z_register"
    incomplete_run.mkdir()  # no decisions.jsonl inside — e.g. a crashed run before start() finished
    complete_run = tmp_path / "20260102T000000Z_register"
    complete_run.mkdir()
    (complete_run / "decisions.jsonl").write_text("{}\n", encoding="utf-8")

    result = tail_log.find_latest_run(str(tmp_path))
    assert result == str(complete_run / "decisions.jsonl")


def test_find_latest_run_returns_none_when_no_runs_exist(tmp_path):
    assert tail_log.find_latest_run(str(tmp_path / "does_not_exist")) is None
    empty = tmp_path / "empty_runs"
    empty.mkdir()
    assert tail_log.find_latest_run(str(empty)) is None


def test_resolve_path_accepts_directory_form(tmp_path):
    run_dir = tmp_path / "20260101T000000Z_followme"
    run_dir.mkdir()
    assert tail_log._resolve_path(str(run_dir)) == os.path.join(str(run_dir), "decisions.jsonl")


def test_resolve_path_passes_through_direct_file_path(tmp_path):
    direct = str(tmp_path / "decisions.jsonl")
    assert tail_log._resolve_path(direct) == direct


def test_tail_file_missing_path_returns_error_not_exception(tmp_path, capsys):
    result = tail_log.tail_file(str(tmp_path / "nonexistent.jsonl"))
    assert result == 1
    assert "does not exist" in capsys.readouterr().err


def test_tail_file_shows_initial_lines_then_follows_new_ones(tmp_path, capsys, monkeypatch):
    """Deliberately does NOT monkeypatch time.sleep: tail_log.time IS the real, process-wide
    `time` module (not a private copy), so patching it globally would also affect any other
    thread in the process calling time.sleep() during this test's window — a real flakiness
    source this suite hit once already (a lingering DebugStreamServer handler thread from an
    earlier test, still polling via its own time.sleep() call, picked up the fake and desynced
    this test's call-count logic when run as part of the full suite, though never when run
    standalone — the tell-tale sign of cross-test global-state leakage). Uses tail_file()'s own
    `_max_idle_polls` test seam instead (module-private, not part of the CLI) plus a REAL
    background thread doing a real short sleep before appending — safe because nothing global is
    touched, only this test's own file and its own thread.
    """
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"ts": 1.0, "frame": 0}\n{"ts": 2.0, "frame": 1}\n', encoding="utf-8")
    monkeypatch.setattr(tail_log, "_POLL_INTERVAL_SECONDS", 0.02)  # safe: a private module constant, not shared global state

    def append_new_line():
        real_time.sleep(0.05)
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"ts": 3.0, "frame": 2}\n')

    writer = threading.Thread(target=append_new_line, daemon=True)
    writer.start()
    result = tail_log.tail_file(str(path), initial_lines=1, _max_idle_polls=50)  # up to ~1s budget
    writer.join(timeout=2.0)

    assert result == 0
    out = capsys.readouterr().out
    assert "frame=1" in out  # last 1 of the 2 pre-existing lines
    assert "frame=0" not in out  # initial_lines=1 — the older pre-existing line is NOT shown
    assert "frame=2" in out  # the newly appended line WAS picked up while following


def test_tail_file_zero_initial_lines_skips_existing_content(tmp_path, capsys, monkeypatch):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"ts": 1.0, "frame": 0}\n', encoding="utf-8")
    monkeypatch.setattr(tail_log, "_POLL_INTERVAL_SECONDS", 0.0)

    result = tail_log.tail_file(str(path), initial_lines=0, _max_idle_polls=1)

    assert result == 0
    assert "frame=0" not in capsys.readouterr().out


def test_tail_file_skips_malformed_line_without_crashing(tmp_path, capsys, monkeypatch):
    path = tmp_path / "decisions.jsonl"
    path.write_text("not valid json at all\n", encoding="utf-8")
    monkeypatch.setattr(tail_log, "_POLL_INTERVAL_SECONDS", 0.0)

    result = tail_log.tail_file(str(path), initial_lines=1, _max_idle_polls=1)

    assert result == 0
    assert "not valid json at all" in capsys.readouterr().out  # shown raw, not silently dropped
