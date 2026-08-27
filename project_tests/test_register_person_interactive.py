"""
Unit tests for register_person.run_interactive() (plans/11_registration_interactive_console.md
chunk 8, extended with the 'follow <name>' command). Drives it with a scripted sequence of
`input()` responses (unittest.mock), monkeypatches registration_data.list_people()/
get_status()/delete_person() and register_person.run() itself so no real camera/model is ever
touched. Run with:

    python -m pytest project_tests/test_register_person_interactive.py -v
"""
import json
import os
from unittest.mock import patch

import pytest

from scripts import register_person
from scripts import registration_data as data
from scripts.run_logging import RunLogger


def _person(name, front=15, back=15, ready=True):
    return data.PersonStatus(
        name=name, raw_front_count=front, raw_back_count=back,
        cropped_front_count=front, cropped_back_count=back,
        has_face_registry=ready, has_target_profile=ready,
    )


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_list_command_prints_people_and_logs(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(data, "list_people", lambda: [_person("Alice"), _person("Bob", ready=False)])
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")

    with patch("builtins.input", side_effect=["list", "quit"]):
        result, chosen_name = register_person.run_interactive(logger=logger)

    assert result == 0
    assert chosen_name is None
    out = capsys.readouterr().out
    assert "Alice" in out and "Bob" in out

    records = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    assert len(records) == 1
    assert records[0]["stage"] == "list"
    assert records[0]["people_count"] == 2


def test_list_command_with_no_people(monkeypatch, capsys):
    monkeypatch.setattr(data, "list_people", lambda: [])
    with patch("builtins.input", side_effect=["list", "quit"]):
        result, chosen_name = register_person.run_interactive()
    assert result == 0
    assert chosen_name is None
    assert "no one registered" in capsys.readouterr().out


def test_register_command_calls_run_and_reports_ok(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_run(name, camera_index, front_samples, back_samples, config_path,
                 show, stream, logger, close_logger_on_exit):
        calls.append((name, close_logger_on_exit))
        return 0

    monkeypatch.setattr(register_person, "run", fake_run)
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")

    with patch("builtins.input", side_effect=["register Alice", "quit"]):
        result, chosen_name = register_person.run_interactive(logger=logger)

    assert result == 0
    assert chosen_name is None  # 'register' alone does NOT select anyone — only 'follow' does
    assert calls == [("Alice", False)]  # close_logger_on_exit=False — REPL owns closing, not run()
    assert "OK" in capsys.readouterr().out


def test_register_command_reports_failed(monkeypatch, capsys):
    monkeypatch.setattr(register_person, "run", lambda *a, **k: 1)
    with patch("builtins.input", side_effect=["register Bob", "quit"]):
        register_person.run_interactive()
    assert "FAILED" in capsys.readouterr().out


def test_delete_confirmed_calls_delete_person(monkeypatch, capsys, tmp_path):
    deleted = []
    monkeypatch.setattr(data, "delete_person", lambda name: deleted.append(name))
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")

    with patch("builtins.input", side_effect=["delete Alice", "y", "quit"]):
        register_person.run_interactive(logger=logger)

    assert deleted == ["Alice"]
    assert "deleted" in capsys.readouterr().out
    records = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    assert records[0]["stage"] == "delete"
    assert records[0]["person_name"] == "Alice"
    assert records[0]["confirmed"] is True


def test_delete_declined_does_not_call_delete_person(monkeypatch, capsys):
    deleted = []
    monkeypatch.setattr(data, "delete_person", lambda name: deleted.append(name))

    with patch("builtins.input", side_effect=["delete Alice", "n", "quit"]):
        register_person.run_interactive()

    assert deleted == []
    assert "cancelled" in capsys.readouterr().out


def test_follow_command_selects_ready_person_and_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(data, "get_status", lambda name: _person(name, ready=True))
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")

    # No 'quit' needed — 'follow' on a ready person exits the loop itself.
    with patch("builtins.input", side_effect=["follow Alice"]):
        result, chosen_name = register_person.run_interactive(logger=logger)

    assert result == 0
    assert chosen_name == "Alice"
    assert "Selected 'Alice'" in capsys.readouterr().out

    records = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    assert records[0]["stage"] == "follow"
    assert records[0]["person_name"] == "Alice"
    assert records[0]["accepted"] is True


def test_follow_command_rejects_not_ready_person_and_stays_in_console(monkeypatch, capsys):
    monkeypatch.setattr(data, "get_status", lambda name: _person(name, ready=False))

    with patch("builtins.input", side_effect=["follow Alice", "quit"]):
        result, chosen_name = register_person.run_interactive()

    assert result == 0
    assert chosen_name is None  # rejected — console did NOT exit on the 'follow' line itself
    assert "isn't fully registered yet" in capsys.readouterr().out


def test_unknown_command_shows_hint(capsys):
    with patch("builtins.input", side_effect=["frobnicate", "quit"]):
        result, chosen_name = register_person.run_interactive()
    assert result == 0
    assert chosen_name is None
    assert "Unknown command" in capsys.readouterr().out


def test_eof_exits_cleanly(tmp_path):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")
    with patch("builtins.input", side_effect=EOFError):
        result, chosen_name = register_person.run_interactive(logger=logger)
    assert result == 0
    assert chosen_name is None
    assert logger._decisions_fh is None  # closed, not left dangling


def test_keyboard_interrupt_exits_cleanly_not_as_traceback(capsys, tmp_path):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result, chosen_name = register_person.run_interactive(logger=logger)  # must NOT raise
    assert result == 0
    assert chosen_name is None
    assert "Interrupted" in capsys.readouterr().out
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["exit_reason"] == "user_quit"


def test_quit_sets_user_quit_exit_reason(tmp_path):
    logger = RunLogger(log_root=str(tmp_path / "runs"))
    run_dir = logger.start("register", ["--interactive"], "camera:0", "config/thresholds.yaml")
    with patch("builtins.input", side_effect=["quit"]):
        register_person.run_interactive(logger=logger)
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["exit_reason"] == "user_quit"
