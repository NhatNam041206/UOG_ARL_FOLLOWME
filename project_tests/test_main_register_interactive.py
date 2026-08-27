"""
Unit tests for main.py's --interactive dispatch and validation (plans/11_registration_
interactive_console.md chunk 10, extended with the 'follow <name>' / auto-select-interactive
fixes) — monkeypatches sys.argv and register_person.run_interactive so no real camera/model is
ever touched. Run with:

    python -m pytest project_tests/test_main_register_interactive.py -v
"""
import sys

import pytest

import main
from scripts import register_person


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"] + argv)
    return main.main()


class _FakeOpenedCap:
    """Just enough of cv2.VideoCapture's interface for main()'s fall-through path to proceed
    past `if not cap.isOpened(): return 1` without a real camera."""

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


def test_interactive_and_person_name_are_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _run_main(["--modules", "register", "--interactive", "--person-name", "Nam"], monkeypatch)
    assert exc_info.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_interactive_and_then_followme_are_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _run_main(["--modules", "register", "--interactive", "--then-followme"], monkeypatch)
    assert exc_info.value.code == 2
    assert "--then-followme has no meaning with --interactive" in capsys.readouterr().err


def test_stream_alone_auto_selects_interactive_instead_of_erroring(monkeypatch, capsys, tmp_path):
    """Behavior change: --stream with --modules register and neither --person-name nor
    --interactive used to be a hard parser.error(). Now it auto-selects --interactive (the
    strictly more capable of the two headless-capable register paths) instead of forcing the
    operator to type --interactive by hand — but still says so out loud, not silently."""
    calls = []

    def fake_run_interactive(camera_index, front_samples, back_samples, config_path, stream, logger):
        calls.append(stream is not None)
        return 0, None

    monkeypatch.setattr(register_person, "run_interactive", fake_run_interactive)

    result = _run_main(["--stream", "--log-dir", str(tmp_path / "runs")], monkeypatch)

    assert result == 0
    assert calls == [True]  # actually reached run_interactive() with a real stream, not rejected
    assert "opening the interactive console" in capsys.readouterr().out


def test_stream_with_explicit_interactive_is_accepted(monkeypatch, tmp_path):
    calls = {}

    def fake_run_interactive(camera_index, front_samples, back_samples, config_path, stream, logger):
        calls["stream_is_none"] = stream is None
        calls["logger_is_none"] = logger is None
        return 0, None

    monkeypatch.setattr(register_person, "run_interactive", fake_run_interactive)
    result = _run_main(
        ["--modules", "register", "--interactive", "--stream", "--log-dir", str(tmp_path / "runs")],
        monkeypatch,
    )
    assert result == 0
    assert calls["stream_is_none"] is False  # --stream was passed, so a real DebugStreamServer was created
    assert calls["logger_is_none"] is False  # a RunLogger is always created for this path


def test_interactive_dispatch_calls_run_interactive_and_returns_its_result(monkeypatch, tmp_path):
    calls = []

    def fake_run_interactive(camera_index, front_samples, back_samples, config_path, stream, logger):
        calls.append((camera_index, front_samples, back_samples, config_path))
        return 7, None  # distinctive return value to prove it's actually propagated, not hardcoded to 0

    monkeypatch.setattr(register_person, "run_interactive", fake_run_interactive)
    result = _run_main(
        ["--modules", "register", "--interactive", "--camera-index", "2",
         "--front-samples", "5", "--back-samples", "6", "--log-dir", str(tmp_path / "runs")],
        monkeypatch,
    )

    assert result == 7
    assert calls == [(2, 5, 6, "config/thresholds.yaml")]


def test_interactive_dispatch_does_not_fall_through_when_nobody_chosen(monkeypatch, tmp_path):
    """No 'follow <name>' used (chosen_name=None) — must return directly, never reach the
    args.mode='camera'; args.modules='followme' fall-through the other two register paths share."""
    monkeypatch.setattr(register_person, "run_interactive", lambda *a, **k: (0, None))
    # If this fell through, main() would next require --mode (pretrigger/followme validation) or
    # try to open a camera for followme — neither of which this test sets up, so a fall-through
    # would raise/fail loudly rather than cleanly returning 0.
    result = _run_main(
        ["--modules", "register", "--interactive", "--log-dir", str(tmp_path / "runs")], monkeypatch,
    )
    assert result == 0


def test_interactive_dispatch_falls_through_to_followme_when_follow_used(monkeypatch, capsys, tmp_path):
    """The actual fix this test file exists to prove: choosing 'follow <name>' inside the console
    must reach the SAME followme camera loop the --person-name/Tkinter paths already fall through
    to — not just return chosen_name and stop."""
    monkeypatch.setattr(register_person, "run_interactive", lambda *a, **k: (0, "Alice"))
    monkeypatch.setattr(main, "open_capture", lambda args: (_FakeOpenedCap(), "camera:0"))

    followme_calls = []

    def fake_run_followme_pipeline(cap, args, source_desc, logger, stream=None):
        followme_calls.append(True)
        return 0

    monkeypatch.setattr(main, "run_followme_pipeline", fake_run_followme_pipeline)

    result = _run_main(
        ["--modules", "register", "--interactive", "--log-dir", str(tmp_path / "runs")], monkeypatch,
    )

    assert result == 0
    assert followme_calls == [True]  # confirms the fall-through into followme mode actually ran
    assert "'Alice' selected" in capsys.readouterr().out
