"""
Log tailer — CLI 2 of plans/11_registration_interactive_console.md's two-CLI design (chunk 9).
CLI 1 (register_person.run_interactive(), or main.py's pretrigger/followme loops) is the only
writer to a run's decisions.jsonl; this file is a strictly read-only observer of it — see
plans/11 §3.1/§3.3.1 for why that split is enforced deliberately, not incidentally.

Works against ANY mode's decisions.jsonl (pretrigger/followme/register) — RunLogger.log_frame()
is schema-agnostic by design (run_logging.py), and different modes log genuinely different fields
(followme logs face_identity/gesture/tracking blocks, pretrigger logs a `people` list, register
logs stage/person_name/phase/...). This file must stay schema-agnostic too (plans/11 invariant
§3.3.6) — it pretty-prints whatever keys a record has, never hard-codes field names tied to one
specific mode.

Usage:
    python tail_log.py runs/20260827T120000Z_followme/decisions.jsonl
    python tail_log.py runs/20260827T120000Z_followme          # directory form, same file assumed
    python tail_log.py --latest                                 # auto-picks the most recently
                                                                  # active run under --log-dir
    python tail_log.py --latest --log-dir /home/pi/register_runs
    python tail_log.py --latest --lines 0                       # skip existing content, follow only
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

_DECISIONS_FILENAME = "decisions.jsonl"
_POLL_INTERVAL_SECONDS = 0.3


def format_record(record: dict) -> str:
    """Pretty-prints one decoded JSON record generically — no assumption about which keys
    exist, since that varies by mode (see this file's own docstring). `ts`, if present, becomes
    a leading HH:MM:SS.mmm timestamp prefix; every other top-level key is rendered as
    `key=value`, with dict/list values compacted to single-line JSON rather than Python's repr
    (keeps the output copy-pasteable back into a JSON tool if needed)."""
    ts = record.get("ts")
    prefix = ""
    if isinstance(ts, (int, float)):
        prefix = f"[{datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]}] "
    parts = []
    for key, value in record.items():
        if key == "ts":
            continue
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, separators=(",", ":"))
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return prefix + " ".join(parts)


def find_latest_run(log_root: str = "runs") -> Optional[str]:
    """Returns the path to the most recently ACTIVE run's decisions.jsonl under `log_root`, or
    None if there isn't one. Ranked by decisions.jsonl's own mtime, not the run directory's —
    RunLogger.log_frame() flushes after every write (run_logging.py), so the file's mtime tracks
    the run's actual last activity; the directory's mtime does NOT update on POSIX when a file
    inside it is merely appended to (only on create/delete/rename), so it would incorrectly rank
    an old, still-running session below a newer, already-finished one."""
    if not os.path.isdir(log_root):
        return None
    candidates = []
    for entry in os.listdir(log_root):
        decisions_path = os.path.join(log_root, entry, _DECISIONS_FILENAME)
        if os.path.isfile(decisions_path):
            candidates.append(decisions_path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_path(raw_path: str) -> str:
    """Accepts either a direct decisions.jsonl path or its containing run directory."""
    if os.path.isdir(raw_path):
        return os.path.join(raw_path, _DECISIONS_FILENAME)
    return raw_path


def tail_file(path: str, initial_lines: int = 10, _max_idle_polls: Optional[int] = None) -> int:
    """Prints the last `initial_lines` existing records (0 = skip straight to following), then
    polls for and prints new ones as they're appended — classic `tail -f`, JSON-aware. Returns a
    plain exit code; never raises on Ctrl+C (caught here, prints a clean message, same convention
    as every other entry point in this project — see register_person.run()'s own docstring for
    why that matters).

    `_max_idle_polls`: internal test-only seam, never exposed via the CLI — stop after this many
    CONSECUTIVE empty polls instead of running forever. Deliberately NOT implemented by
    monkeypatching `time.sleep` in tests: `tail_log.time` is the real, process-wide `time` module
    (not a private copy), so patching it globally would also affect any other thread in the
    process calling `time.sleep()` during the same window — a real flakiness source this project
    hit once already (see test_tail_log.py's own comment on this parameter for the full story).
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist.", file=sys.stderr)
        return 1

    print(f"Tailing {path} — Ctrl+C to stop.")
    idle_polls = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in (lines[-initial_lines:] if initial_lines > 0 else []):
                _print_line(line)
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    if _max_idle_polls is not None:
                        idle_polls += 1
                        if idle_polls >= _max_idle_polls:
                            break
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue
                idle_polls = 0
                _print_line(line)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    return 0


def _print_line(line: str) -> None:
    line = line.strip()
    if not line:
        return
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        print(line)  # a malformed/partial line (e.g. a torn read) — show it raw rather than drop it
        return
    print(format_record(record))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tail a run's decisions.jsonl live — plans/11_registration_interactive_console.md chunk 9."
    )
    parser.add_argument(
        "path", nargs="?",
        help="Path to a decisions.jsonl file, or its containing run directory. Omit when using --latest.",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Auto-pick the most recently active run under --log-dir instead of a specific path.",
    )
    parser.add_argument(
        "--log-dir", default="runs",
        help="Where to look for runs when using --latest (default: runs) — same directory main.py's "
             "own --log-dir/register_person.py's own --log-dir write into.",
    )
    parser.add_argument(
        "--lines", type=int, default=10,
        help="How many existing records to show before following new ones (default 10; 0 = skip "
             "existing content, follow only).",
    )
    args = parser.parse_args()

    if args.latest and args.path:
        parser.error("--latest and a positional path are mutually exclusive.")
    if not args.latest and not args.path:
        parser.error("Provide a path, or pass --latest.")

    if args.latest:
        path = find_latest_run(args.log_dir)
        if path is None:
            print(f"ERROR: no runs with a {_DECISIONS_FILENAME} found under '{args.log_dir}'.", file=sys.stderr)
            return 1
    else:
        path = _resolve_path(args.path)

    return tail_file(path, initial_lines=args.lines)


if __name__ == "__main__":
    sys.exit(main())
