"""
Run-observability utility — cross-cutting, NOT a CV module (no model loading, no
own-instance-isolation concerns per docs/architecture.md rule #2/#3; this is infrastructure every
mode of main.py shares, same category as argparse itself).

Gives every main.py run a self-contained, scp-able folder:

    runs/<UTC_ISO8601>_<mode>/
        meta.json         # run manifest: git commit, argv, resolved thresholds.yaml snapshot,
                           # start/end timestamps, frame count, exit reason, video/stream info
        decisions.jsonl    # one structured JSON record per frame, caller-defined schema

See plans/10_debug_logging_observability.md for the full per-frame schema and design rationale.

Crash/disconnect safety: decisions.jsonl is flushed after every write and meta.json is rewritten
in full (not appended) on both start() and close() — so an SSH disconnect or `kill` mid-run still
leaves a readable, if incomplete, run folder (meta.json's exit_reason simply stays None and
end_ts stays None, which is itself the signal the run didn't exit cleanly).
"""
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

__all__ = ["RunLogger"]


def _git_info() -> Dict[str, Optional[Any]]:
    """Best-effort — returns (None, None) if git isn't available or this isn't a repo (e.g. a
    deployed copy on the Pi with the .git dir stripped), never raises."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        ).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"git_commit": None, "git_dirty": None}


class RunLogger:
    """One instance per pipeline run (mirrors this project's single-process, single-episode
    convention elsewhere) — NOT thread-safe, not meant to be shared across runs."""

    def __init__(self, log_root: str = "runs"):
        self._log_root = log_root
        self.run_dir: Optional[str] = None
        self._decisions_fh = None
        self._meta: Dict[str, Any] = {}

    def start(self, mode: str, argv: Sequence[str], source_desc: str,
              thresholds_config_path: str = "config/thresholds.yaml") -> str:
        """Creates runs/<UTC-timestamp>_<mode>/, opens decisions.jsonl for append, writes the
        initial meta.json (thresholds_snapshot is a full copy of the resolved YAML at this
        instant — not a live reference, so later edits to the file don't retroactively change
        what a past run's manifest says it ran under). Returns the run directory path."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{ts}_{mode}"
        self.run_dir = os.path.join(self._log_root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        thresholds_snapshot: Dict[str, Any] = {}
        if os.path.exists(thresholds_config_path):
            with open(thresholds_config_path, "r", encoding="utf-8") as f:
                thresholds_snapshot = yaml.safe_load(f) or {}

        self._meta = {
            "mode": mode,
            "source": source_desc,
            "argv": list(argv),
            "thresholds_snapshot": thresholds_snapshot,
            "start_ts": datetime.now(timezone.utc).timestamp(),
            "end_ts": None,
            "frame_count": 0,
            "exit_reason": None,
            "video_saved": False, "video_path": None, "video_fps": None, "video_resolution": None,
            "stream_enabled": False, "stream_url": None,
            **_git_info(),
        }
        self._write_meta()

        self._decisions_fh = open(os.path.join(self.run_dir, "decisions.jsonl"), "a", encoding="utf-8")
        return self.run_dir

    def log_frame(self, **fields: Any) -> None:
        """Appends one JSON record as a single line to decisions.jsonl. No fixed schema is
        enforced here (this class is a plain sink) — callers pass whatever keyword fields they
        want; plans/10_debug_logging_observability.md documents the recommended per-frame shape
        used by main.py."""
        if self._decisions_fh is None:
            raise RuntimeError("RunLogger.log_frame() called before start()")
        self._decisions_fh.write(json.dumps(fields, default=str) + "\n")
        self._decisions_fh.flush()

    def set_video_info(self, video_path: str, fps: float, resolution: Tuple[int, int]) -> None:
        """Called once, after opening the debug-video writer, so meta.json records what was
        actually saved without the caller needing a separate manifest field of its own."""
        self._meta["video_saved"] = True
        self._meta["video_path"] = os.path.basename(video_path)
        self._meta["video_fps"] = fps
        self._meta["video_resolution"] = list(resolution)
        self._write_meta()

    def set_stream_info(self, stream_url: str) -> None:
        self._meta["stream_enabled"] = True
        self._meta["stream_url"] = stream_url
        self._write_meta()

    def close(self, frame_count: int, exit_reason: str) -> None:
        """Finalizes meta.json (frame_count, exit_reason, end_ts) and closes decisions.jsonl.
        Safe to call at most once; safe to skip on a hard crash (start()'s meta.json already on
        disk is the fallback record)."""
        self._meta["frame_count"] = frame_count
        self._meta["exit_reason"] = exit_reason
        self._meta["end_ts"] = datetime.now(timezone.utc).timestamp()
        self._write_meta()
        if self._decisions_fh is not None:
            self._decisions_fh.close()
            self._decisions_fh = None

    def _write_meta(self) -> None:
        with open(os.path.join(self.run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, default=str)
