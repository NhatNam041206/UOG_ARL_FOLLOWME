"""
Makes the vendored Autocar tree (modules/autocar/ — vinhh9608-byte/Autocar, commit 27ee33a, kept
completely unmodified, nothing added to that directory either) importable via their OWN internal
absolute imports (`import config`, `from detector.base import PoseDetector`,
`from utils.types import Detection`, etc.), exactly as their own scripts/enroll_person.py
bootstraps itself relative to ITS repo root. modules/autocar/ itself — not `modules/` — must be
on sys.path before anything imports `detector`, `tracker`, `identity`, `utils`, or `config`.

Lives here in followme_orchestrator, not inside modules/autocar/, so that vendored directory
never gains a single file beyond what the clone itself provided.

Safe because nothing else in this project ever does a bare `import config` / `import detector` /
etc. (confirmed: this project loads its own config via `config/thresholds.yaml` + PyYAML, never
as a Python package) — so there is no name collision with anything else that could end up in
sys.modules under these generic names.
"""
import sys
from pathlib import Path

_AUTOCAR_DIR = str(Path(__file__).resolve().parent.parent / "autocar")


def ensure_on_path() -> None:
    if _AUTOCAR_DIR not in sys.path:
        sys.path.insert(0, _AUTOCAR_DIR)
