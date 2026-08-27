import colorsys
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.types import TrackedObject


def id_to_color(track_id: int):
    """Deterministic, well-spread BGR color per track_id (golden-ratio hue step)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


_TARGET_COLOR = (0, 255, 0)     # locked target - bright green, thicker box
_OTHER_COLOR_DIM = (0, 0, 255)  # candidates while still searching for a target - dim red, thin box


def _draw_box(frame: np.ndarray, t: TrackedObject, color, thickness: int, label: str) -> None:
    x1, y1, x2, y2 = t.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_tracks(frame: np.ndarray, tracks: List[TrackedObject], target_id: Optional[int] = None,
                 target_mode: bool = False, target_confidence: Optional[float] = None,
                 candidate_scores: Optional[Dict[int, float]] = None) -> None:
    """target_mode=False (no --target given): generic per-id-colored box for every track,
    unchanged from the original multi-person behavior. The number shown is YOLO's person-
    detection confidence - there's no enrolled profile loaded in this mode.

    target_mode=True (main.py --target): every number shown is re-identify confidence (% match
    against the enrolled profile), never detection confidence.
      - target_id given (locked this frame): draw ONLY that one track (green, thicker, "TARGET"
        label) - every other detected person is skipped entirely, not drawn at all.
        target_confidence (identity.TargetLock.last_verify_score - the most recent real check's
        score) is shown as a percentage; falls back to the detection score if not given.
      - target_id is None (still searching): every candidate is dimmed (red, thin).
        candidate_scores (identity.TargetLock.candidate_scores - each track_id's RUNNING AVERAGE
        similarity-to-target across this acquisition cycle's rounds so far, the same average that
        decides who locks in) supplies the number; a track with no sample yet shows "no face"
        instead of a number.
    """
    if not target_mode:
        for t in tracks:
            _draw_box(frame, t, id_to_color(t.track_id), 2, f"ID {t.track_id} ({t.score:.2f})")
        return

    if target_id is not None:
        target = next((t for t in tracks if t.track_id == target_id), None)
        if target is not None:
            if target_confidence is not None:
                label = f"TARGET ID {target.track_id} ({target_confidence * 100:.0f}%)"
            else:
                label = f"TARGET ID {target.track_id} ({target.score:.2f})"
            _draw_box(frame, target, _TARGET_COLOR, 3, label)
        return

    candidate_scores = candidate_scores or {}
    for t in tracks:
        score = candidate_scores.get(t.track_id)
        label = f"ID {t.track_id} ({score * 100:.0f}%)" if score is not None else f"ID {t.track_id} (no face)"
        _draw_box(frame, t, _OTHER_COLOR_DIM, 1, label)
