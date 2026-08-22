"""
Initial track_id locking: given this frame's ByteTrack detections and the target bbox the caller
wants to lock onto (either the gesture-trigger's initial_person_bbox at start(), or a fresh
recovery-handed bbox at reset()), pick whichever detection is actually that target.

Independently reimplemented center-containment/closest-center disambiguation — the same
DISAMBIGUATION STYLE already used by modules.human_detection_roi's own
`_select_best_detection()` (own-code isolation convention, docs/architecture.md rule #2/#3 — not
imported from there, a fresh implementation of the same idea for a different purpose: picking
ONE detection to LOCK ONTO once, not re-selecting a body for a face every frame).
"""
from typing import Any, Dict, List, Optional, Tuple

BboxXYXY = Tuple[float, float, float, float]


def select_matching_detection(detections: List[Dict[str, Any]], target_bbox_xyxy: BboxXYXY) -> Optional[Dict[str, Any]]:
    """
    Prefer whichever detection's bbox actually CONTAINS target_bbox's center (among those,
    highest confidence); if none contain it, fall back to whichever detection's center is
    closest to target's center. Returns None if `detections` is empty.
    """
    tx1, ty1, tx2, ty2 = target_bbox_xyxy
    target_cx, target_cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0

    best = None
    best_score = None
    for det in detections:
        dx1, dy1, dx2, dy2 = det["bbox"]
        contains = (dx1 <= target_cx <= dx2) and (dy1 <= target_cy <= dy2)
        if contains:
            score = (1, det["confidence"])
        else:
            det_cx, det_cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            dist_sq = (det_cx - target_cx) ** 2 + (det_cy - target_cy) ** 2
            score = (0, -dist_sq)
        if best_score is None or score > best_score:
            best_score = score
            best = det
    return best
