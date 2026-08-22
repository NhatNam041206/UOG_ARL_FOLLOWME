"""
Bbox-only aspect-ratio helper for target re-identification (scripts/enroll_person.py) - captures
a person's bbox width/height ratio at enrollment time, independent of pose keypoints. No longer
consulted during live re-identify (identity/target_lock.py verifies face/head appearance only,
since body/clothing-derived signals break across an outfit change) - kept for the enrolled
profile's own bookkeeping.
"""
import numpy as np


def aspect_ratio_from_bbox(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return float(w / h)
