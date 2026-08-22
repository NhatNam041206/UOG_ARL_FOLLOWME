"""
Splits a person's bbox into a head region and a lower (body) region using the pose keypoints
the detector already returns - no separate face detector or extra model call. Used by
identity/target_lock.py to score head-appearance and lower-body-appearance separately.
"""
from typing import Optional, Tuple

import numpy as np

import config

# COCO-17 indices
_NOSE, _L_EYE, _R_EYE = 0, 1, 2
_L_SHOULDER, _R_SHOULDER = 5, 6


def is_face_visible(keypoints: Optional[np.ndarray], min_conf: float = None) -> bool:
    """Nose + at least one eye confidently detected - a cheap proxy for "facing roughly towards
    the camera" without running an actual face detector."""
    conf = config.REID_FACE_MIN_KEYPOINT_CONF if min_conf is None else min_conf
    if keypoints is None or keypoints.shape[0] <= _R_EYE:
        return False
    nose_ok = keypoints[_NOSE, 2] >= conf
    eye_ok = keypoints[_L_EYE, 2] >= conf or keypoints[_R_EYE, 2] >= conf
    return bool(nose_ok and eye_ok)


def _split_y(bbox: np.ndarray, keypoints: Optional[np.ndarray]) -> float:
    y1, y2 = float(bbox[1]), float(bbox[3])
    if keypoints is not None and keypoints.shape[0] > _R_SHOULDER:
        shoulder_ys = [keypoints[i, 1] for i in (_L_SHOULDER, _R_SHOULDER) if keypoints[i, 2] >= 0.3]
        if shoulder_ys:
            margin = max(2.0, (y2 - y1) * 0.03)
            return float(np.clip(np.mean(shoulder_ys), y1 + margin, y2 - margin))
    return y1 + (y2 - y1) * config.REID_HEAD_SPLIT_FALLBACK_FRACTION


def crop_head_lower(frame: np.ndarray, bbox: np.ndarray, keypoints: Optional[np.ndarray],
                     frame_w: int, frame_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (head_crop, lower_crop) - either may be a degenerate (empty) array for a
    tiny/edge-of-frame bbox; OSNetEmbedder.extract() already handles that (returns a zero vector)."""
    x1, y1, x2, y2 = bbox
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame_w, int(x2)), min(frame_h, int(y2))

    split_y = int(np.clip(_split_y(bbox, keypoints), y1c, y2c))

    head_crop = frame[y1c:split_y, x1c:x2c]
    lower_crop = frame[split_y:y2c, x1c:x2c]
    return head_crop, lower_crop
