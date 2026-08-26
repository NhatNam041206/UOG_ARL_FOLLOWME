"""
Splits a person's bbox into a head region and a lower (body) region using the pose keypoints
the detector already returns - no extra model call needed just for the split. Used by
identity/target_lock.py (and scripts/enroll_person.py) to isolate the head region before running
real face detection (identity/face_recognizer.py) or the back-of-head appearance model
(identity/osnet_embedder.py) on it.
"""
from typing import Optional, Tuple

import numpy as np

import config

# COCO-17 indices
_L_SHOULDER, _R_SHOULDER = 5, 6


def _split_y(bbox: np.ndarray, keypoints: Optional[np.ndarray]) -> float:
    y1, y2 = float(bbox[1]), float(bbox[3])
    if keypoints is not None and keypoints.shape[0] > _R_SHOULDER:
        shoulder_ys = [keypoints[i, 1] for i in (_L_SHOULDER, _R_SHOULDER) if keypoints[i, 2] >= 0.3]
        if shoulder_ys:
            margin = max(2.0, (y2 - y1) * 0.03)
            return float(np.clip(np.mean(shoulder_ys), y1 + margin, y2 - margin))
    return y1 + (y2 - y1) * config.REID_HEAD_SPLIT_FALLBACK_FRACTION


def _head_x_range(bbox: np.ndarray, keypoints: Optional[np.ndarray]) -> Tuple[float, float]:
    """Narrows the head crop's horizontal extent to roughly head width (a fraction of the
    shoulder-to-shoulder distance, centered between the shoulders) instead of the full bbox width
    (~shoulder width). Matters most for the BACK-of-head OSNet path (identity/target_lock.py),
    which embeds this crop directly and has no face detector to ignore the crop's edges the way
    the FRONT path does - without narrowing, the bottom corners of even a "head-only" crop include
    collar/shoulder clothing, letting outfit changes corrupt back-of-head matches the same way
    they used to corrupt front-face ones. Falls back to the full bbox width if shoulders aren't
    confidently detected."""
    x1, x2 = float(bbox[0]), float(bbox[2])
    if keypoints is not None and keypoints.shape[0] > _R_SHOULDER:
        l_conf, r_conf = keypoints[_L_SHOULDER, 2], keypoints[_R_SHOULDER, 2]
        if l_conf >= 0.3 and r_conf >= 0.3:
            l_x, r_x = float(keypoints[_L_SHOULDER, 0]), float(keypoints[_R_SHOULDER, 0])
            center = (l_x + r_x) / 2.0
            half_width = max(abs(r_x - l_x) * config.REID_HEAD_CROP_WIDTH_FRACTION / 2.0, 10.0)
            return center - half_width, center + half_width
    return x1, x2


def crop_head_lower(frame: np.ndarray, bbox: np.ndarray, keypoints: Optional[np.ndarray],
                     frame_w: int, frame_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (head_crop, lower_crop) - either may be a degenerate (empty) array for a
    tiny/edge-of-frame bbox; OSNetEmbedder.extract() already handles that (returns a zero vector).
    head_crop is narrowed to roughly head width (see _head_x_range); lower_crop still spans the
    full bbox width (unused by current matching, kept only for the profile format's sake)."""
    x1, y1, x2, y2 = bbox
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame_w, int(x2)), min(frame_h, int(y2))

    split_y = int(np.clip(_split_y(bbox, keypoints), y1c, y2c))

    hx1, hx2 = _head_x_range(bbox, keypoints)
    hx1c = max(0, int(np.clip(hx1, x1c, x2c)))
    hx2c = min(frame_w, int(np.clip(hx2, x1c, x2c)))

    head_crop = frame[y1c:split_y, hx1c:hx2c]
    lower_crop = frame[split_y:y2c, x1c:x2c]
    return head_crop, lower_crop
