"""
Palm orientation — DEBUG METRIC ONLY, per the confirmed design (spec §3, option 1): this does
NOT gate is_waving in any way. It exists purely so palm-facing-camera-or-not is visible during
calibration/comparison against Methods 1 and 3, in case it turns out useful for a future
revision — that decision is explicitly deferred, not made here.

Heuristic: the 2D winding order (signed area / cross product) of wrist -> index_MCP -> pinky_MCP,
combined with MediaPipe's handedness label, approximates whether the palm or the back of the
hand faces the camera. This sign convention is a best-effort derivation, NOT independently
verified against ground truth — treat this value as a low-confidence hint to eyeball during
testing, not a validated measurement, until checked against real footage.
"""
from typing import Optional

import numpy as np

from .constants import INDEX_MCP, PINKY_MCP, WRIST


def palm_facing_camera_debug(landmarks_px: np.ndarray, handedness: Optional[str]) -> Optional[bool]:
    """Returns True/False, or None if handedness wasn't classified (can't resolve the sign
    convention without knowing which physical hand this is)."""
    if handedness not in ("Left", "Right"):
        return None

    wrist = landmarks_px[WRIST]
    v1 = landmarks_px[INDEX_MCP] - wrist
    v2 = landmarks_px[PINKY_MCP] - wrist
    cross_z = float(v1[0] * v2[1] - v1[1] * v2[0])

    # Image coordinates are y-down. For a "Right" hand with the palm facing the camera, the
    # wrist->index_MCP->pinky_MCP sweep is clockwise (cross_z > 0) in that convention; mirrored
    # for "Left". Unverified — see module docstring.
    if handedness == "Right":
        return cross_z > 0
    return cross_z < 0
