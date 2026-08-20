"""
is_facing_camera raw check (spec §3): a crude four-keypoint visibility proxy, not head-pose
estimation. Per-frame, stateless — the confirmation state machine (confirmation.py) is what
turns a passing frame into a held True.

Known limitation, intentionally not solved here (spec §3): cannot distinguish "facing camera"
from "facing camera at a steep up/down angle" or a partial occlusion that happens to still clear
the confidence floor. Acceptable for MVP.
"""
from typing import List, Tuple

from .config import WaveFacingConfig
from .constants import Keypoint, LEFT_EYE, LEFT_SHOULDER, RIGHT_EYE, RIGHT_SHOULDER

_REQUIRED = (LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER)


def facing_camera_raw(keypoints: List[Keypoint], config: WaveFacingConfig) -> Tuple[bool, float]:
    scores = [keypoints[i].score for i in _REQUIRED]
    min_score = min(scores)
    return min_score >= config.confidence_threshold_facing, min_score
