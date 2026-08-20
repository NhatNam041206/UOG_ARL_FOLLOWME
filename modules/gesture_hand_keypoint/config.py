"""
Internal config loading for the gesture_hand_keypoint module. Not part of the public contract —
external callers use interface.py only.

Redesign: this module no longer detects waving via motion — pure hand-shape sequence
classification only (spec confirmed with the user). All prior motion-related keys
(motion_window_seconds, motion_min_samples, motion_min_direction_changes,
motion_direction_change_angle_deg, motion_min_displacement_px) and wrist_height_fraction are
gone, replaced by the keys below. Still independently tunable from Methods 1 and 3 — none share
config values.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

REQUIRED_KEYS = (
    "confidence_threshold",
    "min_fingers_extended_open",
    "min_fingers_curled_closed",
    "thumb_extension_ratio_threshold",
    "palm_height_fraction",
    "max_transition_gap_seconds",
    "confirmation_duration_seconds",
)


@dataclass
class GestureHandKeypointConfig:
    confidence_threshold: Optional[float] = None          # MediaPipe handedness/detection confidence floor
    min_fingers_extended_open: Optional[int] = None         # of 4 NON-THUMB fingers, how many extended = OPEN (thumb also required, checked separately)
    min_fingers_curled_closed: Optional[int] = None         # of 4 NON-THUMB fingers, how many curled = CLOSED (thumb state ignored — see hand_shape.py)
    thumb_extension_ratio_threshold: Optional[float] = None  # dist(thumb_tip,pinky_MCP)/hand_scale threshold — only gates OPEN
    palm_height_fraction: Optional[float] = None             # palm (wrist) must be in upper fraction of person bbox
    max_transition_gap_seconds: Optional[float] = None       # timeout between sequence transitions
    confirmation_duration_seconds: Optional[float] = None

    # Not a threshold — model bundle path, downloaded into this module's own models/ dir.
    model_path: str = "modules/gesture_hand_keypoint/models/hand_landmarker.task"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> GestureHandKeypointConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("gesture_hand_keypoint", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    model_path = section.get("model_path")
    if model_path:
        kwargs["model_path"] = model_path

    return GestureHandKeypointConfig(**kwargs)
