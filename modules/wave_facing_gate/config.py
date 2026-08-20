"""
Internal config loading for the wave_facing_gate module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# Every key here MUST be calibrated (non-null) before the module will ever report is_waving or
# is_facing_camera as True. Values start as None (matching config/thresholds.yaml's placeholders)
# so a missing or not-yet-calibrated key is trivially detectable, never silently defaulted —
# mirrors modules/emergency_stop/config.py's REQUIRED_KEYS pattern (spec §8: "no threshold in
# this file is a final value").
REQUIRED_KEYS = (
    "confidence_threshold_facing",
    "confidence_threshold_pose",
    "wrist_height_fraction",
    "verticality_threshold_deg",
    "motion_window_seconds",
    "motion_confidence_threshold",
    "motion_min_samples",
    "motion_min_direction_changes",
    "motion_direction_change_angle_deg",
    "motion_min_displacement_px",
    "confirmation_duration_seconds",
)


@dataclass
class WaveFacingConfig:
    confidence_threshold_facing: Optional[float] = None
    confidence_threshold_pose: Optional[float] = None
    wrist_height_fraction: Optional[float] = None
    verticality_threshold_deg: Optional[float] = None
    motion_window_seconds: Optional[float] = None
    motion_confidence_threshold: Optional[float] = None
    motion_min_samples: Optional[int] = None
    motion_min_direction_changes: Optional[int] = None
    motion_direction_change_angle_deg: Optional[float] = None
    motion_min_displacement_px: Optional[float] = None
    confirmation_duration_seconds: Optional[float] = None

    # Fixed by the model choice (spec §2), not a calibration target.
    movenet_input_size: int = 192

    # Not a threshold — where to load MoveNet Lightning from. Defaults to the official TF Hub
    # handle (auto-downloads and caches weights on first use); override to point at a local
    # SavedModel directory instead.
    movenet_tfhub_handle: str = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> WaveFacingConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("wave_facing", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    movenet_input_size = section.get("movenet_input_size")
    if movenet_input_size:
        kwargs["movenet_input_size"] = movenet_input_size

    movenet_tfhub_handle = section.get("movenet_tfhub_handle")
    if movenet_tfhub_handle:
        kwargs["movenet_tfhub_handle"] = movenet_tfhub_handle

    return WaveFacingConfig(**kwargs)
