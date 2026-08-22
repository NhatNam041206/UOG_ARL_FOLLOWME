"""
Internal config loading for the target_tracking module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

REQUIRED_KEYS = (
    "record_duration_seconds",
    "appearance_reverify_interval_seconds",
    "appearance_reverify_similarity_threshold",
    "track_loss_grace_period_seconds",
)


@dataclass
class TargetTrackingConfig:
    record_duration_seconds: Optional[float] = None
    appearance_reverify_interval_seconds: Optional[float] = None
    # Deliberately its OWN key, independent from appearance_verifier.similarity_threshold and
    # target_recovery.appearance_fallback_threshold (plans/05 §4) — never collapse these.
    appearance_reverify_similarity_threshold: Optional[float] = None
    track_loss_grace_period_seconds: Optional[float] = None

    # Not calibration-gated — working defaults, both confirmed with the user (see
    # docs/parameters.md for the design rationale behind each).
    min_recording_crops: int = 3
    appearance_reverify_consecutive_failures: int = 2

    # Not a threshold — this module's own independent YOLO weights path (own-instance isolation,
    # plans/06 §0.3). Same weights file every other module in this repo uses, but never a shared
    # live instance.
    yolo_model_path: str = "yolo11n.onnx"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> TargetTrackingConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("target_tracking", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    for optional_key in ("min_recording_crops", "appearance_reverify_consecutive_failures", "yolo_model_path"):
        value = section.get(optional_key)
        if value is not None:
            kwargs[optional_key] = value

    return TargetTrackingConfig(**kwargs)
