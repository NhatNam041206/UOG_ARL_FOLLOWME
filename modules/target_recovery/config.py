"""
Internal config loading for the target_recovery module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

REQUIRED_KEYS = (
    "face_search_grace_attempts",
    "appearance_fallback_threshold",
    "search_timeout_seconds",
)


@dataclass
class TargetRecoveryConfig:
    # A COUNT of consecutive Path-A-failure frames, NOT a time duration — deliberate (plans/07
    # §4.2): face detection is variable-cost inference, so a time-based gate would give Path A an
    # inconsistent number of real attempts depending on system load. Do not change to a duration.
    face_search_grace_attempts: Optional[int] = None
    # Deliberately its OWN key, independent from appearance_verifier.similarity_threshold and
    # target_tracking.appearance_reverify_similarity_threshold (plans/05 §4) — never collapse.
    appearance_fallback_threshold: Optional[float] = None
    search_timeout_seconds: Optional[float] = None

    # Not a threshold — Path B's own independent YOLO weights path (own-instance isolation,
    # confirmed with the user per plans/07 §4.3's stated default: a fresh instance, not reusing
    # modules.human_detection's).
    yolo_model_path: str = "yolo11n.onnx"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> TargetRecoveryConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("target_recovery", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    yolo_model_path = section.get("yolo_model_path")
    if yolo_model_path:
        kwargs["yolo_model_path"] = yolo_model_path

    return TargetRecoveryConfig(**kwargs)
