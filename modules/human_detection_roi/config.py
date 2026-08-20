"""
Internal config loading for the human_detection_roi module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from .roi import DEFAULT_UPWARD_FRACTION, DEFAULT_WIDTH_FRACTION

# Both calibration-gated (fail-closed) — mirrors modules/emergency_stop and
# modules/wave_facing_gate's convention: uncalibrated degrades to person_found=False on every
# call, never a silent guessed default.
REQUIRED_KEYS = (
    "roi_expansion_factor",
    "detection_confidence_threshold",
)


@dataclass
class HumanDetectionROIConfig:
    roi_expansion_factor: Optional[float] = None
    detection_confidence_threshold: Optional[float] = None

    # Not calibration-gated — working defaults matching roi.py's original fixed ratios, exposed
    # as tunable overrides (per user request) rather than hardcoded constants.
    roi_upward_fraction: float = DEFAULT_UPWARD_FRACTION
    roi_width_fraction: float = DEFAULT_WIDTH_FRACTION

    # Not a threshold — this module's own standalone YOLO instance's weights path. Defaults to
    # the same yolo11n.onnx file every other module in this repo uses.
    yolo_model_path: str = "yolo11n.onnx"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> HumanDetectionROIConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("human_detection_roi", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    for optional_key in ("roi_upward_fraction", "roi_width_fraction"):
        value = section.get(optional_key)
        if value is not None:
            kwargs[optional_key] = value

    yolo_model_path = section.get("yolo_model_path")
    if yolo_model_path:
        kwargs["yolo_model_path"] = yolo_model_path

    return HumanDetectionROIConfig(**kwargs)
