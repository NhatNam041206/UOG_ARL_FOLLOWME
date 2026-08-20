"""
Internal config loading for the emergency_stop module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# Every key here MUST be calibrated (non-null) before the module will ever output anything but
# UNCERTAIN. Values start as None (matching config/thresholds.yaml's placeholders) so a missing
# or not-yet-calibrated key is trivially detectable, never silently defaulted.
REQUIRED_KEYS = (
    "runway_left_line",
    "runway_right_line",
    "roi_buffer_px",
    "size_prefilter_width_px",
    "size_prefilter_height_px",
    "zone_far_boundary",
    "zone_mid_boundary",
    "t_mid_seconds",
    "min_detection_confidence",
    "resume_buffer_seconds",
)


@dataclass
class EStopConfig:
    runway_left_line: Optional[List[List[float]]] = None
    runway_right_line: Optional[List[List[float]]] = None
    roi_buffer_px: Optional[int] = None
    size_prefilter_width_px: Optional[float] = None
    size_prefilter_height_px: Optional[float] = None
    zone_far_boundary: Optional[float] = None
    zone_mid_boundary: Optional[float] = None
    t_mid_seconds: Optional[float] = None
    min_detection_confidence: Optional[float] = None
    resume_buffer_seconds: Optional[float] = None

    # Not a threshold — the detector's own model weights path.
    yolo_model_path: str = "yolo11n.onnx"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> EStopConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("emergency_stop", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    yolo_model_path = section.get("yolo_model_path")
    if yolo_model_path:
        kwargs["yolo_model_path"] = yolo_model_path

    return EStopConfig(**kwargs)
