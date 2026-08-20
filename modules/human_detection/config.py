"""
Internal config loading for the human_detection module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class HumanDetectionConfig:
    # Unlike modules/emergency_stop and modules/wave_facing_gate, this module has no spec
    # demanding a fail-closed-until-calibrated confidence floor (it's not a safety layer, and no
    # empirical-calibration requirement was placed on it) — a standard, working default is
    # sufficient, tunable via thresholds.yaml if it needs adjusting for a given camera/distance.
    confidence_threshold: float = 0.5

    # Not a threshold — this module's own standalone YOLO instance's weights path. Defaults to
    # yolo11n (nano), the fastest variant in the YOLO11 family, per the fast-inference requirement.
    yolo_model_path: str = "yolo11n.onnx"


def load_config(thresholds_path: str = "config/thresholds.yaml") -> HumanDetectionConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("human_detection", {}) or {}

    kwargs = {}
    confidence_threshold = section.get("confidence_threshold")
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold

    yolo_model_path = section.get("yolo_model_path")
    if yolo_model_path:
        kwargs["yolo_model_path"] = yolo_model_path

    return HumanDetectionConfig(**kwargs)
