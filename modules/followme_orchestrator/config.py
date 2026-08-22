"""
Internal config loading for the followme_orchestrator module — the camera.fov_degrees/lens_type/
focus_type additions (plans/08 §2, added to the EXISTING camera: section, not a duplicate block)
and the new steering: section (plans/08 §3/§4). Not part of the public contract — external
callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# fov_degrees gates SteeringController: without a real angle-conversion factor, horizontal_offset
# can never become a real steering angle — fail-closed, same convention as every other module in
# this project (should_move is forced False while actively tracking, until this is set).
# lens_type/focus_type are informational only (plans/08 §2) and never gate anything.
REQUIRED_KEYS = ("fov_degrees", "kp", "ki", "kd", "max_steering_angle_degrees")


@dataclass
class FollowMeOrchestratorConfig:
    fov_degrees: Optional[float] = None
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    max_steering_angle_degrees: Optional[float] = None

    # Not calibration-gated — informational hardware documentation only (plans/08 §2).
    lens_type: Optional[str] = None
    focus_type: Optional[str] = None

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> FollowMeOrchestratorConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    camera_section: Dict[str, Any] = thresholds.get("camera", {}) or {}
    steering_section: Dict[str, Any] = thresholds.get("steering", {}) or {}

    return FollowMeOrchestratorConfig(
        fov_degrees=camera_section.get("fov_degrees"),
        lens_type=camera_section.get("lens_type"),
        focus_type=camera_section.get("focus_type"),
        kp=steering_section.get("kp"),
        ki=steering_section.get("ki"),
        kd=steering_section.get("kd"),
        max_steering_angle_degrees=steering_section.get("max_steering_angle_degrees"),
    )
