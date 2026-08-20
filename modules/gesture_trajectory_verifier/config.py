"""
Internal config loading for the gesture_trajectory_verifier module. Not part of the public
contract — external callers use interface.py only.

All calibration values here are independently tunable from Methods 1 and 2 — none are copied or
shared (spec §0.3/§6).
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# §6's YAML template lists 5 keys; `confidence_threshold` is added here beyond that literal
# list because §2.1's confidence-gating requirement (reuse the SAME PATTERN as Method 1: gate
# closed if any of wrist/elbow/shoulder is below threshold) is NOT itself one of the two
# stop-and-ask items (§2.3 resampling, §4.3 zero/one-reference handling) — it's already-settled
# spec behavior that simply needs a config knob to be implementable at all.
REQUIRED_KEYS = (
    "confidence_threshold",
    "trajectory_window_seconds",
    "min_samples_for_comparison",
    "resample_length",
    "similarity_threshold",
    "confirmation_duration_seconds",
)

# Minimum reference trajectories needed before evaluate() will attempt a real comparison (spec
# §4.3, confirmed with the user): 0 or 1 both count as "not ready" — a single reference offers
# no meaningful "best of set" comparison and calibrating a threshold against one sample isn't
# trustworthy. This is a fixed structural rule, not a per-project calibration value, so it lives
# here as a constant rather than in thresholds.yaml.
MIN_REFERENCE_COUNT = 2


@dataclass
class GestureTrajectoryVerifierConfig:
    confidence_threshold: Optional[float] = None
    trajectory_window_seconds: Optional[float] = None
    min_samples_for_comparison: Optional[int] = None
    resample_length: Optional[int] = None
    similarity_threshold: Optional[float] = None
    confirmation_duration_seconds: Optional[float] = None

    # Not thresholds — working defaults.
    movenet_tfhub_handle: str = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    reference_dir: str = "modules/gesture_trajectory_verifier/reference_trajectories"
    movenet_input_size: int = 192

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> GestureTrajectoryVerifierConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("gesture_trajectory_verifier", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    for optional_key in ("movenet_tfhub_handle", "reference_dir", "movenet_input_size"):
        value = section.get(optional_key)
        if value:
            kwargs[optional_key] = value

    return GestureTrajectoryVerifierConfig(**kwargs)
