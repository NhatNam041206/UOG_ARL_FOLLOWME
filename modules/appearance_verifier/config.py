"""
Internal config loading for the appearance_verifier module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# similarity_threshold is ESPECIALLY uncalibrated (plans/05_appearance_verifier.md §2) — OSNet
# appearance matching has two documented accuracy risks (similar-clothing confusion, cross-domain
# generalization drop) that make a "starting guess" here less trustworthy than usual elsewhere in
# this project. Same fail-closed convention as every other module: uncalibrated -> match_found
# stays False on every call.
REQUIRED_KEYS = ("similarity_threshold",)


@dataclass
class AppearanceVerifierConfig:
    similarity_threshold: Optional[float] = None

    # Not a threshold — which torchreid OSNet variant to build. osnet_x1_0 is the standard,
    # full-size variant (confirmed with the user: real Market1501-pretrained weights, fetched
    # automatically by torchreid on first use — see embedder.py).
    osnet_model_name: str = "osnet_x1_0"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> AppearanceVerifierConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("appearance_verifier", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    osnet_model_name = section.get("osnet_model_name")
    if osnet_model_name:
        kwargs["osnet_model_name"] = osnet_model_name

    return AppearanceVerifierConfig(**kwargs)
