"""
Internal config loading for the face_identity module. Not part of the public contract —
external callers use interface.py only.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# Both calibration thresholds must be non-null before evaluate() will ever report
# is_registered_match=True — mirrors modules/emergency_stop and modules/wave_facing_gate's
# fail-closed convention (uncalibrated degrades to the safe/negative state, never a silent
# default). face_found/face_bbox still work while uncalibrated (needed so the mandatory
# visualization tool, spec §5, is usable before calibration) — only the match verdict is gated.
REQUIRED_KEYS = (
    "similarity_threshold_face_match",
    "face_detection_confidence_threshold",
)


@dataclass
class FaceIdentityConfig:
    similarity_threshold_face_match: Optional[float] = None
    face_detection_confidence_threshold: Optional[float] = None

    # Not thresholds — working defaults pointing at the model files downloaded into this
    # module's own models/ directory (spec §2.1 YuNet, §2.2 EdgeFace-XS).
    yunet_model_path: str = "modules/face_identity/models/face_detection_yunet_2023mar.onnx"
    edgeface_model_path: str = "modules/face_identity/models/edgeface_xs_gamma_06.onnx"

    # Not a threshold — where registered people's .npz entries live (mirrors, in format only,
    # the sibling OSNet project's logs/registry/ .npz-per-person convention — see registry.py's
    # docstring for the isolation note).
    registry_dir: str = "modules/face_identity/registry_data"

    # Not a threshold — where Phase 1 raw capture images live (one subfolder per person), read
    # by Phase 2 to build the registry. See capture_face_images.py / build_face_registry.py.
    raw_captures_dir: str = "modules/face_identity/raw_captures"

    def missing_keys(self) -> List[str]:
        return [k for k in REQUIRED_KEYS if getattr(self, k) is None]


def load_config(thresholds_path: str = "config/thresholds.yaml") -> FaceIdentityConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("face_identity", {}) or {}

    kwargs = {k: section.get(k) for k in REQUIRED_KEYS}

    for optional_key in ("yunet_model_path", "edgeface_model_path", "registry_dir", "raw_captures_dir"):
        value = section.get(optional_key)
        if value:
            kwargs[optional_key] = value

    return FaceIdentityConfig(**kwargs)
