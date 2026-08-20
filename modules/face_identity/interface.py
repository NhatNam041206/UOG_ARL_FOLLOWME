"""
Face Detection + Face Matching module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.face_identity. Everything else in
this package (config.py, constants.py, face_detector.py, alignment.py, embedder.py, matching.py,
registry.py, pipeline.py) is an internal implementation detail and may change without notice.

Pipeline position (spec §1): FIRST stage of a standalone, exploratory face-first Follow-Me
pipeline — full frame -> [THIS MODULE] -> human detection/ROI -> gesture method -> is_waving.

Isolation (spec §0.3, confirmed with the user): this module and its registry are fully
independent of the teammate's OSNet-based Re-ID/tracking pipeline (a separate git repository at
UOG_ARL_FOLLOWME) — no code or live state is shared with it, only the .npz-per-person storage
FORMAT was mirrored (see registry.py's docstring). This module also does not share state with
any gesture-detection module (modules.wave_facing_gate or the Method 2/3 modules) — it only
produces FaceIdentityResult and knows nothing about waving or Stage 2 trigger logic.

Known limitation, not solved here (spec §3): a single frame's match is NOT debounced/confirmed
over time — that temporal confirmation, if needed, is the calling pipeline's job.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import FaceIdentityConfig, load_config
from .pipeline import FaceIdentityPipeline
from .registry import FaceRegistry, RegistryEntry, sanitize_person_name

__all__ = ["FaceIdentityResult", "FaceRegistry", "RegistryEntry", "sanitize_person_name", "evaluate", "configure"]


@dataclass
class FaceIdentityResult:
    face_found: bool
    face_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) in FULL FRAME pixel space
    is_registered_match: bool
    matched_person_name: Optional[str]
    match_confidence: Optional[float]          # embedding similarity score, for debugging/calibration
    face_detection_confidence: Optional[float]


_pipeline_singleton: Optional[FaceIdentityPipeline] = None
_pipeline_config_path: Optional[str] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """
    Optional: (re)initialize the module-level pipeline from a specific config path before the
    first evaluate() call. If never called, evaluate() lazily initializes from the default
    "config/thresholds.yaml" on first use. Mainly useful for tests pointing at a fixture config.
    """
    global _pipeline_singleton, _pipeline_config_path
    config: FaceIdentityConfig = load_config(thresholds_config_path)
    _pipeline_singleton = FaceIdentityPipeline(config)
    _pipeline_config_path = thresholds_config_path


def _get_pipeline() -> FaceIdentityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def evaluate(frame: np.ndarray, registry: FaceRegistry) -> List[FaceIdentityResult]:
    """
    Input: the FULL raw frame (not a crop — this module owns face detection from scratch).

    Returns a list because a frame may contain zero, one, or multiple faces. The caller decides
    what to do with multiple matches — this module reports everything it found, it does not pick
    "the" person.
    """
    results = _get_pipeline().evaluate(frame, registry)
    return [
        FaceIdentityResult(
            face_found=r.face_found,
            face_bbox=r.face_bbox,
            is_registered_match=r.is_registered_match,
            matched_person_name=r.matched_person_name,
            match_confidence=r.match_confidence,
            face_detection_confidence=r.face_detection_confidence,
        )
        for r in results
    ]
