"""
Human Detection (ROI-Scoped) module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.human_detection_roi. Everything
else in this package (config.py, roi.py, detector.py, pipeline.py) is an internal implementation
detail and may change without notice.

Pipeline position (spec §1): second stage of the exploratory face-first Follow-Me pipeline —
Face (modules.face_identity) -> ROI derived from face bbox -> Human detection inside that ROI.

Given a face bbox that modules.face_identity already matched to a registered person, this module
finds that same person's full-body bbox — scoped to a region around the face (spec §2), not the
whole frame. Human detection is only ever triggered once a face has already been matched — it
does not run independently. It does NOT re-verify identity; that already happened upstream. If
the input face bbox wasn't actually a match, that's a face_identity concern, not this module's.

No persistent track_id (confirmed with the user): the ROI crop this module scans shifts every
frame (it follows wherever the matched face currently is), which is not a stable coordinate
frame for a tracker's motion model — ByteTrack-style persistence doesn't combine cleanly with a
per-frame-shifting crop, so this module stays a stateless, per-call, ROI-scoped detection.

Isolation (spec §0.3, confirmed with the user): a fresh, standalone YOLO instance — same
yolo11n.onnx weights file every other module in this repo uses, but no shared live state with
the teammate's OSNet pipeline or with modules.human_detection's own instance. Also does not
reach into modules.face_identity's internals — only its public bbox output is consumed here.

Fallback behavior (spec §2, confirmed): if nothing is found within the scoped ROI, this module
reports person_found=False. It does NOT silently retry against the full frame.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import HumanDetectionROIConfig, load_config
from .pipeline import HumanDetectionROIPipeline

__all__ = ["HumanDetectionResult", "evaluate", "configure"]


@dataclass
class HumanDetectionResult:
    person_found: bool
    person_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h), FULL FRAME pixel space
    detection_confidence: Optional[float]


_pipeline_singleton: Optional[HumanDetectionROIPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first evaluate() call. If never called, evaluate() lazily initializes on first use."""
    global _pipeline_singleton
    config: HumanDetectionROIConfig = load_config(thresholds_config_path)
    _pipeline_singleton = HumanDetectionROIPipeline(config)


def _get_pipeline() -> HumanDetectionROIPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def evaluate(frame: np.ndarray, matched_face_bbox: Tuple[int, int, int, int]) -> HumanDetectionResult:
    """
    Input: the full frame, PLUS the matched face bbox from modules.face_identity (only a face
    that already matched a registered person should reach this module).

    Runs person/body detection SCOPED to a region around matched_face_bbox, not the whole frame.
    """
    result = _get_pipeline().evaluate(frame, matched_face_bbox)
    return HumanDetectionResult(
        person_found=result.person_found,
        person_bbox=result.person_bbox,
        detection_confidence=result.detection_confidence,
    )
