"""
Per-call orchestrator: matched_face_bbox -> compute ROI -> crop -> detect persons within ROI ->
pick the one detection that's actually this person -> convert back to full-frame coords. Not
part of the public contract — external callers use interface.py only.

FLOW: Face -> ROI (derived from face bbox size * roi_expansion_factor) -> Human detection inside
that ROI. Human detection is only ever triggered once a face has already been matched (the
caller only calls evaluate() per matched face) — it does not run independently of face matching.
"""
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .config import HumanDetectionROIConfig
from .detector import HumanDetectorROI
from .roi import compute_roi

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public HumanDetectionResult dataclass) so this internal
    module has no import-time dependency on interface.py — mirrors this project's established
    tuple-return convention to avoid an import cycle."""
    person_found: bool
    person_bbox: Optional[Tuple[int, int, int, int]]
    detection_confidence: Optional[float]


def _select_best_detection(detections_full_frame: List[Dict[str, Any]],
                             face_bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
    """
    Disambiguates between multiple person detections landing in the same ROI (e.g. a crowd) —
    the whole reason to ROI-scope in the first place is "reduce the chance of picking up a
    different person's body," so picking arbitrarily/by-confidence-alone would partially defeat
    that purpose. Prefer whichever detection's bbox actually CONTAINS the face bbox's center
    (i.e. this detection's head region is where the matched face was); among those, prefer
    higher confidence. If none contain it (occlusion/detection noise near the edge), fall back
    to the detection whose center is closest to the face center.
    """
    fx, fy, fw, fh = face_bbox
    face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0

    best = None
    best_score = None
    for det in detections_full_frame:
        dx1, dy1, dx2, dy2 = det["bbox"]
        contains = (dx1 <= face_cx <= dx2) and (dy1 <= face_cy <= dy2)
        if contains:
            score = (1, det["confidence"])
        else:
            det_cx, det_cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            dist_sq = (det_cx - face_cx) ** 2 + (det_cy - face_cy) ** 2
            score = (0, -dist_sq)
        if best_score is None or score > best_score:
            best_score = score
            best = det
    return best


class HumanDetectionROIPipeline:
    def __init__(self, config: HumanDetectionROIConfig):
        self.config = config
        self.detector = HumanDetectorROI(config.yolo_model_path)

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"human_detection_roi: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — evaluate() will report person_found=False on every "
                f"call until config/thresholds.yaml's human_detection_roi section is filled in."
            )

    def evaluate(self, frame: np.ndarray, matched_face_bbox: Tuple[int, int, int, int]) -> PipelineResult:
        missing = self.config.missing_keys()
        if missing:
            return PipelineResult(person_found=False, person_bbox=None, detection_confidence=None)

        if frame is None or getattr(frame, "size", 0) == 0:
            return PipelineResult(person_found=False, person_bbox=None, detection_confidence=None)

        rx1, ry1, rx2, ry2 = compute_roi(
            matched_face_bbox, frame.shape, self.config.roi_expansion_factor,
            upward_fraction=self.config.roi_upward_fraction, width_fraction=self.config.roi_width_fraction,
        )
        if rx2 <= rx1 or ry2 <= ry1:
            return PipelineResult(person_found=False, person_bbox=None, detection_confidence=None)

        roi_crop = frame[ry1:ry2, rx1:rx2]
        raw_detections = self.detector.detect(roi_crop, self.config.detection_confidence_threshold)

        # If nothing is found in the scoped ROI, report person_found=False — do NOT silently
        # retry against the full frame. That would defeat ROI-scoping's purpose and risk
        # picking up a different person entirely.
        if not raw_detections:
            return PipelineResult(person_found=False, person_bbox=None, detection_confidence=None)

        detections_full_frame = [
            {"bbox": (d["bbox"][0] + rx1, d["bbox"][1] + ry1, d["bbox"][2] + rx1, d["bbox"][3] + ry1),
             "confidence": d["confidence"]}
            for d in raw_detections
        ]

        best = _select_best_detection(detections_full_frame, matched_face_bbox)
        if best is None:
            return PipelineResult(person_found=False, person_bbox=None, detection_confidence=None)

        bx1, by1, bx2, by2 = best["bbox"]
        person_bbox = (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1))  # (x, y, w, h)
        return PipelineResult(person_found=True, person_bbox=person_bbox, detection_confidence=best["confidence"])
