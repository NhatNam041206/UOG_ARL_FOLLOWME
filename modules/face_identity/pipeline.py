"""
Per-frame orchestrator: full frame -> YuNet detect (all faces) -> confidence post-filter ->
per-face 5-point align -> EdgeFace embed -> match against registry -> FaceIdentityResult per
face. Not part of the public contract — external callers use interface.py only.
"""
import logging
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .alignment import align_face
from .config import FaceIdentityConfig
from .embedder import EdgeFaceEmbedder
from .face_detector import YuNetFaceDetector
from .matching import match_face
from .registry import FaceRegistry

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """
    Plain-primitive result kept here (not the public FaceIdentityResult dataclass) so this
    internal module has no import-time dependency on interface.py — mirrors
    modules/emergency_stop and modules/wave_facing_gate's tuple-return convention to avoid an
    import cycle.
    """
    face_found: bool
    face_bbox: Tuple[int, int, int, int]
    is_registered_match: bool
    matched_person_name: Optional[str]
    match_confidence: Optional[float]
    face_detection_confidence: float


class FaceIdentityPipeline:
    def __init__(self, config: FaceIdentityConfig):
        self.config = config
        self.detector = YuNetFaceDetector(config.yunet_model_path)
        self.embedder = EdgeFaceEmbedder(config.edgeface_model_path)

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"face_identity: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — evaluate() will report face_found=False on every "
                f"frame until config/thresholds.yaml's face_identity section is fully filled in."
            )

    def evaluate(self, frame: np.ndarray, registry: FaceRegistry) -> List[PipelineResult]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        missing = self.config.missing_keys()
        if missing:
            return []

        raw_faces = self.detector.detect(frame)
        confidence_floor = self.config.face_detection_confidence_threshold
        registry_entries = registry.load_all()

        results: List[PipelineResult] = []
        for face in raw_faces:
            if face.score < confidence_floor:
                continue

            aligned = align_face(frame, face.landmarks)
            if aligned is None:
                # Landmarks too degenerate to align (rare) — still report the detection itself,
                # just without a match verdict, rather than dropping the face entirely.
                results.append(PipelineResult(
                    face_found=True, face_bbox=face.bbox, is_registered_match=False,
                    matched_person_name=None, match_confidence=None,
                    face_detection_confidence=face.score,
                ))
                continue

            embedding = self.embedder.embed(aligned)
            is_match, person_name, score = match_face(
                embedding, registry_entries, self.config.similarity_threshold_face_match,
            )
            results.append(PipelineResult(
                face_found=True, face_bbox=face.bbox, is_registered_match=is_match,
                matched_person_name=person_name, match_confidence=score,
                face_detection_confidence=face.score,
            ))

        return results
