"""
Orchestrator: crop(s) -> OSNet embed -> cosine similarity against a reference set ->
AppearanceVerifierResult. Not part of the public contract — external callers use interface.py
only.
"""
import logging
from dataclasses import dataclass, field
from typing import List, NamedTuple

import numpy as np

from .config import AppearanceVerifierConfig
from .embedder import OSNetEmbedder
from .matching import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ReferenceEmbeddingSet:
    """The object AppearanceVerifierResult's callers pass around opaquely (e.g.
    modules.target_tracking.TrackingResult.reference_set) — built once by build_reference_set(),
    reused across many verify() calls."""
    embeddings: List[np.ndarray] = field(default_factory=list)


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public AppearanceVerifierResult dataclass) so this
    internal module has no import-time dependency on interface.py — mirrors this project's
    established tuple-return convention to avoid an import cycle."""
    match_found: bool
    best_similarity_score: float
    reference_frame_count: int


class AppearanceVerifierPipeline:
    def __init__(self, config: AppearanceVerifierConfig):
        self.config = config
        self.embedder = OSNetEmbedder(config.osnet_model_name)

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"appearance_verifier: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — verify() will report match_found=False on every call "
                f"until config/thresholds.yaml's appearance_verifier section is filled in."
            )

    def build_reference_set(self, person_crops: List[np.ndarray]) -> ReferenceEmbeddingSet:
        """Embeds every provided crop ONCE (spec §3: do not re-embed reference crops per
        comparison — wasted repeated work for data that doesn't change during an episode).
        Embedding needs no calibrated threshold, so this always runs regardless of config state
        — only verify()'s match_found verdict is fail-closed."""
        embeddings = [
            self.embedder.embed(crop) for crop in person_crops
            if crop is not None and getattr(crop, "size", 0) > 0
        ]
        return ReferenceEmbeddingSet(embeddings=embeddings)

    def verify(self, candidate_crop: np.ndarray, reference_set: ReferenceEmbeddingSet) -> PipelineResult:
        reference_count = len(reference_set.embeddings) if reference_set is not None else 0

        # "Not ready" floor (spec §5, mirrors gesture_trajectory_verifier's MIN_REFERENCE_COUNT
        # pattern): an empty reference set gets a distinct, visibly-zero reference_frame_count
        # rather than a meaningless comparison — best_similarity_score stays a real placeholder
        # (0.0) since the public contract types it as a plain float, never None/NaN; the caller
        # distinguishes "not ready" from "compared, didn't match" via reference_frame_count, not
        # via the score itself.
        if reference_count == 0:
            return PipelineResult(match_found=False, best_similarity_score=0.0, reference_frame_count=0)

        if candidate_crop is None or getattr(candidate_crop, "size", 0) == 0:
            return PipelineResult(match_found=False, best_similarity_score=0.0, reference_frame_count=reference_count)

        candidate_embedding = self.embedder.embed(candidate_crop)
        best_score = max(cosine_similarity(candidate_embedding, ref) for ref in reference_set.embeddings)

        missing = self.config.missing_keys()
        if missing:
            return PipelineResult(match_found=False, best_similarity_score=best_score, reference_frame_count=reference_count)

        match_found = best_score >= self.config.similarity_threshold
        return PipelineResult(match_found=match_found, best_similarity_score=best_score, reference_frame_count=reference_count)
