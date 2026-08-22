"""
Per-frame orchestrator for a single active recovery/search episode: Path A (face_identity ->
human_detection_roi, primary) tried every frame; Path B (whole-frame detection ->
appearance_verifier.verify(), fallback) additionally tried once face_search_grace_attempts
consecutive Path-A failures have accumulated this episode. Not part of the public contract —
external callers use interface.py only.

Only ONE search episode is tracked at a time — start()/update() take no track_id, mirroring
modules.target_tracking's single-episode design (this module receives that module's LOST handoff
directly, per plans/07 §2).
"""
import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

from modules.appearance_verifier.interface import ReferenceEmbeddingSet, verify as appearance_verify
from modules.face_identity.interface import evaluate as evaluate_face
from modules.human_detection_roi.interface import evaluate as evaluate_person

from .config import TargetRecoveryConfig
from .detector import RecoveryCandidateDetector

logger = logging.getLogger(__name__)

SEARCHING = "SEARCHING"
REACQUIRED = "REACQUIRED"
TIMEOUT = "TIMEOUT"

BboxXYWH = Tuple[int, int, int, int]


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public RecoveryResult dataclass) so this internal module
    has no import-time dependency on interface.py — mirrors this project's established
    tuple-return convention to avoid an import cycle."""
    status: str
    reacquired_person_bbox: Optional[BboxXYWH]
    reacquired_via: Optional[str]
    face_search_fail_count: int
    elapsed_search_seconds: float


@dataclass
class _EpisodeState:
    active: bool = False
    reference_set: Optional[ReferenceEmbeddingSet] = None
    target_person_name: Optional[str] = None
    search_start_time: Optional[float] = None
    face_search_fail_count: int = 0
    # Once REACQUIRED/TIMEOUT, stay terminal (spec §2: "terminal for a given search episode")
    # until start() is called again for a new episode.
    terminal_status: Optional[str] = None


class TargetRecoveryPipeline:
    def __init__(self, config: TargetRecoveryConfig):
        self.config = config
        self.path_b_detector = RecoveryCandidateDetector(config.yolo_model_path)
        self._episode = _EpisodeState()

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"target_recovery: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — Path B and the overall search timeout stay gated off "
                f"(status will only ever report SEARCHING or REACQUIRED via Path A) until "
                f"config/thresholds.yaml's target_recovery section is filled in."
            )

    def start(self, reference_set: ReferenceEmbeddingSet, target_person_name: str, timestamp: float) -> None:
        self._episode = _EpisodeState(
            active=True, reference_set=reference_set, target_person_name=target_person_name,
            search_start_time=timestamp, face_search_fail_count=0,
        )

    def update(self, frame: np.ndarray, registry, timestamp: float) -> PipelineResult:
        episode = self._episode

        if not episode.active:
            # start() was never called for this episode — report a harmless zeroed SEARCHING
            # result rather than crashing (mirrors this project's "plumbing runs, only the
            # verdict is gated" convention, applied here to "no episode started yet").
            return PipelineResult(SEARCHING, None, None, 0, 0.0)

        if episode.terminal_status is not None:
            elapsed = timestamp - episode.search_start_time
            return PipelineResult(episode.terminal_status, None, None, episode.face_search_fail_count, elapsed)

        elapsed = timestamp - episode.search_start_time

        # --- Path A: face-based re-acquisition (primary, ALWAYS tried first every frame) ---
        face_results = evaluate_face(frame, registry)
        target_match = next(
            (r for r in face_results if r.is_registered_match and r.matched_person_name == episode.target_person_name),
            None,
        )
        if target_match is not None:
            # The face-detectability question face_search_grace_attempts tracks (spec §4.2) is
            # answered "yes" here regardless of whether the body-scoping step below also
            # succeeds this exact frame — reset unconditionally on a correct face match.
            episode.face_search_fail_count = 0
            person_result = evaluate_person(frame, target_match.face_bbox)
            if person_result.person_found:
                episode.terminal_status = REACQUIRED
                return PipelineResult(REACQUIRED, person_result.person_bbox, "face_match", 0, elapsed)
            # Face matched but the body wasn't found this frame (rare: face visible, body
            # occluded) — not a complete reacquisition yet, keep searching.
        else:
            episode.face_search_fail_count += 1

        # --- Path B: appearance-based fallback (only once the grace-attempts COUNT is hit) ---
        missing = self.config.missing_keys()
        if not missing and episode.face_search_fail_count >= self.config.face_search_grace_attempts:
            reacquired = self._try_path_b(frame, episode)
            if reacquired is not None:
                episode.terminal_status = REACQUIRED
                return PipelineResult(REACQUIRED, reacquired, "appearance_fallback", episode.face_search_fail_count, elapsed)

        # --- Overall search timeout, checked every frame regardless of which path was tried ---
        if self.config.search_timeout_seconds is not None and elapsed >= self.config.search_timeout_seconds:
            episode.terminal_status = TIMEOUT
            return PipelineResult(TIMEOUT, None, None, episode.face_search_fail_count, elapsed)

        return PipelineResult(SEARCHING, None, None, episode.face_search_fail_count, elapsed)

    def _try_path_b(self, frame: np.ndarray, episode: _EpisodeState) -> Optional[BboxXYWH]:
        """Whole-frame candidate detection -> appearance_verifier.verify() per candidate -> the
        BEST candidate clearing appearance_fallback_threshold, or None. On a match, the candidate
        bbox is used DIRECTLY as the reacquired bbox (spec §4.3) — human_detection_roi is
        deliberately NOT re-run here, since Path B already found a body bbox directly; re-running
        ROI detection on a region already known to contain the target would be pure waste."""
        candidates = self.path_b_detector.detect(frame)
        if not candidates:
            return None

        frame_h, frame_w = frame.shape[:2]
        best_bbox: Optional[BboxXYWH] = None
        best_score: Optional[float] = None

        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            result = appearance_verify(crop, episode.reference_set)
            if result.reference_frame_count == 0:
                continue  # no reference set to compare against — cannot use this path at all
            if result.best_similarity_score < self.config.appearance_fallback_threshold:
                continue
            if best_score is None or result.best_similarity_score > best_score:
                best_score = result.best_similarity_score
                best_bbox = (x1, y1, x2 - x1, y2 - y1)

        return best_bbox
