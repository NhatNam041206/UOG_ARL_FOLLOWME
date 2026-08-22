"""
Per-frame orchestrator for a single active follow-me episode: lock onto one target's ByteTrack
track_id -> RECORDING crop buffer -> appearance_verifier.build_reference_set() -> TRACKING
(horizontal deviation + periodic appearance_verifier.verify() re-check) -> LOST handoff. Not part
of the public contract — external callers use interface.py only.

Only ONE episode is tracked at a time — start()/update()/reset() take no track_id (unlike every
other module in this project, which is keyed per track_id/person) because this module manages a
single locked follow-me target by design (plans/06 §0.3). A fresh start() or reset() call always
replaces whatever episode state existed before.

RECORDING keeps running this module's own tracker every frame (not just a static crop from
start() time) so the collected reference crops follow the target if they move during the
recording window — the same "is the locked track_id still being seen" question applies during
RECORDING as during TRACKING, so track_loss_grace_period_seconds is checked uniformly across both
states (an extension beyond the spec's literal §4.4 wording, which is written under §4 TRACKING,
but follows directly from RECORDING needing the same continuous tracking §4.1 describes).
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from modules.appearance_verifier.interface import (
    ReferenceEmbeddingSet, build_reference_set, verify as appearance_verify,
)

from .config import TargetTrackingConfig
from .locking import select_matching_detection
from .tracker import TargetTracker

logger = logging.getLogger(__name__)

RECORDING = "RECORDING"
TRACKING = "TRACKING"
LOST = "LOST"

BboxXYWH = Tuple[int, int, int, int]


def compute_horizontal_offset(person_bbox: BboxXYWH, frame_width: int) -> float:
    """Normalized -1.0 (frame-left edge) to +1.0 (frame-right edge), 0.0 = bbox center exactly on
    frame's vertical centerline. Deliberately NOT a true angle — FOV-based angle conversion is
    the downstream steering layer's job (spec §4.2's explicit architecture boundary), not this
    module's; camera.fov_degrees is intentionally never referenced here."""
    x, y, w, h = person_bbox
    bbox_center_x = x + w / 2.0
    frame_center_x = frame_width / 2.0
    pixel_offset = bbox_center_x - frame_center_x
    return pixel_offset / (frame_width / 2.0)


def _to_xyxy(bbox_xywh: BboxXYWH) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return (float(x), float(y), float(x + w), float(y + h))


@dataclass
class _EpisodeState:
    state: str = LOST                              # LOST = no episode active until start() is called
    locked_track_id: Optional[int] = None
    pending_lock_bbox: Optional[BboxXYWH] = None    # set by start()/reset(); consumed once locked
    recording_start_time: Optional[float] = None
    recording_crops: List[np.ndarray] = field(default_factory=list)
    reference_set: Optional[ReferenceEmbeddingSet] = None
    last_seen_time: Optional[float] = None          # for track_loss_grace_period_seconds
    last_bbox: Optional[BboxXYWH] = None            # last known bbox, for grace-period frames
    last_horizontal_offset: Optional[float] = None
    last_reverify_time: Optional[float] = None
    reverify_consecutive_failures: int = 0
    last_reverify_score: Optional[float] = None     # debug/visualization only (spec §7)
    last_reverify_pass: Optional[bool] = None        # debug/visualization only (spec §7)


class PipelineResult:
    """Plain container (not the public TrackingResult dataclass) so this internal module has no
    import-time dependency on interface.py — mirrors this project's tuple-return convention to
    avoid an import cycle. A plain class rather than NamedTuple since a field holds a mutable
    ReferenceEmbeddingSet object, not just primitives."""
    __slots__ = (
        "target_locked", "horizontal_offset", "person_bbox", "state", "reference_set",
        "last_reverify_score", "last_reverify_pass",
    )

    def __init__(self, target_locked, horizontal_offset, person_bbox, state, reference_set,
                 last_reverify_score=None, last_reverify_pass=None):
        self.target_locked = target_locked
        self.horizontal_offset = horizontal_offset
        self.person_bbox = person_bbox
        self.state = state
        self.reference_set = reference_set
        # Debug/visualization only (spec §7) — the most recent periodic appearance re-verify
        # outcome this episode, if one has run yet; stays populated across frames between
        # re-verify checks (not just on the exact frame it ran), same "debug fields always
        # reflect the latest known value" convention used elsewhere in this project.
        self.last_reverify_score = last_reverify_score
        self.last_reverify_pass = last_reverify_pass


class TargetTrackingPipeline:
    def __init__(self, config: TargetTrackingConfig):
        self.config = config
        self.tracker = TargetTracker(config.yolo_model_path)
        self._episode = _EpisodeState()

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"target_tracking: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — episodes will not progress meaningfully past RECORDING "
                f"until config/thresholds.yaml's target_tracking section is filled in."
            )

    def start(self, initial_person_bbox: BboxXYWH, frame: np.ndarray, timestamp: float) -> None:
        self._episode = _EpisodeState(
            state=RECORDING,
            pending_lock_bbox=initial_person_bbox,
            recording_start_time=timestamp,
            last_seen_time=timestamp,
        )

    def reset(self, fresh_person_bbox: BboxXYWH, frame: np.ndarray, timestamp: float) -> None:
        """Re-enters RECORDING with a freshly re-acquired bbox — identical to a fresh start().
        NOTE: plans/06_target_tracking.md §0.3 drafted reset()'s signature as taking no
        parameters at all, but its own docstring says it "hands a fresh bbox back" — an internal
        spec inconsistency. Resolved here per §0.4's instruction to flag and use the described
        behavior rather than the literal (parameterless) signature, which cannot actually do what
        its own docstring says."""
        self.start(fresh_person_bbox, frame, timestamp)

    def update(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        episode = self._episode

        if episode.state == LOST:
            # Nothing further until reset() is called (spec §5) — cheap no-op.
            return PipelineResult(False, None, None, LOST, episode.reference_set)

        if frame is None or getattr(frame, "size", 0) == 0:
            return self._missing_this_frame(episode, timestamp)

        frame_h, frame_w = frame.shape[:2]
        detections = self.tracker.track(frame)  # every visible person, (x1,y1,x2,y2)

        if episode.locked_track_id is None:
            matched = select_matching_detection(detections, _to_xyxy(episode.pending_lock_bbox))
            if matched is None:
                # Haven't been able to lock onto a track_id yet this episode — keep waiting,
                # subject to the same missing-target grace period as an established lock, so a
                # target that's never detected still eventually resolves to LOST rather than
                # stalling the episode forever.
                return self._missing_this_frame(episode, timestamp)
            episode.locked_track_id = matched["track_id"]
            episode.pending_lock_bbox = None

        current = next((d for d in detections if d["track_id"] == episode.locked_track_id), None)
        if current is None:
            return self._missing_this_frame(episode, timestamp)

        episode.last_seen_time = timestamp
        x1, y1, x2, y2 = current["bbox"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return self._missing_this_frame(episode, timestamp)
        person_bbox: BboxXYWH = (x1, y1, x2 - x1, y2 - y1)

        if episode.state == RECORDING:
            return self._update_recording(episode, frame, person_bbox, frame_w, timestamp)
        return self._update_tracking(episode, frame, person_bbox, frame_w, timestamp)

    def _missing_this_frame(self, episode: _EpisodeState, timestamp: float) -> PipelineResult:
        """Locked (or not-yet-locked pending) target missing this frame — tolerated up to
        track_loss_grace_period_seconds (spec §4.4) before declaring LOST. If that key is itself
        uncalibrated, the episode simply cannot time out via this path (fail-closed applies to
        the LOST *decision*, not to silently guessing a grace period)."""
        grace = self.config.track_loss_grace_period_seconds
        if grace is not None and episode.last_seen_time is not None and (timestamp - episode.last_seen_time) >= grace:
            episode.state = LOST
            return PipelineResult(False, None, None, LOST, episode.reference_set)
        return PipelineResult(
            True, episode.last_horizontal_offset, episode.last_bbox, episode.state, episode.reference_set,
            last_reverify_score=episode.last_reverify_score, last_reverify_pass=episode.last_reverify_pass,
        )

    def _update_recording(self, episode: _EpisodeState, frame: np.ndarray, person_bbox: BboxXYWH,
                            frame_w: int, timestamp: float) -> PipelineResult:
        x, y, w, h = person_bbox
        crop = frame[y:y + h, x:x + w]
        if crop.size > 0:
            # .copy(): frame is reused/overwritten by the caller's next cap.read() before
            # build_reference_set() runs later — a bare view would go stale by then.
            episode.recording_crops.append(crop.copy())

        episode.last_bbox = person_bbox
        episode.last_horizontal_offset = compute_horizontal_offset(person_bbox, frame_w)

        if self.config.record_duration_seconds is None:
            # Uncalibrated -> RECORDING can never elapse -> never transitions to TRACKING
            # (fail-closed: stays in RECORDING indefinitely rather than guessing a duration).
            return PipelineResult(True, episode.last_horizontal_offset, person_bbox, RECORDING, None)

        elapsed = timestamp - episode.recording_start_time
        if elapsed < self.config.record_duration_seconds:
            return PipelineResult(True, episode.last_horizontal_offset, person_bbox, RECORDING, None)

        if len(episode.recording_crops) < self.config.min_recording_crops:
            # Too few usable crops (confirmed with the user): EXTEND RECORDING rather than
            # building a fragile reference set — push the window forward, keep whatever crops
            # were already collected, keep collecting more.
            episode.recording_start_time = timestamp
            return PipelineResult(True, episode.last_horizontal_offset, person_bbox, RECORDING, None)

        reference_set = build_reference_set(episode.recording_crops)
        episode.reference_set = reference_set
        episode.state = TRACKING
        episode.last_reverify_time = timestamp  # don't reverify on the very first TRACKING frame
        episode.reverify_consecutive_failures = 0
        return PipelineResult(True, episode.last_horizontal_offset, person_bbox, TRACKING, reference_set)

    def _update_tracking(self, episode: _EpisodeState, frame: np.ndarray, person_bbox: BboxXYWH,
                           frame_w: int, timestamp: float) -> PipelineResult:
        episode.last_bbox = person_bbox
        episode.last_horizontal_offset = compute_horizontal_offset(person_bbox, frame_w)

        reverify_keys_ready = (
            self.config.appearance_reverify_interval_seconds is not None
            and self.config.appearance_reverify_similarity_threshold is not None
        )
        if reverify_keys_ready and episode.last_reverify_time is not None:
            elapsed_since_reverify = timestamp - episode.last_reverify_time
            if elapsed_since_reverify >= self.config.appearance_reverify_interval_seconds:
                x, y, w, h = person_bbox
                crop = frame[y:y + h, x:x + w]
                result = appearance_verify(crop, episode.reference_set)
                episode.last_reverify_time = timestamp
                episode.last_reverify_score = result.best_similarity_score

                # Deliberately recomputed against THIS module's OWN
                # appearance_reverify_similarity_threshold, not result.match_found (which uses
                # appearance_verifier's own similarity_threshold — a separate, independently
                # tunable key per plans/05 §4; never conflate the two).
                passed = (
                    result.reference_frame_count > 0
                    and result.best_similarity_score >= self.config.appearance_reverify_similarity_threshold
                )
                episode.last_reverify_pass = passed

                if passed:
                    episode.reverify_consecutive_failures = 0
                else:
                    episode.reverify_consecutive_failures += 1
                    # 2 consecutive failures required before declaring LOST (confirmed with the
                    # user) — a single bad-lighting/occlusion frame doesn't end tracking.
                    if episode.reverify_consecutive_failures >= self.config.appearance_reverify_consecutive_failures:
                        episode.state = LOST
                        return PipelineResult(False, None, None, LOST, episode.reference_set)

        return PipelineResult(
            True, episode.last_horizontal_offset, person_bbox, TRACKING, episode.reference_set,
            last_reverify_score=episode.last_reverify_score, last_reverify_pass=episode.last_reverify_pass,
        )
