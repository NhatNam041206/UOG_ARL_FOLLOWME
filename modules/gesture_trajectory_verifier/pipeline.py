"""
Per-track orchestrator: person crop -> MoveNet -> per-arm trajectory buffer -> normalize+resample
-> compare against every reference trajectory -> best (arm, reference_id, score) -> confirmation
state machine -> GestureMethodResult. Not part of the public contract — external callers use
interface.py only.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

from .config import MIN_REFERENCE_COUNT, GestureTrajectoryVerifierConfig
from .confirmation import ConfirmationTracker, GREEN
from .constants import ARM_KEYPOINTS
from .normalization import normalize_trajectory
from .pose_estimator import MoveNetPoseEstimator
from .preprocessing import decode_keypoints, preprocess_crop
from .reference_store import ReferenceTrajectory, ReferenceTrajectoryStore
from .resampling import resample_time_based
from .similarity import flatten_trajectory, trajectory_similarity
from .trajectory_buffer import TrajectoryBuffer, update_trajectory_buffer

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public GestureMethodResult dataclass) so this internal
    module has no import-time dependency on interface.py."""
    track_id: int
    is_waving: bool
    waving_state: str
    confidence_debug: Optional[float]
    matched_reference_id: Optional[str]
    arm: Optional[str]
    reference_count: int
    keypoints_raw: Optional[object]
    keypoints_decoded: Optional[object]  # list of Keypoint, crop-pixel space — for draw_debug()


@dataclass
class _TrackState:
    motion_buffers: Dict[str, TrajectoryBuffer] = field(default_factory=lambda: {"left": TrajectoryBuffer(), "right": TrajectoryBuffer()})
    waving_tracker: ConfirmationTracker = field(default_factory=ConfirmationTracker)


def gesture_candidate_this_frame(
    left_buffer: TrajectoryBuffer, right_buffer: TrajectoryBuffer,
    reference_entries: "list[ReferenceTrajectory]", bbox_height: float,
    config: GestureTrajectoryVerifierConfig,
) -> Tuple[bool, Optional[str], Optional[str], Optional[float]]:
    """
    Computes similarity for whichever arm(s) have enough buffered samples, against every
    reference trajectory. Returns (is_waving_candidate, arm, reference_id, score) — the best
    triple seen. If the reference set is too small (spec §4.3, confirmed: fewer than
    MIN_REFERENCE_COUNT=2), returns (False, None, None, None) unconditionally — never computes
    a technically-real-but-untrustworthy score against 0-1 references.
    """
    if len(reference_entries) < MIN_REFERENCE_COUNT:
        return False, None, None, None

    best_arm: Optional[str] = None
    best_ref_id: Optional[str] = None
    best_score: Optional[float] = None

    for arm, buffer in (("left", left_buffer), ("right", right_buffer)):
        if len(buffer.samples) < config.min_samples_for_comparison:
            continue
        normalized = normalize_trajectory(buffer.samples, bbox_height)
        resampled = resample_time_based(normalized, config.resample_length)
        if not resampled:
            continue
        flat = flatten_trajectory(resampled)

        for ref in reference_entries:
            score = trajectory_similarity(flat, ref.flat_vector)
            if best_score is None or score > best_score:
                best_score = score
                best_arm = arm
                best_ref_id = ref.reference_id

    if best_arm is None:
        return False, None, None, None
    is_candidate = best_score >= config.similarity_threshold
    return is_candidate, best_arm, best_ref_id, best_score


class GestureTrajectoryVerifierPipeline:
    def __init__(self, config: GestureTrajectoryVerifierConfig):
        self.config = config
        self.pose_estimator = MoveNetPoseEstimator(config.movenet_tfhub_handle)
        self.reference_store = ReferenceTrajectoryStore(config.reference_dir)
        self._tracks: Dict[int, _TrackState] = {}

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"gesture_trajectory_verifier: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — evaluate() will report is_waving=False on every call "
                f"until config/thresholds.yaml's gesture_trajectory_verifier section is filled in."
            )

    def release_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def evaluate(self, track_id: int, person_crop_bgr, timestamp: Optional[float] = None) -> PipelineResult:
        if timestamp is None:
            timestamp = time.time()
        state = self._tracks.setdefault(track_id, _TrackState())

        missing = self.config.missing_keys()
        if missing:
            return self._no_signal_result(track_id, state, timestamp)

        pre = preprocess_crop(person_crop_bgr, self.config.movenet_input_size)
        if pre is None:
            return self._no_signal_result(track_id, state, timestamp)

        raw_keypoints = self.pose_estimator.estimate(pre.tensor)
        keypoints = decode_keypoints(raw_keypoints, pre)

        for arm, (wrist_idx, elbow_idx, shoulder_idx) in ARM_KEYPOINTS.items():
            update_trajectory_buffer(
                state.motion_buffers[arm], keypoints[wrist_idx], keypoints[elbow_idx],
                keypoints[shoulder_idx], timestamp, self.config,
            )

        reference_entries = self.reference_store.load_all()
        waving_raw, arm, ref_id, score = gesture_candidate_this_frame(
            state.motion_buffers["left"], state.motion_buffers["right"],
            reference_entries, pre.orig_h, self.config,
        )

        waving_state = state.waving_tracker.update(waving_raw, timestamp, self.config)
        return PipelineResult(
            track_id=track_id, is_waving=(waving_state == GREEN), waving_state=waving_state,
            confidence_debug=score, matched_reference_id=ref_id, arm=arm,
            reference_count=len(reference_entries), keypoints_raw=raw_keypoints,
            keypoints_decoded=keypoints,
        )

    def _no_signal_result(self, track_id: int, state: _TrackState, timestamp: float) -> PipelineResult:
        waving_state = state.waving_tracker.update(False, timestamp, self.config)
        return PipelineResult(
            track_id=track_id, is_waving=False, waving_state=waving_state,
            confidence_debug=None, matched_reference_id=None, arm=None,
            reference_count=0, keypoints_raw=None, keypoints_decoded=None,
        )
