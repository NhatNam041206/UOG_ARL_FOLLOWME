"""
Per-track orchestrator: crop -> preprocess -> MoveNet -> decode -> {facing raw check, Gate A +
Gate B per arm} -> confirmation state machines -> GestureFacingResult. Not part of the public
contract — external callers use interface.py only.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional

from .config import WaveFacingConfig
from .confirmation import ConfirmationTracker, GREEN
from .constants import ARM_KEYPOINTS
from .facing import facing_camera_raw
from .gate_a import gate_a_pass
from .gate_b import MotionBuffer, gate_b_pass, update_motion_buffer
from .pose_estimator import MoveNetPoseEstimator
from .preprocessing import decode_keypoints, preprocess_crop

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """
    Plain-primitive result kept here (not the public GestureFacingResult dataclass) so this
    internal module has no import-time dependency on interface.py — mirrors
    modules/emergency_stop/pipeline.py's tuple-return convention to avoid an import cycle
    (interface.py constructs WaveFacingPipeline, so the dependency can't run the other way).
    """
    track_id: int
    is_waving: bool
    is_facing_camera: bool
    waving_state: str
    facing_state: str
    wave_arm: Optional[str]
    facing_confidence_min: Optional[float]
    keypoints_raw: object
    keypoints_decoded: Optional[object]  # list of Keypoint for debugging/visualization
    gate_a_passes: Optional[dict]        # per-arm Gate A pass/fail for debugging


@dataclass
class _TrackState:
    """
    Per-track_id state (spec §7). A new track_id (e.g. after the upstream tracker loses and
    reacquires the same physical person) naturally gets a fresh _TrackState — no cross-track
    identity stitching, by design.
    """
    motion_buffers: Dict[str, MotionBuffer] = field(default_factory=lambda: {"left": MotionBuffer(), "right": MotionBuffer()})
    waving_tracker: ConfirmationTracker = field(default_factory=ConfirmationTracker)
    facing_tracker: ConfirmationTracker = field(default_factory=ConfirmationTracker)
    last_wave_arm: Optional[str] = None


class WaveFacingPipeline:
    def __init__(self, config: WaveFacingConfig):
        self.config = config
        self.pose_estimator = MoveNetPoseEstimator(config.movenet_tfhub_handle)
        self._tracks: Dict[int, _TrackState] = {}

        # Exposed for external latency benchmarking (spec §11 calibration checklist: MoveNet
        # inference latency must be measured on target hardware, not trusted from literature).
        self.last_latency_ms: float = 0.0

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"wave_facing_gate: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — is_waving/is_facing_camera will stay False on every "
                f"frame until config/thresholds.yaml's wave_facing section is fully filled in."
            )

    def reset_track(self, track_id: int) -> None:
        """External hygiene hook: drop a track's state (motion buffers, both confirmation
        trackers) once the caller knows that track_id is no longer active, to bound memory. Not
        required for correctness — a track_id that simply stops appearing just stops accumulating
        new state."""
        self._tracks.pop(track_id, None)

    def process(self, track_id: int, crop_bgr, timestamp: Optional[float] = None):
        t_start = time.time()
        if timestamp is None:
            timestamp = t_start
        state = self._tracks.setdefault(track_id, _TrackState())

        try:
            missing = self.config.missing_keys()
            if missing:
                return self._no_signal_result(track_id, state, timestamp, keypoints_raw=None)

            pre = preprocess_crop(crop_bgr, self.config.movenet_input_size)
            if pre is None:
                return self._no_signal_result(track_id, state, timestamp, keypoints_raw=None)

            raw_keypoints = self.pose_estimator.estimate(pre.tensor)
            keypoints = decode_keypoints(raw_keypoints, pre)

            facing_raw, facing_conf_min = facing_camera_raw(keypoints, self.config)

            waving_raw = False
            wave_arm = None
            gate_a_passes = {}
            for side, (wrist_idx, elbow_idx, shoulder_idx) in ARM_KEYPOINTS.items():
                wrist, elbow, shoulder = keypoints[wrist_idx], keypoints[elbow_idx], keypoints[shoulder_idx]
                buffer = state.motion_buffers[side]
                update_motion_buffer(buffer, wrist, timestamp, self.config)

                a_pass = gate_a_pass(wrist, elbow, shoulder, pre.orig_h, self.config)
                gate_a_passes[side] = a_pass
                b_pass = gate_b_pass(buffer, self.config)
                if a_pass and b_pass and not waving_raw:
                    waving_raw = True
                    wave_arm = side

            if wave_arm is not None:
                state.last_wave_arm = wave_arm

            waving_state = state.waving_tracker.update(waving_raw, timestamp, self.config)
            facing_state = state.facing_tracker.update(facing_raw, timestamp, self.config)

            return self._result(
                track_id=track_id,
                waving_state=waving_state,
                facing_state=facing_state,
                wave_arm=state.last_wave_arm,
                facing_confidence_min=facing_conf_min,
                keypoints_raw=raw_keypoints,
                keypoints_decoded=keypoints,
                gate_a_passes=gate_a_passes,
            )
        finally:
            self.last_latency_ms = (time.time() - t_start) * 1000.0

    def _no_signal_result(self, track_id: int, state: _TrackState, timestamp: float, keypoints_raw):
        # Route through the trackers so RED/interruption bookkeeping stays in one place — an
        # unevaluable frame (missing config, empty crop) simply fails the raw condition.
        waving_state = state.waving_tracker.update(False, timestamp, self.config)
        facing_state = state.facing_tracker.update(False, timestamp, self.config)
        return self._result(
            track_id=track_id,
            waving_state=waving_state,
            facing_state=facing_state,
            wave_arm=state.last_wave_arm,
            facing_confidence_min=None,
            keypoints_raw=keypoints_raw,
            keypoints_decoded=None,
            gate_a_passes=None,
        )

    def _result(self, track_id, waving_state, facing_state, wave_arm, facing_confidence_min,
                 keypoints_raw, keypoints_decoded, gate_a_passes):
        return PipelineResult(
            track_id=track_id,
            is_waving=(waving_state == GREEN),
            is_facing_camera=(facing_state == GREEN),
            waving_state=waving_state,
            facing_state=facing_state,
            wave_arm=wave_arm,
            facing_confidence_min=facing_confidence_min,
            keypoints_raw=keypoints_raw,
            keypoints_decoded=keypoints_decoded,
            gate_a_passes=gate_a_passes,
        )
