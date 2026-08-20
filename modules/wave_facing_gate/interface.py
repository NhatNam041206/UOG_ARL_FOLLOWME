"""
Wave Gesture + Facing-Camera Gate module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.wave_facing_gate. Everything else in
this package (config.py, constants.py, preprocessing.py, pose_estimator.py, facing.py, gate_a.py,
gate_b.py, confirmation.py, pipeline.py) is an internal implementation detail and may change
without notice.

Scope (see the module spec for full detail): this module receives one cropped bbox image per
call for one already-verified, registered track (output of the teammate's detection/tracking/
Re-ID pipeline). It does NOT run detection, tracking, or identity verification, and it does NOT
compute the final Stage-2 trigger — it exposes two independent, individually-debounced signals
(`is_waving`, `is_facing_camera`); the caller ANDs those with its own `registered_person` check:

    trigger = (registered_person and result.is_waving and result.is_facing_camera)

That combination lives in Stage 2 orchestration code, not here.

Crop convention: a raw BGR crop (numpy.ndarray from cv2), the standard OpenCV convention used
elsewhere in this codebase (see modules/emergency_stop/interface.py).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import WaveFacingConfig, load_config
from .constants import Keypoint
from .pipeline import WaveFacingPipeline
from . import visualize


@dataclass
class GestureFacingResult:
    track_id: int
    is_waving: bool           # True only when the waving ConfirmationTracker == GREEN
    is_facing_camera: bool    # True only when the facing ConfirmationTracker == GREEN
    waving_state: str         # "RED" | "YELLOW" | "GREEN" — for bbox color visualization
    facing_state: str         # "RED" | "YELLOW" | "GREEN" — for bbox color visualization
    wave_arm: Optional[str]   # "left" | "right" | None — which arm's Gate A+B passed most recently
    facing_confidence_min: Optional[float]
    keypoints_raw: object      # raw MoveNet [17, 3] output for this frame, for debugging
    # Optional debug visualization fields (only populated if process_frame was called with debug=True):
    keypoints_decoded: Optional[List[Keypoint]] = None  # decoded to bbox pixel space
    gate_a_passes: Optional[Dict[str, bool]] = None     # per-arm Gate A pass/fail

    def draw_debug(self, frame: np.ndarray) -> None:
        """
        Draw pose keypoints, skeleton, arm vectors, and gate state onto the frame for debugging.
        Only call this if keypoints_decoded and gate_a_passes are available (they are None for
        frames that couldn't be evaluated, e.g. missing config or empty crop).
        """
        if self.keypoints_decoded is None or self.gate_a_passes is None:
            return
        visualize.draw_keypoints(frame, self.keypoints_decoded)
        visualize.draw_skeleton(frame, self.keypoints_decoded)
        visualize.draw_arm_vectors(frame, self.keypoints_decoded, self.gate_a_passes)


class WaveFacingGateModule:
    """
    Owns all per-track state (motion buffers, confirmation trackers, own MoveNet instance).
    Create one instance and call process_frame() once per (track_id, crop) per frame — a given
    track_id's state persists across calls; a new track_id (e.g. the upstream tracker lost and
    reacquired the same physical person) starts fresh with no cross-track identity stitching
    (spec §7, a deliberate simplification).
    """

    def __init__(self, thresholds_config_path: str = "config/thresholds.yaml"):
        config: WaveFacingConfig = load_config(thresholds_config_path)
        self._pipeline = WaveFacingPipeline(config)

    def process_frame(self, track_id: int, crop, timestamp: Optional[float] = None) -> GestureFacingResult:
        result = self._pipeline.process(track_id, crop, timestamp)
        return GestureFacingResult(
            track_id=result.track_id,
            is_waving=result.is_waving,
            is_facing_camera=result.is_facing_camera,
            waving_state=result.waving_state,
            facing_state=result.facing_state,
            wave_arm=result.wave_arm,
            facing_confidence_min=result.facing_confidence_min,
            keypoints_raw=result.keypoints_raw,
            keypoints_decoded=result.keypoints_decoded,
            gate_a_passes=result.gate_a_passes,
        )

    def reset_track(self, track_id: int) -> None:
        """Drop a track_id's accumulated state once the caller knows it's no longer active."""
        self._pipeline.reset_track(track_id)

    @property
    def last_latency_ms(self) -> float:
        """Measured wall-clock time for the most recent process_frame() call, in milliseconds —
        exposed for the MoveNet inference latency benchmarking called out in the module spec's
        calibration checklist (spec §11)."""
        return self._pipeline.last_latency_ms
