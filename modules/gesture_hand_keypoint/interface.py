"""
Gesture Detection — Method 2, Hand Keypoint — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.gesture_hand_keypoint. Everything
else in this package (config.py, constants.py, hand_detector.py, hand_shape.py, bbox_context.py,
sequence_state_machine.py, palm_orientation.py, confirmation.py, pipeline.py) is an internal
implementation detail.

REDESIGN: this module no longer detects "waving" via motion of any kind — no wrist motion, no
trajectory, no arm geometry. It is now a PURE hand-shape sequence classifier, using only
MediaPipe Hands landmark geometry:

    A valid gesture = OPEN -> CLOSED -> OPEN -> CLOSED, each transition classified purely from
    hand-shape (finger extension geometry), within a bounded time window
    (max_transition_gap_seconds) between consecutive transitions. Must start from OPEN.

    WAITING_OPEN -> (open) -> WAITING_CLOSE_1 -> (closed) -> WAITING_OPEN_2 -> (open) ->
    WAITING_CLOSE_2 -> (closed) -> CONFIRMED

Every OPEN/CLOSED read counted toward the sequence must ALSO independently clear a palm-height
gate: the palm (wrist landmark) must be in the upper `palm_height_fraction` of the person's
FULL-FRAME bbox (from modules.human_detection_roi's output — see `person_bbox_full_frame` below),
not just somewhere in the crop. Failing the height gate is an IMMEDIATE reset to WAITING_OPEN
(confirmed with the user), stricter than a merely-non-advancing frame.

CONFIRMED (reaching the end of the sequence) feeds into the same shared RED/YELLOW/GREEN
confirmation pattern used elsewhere in this project — see pipeline.py for exactly how a
momentary completion event is reconciled with that continuous-condition-style tracker.

One of THREE interchangeable gesture-detection methods — Method 1 is modules.wave_facing_gate,
Method 3 is modules.gesture_trajectory_verifier. Both are untouched by this redesign. This
module shares NO code or state with either.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import GestureHandKeypointConfig, load_config
from .pipeline import GestureHandKeypointPipeline

__all__ = ["GestureMethodResult", "evaluate", "release_track", "configure"]


@dataclass
class GestureMethodResult:
    track_id: int
    is_waving: bool                          # debounced, confirmed result (RED/YELLOW/GREEN)
    waving_state: str                         # "RED" | "YELLOW" | "GREEN"
    method_name: str = "hand_keypoint"        # fixed for this module
    confidence_debug: Optional[float] = None  # best hand's handedness/detection confidence
    keypoints_raw: Optional[object] = None    # raw MediaPipe per-hand landmark data, for debugging
    # Method-2-specific extra fields, beyond the 3-method shared contract:
    palm_facing_camera_debug: Optional[bool] = None  # DEBUG ONLY, does not gate is_waving
    sequence_stage: str = "WAITING_OPEN"       # WAITING_OPEN | WAITING_CLOSE_1 | WAITING_OPEN_2 |
                                                 # WAITING_CLOSE_2 | CONFIRMED — furthest-advanced
                                                 # hand's stage this frame, for debugging the sequence
    open_count: int = 0                        # opens consumed so far in the CURRENT attempt (0-2),
                                                 # a pure function of sequence_stage — debug/logging only
    close_count: int = 0                       # closes consumed so far in the CURRENT attempt (0-2)
    total_confirmed_count: int = 0             # cumulative CONFIRMED pulses for this track_id,
                                                 # session lifetime (since the last release_track())

    def draw_debug(self, frame: np.ndarray,
                    person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Draws all 21 hand landmarks + skeleton (colored yellow=OPEN/green=CLOSED/gray=NEITHER)
        + per-finger extended/curled coloring for every hand detected this frame, using
        keypoints_raw — no extra inference. Also draws a red dotted line at the
        palm_height_fraction calibration cutoff, in crop-local coordinates — pass the SAME
        `person_bbox_full_frame` given to evaluate() for this to line up correctly (defaults to
        treating the whole crop as its own bbox, same fallback evaluate() uses). No-ops entirely
        if there's nothing to draw (e.g. missing config, no hand detected that frame).
        """
        from . import visualize
        from .bbox_context import BboxContext
        config = _get_pipeline().config
        bbox = (BboxContext.from_person_bbox(person_bbox_full_frame) if person_bbox_full_frame is not None
                else BboxContext.whole_crop_as_bbox(frame.shape))
        if config.palm_height_fraction is not None:
            visualize.draw_palm_height_threshold(frame, bbox, config)
        if self.keypoints_raw:
            visualize.draw_hand_debug(frame, self.keypoints_raw, config)


_pipeline_singleton: Optional[GestureHandKeypointPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first evaluate() call. If never called, evaluate() lazily initializes on first use."""
    global _pipeline_singleton
    config: GestureHandKeypointConfig = load_config(thresholds_config_path)
    _pipeline_singleton = GestureHandKeypointPipeline(config)


def _get_pipeline() -> GestureHandKeypointPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def evaluate(track_id: int, person_crop_bgr: np.ndarray, timestamp: float,
             person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> GestureMethodResult:
    """
    Input: a person bbox crop (from modules.human_detection_roi), already face-matched upstream.
    This module does its own hand detection/localization within that crop.

    `person_bbox_full_frame`: the SAME (x, y, w, h) tuple modules.human_detection_roi's
    HumanDetectionResult.person_bbox returns — needed for the palm-height gate, which measures
    against the person's full-frame bbox height, not just the crop. If omitted (e.g. this
    module's own standalone test script, run without a real upstream detector), the whole crop
    is treated as its own bbox at offset (0, 0) — a documented simplification, not a silent
    approximation.
    """
    result = _get_pipeline().evaluate(track_id, person_crop_bgr, timestamp, person_bbox_full_frame)
    return GestureMethodResult(
        track_id=result.track_id,
        is_waving=result.is_waving,
        waving_state=result.waving_state,
        confidence_debug=result.confidence_debug,
        keypoints_raw=result.hands_raw,
        palm_facing_camera_debug=result.palm_facing_camera_debug,
        sequence_stage=result.sequence_stage,
        open_count=result.open_count,
        close_count=result.close_count,
        total_confirmed_count=result.total_confirmed_count,
    )


def release_track(track_id: int) -> None:
    """Drop a track_id's accumulated state (sequence machines, confirmation tracker) once the
    caller knows it's no longer active."""
    _get_pipeline().release_track(track_id)
