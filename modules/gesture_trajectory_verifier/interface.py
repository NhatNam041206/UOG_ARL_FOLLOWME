"""
Gesture Detection — Method 3, Trajectory Verifier — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.gesture_trajectory_verifier.
Everything else in this package is an internal implementation detail.

Third of three interchangeable gesture-detection methods (spec §0.3) — Method 1 is
modules.wave_facing_gate, Method 2 is modules.gesture_hand_keypoint. Shares NO code or state
with either; only the underlying MoveNet Lightning MODEL is reused (spec §0.3: reusing the model
itself is fine, reusing another method's code/classes operating on it is not).

Design (spec §1-§4): tracks wrist+elbow+shoulder for ONE arm at a time (both arms computed and
compared independently each cycle, whichever scores higher wins), normalizes (translate to
own-start, scale by bbox height at capture) and resamples to a fixed length — TIME-BASED
resampling, confirmed with the user over arc-length-based (spec §2.3) — then compares via cosine
similarity against a small SHARED, GENERIC reference trajectory set (not per-person, confirmed
design, spec §4).

"Not ready" signal (spec §4.3, confirmed with the user): if the reference set has fewer than 2
entries, evaluate() returns is_waving=False with reference_count reflecting the actual (0 or 1)
count and confidence_debug/matched_reference_id both None — distinguishable from a genuine
non-match (which would have a real confidence_debug score and reference_count >= 2), so this
never gets misread as "evaluated and didn't match" during calibration.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GestureTrajectoryVerifierConfig, load_config
from .pipeline import GestureTrajectoryVerifierPipeline

__all__ = ["GestureMethodResult", "evaluate", "release_track", "configure"]


@dataclass
class GestureMethodResult:
    track_id: int
    is_waving: bool
    waving_state: str                                # "RED" | "YELLOW" | "GREEN"
    method_name: str = "trajectory_verifier"          # fixed for this module
    confidence_debug: Optional[float] = None          # best similarity score against the reference set
    matched_reference_id: Optional[str] = None        # which reference trajectory scored best, for debugging
    keypoints_raw: Optional[object] = None
    # Method-3-specific extra fields, beyond the 3-method shared contract:
    arm: Optional[str] = None                         # "left" | "right" — which arm produced the best score
    reference_count: int = 0                          # size of the reference set this evaluation used;
                                                         # < 2 means "not ready" (spec §4.3), not a real non-match


_pipeline_singleton: Optional[GestureTrajectoryVerifierPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first evaluate() call. If never called, evaluate() lazily initializes on first use."""
    global _pipeline_singleton
    config: GestureTrajectoryVerifierConfig = load_config(thresholds_config_path)
    _pipeline_singleton = GestureTrajectoryVerifierPipeline(config)


def _get_pipeline() -> GestureTrajectoryVerifierPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def evaluate(track_id: int, person_crop_bgr: np.ndarray, timestamp: float) -> GestureMethodResult:
    """Input: person crop from modules.human_detection_roi, same as Methods 1 and 2."""
    result = _get_pipeline().evaluate(track_id, person_crop_bgr, timestamp)
    return GestureMethodResult(
        track_id=result.track_id,
        is_waving=result.is_waving,
        waving_state=result.waving_state,
        confidence_debug=result.confidence_debug,
        matched_reference_id=result.matched_reference_id,
        keypoints_raw=result.keypoints_raw,
        arm=result.arm,
        reference_count=result.reference_count,
    )


def release_track(track_id: int) -> None:
    """Drop a track_id's accumulated state (motion buffers, confirmation tracker)."""
    _get_pipeline().release_track(track_id)
