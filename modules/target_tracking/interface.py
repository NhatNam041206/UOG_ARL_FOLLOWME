"""
Target Recording + Tracking — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.target_tracking. Everything else in
this package (config.py, tracker.py, locking.py, pipeline.py) is an internal implementation
detail and may change without notice.

Purpose (plans/06_target_tracking.md): once a gesture trigger is confirmed (is_waving reaches
GREEN in whichever gesture method is active), this module takes over — locks onto that person as
"the target," records a short set of reference appearance frames, then tracks them continuously
frame-to-frame, reporting how far off-center they are (for downstream steering) every frame.

    gesture trigger GREEN -> start(initial_bbox, frame, ts)
            |
            v
       RECORDING  (record_duration_seconds; extends if too few usable crops were collected)
            |
            v
       TRACKING   (horizontal_offset + periodic appearance re-verify)
            |
            v
       LOST -> caller hands off to modules.target_recovery (specced/built separately)
            | (recovery re-acquires, calls reset() with a fresh bbox)
            v
       RECORDING again (loop)

If tracking is lost, this module's job ends at declaring LOST and providing the reference_set
modules.target_recovery needs — it does not implement recovery/search itself.

Isolation: this module runs its OWN independent YOLO+ByteTrack instance (tracker.py) — never
shared with modules.human_detection's, modules.emergency_stop's, or modules.human_detection_roi's
own separate instances, even though all load the same yolo11n.onnx weights file. It calls
modules.appearance_verifier's public interface.py functions for the periodic re-verification
described below — that is the sanctioned, intended use of that module's public contract, not a
violation of isolation (isolation forbids sharing state/instances, not calling another module's
public functions).

Manages a SINGLE active follow-me episode at a time — start()/update()/reset() take no track_id,
unlike every other module in this project.

KNOWN LIMITATIONS:
  - ByteTrack's track_id continuity is motion-based, not identity-verified (same caveat
    documented for modules.human_detection). In a crowd, ByteTrack can silently reassign the
    locked track_id to a different nearby person after an occlusion, without ever reporting a
    track loss. The periodic appearance re-verification below exists specifically to catch this
    — it is not a general accuracy improvement, it is the mitigation for this exact failure mode.
    Two consecutive failed re-verifies (not one) are required before declaring LOST, so a single
    bad-lighting/occlusion frame doesn't trigger an unnecessary full recovery cycle.
  - No true-angle/FOV-based steering computation — horizontal_offset is a normalized -1..+1
    value, not a real angle. Converting to an angle using the camera's FOV is deliberately the
    downstream steering layer's job, not this module's (spec §4.2's explicit architecture
    boundary) — camera.fov_degrees never appears in this module's config.
  - No PID or any control-loop logic whatsoever.
"""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from .config import TargetTrackingConfig, load_config
from .pipeline import TargetTrackingPipeline

__all__ = ["TrackingResult", "start", "update", "reset", "configure"]

_STATE_COLOR = {"RECORDING": (0, 220, 255), "TRACKING": (0, 200, 0), "LOST": (0, 0, 255)}


@dataclass
class TrackingResult:
    target_locked: bool                                    # True while in RECORDING or TRACKING; False once LOST
    horizontal_offset: Optional[float]                      # normalized -1.0..+1.0, 0.0 = centered; None if target_locked is False
    person_bbox: Optional[Tuple[int, int, int, int]]         # (x, y, w, h), full-frame pixel space
    state: Literal["RECORDING", "TRACKING", "LOST"]
    reference_set: Optional[object]                          # appearance_verifier.ReferenceEmbeddingSet, once RECORDING
                                                               # completes — forward this into modules.target_recovery on LOST
    # Debug/visualization only — the most recent periodic appearance re-verify outcome this
    # episode, if one has run yet (see modules.appearance_verifier.verify(), called internally).
    last_reverify_score: Optional[float] = None
    last_reverify_pass: Optional[bool] = None

    def draw_debug(self, frame: np.ndarray) -> None:
        """
        Draws the tracked bbox (colored by state) and a vertical frame-center reference line,
        directly onto `frame` (full-frame coordinates) — plus the last re-verify score/pass, if
        one has run. Externally callable so any caller gets the identical overlay
        modules/target_tracking/visualize_target_tracking.py already draws, without
        re-implementing it. No-ops (bbox only) if person_bbox is None.
        """
        color = _STATE_COLOR.get(self.state, (255, 255, 255))
        frame_h, frame_w = frame.shape[:2]
        cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (120, 120, 120), 1)

        if self.person_bbox is not None:
            x, y, w, h = self.person_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        offset_str = f"{self.horizontal_offset:+.2f}" if self.horizontal_offset is not None else "None"
        cv2.putText(frame, f"tracking: state={self.state} offset={offset_str}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.last_reverify_score is not None:
            cv2.putText(frame, f"reverify: score={self.last_reverify_score:.4f} pass={self.last_reverify_pass}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


_pipeline_singleton: Optional[TargetTrackingPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first start() call. If never called, lazily initializes on first use."""
    global _pipeline_singleton
    config: TargetTrackingConfig = load_config(thresholds_config_path)
    _pipeline_singleton = TargetTrackingPipeline(config)


def _get_pipeline() -> TargetTrackingPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def start(initial_person_bbox: Tuple[int, int, int, int], frame: np.ndarray, timestamp: float) -> None:
    """Called once, when the gesture trigger first goes GREEN. Locks onto initial_person_bbox
    ((x, y, w, h), full-frame pixel space — same convention as modules.human_detection_roi's
    person_bbox, the pipeline's own upstream source for this value) as the target and begins the
    RECORDING phase."""
    _get_pipeline().start(initial_person_bbox, frame, timestamp)


def update(frame: np.ndarray, timestamp: float) -> TrackingResult:
    """Called once per frame while this module owns the active follow-me episode (from start()
    until state becomes LOST and the caller has taken over)."""
    result = _get_pipeline().update(frame, timestamp)
    return TrackingResult(
        target_locked=result.target_locked,
        horizontal_offset=result.horizontal_offset,
        person_bbox=result.person_bbox,
        state=result.state,
        reference_set=result.reference_set,
        last_reverify_score=result.last_reverify_score,
        last_reverify_pass=result.last_reverify_pass,
    )


def reset(fresh_person_bbox: Tuple[int, int, int, int], frame: np.ndarray, timestamp: float) -> None:
    """Called once modules.target_recovery reports REACQUIRED, with the freshly re-acquired
    bbox — re-enters RECORDING with that new bbox, same as a fresh start(). (NOTE: this
    signature intentionally differs from plans/06_target_tracking.md §0.3's literal draft, which
    omitted the bbox/frame/timestamp parameters despite reset()'s own docstring describing a
    fresh bbox being handed back — see pipeline.py's reset() for the full note.)"""
    _get_pipeline().reset(fresh_person_bbox, frame, timestamp)
