"""
Target Recovery / Search — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.target_recovery. Everything else in
this package (config.py, detector.py, pipeline.py) is an internal implementation detail and may
change without notice.

Purpose (plans/07_target_recovery.md): when modules.target_tracking reports state == LOST, this
module takes over — searches the full frame (not the narrow region tracking was using) to
re-acquire the same target, via two paths of different strength and cost:

    target_tracking reports LOST, hands off reference_set
            |
            v
       start(reference_set, target_person_name, timestamp)
            |
            v
       SEARCHING loop, each frame via update():
            +-- Path A (primary, ALWAYS tried first): face_identity.evaluate() -> a match for
            |     target_person_name specifically? -> human_detection_roi.evaluate() -> fresh
            |     person_bbox -> REACQUIRED, reacquired_via="face_match"
            +-- Path B (fallback, only once face_search_fail_count >= face_search_grace_attempts
            |     CONSECUTIVE Path-A failures): whole-frame detection ->
            |     appearance_verifier.verify() per candidate -> best candidate clearing
            |     appearance_fallback_threshold -> REACQUIRED, reacquired_via="appearance_fallback"
            +-- elapsed_search_seconds >= search_timeout_seconds (checked every frame) -> TIMEOUT

REACQUIRED and TIMEOUT are terminal for a given episode. The caller (orchestration layer) is
responsible for calling modules.target_tracking.reset() with the reacquired bbox on REACQUIRED,
or propagating a stop signal on TIMEOUT — this module does not call target_tracking itself (no
cross-module instance sharing) and produces no robot commands of its own.

`target_person_name` (added beyond plans/07_target_recovery.md's literally drafted start()
signature, confirmed with the user — see plans/07 §0.5's own audit item #1, which flagged this as
a likely real gap): face_identity.evaluate() can return multiple registered people's matches in a
crowd; Path A needs to know WHICH one is actually this episode's target rather than accepting any
registered match.

Isolation: orchestrates calls to THREE existing modules' public interfaces (face_identity,
human_detection_roi, appearance_verifier) — the intended, sanctioned pattern (calling public
interface.py contracts is not a violation of isolation; sharing live state/instances is). Path B
runs its OWN independent whole-frame YOLO instance (detector.py), never
modules.human_detection's.

KNOWN LIMITATIONS:
  - `face_search_grace_attempts` is a COUNT of consecutive Path-A-failure frames, not a time
    duration (plans/07 §4.2). Face detection (YuNet, full-frame) is variable-cost inference — a
    time-based gate would give Path A an inconsistent number of real attempts depending on system
    load that cycle. A count-based gate ties the threshold to actual attempts made, independent
    of frame rate. Do not change this to a duration.
  - Path B inherits BOTH of modules.appearance_verifier's named accuracy risks — similar-clothing
    confusion and cross-domain generalization drop (see docs/modules.md's appearance_verifier
    section for the full detail; not re-explained here).
"""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from .config import TargetRecoveryConfig, load_config
from .pipeline import TargetRecoveryPipeline

__all__ = ["RecoveryResult", "start", "update", "configure"]

_STATUS_COLOR = {"SEARCHING": (0, 220, 255), "REACQUIRED": (0, 200, 0), "TIMEOUT": (0, 0, 255)}


@dataclass
class RecoveryResult:
    status: Literal["SEARCHING", "REACQUIRED", "TIMEOUT"]
    reacquired_person_bbox: Optional[Tuple[int, int, int, int]]  # full-frame pixel space; populated ONLY when status == REACQUIRED
    reacquired_via: Optional[Literal["face_match", "appearance_fallback"]]  # which path succeeded
    face_search_fail_count: int    # current consecutive Path-A-failure count, for debug/visualization
    elapsed_search_seconds: float  # for debug/visualization and the visible search-timeout countdown

    def draw_debug(self, frame: np.ndarray) -> None:
        """
        Draws the current search status, elapsed time, and consecutive fail count onto `frame`
        (full-frame coordinates); on REACQUIRED, also draws the reacquired bbox and which path
        succeeded. Externally callable so any caller gets the identical overlay
        modules/target_recovery/visualize_target_recovery.py already draws, without
        re-implementing it.
        """
        color = _STATUS_COLOR.get(self.status, (255, 255, 255))
        cv2.putText(frame, f"recovery: status={self.status} fails={self.face_search_fail_count} "
                            f"elapsed={self.elapsed_search_seconds:.1f}s", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.status == "REACQUIRED" and self.reacquired_person_bbox is not None:
            x, y, w, h = self.reacquired_person_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, f"REACQUIRED via {self.reacquired_via}", (x, max(15, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


_pipeline_singleton: Optional[TargetRecoveryPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first start() call. If never called, lazily initializes on first use."""
    global _pipeline_singleton
    config: TargetRecoveryConfig = load_config(thresholds_config_path)
    _pipeline_singleton = TargetRecoveryPipeline(config)


def _get_pipeline() -> TargetRecoveryPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def start(reference_set: object, target_person_name: str, timestamp: float) -> None:
    """
    Called once when modules.target_tracking reports state == LOST. `reference_set` is the
    TrackingResult.reference_set handed off from that module (built by
    modules.appearance_verifier.build_reference_set() during that module's RECORDING phase).
    `target_person_name` identifies WHICH registered person this recovery episode is for — see
    this module's docstring above for why this parameter exists.
    """
    _get_pipeline().start(reference_set, target_person_name, timestamp)


def update(frame: np.ndarray, registry: object, timestamp: float) -> RecoveryResult:
    """
    Called once per frame while searching. `registry` is the FaceRegistry needed by
    face_identity.evaluate() — pass the SAME object the main face_first pipeline already loads
    and passes to face_identity elsewhere; do not build a second registry instance.
    """
    result = _get_pipeline().update(frame, registry, timestamp)
    return RecoveryResult(
        status=result.status,
        reacquired_person_bbox=result.reacquired_person_bbox,
        reacquired_via=result.reacquired_via,
        face_search_fail_count=result.face_search_fail_count,
        elapsed_search_seconds=result.elapsed_search_seconds,
    )
