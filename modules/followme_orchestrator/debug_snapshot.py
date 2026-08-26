"""
Pure data-shaping helper — assembles the plain dict logged per frame (see
plans/10_debug_logging_observability.md) from this package's already-computed result objects.

Deliberately its OWN file with NO imports from any CV module (face_identity,
gesture_hand_keypoint, autocar_adapter, etc.) and no model loading of any kind — this is what
makes it independently unit-testable without mediapipe/onnxruntime/etc. installed, unlike the
rest of this package. pipeline.py is the only real caller; it passes its own already-computed
result objects in.
"""
from typing import Any, Dict, Optional

__all__ = ["build_debug_snapshot"]


def build_debug_snapshot(face_result: Optional[Any] = None, person_result: Optional[Any] = None,
                          gesture_result: Optional[Any] = None,
                          tracking_result: Optional[Any] = None) -> Dict[str, Optional[dict]]:
    """
    Each argument is duck-typed against this package's own result objects — FaceIdentityResult,
    HumanDetectionResult, gesture_hand_keypoint.GestureMethodResult,
    autocar_adapter.TrackingResult — and only needs to expose the specific attributes read below,
    not be the real class (that's what keeps this testable with plain fakes). Pass None for
    whichever phase(s) did not run this frame — mirrors draw_debug()'s own "only whichever
    phase(s) actually ran this step() call" convention; the corresponding output key is then None
    too, rather than a block of stale/fabricated values.
    """
    face_identity = None
    if face_result is not None:
        face_identity = {
            "face_found": face_result.face_found,
            "matched_person_name": face_result.matched_person_name,
            "match_confidence": face_result.match_confidence,
        }

    human_detection_roi = None
    if person_result is not None:
        human_detection_roi = {
            "person_found": person_result.person_found,
            "detection_confidence": person_result.detection_confidence,
        }

    gesture = None
    if gesture_result is not None:
        gesture = {
            "waving_state": gesture_result.waving_state,
            "sequence_stage": gesture_result.sequence_stage,
            "open_count": gesture_result.open_count,
            "close_count": gesture_result.close_count,
            "total_confirmed_count_session": gesture_result.total_confirmed_count,
        }

    tracking = None
    if tracking_result is not None:
        tracking = {
            "target_locked": tracking_result.target_locked,
            "state": tracking_result.state,
            "horizontal_offset": tracking_result.horizontal_offset,
            "just_reacquired": tracking_result.just_reacquired,
        }

    return {
        "face_identity": face_identity,
        "human_detection_roi": human_detection_roi,
        "gesture": gesture,
        "tracking": tracking,
    }
