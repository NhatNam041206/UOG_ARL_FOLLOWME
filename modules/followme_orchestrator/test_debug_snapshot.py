"""
Unit tests for debug_snapshot.build_debug_snapshot — pure dict-shaping logic, no CV models, no
video file, no mediapipe/onnxruntime dependency (this file and debug_snapshot.py both
deliberately avoid importing anything from this package's real pipeline.py, which DOES need those
— see debug_snapshot.py's own docstring). Uses plain duck-typed fakes for the four result types.

Run with:
    python -m pytest modules/followme_orchestrator/test_debug_snapshot.py -v
"""
from dataclasses import dataclass
from typing import Optional

from modules.followme_orchestrator.debug_snapshot import build_debug_snapshot


@dataclass
class _FakeFaceResult:
    face_found: bool
    matched_person_name: Optional[str]
    match_confidence: Optional[float]


@dataclass
class _FakePersonResult:
    person_found: bool
    detection_confidence: Optional[float]


@dataclass
class _FakeGestureResult:
    waving_state: str
    sequence_stage: str
    open_count: int
    close_count: int
    total_confirmed_count: int


@dataclass
class _FakeTrackingResult:
    target_locked: bool
    state: str
    horizontal_offset: Optional[float]
    just_reacquired: bool


def test_all_none_when_nothing_ran():
    snapshot = build_debug_snapshot()
    assert snapshot == {
        "face_identity": None, "human_detection_roi": None, "gesture": None, "tracking": None,
    }


def test_face_and_person_only():
    face = _FakeFaceResult(face_found=True, matched_person_name="alice", match_confidence=0.87)
    person = _FakePersonResult(person_found=True, detection_confidence=0.91)

    snapshot = build_debug_snapshot(face_result=face, person_result=person)

    assert snapshot["face_identity"] == {
        "face_found": True, "matched_person_name": "alice", "match_confidence": 0.87,
    }
    assert snapshot["human_detection_roi"] == {"person_found": True, "detection_confidence": 0.91}
    assert snapshot["gesture"] is None
    assert snapshot["tracking"] is None


def test_gesture_fields_mapped_correctly():
    gesture = _FakeGestureResult(
        waving_state="GREEN", sequence_stage="CONFIRMED",
        open_count=2, close_count=2, total_confirmed_count=3,
    )

    snapshot = build_debug_snapshot(gesture_result=gesture)

    assert snapshot["gesture"] == {
        "waving_state": "GREEN", "sequence_stage": "CONFIRMED",
        "open_count": 2, "close_count": 2, "total_confirmed_count_session": 3,
    }


def test_tracking_fields_mapped_correctly():
    tracking = _FakeTrackingResult(
        target_locked=True, state="TRACKING", horizontal_offset=0.12, just_reacquired=False,
    )

    snapshot = build_debug_snapshot(tracking_result=tracking)

    assert snapshot["tracking"] == {
        "target_locked": True, "state": "TRACKING", "horizontal_offset": 0.12, "just_reacquired": False,
    }


def test_all_four_populated_together():
    face = _FakeFaceResult(face_found=True, matched_person_name="bob", match_confidence=0.7)
    person = _FakePersonResult(person_found=True, detection_confidence=0.8)
    gesture = _FakeGestureResult(
        waving_state="YELLOW", sequence_stage="WAITING_CLOSE_1",
        open_count=1, close_count=0, total_confirmed_count=0,
    )
    tracking = _FakeTrackingResult(
        target_locked=False, state="SEARCHING", horizontal_offset=None, just_reacquired=False,
    )

    snapshot = build_debug_snapshot(face, person, gesture, tracking)

    assert all(snapshot[key] is not None for key in ("face_identity", "human_detection_roi", "gesture", "tracking"))
    assert snapshot["tracking"]["state"] == "SEARCHING"
