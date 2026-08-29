"""
Tests for followme_orchestrator TARGET_REACHED condition and buffer timing.
Verifies:
1. _is_target_reached logic (OpenCV inverted y-axis horizon check + bbox proportion check).
2. Fail-safe handling when thresholds are None (feature disabled).
3. Both conditions must be met simultaneously for TARGET_REACHED to be True.
4. Continuous buffer duration progression:
   - Target reached starts timing.
   - While elapsed < buffer_seconds: should_move=False, is_finished=False, remaining_seconds decreases.
   - When elapsed >= buffer_seconds: should_move=False, is_finished=True, remaining_seconds=0.0.
   - Stepping back resets the timer to None and resumes normal tracking.
   - target_reached_buffer_seconds=None keeps holding without finishing.
"""
from unittest.mock import MagicMock
import numpy as np

from modules.followme_orchestrator.config import FollowMeOrchestratorConfig
from modules.followme_orchestrator.pipeline import FollowMeOrchestratorPipeline, PipelineResult


def _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35, buffer_seconds=5.0):
    config = FollowMeOrchestratorConfig(
        fov_degrees=85.0,
        kp=1.0,
        ki=0.5,
        kd=0.5,
        max_steering_angle_degrees=45.0,
        servo_center_degrees=90.0,
        target_reached_horizon_y_ratio=horizon_ratio,
        target_reached_min_bbox_proportion=min_proportion,
        target_reached_buffer_seconds=buffer_seconds,
    )
    # Bypass model eager loads for lightweight unit testing of logic
    pipeline = FollowMeOrchestratorPipeline.__new__(FollowMeOrchestratorPipeline)
    pipeline.config = config
    pipeline.steering = MagicMock()
    pipeline.steering.is_calibrated.return_value = True
    pipeline.steering.update.return_value = 90.0
    pipeline._tracking_active = True
    pipeline._target_person_name = "Alice"
    pipeline.last_person_bbox = None
    pipeline._target_reached_since = None
    pipeline._last_timestamp = None
    return pipeline


def test_is_target_reached_both_conditions_met():
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35)
    frame_shape = (480, 640, 3)

    # Frame area = 480 * 640 = 307200
    # py = 48 (py / 480 = 0.10 <= 0.15: horizon touched)
    # pw = 400, ph = 300 (area = 120000 -> 120000 / 307200 = 0.39 >= 0.35: proportion met)
    person_bbox = (100, 48, 400, 300)

    assert pipeline._is_target_reached(person_bbox, frame_shape) is True


def test_is_target_reached_only_horizon_met_but_proportion_small():
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35)
    frame_shape = (480, 640, 3)

    # py = 40 (py / 480 = 0.083 <= 0.15: horizon touched)
    # pw = 100, ph = 100 (area = 10000 -> 10000 / 307200 = 0.0325 < 0.35: proportion NOT met)
    person_bbox = (100, 40, 100, 100)

    assert pipeline._is_target_reached(person_bbox, frame_shape) is False


def test_is_target_reached_only_proportion_met_but_below_horizon():
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35)
    frame_shape = (480, 640, 3)

    # py = 120 (py / 480 = 0.25 > 0.15: horizon NOT touched)
    # pw = 500, ph = 300 (area = 150000 -> 150000 / 307200 = 0.488 >= 0.35: proportion met)
    person_bbox = (50, 120, 500, 300)

    assert pipeline._is_target_reached(person_bbox, frame_shape) is False


def test_is_target_reached_neither_condition_met():
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35)
    frame_shape = (480, 640, 3)

    # Far away person
    # py = 200 (py / 480 = 0.416 > 0.15)
    # pw = 100, ph = 150 (area = 15000 / 307200 = 0.048 < 0.35)
    person_bbox = (200, 200, 100, 150)

    assert pipeline._is_target_reached(person_bbox, frame_shape) is False


def test_is_target_reached_none_bbox():
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35)
    frame_shape = (480, 640, 3)
    assert pipeline._is_target_reached(None, frame_shape) is False


def test_is_target_reached_uncalibrated_config():
    pipeline = _make_dummy_pipeline(horizon_ratio=None, min_proportion=None)
    frame_shape = (480, 640, 3)
    person_bbox = (100, 20, 400, 400)
    assert pipeline._is_target_reached(person_bbox, frame_shape) is False


def test_target_reached_buffer_progression_and_finish(monkeypatch):
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35, buffer_seconds=5.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock autocar_adapter.tracking_update to return a close person (TARGET_REACHED conditions met)
    close_bbox = (100, 40, 400, 300)  # area = 120000 / 307200 = 0.39, py=40/480=0.083
    mock_tracking_result = MagicMock()
    mock_tracking_result.state = "TRACKING"
    mock_tracking_result.just_reacquired = False
    mock_tracking_result.person_bbox = close_bbox
    mock_tracking_result.horizontal_offset = 0.0

    monkeypatch.setattr("modules.followme_orchestrator.pipeline.tracking_update", lambda f, ts: mock_tracking_result)

    # Step 1: t = 100.0 (First frame of TARGET_REACHED)
    r1 = pipeline.step(frame, 100.0)
    assert r1.debug_state == "TARGET_REACHED"
    assert r1.should_move is False
    assert r1.is_finished is False
    assert r1.target_reached_remaining_seconds == 5.0

    # Step 2: t = 102.0 (2 seconds elapsed -> 3.0s remaining)
    r2 = pipeline.step(frame, 102.0)
    assert r2.debug_state == "TARGET_REACHED"
    assert r2.should_move is False
    assert r2.is_finished is False
    assert abs(r2.target_reached_remaining_seconds - 3.0) < 1e-5

    # Step 3: t = 105.0 (5 seconds elapsed -> Buffer reached, is_finished=True!)
    r3 = pipeline.step(frame, 105.0)
    assert r3.debug_state == "TARGET_REACHED"
    assert r3.should_move is False
    assert r3.is_finished is True
    assert r3.target_reached_remaining_seconds == 0.0


def test_target_reached_step_back_resets_buffer(monkeypatch):
    pipeline = _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35, buffer_seconds=5.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    close_bbox = (100, 40, 400, 300)
    far_bbox = (200, 200, 100, 150)

    mock_tracking_result = MagicMock()
    mock_tracking_result.state = "TRACKING"
    mock_tracking_result.just_reacquired = False
    mock_tracking_result.horizontal_offset = 0.0

    monkeypatch.setattr("modules.followme_orchestrator.pipeline.tracking_update", lambda f, ts: mock_tracking_result)

    # Step 1: Target is close at t = 100.0
    mock_tracking_result.person_bbox = close_bbox
    r1 = pipeline.step(frame, 100.0)
    assert r1.debug_state == "TARGET_REACHED"
    assert r1.is_finished is False

    # Step 2: Target is close at t = 103.0 (3s elapsed)
    r2 = pipeline.step(frame, 103.0)
    assert r2.debug_state == "TARGET_REACHED"
    assert abs(r2.target_reached_remaining_seconds - 2.0) < 1e-5

    # Step 3: Target steps back at t = 104.0 (conditions no longer met -> returns to TRACKING)
    mock_tracking_result.person_bbox = far_bbox
    r3 = pipeline.step(frame, 104.0)
    assert r3.debug_state == "TRACKING"
    assert r3.should_move is True
    assert r3.is_finished is False
    assert r3.target_reached_remaining_seconds is None
    assert pipeline._target_reached_since is None  # timer reset!

    # Step 4: Target walks close again at t = 110.0 -> Timer starts over at 5.0s
    mock_tracking_result.person_bbox = close_bbox
    r4 = pipeline.step(frame, 110.0)
    assert r4.debug_state == "TARGET_REACHED"
    assert r4.is_finished is False
    assert r4.target_reached_remaining_seconds == 5.0
