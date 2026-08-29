"""
Tests for followme_orchestrator TARGET_REACHED condition.
Verifies:
1. _is_target_reached logic (OpenCV inverted y-axis horizon check + bbox proportion check).
2. Fail-safe handling when thresholds are None (feature disabled).
3. Both conditions must be met simultaneously for TARGET_REACHED to be True.
"""
from modules.followme_orchestrator.config import FollowMeOrchestratorConfig
from modules.followme_orchestrator.pipeline import FollowMeOrchestratorPipeline, PipelineResult


def _make_dummy_pipeline(horizon_ratio=0.15, min_proportion=0.35):
    config = FollowMeOrchestratorConfig(
        fov_degrees=85.0,
        kp=1.0,
        ki=0.5,
        kd=0.5,
        max_steering_angle_degrees=45.0,
        servo_center_degrees=90.0,
        target_reached_horizon_y_ratio=horizon_ratio,
        target_reached_min_bbox_proportion=min_proportion,
    )
    # Bypass model eager loads for lightweight unit testing of logic
    pipeline = FollowMeOrchestratorPipeline.__new__(FollowMeOrchestratorPipeline)
    pipeline.config = config
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
