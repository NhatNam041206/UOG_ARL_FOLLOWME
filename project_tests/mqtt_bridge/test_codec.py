"""
Unit tests for mqtt_bridge.codec.encode — pure logic, no MQTT broker or network needed (mirrors
project_tests/gesture_hand_keypoint/test_sequence_counts.py's approach: real network-facing behavior
stays untested here, this file covers the string-encoding boundary in isolation).

Run with:
    python -m pytest project_tests/mqtt_bridge/test_codec.py -v
"""
import pytest

from modules.mqtt_bridge.codec import encode

_CENTER = 90.0


def test_moving_encodes_given_angle_and_is_moving_1():
    assert encode(True, 95.3, _CENTER) == "95,1"


def test_not_moving_encodes_center_angle_and_is_moving_0():
    assert encode(False, 130.0, _CENTER) == "90,0"


def test_none_angle_encodes_center_angle():
    assert encode(True, None, _CENTER) == "90,1"


def test_none_angle_while_not_moving_encodes_center_angle_and_is_moving_0():
    assert encode(False, None, _CENTER) == "90,0"


def test_angle_rounds_to_nearest_integer():
    assert encode(True, 44.6, _CENTER) == "45,1"
    assert encode(True, 44.4, _CENTER) == "44,1"


def test_boundary_zero_is_valid():
    assert encode(True, 0.0, _CENTER) == "0,1"


def test_boundary_180_is_valid():
    assert encode(True, 180.0, _CENTER) == "180,1"


def test_out_of_range_low_raises():
    with pytest.raises(ValueError):
        encode(True, -1.0, _CENTER)


def test_out_of_range_high_raises():
    with pytest.raises(ValueError):
        encode(True, 181.0, _CENTER)


def test_out_of_range_center_raises():
    with pytest.raises(ValueError):
        encode(False, None, 200.0)
