"""
Gate A — static pose gate (spec §4.1): per-frame, no memory. Does a single frame's arm geometry
look like a raised, vertically-extended arm? Deliberately knows nothing about motion (gate_b.py)
or about previous frames — see spec §5 on the two gates' independence.
"""
from math import atan2, degrees

from .config import WaveFacingConfig
from .constants import Keypoint


def wrist_above_bbox_fraction(wrist_y_px: float, bbox_height_px: float, config: WaveFacingConfig) -> bool:
    return wrist_y_px < config.wrist_height_fraction * bbox_height_px


def wrist_above_elbow(wrist: Keypoint, elbow: Keypoint) -> bool:
    return wrist.y < elbow.y


def angle_from_vertical_deg(p_from: Keypoint, p_to: Keypoint) -> float:
    dx = p_to.x - p_from.x
    dy = p_to.y - p_from.y
    return degrees(atan2(abs(dx), abs(dy)))


def arm_is_vertical(wrist: Keypoint, elbow: Keypoint, shoulder: Keypoint, config: WaveFacingConfig) -> bool:
    # Both wrist->elbow AND wrist->shoulder must independently clear the threshold — deliberate
    # redundancy so one noisy keypoint doesn't singlehandedly pass the check (spec §4.1).
    angle_wrist_elbow = angle_from_vertical_deg(wrist, elbow)
    angle_wrist_shoulder = angle_from_vertical_deg(wrist, shoulder)
    return (angle_wrist_elbow <= config.verticality_threshold_deg
            and angle_wrist_shoulder <= config.verticality_threshold_deg)


def gate_a_pass(wrist: Keypoint, elbow: Keypoint, shoulder: Keypoint, bbox_height_px: float,
                 config: WaveFacingConfig) -> bool:
    # Fail closed if any required keypoint is below the confidence floor (spec §4.1) — a low
    # confidence reading is treated as "cannot evaluate", not as a pass.
    if min(wrist.score, elbow.score, shoulder.score) < config.confidence_threshold_pose:
        return False
    return (wrist_above_bbox_fraction(wrist.y, bbox_height_px, config)
            and wrist_above_elbow(wrist, elbow)
            and arm_is_vertical(wrist, elbow, shoulder, config))
