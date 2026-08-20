"""
Optional pose visualization for debugging (spec §11: calibration checklist).
Draws keypoints, skeleton, arm gates, and confirmation state onto the frame — purely diagnostic,
not part of the module's core output contract. Call this only if you're actively tuning thresholds
and need to see what MoveNet is detecting and what the gates are doing.

All drawing coordinates assume the input frame/crop is in the caller's coordinate space (not
model-input space) — keypoints are already decoded to bbox-pixel space by preprocessing.py.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .constants import ARM_KEYPOINTS, Keypoint

# Skeleton pairs: (from_idx, to_idx) — the standard MoveNet 17-point skeleton topology.
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

_KEYPOINT_RADIUS = 4
_SKELETON_THICKNESS = 2
_KEYPOINT_CONFIDENCE_THRESHOLD = 0.2  # only draw keypoints above this threshold
_COLORS = {
    "keypoint_high": (0, 255, 0),      # green: high confidence
    "keypoint_low": (0, 165, 255),     # orange: low confidence but above threshold
    "skeleton": (100, 100, 100),       # dark gray
    "arm_left": (255, 0, 0),           # blue (for left arm)
    "arm_right": (0, 0, 255),          # red (for right arm)
    "text_info": (255, 255, 255),      # white
    "bbox": (200, 200, 200),           # light gray
}


def draw_keypoints(frame: np.ndarray, keypoints: List[Keypoint]) -> None:
    """Draw all 17 keypoints as circles, colored by confidence."""
    for kp in keypoints:
        if kp.score < _KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        x, y = int(kp.x), int(kp.y)
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            color = _COLORS["keypoint_high"] if kp.score > 0.5 else _COLORS["keypoint_low"]
            cv2.circle(frame, (x, y), _KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, (x, y), _KEYPOINT_RADIUS, (0, 0, 0), 1)


def draw_skeleton(frame: np.ndarray, keypoints: List[Keypoint]) -> None:
    """Draw skeleton connections between keypoints."""
    for from_idx, to_idx in SKELETON_PAIRS:
        from_kp = keypoints[from_idx]
        to_kp = keypoints[to_idx]
        if from_kp.score >= _KEYPOINT_CONFIDENCE_THRESHOLD and to_kp.score >= _KEYPOINT_CONFIDENCE_THRESHOLD:
            from_pt = (int(from_kp.x), int(from_kp.y))
            to_pt = (int(to_kp.x), int(to_kp.y))
            if (0 <= from_pt[0] < frame.shape[1] and 0 <= from_pt[1] < frame.shape[0] and
                0 <= to_pt[0] < frame.shape[1] and 0 <= to_pt[1] < frame.shape[0]):
                cv2.line(frame, from_pt, to_pt, _COLORS["skeleton"], _SKELETON_THICKNESS)


def draw_arm_vectors(frame: np.ndarray, keypoints: List[Keypoint], gate_a_passes: dict) -> None:
    """Draw arm vectors (wrist->elbow->shoulder) with colors indicating Gate A pass/fail."""
    for side, (wrist_idx, elbow_idx, shoulder_idx) in ARM_KEYPOINTS.items():
        wrist = keypoints[wrist_idx]
        elbow = keypoints[elbow_idx]
        shoulder = keypoints[shoulder_idx]

        if wrist.score < _KEYPOINT_CONFIDENCE_THRESHOLD or elbow.score < _KEYPOINT_CONFIDENCE_THRESHOLD:
            continue

        # Color based on Gate A result for this arm.
        gate_a_pass = gate_a_passes.get(side, False)
        color = _COLORS["arm_left" if side == "left" else "arm_right"]
        line_thickness = 3 if gate_a_pass else 1

        wrist_pt = (int(wrist.x), int(wrist.y))
        elbow_pt = (int(elbow.x), int(elbow.y))
        shoulder_pt = (int(shoulder.x), int(shoulder.y))

        # Draw wrist->elbow, elbow->shoulder
        if (0 <= wrist_pt[0] < frame.shape[1] and 0 <= wrist_pt[1] < frame.shape[0] and
            0 <= elbow_pt[0] < frame.shape[1] and 0 <= elbow_pt[1] < frame.shape[0]):
            cv2.line(frame, wrist_pt, elbow_pt, color, line_thickness, cv2.LINE_AA)

        if (0 <= elbow_pt[0] < frame.shape[1] and 0 <= elbow_pt[1] < frame.shape[0] and
            0 <= shoulder_pt[0] < frame.shape[1] and 0 <= shoulder_pt[1] < frame.shape[0]):
            cv2.line(frame, elbow_pt, shoulder_pt, color, line_thickness, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    """Draw a bounding box rectangle."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_info_text(frame: np.ndarray, texts: List[str], start_y: int = 10) -> None:
    """Draw debug info text lines on the frame."""
    y = start_y
    for text in texts:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLORS["text_info"], 1)
        y += 20
