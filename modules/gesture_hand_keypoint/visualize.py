"""
Optional hand keypoint visualization for debugging — mirrors modules/wave_facing_gate/visualize.py's
pattern. Draws all 21 MediaPipe landmarks + the standard hand skeleton (colored by the current
OPEN/CLOSED/NEITHER classification), per-finger extended/curled coloring (the exact signal
hand_shape.py's classification uses, including the thumb's distance-based test), and a
calibration line for the palm-height gate. Purely diagnostic, not part of the module's core
output contract.
"""
from typing import List

import cv2

from .bbox_context import BboxContext
from .config import GestureHandKeypointConfig
from .constants import FINGER_TIP_PIP_PAIRS, THUMB_TIP, WRIST
from .hand_detector import DetectedHand
from .hand_shape import CLOSED, NEITHER, OPEN, classify_hand_shape, is_finger_extended, is_thumb_extended

# Standard MediaPipe Hands 21-point skeleton topology (thumb, index, middle, ring, pinky chains
# off the wrist, plus the palm-base connections between each finger's MCP joint).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle (+ palm: index MCP -> middle MCP)
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring (+ palm: middle MCP -> ring MCP)
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky (+ palm: ring MCP -> pinky MCP)
    (0, 17),                                  # palm closure: wrist -> pinky MCP
]

_KEYPOINT_COLOR = (0, 255, 0)
_SKELETON_COLOR = (180, 180, 180)      # fallback/neutral, unused now that skeleton is shape-colored
_EXTENDED_COLOR = (0, 255, 0)           # per-finger edge, when hand_shape.py counts it as extended
_CURLED_COLOR = (0, 0, 255)             # per-finger edge, when curled
_OPEN_HAND_COLOR = (0, 255, 255)        # yellow (BGR) — whole hand, classified OPEN
_CLOSED_HAND_COLOR = (0, 200, 0)        # green (BGR) — whole hand, classified CLOSED
_NEITHER_HAND_COLOR = (200, 200, 200)   # gray — ambiguous/in-between, neither cleanly
_THRESHOLD_LINE_COLOR = (0, 0, 255)     # red — palm-height gate calibration line
_KEYPOINT_RADIUS = 4

_HAND_SHAPE_COLOR = {OPEN: _OPEN_HAND_COLOR, CLOSED: _CLOSED_HAND_COLOR, NEITHER: _NEITHER_HAND_COLOR}

# Config keys classify_hand_shape()/is_thumb_extended() need — comparing against a still-null
# (uncalibrated) threshold raises a TypeError, so classification-dependent coloring is skipped
# (falls back to neutral gray) rather than crashing, keeping keypoints visible pre-calibration —
# same "detection/drawing works uncalibrated, only the decision is gated" precedent as elsewhere.
_REQUIRED_FOR_CLASSIFICATION = ("min_fingers_extended_open", "min_fingers_curled_closed", "thumb_extension_ratio_threshold")


def _config_ready_for_classification(config: GestureHandKeypointConfig) -> bool:
    return all(getattr(config, k) is not None for k in _REQUIRED_FOR_CLASSIFICATION)


def _in_bounds(frame, x: int, y: int) -> bool:
    return 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]


def draw_keypoints(frame, landmarks_px, color=_KEYPOINT_COLOR) -> None:
    for x, y in landmarks_px:
        xi, yi = int(x), int(y)
        if _in_bounds(frame, xi, yi):
            cv2.circle(frame, (xi, yi), _KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, (xi, yi), _KEYPOINT_RADIUS, (0, 0, 0), 1)


def draw_skeleton(frame, landmarks_px, color=_SKELETON_COLOR) -> None:
    for a, b in HAND_CONNECTIONS:
        xa, ya = landmarks_px[a]
        xb, yb = landmarks_px[b]
        pa, pb = (int(xa), int(ya)), (int(xb), int(yb))
        if _in_bounds(frame, *pa) and _in_bounds(frame, *pb):
            cv2.line(frame, pa, pb, color, 2)


def draw_finger_extension(frame, landmarks_px, config: GestureHandKeypointConfig) -> None:
    """Highlights each finger's key edge green if hand_shape.py's classification would count it
    as extended, red if curled — makes the exact per-finger OPEN/CLOSED signal visible, thumb
    included. Independent of the overall hand-shape color (see draw_hand_debug)."""
    for tip_idx, pip_idx in FINGER_TIP_PIP_PAIRS.values():
        extended = is_finger_extended(landmarks_px, tip_idx, pip_idx)
        color = _EXTENDED_COLOR if extended else _CURLED_COLOR
        tip, pip = landmarks_px[tip_idx], landmarks_px[pip_idx]
        p_tip, p_pip = (int(tip[0]), int(tip[1])), (int(pip[0]), int(pip[1]))
        if _in_bounds(frame, *p_tip) and _in_bounds(frame, *p_pip):
            cv2.line(frame, p_pip, p_tip, color, 3)

    # Thumb: distance-based test (thumb_tip to pinky_MCP), drawn as a line from thumb tip to
    # wrist so its color is visible even though the geometric test itself uses pinky_MCP as the
    # reference point, not the wrist — this edge is just for display.
    thumb_extended = is_thumb_extended(landmarks_px, config)
    color = _EXTENDED_COLOR if thumb_extended else _CURLED_COLOR
    tip, wrist = landmarks_px[THUMB_TIP], landmarks_px[WRIST]
    p_tip, p_wrist = (int(tip[0]), int(tip[1])), (int(wrist[0]), int(wrist[1]))
    if _in_bounds(frame, *p_tip) and _in_bounds(frame, *p_wrist):
        cv2.line(frame, p_wrist, p_tip, color, 3)


def draw_hand_debug(frame, hands: List[DetectedHand], config: GestureHandKeypointConfig) -> None:
    """Draws, per detected hand: the skeleton + keypoints colored by the CURRENT classification
    (yellow = OPEN, green = CLOSED, gray = NEITHER/ambiguous), plus the per-finger
    extended/curled highlight edges (a separate, more granular diagnostic — see
    draw_finger_extension). Falls back to a neutral gray skeleton/keypoints, with no per-finger
    coloring, if classification's own thresholds aren't calibrated yet — keypoints stay visible
    either way."""
    ready = _config_ready_for_classification(config)
    for hand in hands:
        if ready:
            shape = classify_hand_shape(hand.landmarks_px, config)
            color = _HAND_SHAPE_COLOR[shape]
            draw_finger_extension(frame, hand.landmarks_px, config)
        else:
            color = _SKELETON_COLOR
        draw_skeleton(frame, hand.landmarks_px, color)
        draw_keypoints(frame, hand.landmarks_px, color)


def _draw_dotted_hline(frame, y: int, x_start: int, x_end: int, color,
                        thickness: int = 2, dash_len: int = 10, gap_len: int = 8) -> None:
    x = x_start
    while x < x_end:
        x2 = min(x + dash_len, x_end)
        cv2.line(frame, (x, y), (x2, y), color, thickness)
        x += dash_len + gap_len


def draw_palm_height_threshold(frame, bbox: BboxContext, config: GestureHandKeypointConfig) -> None:
    """
    Draws a red dotted horizontal line, in CROP-LOCAL coordinates, at the palm_height_fraction
    cutoff the palm-height gate checks against — for visually calibrating that threshold. Drawn
    at `palm_height_fraction * bbox.bbox_height` down from the crop's own top edge, which is
    correct as long as the crop IS the person bbox (the current architecture's invariant) —
    bbox.bbox_height is used explicitly rather than frame.shape so this stays correct even if
    that invariant ever changes (see bbox_context.py).
    """
    y = int(config.palm_height_fraction * bbox.bbox_height)
    if 0 <= y < frame.shape[0]:
        _draw_dotted_hline(frame, y, 0, frame.shape[1], _THRESHOLD_LINE_COLOR)
