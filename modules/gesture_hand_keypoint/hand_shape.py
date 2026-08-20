"""
Hand-shape classification (redesign, replaces the old motion-based Gate A/B entirely — this
module no longer detects waving via motion of any kind, per confirmed design: pure hand-shape
sequence classification, only MediaPipe landmark geometry, no wrist motion/trajectory/arm
geometry).

Per-frame, no memory — classifies a single hand's landmarks as OPEN (all-fingers-extended open
palm), CLOSED (fist), or NEITHER (in-between/ambiguous).

Per-finger extension tests, confirmed with the user:
  - Index/middle/ring/pinky: tip farther from wrist than PIP joint = extended (reused from the
    prior design's Gate A heuristic — this part was never motion-based, it's pure geometry).
  - Thumb (previously excluded — its PIP/MCP joints don't radiate from the wrist the way the
    other four fingers' do): distance(thumb_tip, pinky_MCP) normalized by hand scale
    (distance(wrist, middle_MCP)) exceeds `thumb_extension_ratio_threshold` = extended. Large
    when the thumb sticks out away from the hand (open); small when tucked toward the palm/
    pinky side (fist). Confirmed with the user over an angle-based alternative.

Calibrated against reference open-palm / fist photos: a natural fist rests the thumb OVER the
curled fingers (not tucked into the palm), so thumb_tip stays roughly as far from pinky_MCP as
in a half-open hand — the thumb-extension test does NOT reliably read "curled" for a real fist.
So the thumb is only required for OPEN (a genuine open palm does spread the thumb away from the
hand); CLOSED is judged purely from the 4 non-thumb fingers being curled, thumb state ignored.
"""
from math import dist
from typing import Literal, Optional

import numpy as np

from .config import GestureHandKeypointConfig
from .constants import FINGER_TIP_PIP_PAIRS, MIDDLE_MCP, PINKY_MCP, THUMB_TIP, WRIST

HandShape = Literal["OPEN", "CLOSED", "NEITHER"]

OPEN: HandShape = "OPEN"
CLOSED: HandShape = "CLOSED"
NEITHER: HandShape = "NEITHER"


def is_finger_extended(landmarks_px: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
    wrist = landmarks_px[WRIST]
    return dist(landmarks_px[tip_idx], wrist) > dist(landmarks_px[pip_idx], wrist)


def is_thumb_extended(landmarks_px: np.ndarray, config: GestureHandKeypointConfig) -> bool:
    hand_scale = dist(landmarks_px[WRIST], landmarks_px[MIDDLE_MCP])
    if hand_scale < 1e-6:
        return False
    ratio = dist(landmarks_px[THUMB_TIP], landmarks_px[PINKY_MCP]) / hand_scale
    return ratio > config.thumb_extension_ratio_threshold


def count_extended_fingers(landmarks_px: np.ndarray, config: GestureHandKeypointConfig) -> int:
    """Of the 4 NON-THUMB fingers only, how many are classified as extended this frame. The
    thumb is judged separately (see is_thumb_extended) since it's only a reliable signal for
    OPEN, not for CLOSED — see module docstring."""
    return sum(
        is_finger_extended(landmarks_px, tip_idx, pip_idx)
        for tip_idx, pip_idx in FINGER_TIP_PIP_PAIRS.values()
    )


def classify_hand_shape(landmarks_px: np.ndarray, config: GestureHandKeypointConfig) -> HandShape:
    """
    OPEN if at least `min_fingers_extended_open` of the 4 non-thumb fingers are extended AND the
    thumb is also extended (a genuine open palm spreads the thumb away from the hand).

    CLOSED if at least `min_fingers_curled_closed` of the 4 non-thumb fingers are curled (i.e.
    at most `4 - min_fingers_curled_closed` extended) — thumb state is NOT checked, since a
    natural fist rests the thumb over the curled fingers rather than tucking it fully into the
    palm, which does not reliably read as "curled" by the distance-based thumb test.

    Otherwise NEITHER (ambiguous/in-between — does not advance the sequence state machine, per
    the "no partial credit outside a clean OPEN/CLOSED read" design).
    """
    extended_count = count_extended_fingers(landmarks_px, config)
    if extended_count >= config.min_fingers_extended_open and is_thumb_extended(landmarks_px, config):
        return OPEN
    if (4 - extended_count) >= config.min_fingers_curled_closed:
        return CLOSED
    return NEITHER
