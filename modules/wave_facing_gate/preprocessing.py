"""
Bbox crop -> MoveNet input tensor, and the reverse: MoveNet's per-keypoint (y, x) -> the bbox
crop's own pixel coordinate space.

Pad-to-square, not stretch-resize (spec §2, confirmed with the user rather than silently
defaulted): stretch-resize would distort limb angles, directly corrupting Gate A's verticality
check (gate_a.py). The letterbox padding this introduces must be undone before any keypoint is
used for a bbox-relative measurement (wrist-height fraction, motion in bbox pixel space, etc.) —
this conversion happens once per frame, for all 17 keypoints, not duplicated per-arm/per-gate
(spec §4.1 implementation note).
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .constants import Keypoint, NUM_KEYPOINTS


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray   # [input_size, input_size, 3] uint8 RGB, ready for the model
    pad_top: int          # padding added inside the square, in square-pixel units
    pad_left: int
    square_size: int      # side length of the padded square, before resizing to input_size
    orig_h: int            # original bbox crop dimensions, in pixels
    orig_w: int


def preprocess_crop(crop_bgr: Optional[np.ndarray], input_size: int) -> Optional[PreprocessResult]:
    """
    Returns None for an empty/None/degenerate crop (fail-closed per spec §10 item 1, confirmed
    with the user) — the caller treats this as "no keypoints" rather than raising.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return None

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    square_size = max(orig_h, orig_w)
    pad_top = (square_size - orig_h) // 2
    pad_bottom = square_size - orig_h - pad_top
    pad_left = (square_size - orig_w) // 2
    pad_right = square_size - orig_w - pad_left
    padded = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    resized = cv2.resize(padded, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    return PreprocessResult(
        tensor=resized,
        pad_top=pad_top,
        pad_left=pad_left,
        square_size=square_size,
        orig_h=orig_h,
        orig_w=orig_w,
    )


def decode_keypoints(raw_keypoints: np.ndarray, pre: PreprocessResult) -> List[Keypoint]:
    """
    `raw_keypoints`: MoveNet's [17, 3] output, each row (y, x, score) normalized to [0, 1]
    against the padded-square model input. Returns 17 Keypoints in ARM_KEYPOINTS/NOSE/etc. index
    order, with x/y converted to pixel offsets in the ORIGINAL bbox crop's coordinate space
    (undoing the square-resize and the letterbox padding).

    Points that fell inside the letterbox padding (score usually low there anyway) decode to
    pixel coordinates outside [0, orig_w]/[0, orig_h] — left unclamped, since every consumer
    (Gate A's fraction/angle checks, Gate B's displacement) already gates on confidence first and
    only compares relative positions, not absolute bounds.
    """
    keypoints: List[Keypoint] = []
    for i in range(NUM_KEYPOINTS):
        y_norm, x_norm, score = raw_keypoints[i]
        y_square_px = float(y_norm) * pre.square_size
        x_square_px = float(x_norm) * pre.square_size
        x_px = x_square_px - pre.pad_left
        y_px = y_square_px - pre.pad_top
        keypoints.append(Keypoint(x=x_px, y=y_px, score=float(score)))
    return keypoints
