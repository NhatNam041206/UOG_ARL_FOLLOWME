"""
Bbox crop -> MoveNet input tensor, and the reverse: MoveNet's per-keypoint (y, x) -> the crop's
own pixel coordinate space. Reimplemented independently here (spec §0.3: this counts as "logic
that operates on the model's output", not "the model" itself, so it must be a fresh
implementation, not an import from modules.wave_facing_gate.preprocessing) — same pad-to-square
approach (not stretch-resize) since this module's keypoints feed a shape-sensitive trajectory
comparison, same reasoning as Method 1's angle-based Gate A.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray
    pad_top: int
    pad_left: int
    square_size: int
    orig_h: int
    orig_w: int


def preprocess_crop(crop_bgr: Optional[np.ndarray], input_size: int) -> Optional[PreprocessResult]:
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
        tensor=resized, pad_top=pad_top, pad_left=pad_left,
        square_size=square_size, orig_h=orig_h, orig_w=orig_w,
    )


def decode_keypoints(raw_keypoints: np.ndarray, pre: PreprocessResult) -> List[Keypoint]:
    keypoints: List[Keypoint] = []
    for i in range(raw_keypoints.shape[0]):
        y_norm, x_norm, score = raw_keypoints[i]
        y_square_px = float(y_norm) * pre.square_size
        x_square_px = float(x_norm) * pre.square_size
        keypoints.append(Keypoint(
            x=x_square_px - pre.pad_left,
            y=y_square_px - pre.pad_top,
            score=float(score),
        ))
    return keypoints
