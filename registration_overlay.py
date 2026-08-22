"""
Registration overlay layer — Layer 2 of register_person's three-layer design (data / overlay /
interaction; see register_person.py's own docstring for the full split). Every function here
takes a frame (or a frame shape) plus plain state and RETURNS an annotated/cropped image. No
camera reads, no display of any kind (neither cv2.imshow nor Tkinter widgets), no file I/O
anywhere in this file — register_person.py (Layer 3) calls these and is the only place anything
is actually shown to the operator.

Capture itself runs no detection (see register_person.py's module docstring) — these functions
only draw the ROI box, the countdown, and progress text; there is no detection bbox to draw
during capture, since detection only happens afterward, in registration_data.build_target_profile.
"""
from typing import Sequence, Tuple

import cv2
import numpy as np

RoiPercent = Sequence[float]  # [x1, y1, x2, y2], each a 0.0-1.0 fraction of frame width/height
RoiPx = Tuple[int, int, int, int]

_ROI_COLOR = (0, 220, 220)
_INSTRUCTION_COLOR = (0, 200, 0)


def roi_to_px(roi_percent: RoiPercent, frame_w: int, frame_h: int) -> RoiPx:
    x1p, y1p, x2p, y2p = roi_percent
    return int(x1p * frame_w), int(y1p * frame_h), int(x2p * frame_w), int(y2p * frame_h)


def crop_to_roi(frame: np.ndarray, roi_percent: RoiPercent) -> np.ndarray:
    """What actually gets saved to disk — the frame cropped down to the configured ROI box, so a
    bystander elsewhere in the scene never ends up in the saved file at all."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_px(roi_percent, frame_w, frame_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    return frame[y1:y2, x1:x2].copy()


def _banner(image: np.ndarray, lines, color) -> None:
    y = 40
    for text in lines:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x = max(10, (image.shape[1] - text_w) // 2)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        y += 40


def draw_roi(image: np.ndarray, roi_percent: RoiPercent) -> RoiPx:
    """Draws the ROI box (what the operator sees live) and returns its pixel coordinates."""
    frame_h, frame_w = image.shape[:2]
    roi_px = roi_to_px(roi_percent, frame_w, frame_h)
    cv2.rectangle(image, roi_px[:2], roi_px[2:], _ROI_COLOR, 2)
    return roi_px


def draw_countdown(frame: np.ndarray, roi_percent: RoiPercent, instruction: str,
                    remaining_seconds: float) -> np.ndarray:
    display = frame.copy()
    draw_roi(display, roi_percent)
    _banner(display, [instruction, f"Starting in {remaining_seconds:.1f}s"], _INSTRUCTION_COLOR)
    return display


def draw_capture(frame: np.ndarray, roi_percent: RoiPercent, instruction: str,
                  saved: int, needed: int) -> np.ndarray:
    display = frame.copy()
    draw_roi(display, roi_percent)
    _banner(display, [instruction, f"{saved}/{needed}"], _INSTRUCTION_COLOR)
    return display
