"""
Palm height gate + the crop-to-full-frame coordinate conversion it needs. Reimplemented
independently here (not imported from modules.wave_facing_gate) per the project's isolation
convention — same conceptual "BboxContext" role, own code.

The palm keypoint (confirmed with the user: MediaPipe landmark 0, the wrist — same anchor point
already used elsewhere for height checks) must be in the upper `palm_height_fraction` of the
PERSON'S full-frame bbox (from modules.human_detection_roi's output), not just the hand-crop's
own height — those aren't always the same thing if the crop pipeline ever changes (padding,
resizing), so the conversion is done explicitly rather than assumed.
"""
from dataclasses import dataclass

import numpy as np

from .config import GestureHandKeypointConfig
from .constants import WRIST


@dataclass(frozen=True)
class BboxContext:
    """The person crop's own placement within the full frame. `offset_x`/`offset_y`: the crop's
    top-left corner in full-frame pixels. `bbox_width`/`bbox_height`: the person bbox's size —
    from modules.human_detection_roi's `person_bbox`, NOT necessarily the same as the crop's own
    .shape if the crop pipeline ever pads/resizes (currently it doesn't, but this keeps the
    height check correct even if that changes)."""
    offset_x: int
    offset_y: int
    bbox_width: int
    bbox_height: int

    @staticmethod
    def from_person_bbox(person_bbox_full_frame) -> "BboxContext":
        x, y, w, h = person_bbox_full_frame
        return BboxContext(offset_x=x, offset_y=y, bbox_width=w, bbox_height=h)

    @staticmethod
    def whole_crop_as_bbox(crop_shape) -> "BboxContext":
        """Fallback when no real person_bbox is available (e.g. this module's own standalone
        test script, run without modules.human_detection_roi) — treats the whole crop as its
        own bbox, offset (0, 0). Same "whole frame/crop as bbox" simplification used elsewhere
        in this project's isolated test scripts."""
        h, w = crop_shape[0], crop_shape[1]
        return BboxContext(offset_x=0, offset_y=0, bbox_width=w, bbox_height=h)


def palm_height_gate_pass(landmarks_px: np.ndarray, bbox: BboxContext, config: GestureHandKeypointConfig) -> bool:
    """
    True if the palm keypoint (wrist landmark) is in the upper `palm_height_fraction` of the
    person's full-frame bbox. `landmarks_px` are in crop-local pixel space; converted to
    full-frame explicitly before comparing against the bbox, per the coordinate-conversion
    discipline used elsewhere in this pipeline.
    """
    wrist_x_local, wrist_y_local = landmarks_px[WRIST]
    wrist_y_full_frame = wrist_y_local + bbox.offset_y
    bbox_top = bbox.offset_y
    threshold_y = bbox_top + config.palm_height_fraction * bbox.bbox_height
    return wrist_y_full_frame < threshold_y
