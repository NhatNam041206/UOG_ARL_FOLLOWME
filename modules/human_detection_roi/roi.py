"""
ROI-scoping strategy (spec §2, mandatory stop-and-ask, resolved with the user): expand primarily
downward and laterally from the face bbox, since the body extends mostly below the face, not
above it. `roi_expansion_factor` drives the overall size; `roi_upward_fraction` and
`roi_width_fraction` (both config, optional overrides with working defaults) split that budget
between up/down and control width — exposed as tunable knobs, not hardcoded, per user request.
"""
from typing import Tuple

# Defaults for the up/down/width split, used unless overridden via config (see config.py). Most
# of the expansion goes downward (body below the face); a small upward allowance covers
# hair/head-top that YuNet's face bbox doesn't include. Width grows slower than height since a
# body silhouette is taller than it is wide.
DEFAULT_UPWARD_FRACTION = 0.15
DEFAULT_WIDTH_FRACTION = 0.6


def compute_roi(
    face_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    roi_expansion_factor: float,
    upward_fraction: float = DEFAULT_UPWARD_FRACTION,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
) -> Tuple[int, int, int, int]:
    """
    `face_bbox`: (x, y, w, h) in full-frame pixel space. `frame_shape`: (height, width, ...) —
    a numpy .shape tuple works directly. `upward_fraction`: share of the total height budget
    (`face_h * roi_expansion_factor`) allocated ABOVE the face; the rest goes below.
    `width_fraction`: ROI width as a fraction of `face_w * roi_expansion_factor` (NOT the height
    budget — width scales off the face's own width, same as the original fixed-ratio version, to
    avoid silently changing established sizing behavior while just exposing it as tunable).
    Returns (x1, y1, x2, y2), clipped to frame bounds.
    """
    x, y, w, h = face_bbox
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    roi_height = h * roi_expansion_factor
    up = roi_height * upward_fraction
    down = roi_height - up

    roi_width = w * roi_expansion_factor * width_fraction
    cx = x + w / 2.0

    x1 = int(round(cx - roi_width / 2.0))
    x2 = int(round(cx + roi_width / 2.0))
    y1 = int(round(y - up))
    y2 = int(round(y + h + down))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    return (x1, y1, x2, y2)
