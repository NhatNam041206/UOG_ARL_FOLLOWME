from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """One raw detector output for a single frame."""
    bbox: np.ndarray                       # [x1, y1, x2, y2] in pixels
    score: float
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, confidence - COCO order


@dataclass
class TrackedObject:
    """One tracker output: a detection resolved to a persistent-within-session track_id."""
    track_id: int
    bbox: np.ndarray                       # [x1, y1, x2, y2] in pixels
    score: float
    keypoints: Optional[np.ndarray] = None
