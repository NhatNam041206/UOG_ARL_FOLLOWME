"""
5-point face alignment (spec §3: "use whatever the embedding model's own documentation
recommends"). EdgeFace's own reference inference pipeline (yakhyo/edgeface-onnx, via the
yakhyo/uniface alignment utility) warps each face to a canonical 112x112 pose using a similarity
transform (rotation + uniform scale + translation) estimated from 5 landmarks against a fixed
reference template — see constants.ARCFACE_TEMPLATE_112.

Reimplemented here with cv2.estimateAffinePartial2D (already a project dependency) rather than
importing the `uniface` package, to avoid pulling in a new third-party dependency for a few dozen
lines of geometry — cv2.estimateAffinePartial2D solves the same similarity-transform estimation
problem (it's OpenCV's partial-affine, i.e. rotation+scale+translation only, no shear), so the
result is equivalent, not a different algorithm.
"""
from typing import Optional

import cv2
import numpy as np

from .constants import ARCFACE_TEMPLATE_112, EDGEFACE_INPUT_SIZE


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    `landmarks`: (5, 2) float32 in FULL FRAME pixel space, YuNet's fixed order (left_eye,
    right_eye, nose, left_mouth, right_mouth) — matches ARCFACE_TEMPLATE_112's point order.
    Returns a (112, 112, 3) BGR aligned face crop, or None if the transform can't be estimated
    (degenerate/collinear landmarks).
    """
    transform, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    if transform is None:
        return None
    return cv2.warpAffine(frame, transform, (EDGEFACE_INPUT_SIZE, EDGEFACE_INPUT_SIZE), borderValue=0.0)
