"""
MoveNet Lightning singlepose keypoint layout (spec §2) and the shared per-keypoint value type.
Fixed by the model's own output contract — do not reorder.
"""
from dataclasses import dataclass

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17

ARM_KEYPOINTS = {
    "left": (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
    "right": (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER),
}


@dataclass(frozen=True)
class Keypoint:
    """
    One decoded keypoint, already converted to the bbox crop's own pixel coordinate space (NOT
    the 192x192 model-input space) — see preprocessing.py's decode step. `x`/`y` are pixel
    offsets from the bbox crop's top-left corner; `score` is MoveNet's raw confidence, [0.0, 1.0].
    """
    x: float
    y: float
    score: float
