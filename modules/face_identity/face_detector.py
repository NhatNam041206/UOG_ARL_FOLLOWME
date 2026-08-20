"""
YuNet face detector wrapper (spec §2.1) via cv2.FaceDetectorYN — bundled with OpenCV, no new
dependency. Deliberately its own detector instance, same "own model instance" convention used by
every other module in this project (modules/emergency_stop/detection.py,
modules/human_detection/detector.py) — never shared live state with another module.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# cv2.FaceDetectorYN internal NMS/top-K knobs — not calibration targets, fixed at YuNet's own
# documented defaults.
_NMS_THRESHOLD = 0.3
_TOP_K = 5000

# cv2.FaceDetectorYN_create requires a numeric score_threshold at CONSTRUCTION time, but
# config.face_detection_confidence_threshold is a calibration-gated value that starts as None
# (fail-closed, see config.py) — it can't be used to construct the detector before it's
# calibrated. So this detector is always constructed with a low, permissive, non-calibration
# floor (filters out only near-zero garbage), and the real calibrated threshold is applied as a
# post-filter by the caller (pipeline.py), decoupling "can we build the detector at all" from
# "do we trust this particular detection enough to act on it".
_CONSTRUCTION_SCORE_FLOOR = 0.2


@dataclass(frozen=True)
class DetectedFace:
    bbox: Tuple[int, int, int, int]              # (x, y, w, h), full-frame pixel space
    landmarks: np.ndarray                         # (5, 2) float32: left_eye, right_eye, nose,
                                                    # left_mouth, right_mouth (YuNet's fixed order)
    score: float


class YuNetFaceDetector:
    def __init__(self, model_path: str):
        # input_size is set per-frame in detect() (setInputSize) since frame size can vary;
        # (320, 320) here is just the constructor's required initial placeholder.
        self._detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), _CONSTRUCTION_SCORE_FLOOR, _NMS_THRESHOLD, _TOP_K,
        )
        logger.info(f"face_identity: loaded YuNet from '{model_path}' (construction floor={_CONSTRUCTION_SCORE_FLOOR})")

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []

        results: List[DetectedFace] = []
        for row in faces:
            # YuNet row layout (15 values): x, y, w, h, then 5x(landmark_x, landmark_y), then score.
            x, y, bw, bh = row[0:4]
            landmarks = row[4:14].reshape(5, 2).astype(np.float32)
            score = float(row[14])
            results.append(DetectedFace(
                bbox=(int(round(x)), int(round(y)), int(round(bw)), int(round(bh))),
                landmarks=landmarks,
                score=score,
            ))
        return results
