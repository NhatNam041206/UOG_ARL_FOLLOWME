"""
MediaPipe Hands wrapper (spec §2), via the modern Tasks API (mediapipe==1.0.1 dropped the legacy
mp.solutions.hands.Hands API entirely — HandLandmarker + hand_landmarker.task is the only
surface available at the installed version). Deliberately its own model instance, same
convention as every other module in this project.

Known limitation, not solved here (spec §2), carried into calibration testing: MediaPipe's palm
detector has shown degraded accuracy in low-light/low-resolution conditions in published
benchmarks (as low as ~58% in one clinical low-light study) — directly relevant to a crowded,
variably-lit campus environment. Not something to fix in this module; measure it empirically
against Methods 1 and 3 during calibration (spec §7), don't assume it away.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .constants import NUM_LANDMARKS

logger = logging.getLogger(__name__)

_NUM_HANDS = 2

# HandLandmarkerOptions requires numeric confidence floors at CONSTRUCTION time, but
# config.confidence_threshold is calibration-gated (may be None). Same decoupling pattern used
# elsewhere in this project (modules/face_identity's YuNet, modules/human_detection_roi's YOLO):
# construct with a low, permissive, non-calibration floor; the real threshold is applied as a
# post-filter on handedness confidence by the caller (pipeline.py).
_CONSTRUCTION_CONFIDENCE_FLOOR = 0.3


@dataclass(frozen=True)
class DetectedHand:
    landmarks_px: np.ndarray        # (21, 2) float32, crop-pixel space (x, y)
    handedness: Optional[str]       # "Left" | "Right" per MediaPipe's classification, or None
    handedness_confidence: float     # MediaPipe gives no separate per-landmark confidence; this
                                       # is the closest available per-hand confidence signal


class HandLandmarkerWrapper:
    def __init__(self, model_path: str):
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=_NUM_HANDS,
            min_hand_detection_confidence=_CONSTRUCTION_CONFIDENCE_FLOOR,
            min_hand_presence_confidence=_CONSTRUCTION_CONFIDENCE_FLOOR,
            min_tracking_confidence=_CONSTRUCTION_CONFIDENCE_FLOOR,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info(f"gesture_hand_keypoint: loaded HandLandmarker from '{model_path}'")

    def detect(self, crop_bgr: np.ndarray) -> List[DetectedHand]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        h, w = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hands: List[DetectedHand] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks_px = np.array(
                [[lm.x * w, lm.y * h] for lm in hand_landmarks], dtype=np.float32
            )
            if landmarks_px.shape != (NUM_LANDMARKS, 2):
                continue
            handedness_label: Optional[str] = None
            handedness_score = 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                handedness_label = result.handedness[i][0].category_name
                handedness_score = result.handedness[i][0].score
            hands.append(DetectedHand(
                landmarks_px=landmarks_px,
                handedness=handedness_label,
                handedness_confidence=handedness_score,
            ))
        return hands

    def close(self) -> None:
        self._landmarker.close()
