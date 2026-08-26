"""
Real face detection + face recognition - YuNet (cv2.FaceDetectorYN) finds an actual face bounding
box + 5 landmarks inside a crop, SFace (cv2.FaceRecognizerSF) turns the aligned face into an
embedding for identity comparison. Both are small ONNX models bundled through OpenCV's own DNN
module (opencv-python >= 4.5.4, no extra ML framework needed) - CPU-friendly, meant for the
Jetson Nano target same as the rest of this pipeline.

Replaces the old approach of running the general-purpose OSNet appearance model on a keypoint-
guessed head-region rectangle (identity/osnet_embedder.py) for the FRONT-face path: that rectangle
spans the full bbox width down to the shoulder line, so its lower portion routinely includes
collar/shoulder clothing - appearance the person controls just by changing what they wear, which
was corrupting matches. A real face detector finds the actual face, nothing else.

identity/osnet_embedder.py is still used, separately, for the BACK-of-head path (identity/
target_lock.py) - no face detector can find a face that isn't there when someone's facing away.
"""
from typing import Optional

import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, detector_model_path: str, recognizer_model_path: str,
                 score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
        # input_size is set per-call in detect_best_face() (varies with each crop's shape), so an
        # arbitrary placeholder is fine here.
        self._detector = cv2.FaceDetectorYN.create(
            detector_model_path, "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._recognizer = cv2.FaceRecognizerSF.create(recognizer_model_path, "")

    def detect_best_face(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Runs YuNet on `crop` and returns the highest-scoring detected face as a 15-value row
        (x, y, w, h, then 5 landmark (x, y) pairs, then score) - the format alignCrop()/extract()
        expect - or None if no face was found (e.g. the person is facing away, or this crop is
        degenerate/empty)."""
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(crop)
        if faces is None or len(faces) == 0:
            return None
        return faces[np.argmax(faces[:, -1])]

    def extract(self, crop: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Aligns and crops the face out of `crop` using YuNet's landmarks, then returns SFace's
        raw feature vector for it, flattened to 1-D (128,) for consistency with how every other
        embedding in this codebase is stored/passed around. Compare with `compare()`, not a plain
        dot product - SFace's own match() handles the normalization."""
        aligned = self._recognizer.alignCrop(crop, face_row)
        return self._recognizer.feature(aligned).flatten()

    def compare(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two SFace features, via SFace's own match() (not a manual dot
        product). Official guidance (github.com/opencv/opencv_zoo face_recognition_sface): >=
        ~0.363 cosine is "same person" - see config.FACE_SIMILARITY_THRESHOLD."""
        return float(self._recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))
