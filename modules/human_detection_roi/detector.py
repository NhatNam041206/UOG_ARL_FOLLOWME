"""
Standalone person detector for the human_detection_roi module.

Deliberately its own fresh YOLO instance (same yolo11n.onnx weights file every other module in
this repo uses — confirmed with the user that reusing the WEIGHTS FILE is fine, distinct from
sharing a live model/pipeline object, which stays forbidden per the isolation rule). This module
does its own single-frame `.predict()` (NOT `.track()`) — no ByteTrack/persist=True: the ROI
crop this module scans shifts every frame (it follows the matched face), which is not a stable
coordinate frame for a tracker's motion model to work against, so this stays a stateless,
per-call, ROI-scoped detection rather than a persistent multi-frame track (confirmed with the
user — see pipeline.py's module docstring).

Detection scope: COCO class 0 (person) only, same reasoning as modules/human_detection.
"""
import logging
from typing import Any, Dict, List

from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class HumanDetectorROI:
    def __init__(self, model_path: str):
        # Confidence threshold is NOT fixed at construction (unlike YuNet's cv2 API) — ultralytics
        # accepts `conf` per predict() call, so model loading stays fully decoupled from the
        # calibration-gated threshold (which may still be None/uncalibrated at construction time).
        self.model = YOLO(model_path, task="detect")
        logger.info(f"human_detection_roi: loaded detector from '{model_path}' (person-only)")

    def detect(self, roi_crop, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {'bbox': (x1,y1,x2,y2), 'confidence'} — bbox in the SAME
        coordinate space as `roi_crop` (i.e. ROI-local, NOT full-frame — the caller converts).
        """
        if roi_crop is None or roi_crop.size == 0:
            return []

        results = self.model.predict(
            roi_crop, classes=[_PERSON_CLASS_ID], conf=confidence_threshold, verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
            detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        return detections
