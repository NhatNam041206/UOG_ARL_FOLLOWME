"""
Standalone person detector + tracker for the human_detection module.

Deliberately its own model + tracker instance — this module owns its own `ultralytics.YOLO(...)`
load and its own ByteTrack state, same convention as modules/emergency_stop/detection.py, so
detector/tracker state is never accidentally shared across modules.

Detection scope: COCO class 0 (person) ONLY, filtered at inference time via `classes=[0]` — this
is both the definition of "human detection" here and a real speed win (skips postprocessing for
the other 79 COCO classes). Uses yolo11n (nano) by default, the fastest variant in the
ultralytics YOLO11 family, per the fast-inference-time requirement.
"""
import logging
from typing import Any, Dict, List

from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0  # COCO class index for "person"


class HumanDetector:
    def __init__(self, model_path: str, confidence_threshold: float):
        self.model = YOLO(model_path, task="detect")
        self.confidence_threshold = confidence_threshold
        logger.info(f"human_detection: loaded detector from '{model_path}' (person-only, conf>={confidence_threshold})")

    def track(self, frame) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {'track_id', 'bbox': (x1,y1,x2,y2), 'confidence'} — bbox in the
        SAME coordinate space as `frame`. track_id is stable across calls for the same physical
        person (ByteTrack, persist=True) as long as this same HumanDetector instance keeps
        receiving a continuous frame stream — callers that key per-person state off track_id
        (e.g. modules.wave_facing_gate) depend on that continuity.
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[_PERSON_CLASS_ID], conf=self.confidence_threshold, verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0

            if box.id is not None:
                track_id = int(box.id[0].cpu().item())
            else:
                track_id = idx + 1000
                logger.warning(f"human_detection: detection missing track_id, fallback id {track_id}")

            detections.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })

        return detections
