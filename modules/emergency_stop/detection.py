"""
Standalone object detector + tracker for the emergency_stop module.

Deliberately its own model + tracker instance (per the module spec: this module must never share
a detector/tracker instance with any other module) — its own `ultralytics.YOLO(...)` load and
its own ByteTrack state.

Detection scope (confirmed with the user): NO class filter — this module detects any COCO
object, not just persons. Its job is generic collision avoidance against anything in the runway,
so restricting to a single class (e.g. persons only) would leave a real safety gap.
"""
import logging
from typing import Any, Dict, List

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class EStopDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect")
        logger.info(f"emergency_stop: loaded standalone detector from '{model_path}' (all classes)")

    def track(self, frame) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {'track_id', 'bbox': (x1,y1,x2,y2), 'confidence', 'class_id',
        'ground_contact': (x, y)} — bbox/ground_contact in the SAME coordinate space as `frame`
        (the caller converts ROI-local coordinates back to full-frame).
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
            class_id = int(box.cls[0].cpu().item()) if box.cls is not None else -1

            if box.id is not None:
                track_id = int(box.id[0].cpu().item())
            else:
                track_id = idx + 1000
                logger.warning(f"emergency_stop: detection missing track_id, fallback id {track_id}")

            # Ground-contact point: bottom-center of bbox (spec §3.2) — approximates where the
            # object touches the ground, NOT the bbox centroid.
            ground_contact = ((x1 + x2) / 2.0, y2)

            detections.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": class_id,
                "ground_contact": ground_contact,
            })

        return detections
