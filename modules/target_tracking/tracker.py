"""
Standalone person detector + tracker for the target_tracking module.

Deliberately its own YOLO+ByteTrack instance — own-instance isolation convention
(docs/architecture.md rule #2) — even though it loads the same yolo11n.onnx weights file as
modules.human_detection, modules.emergency_stop, and modules.human_detection_roi's own separate
instances (confirmed acceptable: reusing a WEIGHTS FILE is fine, sharing a live instance is not).

Uses persistent `.track(persist=True)` (NOT human_detection_roi's stateless single-frame
`.predict()`) — this module's tracker follows one locked target continuously frame-to-frame while
an episode is active, closer in spirit to modules.human_detection's "track everyone" usage, but
its own separate instance and calling site (confirmed with the user per plans/06 §0.3/§0.4:
ByteTrack itself still tracks every person it sees each frame — this module doesn't change
ByteTrack's own behavior, it just filters the returned detections down to whichever track_id was
locked at start()/reset() time; see locking.py).
"""
import logging
from typing import Any, Dict, List

from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class TargetTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect")
        logger.info(f"target_tracking: loaded standalone tracker from '{model_path}' (person-only)")

    def track(self, frame) -> List[Dict[str, Any]]:
        """
        Returns list of {'track_id', 'bbox': (x1,y1,x2,y2), 'confidence'} in frame pixel space —
        EVERY person ByteTrack currently sees, not just the locked one (filtering to the locked
        track_id is the caller's job, see locking.py/pipeline.py). track_id continuity depends on
        calling this every frame on a continuous stream from this same instance (mirrors
        modules.human_detection's documented caveat).
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[_PERSON_CLASS_ID], verbose=False,
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
                logger.warning(f"target_tracking: detection missing track_id, fallback id {track_id}")

            detections.append({"track_id": track_id, "bbox": (x1, y1, x2, y2), "confidence": conf})

        return detections
