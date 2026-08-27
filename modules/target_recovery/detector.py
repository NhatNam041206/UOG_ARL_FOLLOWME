"""
Standalone whole-frame person detector for target_recovery's Path B fallback (plans/07 §4.3).

Deliberately its own independent YOLO instance (own-instance isolation, docs/architecture.md
rule #2) — confirmed with the user per the spec's own stated default: a fresh instance, not a
reuse of modules.human_detection's existing whole-frame detection call, even though both load the
same yolo11n.onnx weights file.

Stateless, single-frame `.predict()` (no ByteTrack) — Path B doesn't need track continuity, only
"every person bbox visible in this frame right now," to run appearance_verifier.verify() against
each candidate. Mirrors modules.human_detection_roi's single-frame detector in spirit (own
independent implementation, not shared).
"""
import logging
from typing import Any, Dict, List

from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class RecoveryCandidateDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect")
        logger.info(f"target_recovery: loaded standalone Path B detector from '{model_path}' (person-only)")

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Returns list of {'bbox': (x1,y1,x2,y2), 'confidence'} in full-frame pixel space —
        every person visible, unfiltered; the caller (pipeline.py) runs appearance_verifier
        against each candidate."""
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(frame, classes=[_PERSON_CLASS_ID], verbose=False)
        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
            detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        return detections
