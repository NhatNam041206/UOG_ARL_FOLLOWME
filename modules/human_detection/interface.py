"""
Human Detection module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.human_detection. Everything else in
this package (config.py, detector.py) is an internal implementation detail and may change
without notice.

Scope: detects and tracks people (COCO person class only) in a full frame, returning one bbox +
stable track_id per person, at fast (nano-model, person-only) inference. Does NOT do pose
estimation, gesture recognition, facing-camera checks, or identity verification/Re-ID —
ByteTrack's track_id is motion-continuity only, not a verified identity. Those are other
modules' jobs; e.g. modules.wave_facing_gate consumes this module's per-person bboxes+track_ids,
one call per detected person, to independently evaluate is_waving/is_facing_camera for each.

Crop convention: a raw BGR frame (numpy.ndarray from cv2), the standard OpenCV convention used
elsewhere in this codebase (see modules/emergency_stop/interface.py).
"""
import time
from dataclasses import dataclass
from typing import List, Tuple

from .config import HumanDetectionConfig, load_config
from .detector import HumanDetector


@dataclass
class PersonDetection:
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in the input frame's pixel space
    confidence: float


class HumanDetectionModule:
    """
    Owns its own YOLO model + ByteTrack instance. Create one instance and call detect() once per
    frame — track_id continuity across calls depends on calling it on a continuous frame stream
    from the same instance (skipping frames or reinstantiating resets tracking).
    """

    def __init__(self, thresholds_config_path: str = "config/thresholds.yaml"):
        config: HumanDetectionConfig = load_config(thresholds_config_path)
        self._detector = HumanDetector(config.yolo_model_path, config.confidence_threshold)
        self.last_latency_ms: float = 0.0

    def detect(self, frame) -> List[PersonDetection]:
        t_start = time.time()
        try:
            raw = self._detector.track(frame)
            return [
                PersonDetection(track_id=d["track_id"], bbox=d["bbox"], confidence=d["confidence"])
                for d in raw
            ]
        finally:
            self.last_latency_ms = (time.time() - t_start) * 1000.0
