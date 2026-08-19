import os
import logging
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, model_path: str, imgsz: Optional[int] = None):
        """
        Wrap YOLO model for person tracking using Ultralytics ByteTrack.

        Args:
            imgsz: Inference resolution passed through to model.track(). None (default) leaves
                Ultralytics' own default (640) untouched — existing callers see no behavior
                change. Lowering this (e.g. 416/320) trades detection recall/localization
                precision for latency; see README perf section for the measured tradeoff.
        """
        self.model_path = model_path
        self.imgsz = imgsz

        # Verify model file existence if a custom file path is provided
        if not os.path.exists(model_path):
            # Check if it's a default ultralytics model name (e.g. 'yolo11n.pt')
            # If path contains directory separators or explicit non-existent file path, raise error
            abs_path = os.path.abspath(model_path)
            if os.path.dirname(model_path) != "" or not (model_path.endswith(".pt") or model_path.endswith(".onnx")):
                raise FileNotFoundError(
                    f"YOLO model file not found at path: {abs_path}. Please check config settings."
                )
            logger.info(f"Model path '{model_path}' not found locally; Ultralytics will attempt download.")

        try:
            self.model = YOLO(model_path, task="detect")
            logger.info(f"Successfully loaded YOLO model from '{model_path}'")
        except Exception as e:
            abs_path = os.path.abspath(model_path)
            raise FileNotFoundError(
                f"Failed to load YOLO model from path '{abs_path}'. Error: {e}"
            ) from e

    def track(self, frame) -> List[Dict[str, Any]]:
        """
        Track persons (COCO class 0) in the input frame.
        
        Returns:
            List of dicts: [{'track_id': int, 'bbox': (x1, y1, x2, y2), 'confidence': float}]
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to YoloDetector.track()")
            return []

        # Run tracking restricted to class 0 (person) with persistence. Explicitly request
        # ByteTrack (not Ultralytics' unspecified-tracker default, BoT-SORT) — this was always
        # the documented intent (see class docstring / plan.md), but the tracker was never
        # actually pinned, so BoT-SORT ran silently instead. BoT-SORT's GMC (global motion
        # compensation) step builds an optical-flow pyramid between consecutive frames and
        # asserts they're the same size — our ROI-constrained detection deliberately feeds
        # differently-sized crops between calls, which made GMC fail (falling back to identity,
        # harmless but noisy) on effectively every ROI/full-frame transition. ByteTrack has no
        # GMC step at all, so this failure mode can't happen, and we don't lose anything: we
        # already do our own OSNet-based re-id externally, so BoT-SORT's optional appearance
        # matching (with_reid, disabled by default anyway) wasn't buying us anything either.
        track_kwargs = {}
        if self.imgsz is not None:
            track_kwargs["imgsz"] = self.imgsz
        results = self.model.track(
            frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False, **track_kwargs
        )

        detections = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            # Coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Confidence score
            conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
            
            # Track ID handling
            if box.id is not None:
                track_id = int(box.id[0].cpu().item())
            else:
                track_id = idx + 1000  # Fallback temporary ID
                logger.warning(
                    f"Detection at bbox ({x1}, {y1}, {x2}, {y2}) missing track_id. "
                    f"Assigned fallback temporary ID: {track_id}"
                )
            
            detections.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })
            
        return detections
