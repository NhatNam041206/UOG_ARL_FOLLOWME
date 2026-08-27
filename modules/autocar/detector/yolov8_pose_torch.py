"""
PC-side pose detector backend: Ultralytics YOLOv8n-pose running directly on
PyTorch. For development/testing on a regular PC only - the Jetson Nano's
JetPack 4.6 runtime is Python 3.6, and Ultralytics requires Python >= 3.8, so
this backend cannot run on the Nano itself. A TensorRT-engine backend behind
the same PoseDetector interface is the way to deploy on-device later.
"""
from typing import List

import numpy as np
from ultralytics import YOLO

import config
from detector.base import PoseDetector
from utils.types import Detection


class YOLOv8PoseTorch(PoseDetector):
    def __init__(self, model_path: str = None, conf: float = None,
                 imgsz: int = None, device: str = "cpu"):
        self.model = YOLO(model_path or config.POSE_MODEL_PATH)
        self.conf = config.DETECT_CONF if conf is None else conf
        self.imgsz = config.DETECT_IMGSZ if imgsz is None else imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],  # person only
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None

        detections = []
        for i in range(len(boxes)):
            detections.append(Detection(
                bbox=boxes[i],
                score=float(scores[i]),
                keypoints=kpts[i] if kpts is not None else None,
            ))
        return detections
