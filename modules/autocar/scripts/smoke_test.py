"""Non-GUI sanity check: run detector + tracker on a synthetic frame sequence
to catch import/shape errors without needing a camera. Not a real accuracy
test - just proves the wiring works end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from detector.yolov8_pose_torch import YOLOv8PoseTorch
from tracker.byte_tracker import BYTETracker

detector = YOLOv8PoseTorch(device="cpu")
tracker = BYTETracker()

frame = np.zeros((480, 640, 3), dtype=np.uint8)

for i in range(5):
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    print(f"frame {i}: {len(detections)} detections -> {len(tracks)} tracks")
    for t in tracks:
        kp_shape = None if t.keypoints is None else t.keypoints.shape
        print(f"  id={t.track_id} score={t.score:.2f} bbox={t.bbox} keypoints={kp_shape}")

print("smoke test OK")
