"""Exercise BYTETracker's actual state machine with synthetic Detections
(no model involved): new track spawn, steady update, occlusion (a few frames
with the person entirely missing), and re-match after reappearing nearby."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tracker.byte_tracker import BYTETracker
from utils.types import Detection

tracker = BYTETracker()


def make_det(x, y, w=50, h=120, score=0.9):
    kpts = np.zeros((17, 3), dtype=float)
    kpts[:, 0] = x + w / 2
    kpts[:, 1] = y + h / 2
    kpts[:, 2] = 0.9
    return Detection(bbox=np.array([x, y, x + w, y + h], dtype=float), score=score, keypoints=kpts)


# person A walks right for 5 frames
for i in range(5):
    dets = [make_det(100 + i * 5, 100)]
    tracks = tracker.update(dets)
    ids = [t.track_id for t in tracks]
    print(f"frame {i}: A moving -> track ids {ids}")
    assert len(ids) == 1, "expected exactly 1 track while A is visible"

first_id = tracker.tracked_tracks[0].track_id if tracker.tracked_tracks else None

# person A occluded for 3 frames (buffer is 30, should survive)
for i in range(5, 8):
    tracks = tracker.update([])
    print(f"frame {i}: occluded -> {len(tracks)} active tracks (lost kept={len(tracker.lost_tracks)})")
    assert len(tracks) == 0
    assert len(tracker.lost_tracks) == 1, "track should be held in lost_tracks during occlusion"

# person A reappears nearby -> should re-use the same track_id, not spawn a new one
dets = [make_det(120, 100)]
tracks = tracker.update(dets)
print(f"frame 8: reappear -> ids {[t.track_id for t in tracks]} (expected {first_id})")
assert len(tracks) == 1
assert tracks[0].track_id == first_id, "re-appearing person should keep the same track_id"

# a second, distant person enters -> must get a *different* id
dets = [make_det(130, 100), make_det(500, 300)]
tracks = tracker.update(dets)
ids = sorted(t.track_id for t in tracks)
print(f"frame 9: two people -> ids {ids}")
assert len(ids) == 2 and len(set(ids)) == 2

print("tracker smoke test OK")
