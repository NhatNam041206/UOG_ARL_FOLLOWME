"""
Sticky target lock + pluggable re-acquisition strategy for demo_wave_trigger.py's --any-person
mode. Mirrors src/pipeline.py's own sticky-target pattern (active_target_id + grace-period
tolerance before giving up) but WITHOUT any identity verification — --any-person mode's whole
point is to work without a registered person. "Sticky" here means "keep following the same
YOLO track_id", not "verified as a specific known person".

Fixes the symptom where naively picking the largest bbox every frame made the tracked subject
visibly jump between two simultaneously-present people whenever their relative bbox sizes
crossed over. Once a track_id is locked on, it stays locked (even if briefly missing from
detection, up to `lock_grace_frames`) until genuinely lost, at which point one of 4 configurable
re-acquisition strategies picks the next primary:

  - "largest_bbox": whoever has the biggest bbox (original naive behavior, zero extra cost).
  - "position": whoever is spatially closest to the lost target's last known bbox (center
    distance + size-ratio difference). No extra dependency, near-zero cost.
  - "histogram": whoever's HSV color-histogram best matches the lost target's cached histogram
    (classical lightweight appearance cue, no ML model, microseconds).
  - "osnet": whoever's OSNet re-id embedding best matches the lost target's cached embedding
    (reuses src/verifier.py's OSNetVerifier, in-memory only — NOT the registry — most robust,
    but pays the same per-crop cost as the real pipeline's identity check).

All 4 are selectable at runtime (config or CLI) so cost/behavior can be A/B'd directly, e.g. on
a Raspberry Pi 5.
"""
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]

VALID_METHODS = ("largest_bbox", "position", "histogram", "osnet")


def _bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    fh, fw = frame.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(fw, x2), min(fh, y2)
    return frame[cy1:cy2, cx1:cx2]


def _hsv_histogram(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Classical lightweight appearance signature: 2D Hue/Saturation histogram (Value/brightness
    channel dropped for lighting invariance), min-max normalized so HISTCMP_BHATTACHARYYA is
    comparable across crops of different sizes.
    """
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


class AnyPersonTracker:
    """
    Stateful, single-target sticky lock over raw YoloDetector.track() output (no identity
    verification). Call `update()` once per frame with that frame's detections; read
    `locked_track_id` for the sticky identity (persists through grace-period frames even though
    `update()` returns found=False during that window) and `last_event` for what happened.
    """

    def __init__(
        self,
        method: str,
        lock_grace_frames: int,
        histogram_similarity_min: float,
        osnet_similarity_min: float,
        osnet_verifier: Optional[Any] = None,
    ):
        method = (method or "largest_bbox").strip().lower()
        if method not in VALID_METHODS:
            raise ValueError(f"Unsupported reacquisition_method '{method}'. Supported: {VALID_METHODS}")
        if method == "osnet" and osnet_verifier is None:
            raise ValueError("method='osnet' requires an osnet_verifier instance.")

        self.method = method
        self.lock_grace_frames = lock_grace_frames
        self.histogram_similarity_min = histogram_similarity_min
        self.osnet_similarity_min = osnet_similarity_min
        self.osnet_verifier = osnet_verifier

        self.locked_track_id: Optional[int] = None
        self.last_known_bbox: Optional[BBox] = None
        self.last_known_hist: Optional[np.ndarray] = None
        self.last_known_embedding: Optional[np.ndarray] = None
        self.miss_streak: int = 0

        self.last_lock_ms: float = 0.0
        self.last_event: str = "none"  # "held" | "grace" | "reacquired" | "lost"

    def reset(self) -> None:
        self.locked_track_id = None
        self.last_known_bbox = None
        self.last_known_hist = None
        self.last_known_embedding = None
        self.miss_streak = 0
        self.last_event = "none"

    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Tuple[bool, Optional[int], Optional[BBox]]:
        t0 = time.time()
        result = self._update(frame, detections)
        self.last_lock_ms = (time.time() - t0) * 1000.0
        return result

    def _update(self, frame, detections):
        if self.locked_track_id is not None:
            match = next((d for d in detections if d["track_id"] == self.locked_track_id), None)
            if match is not None:
                self.miss_streak = 0
                self.last_known_bbox = match["bbox"]
                self.last_event = "held"
                return True, self.locked_track_id, match["bbox"]

            self.miss_streak += 1
            if self.miss_streak <= self.lock_grace_frames:
                # Still within tolerance — stay locked (locked_track_id unchanged), report
                # nothing this frame rather than grabbing a different person while we wait for
                # the real target to reappear.
                self.last_event = "grace"
                return False, None, None

            # Grace period exceeded — lock is genuinely broken, fall through to reacquire.
            self.locked_track_id = None

        if not detections:
            self.last_event = "lost"
            return False, None, None

        primary = self._reacquire(frame, detections)
        self._acquire(frame, primary)
        self.last_event = "reacquired"
        return True, primary["track_id"], primary["bbox"]

    def _reacquire(self, frame, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Cold start (no cached signature yet) always falls back to largest_bbox, regardless of
        # configured method — there is nothing to compare candidates against yet.
        if self.method == "largest_bbox" or self.last_known_bbox is None:
            return max(detections, key=lambda d: _bbox_area(d["bbox"]))

        if self.method == "position":
            return min(detections, key=lambda d: self._position_score(d["bbox"]))

        if self.method == "histogram":
            best = self._best_by_histogram(frame, detections)
            return best if best is not None else max(detections, key=lambda d: _bbox_area(d["bbox"]))

        # method == "osnet"
        best = self._best_by_osnet(frame, detections)
        return best if best is not None else max(detections, key=lambda d: _bbox_area(d["bbox"]))

    def _position_score(self, bbox: BBox) -> float:
        """Lower is better: normalized center-distance + relative size-difference vs last known."""
        lcx, lcy = _bbox_center(self.last_known_bbox)
        ccx, ccy = _bbox_center(bbox)
        center_dist = ((ccx - lcx) ** 2 + (ccy - lcy) ** 2) ** 0.5
        diag_sq = (self.last_known_bbox[2] - self.last_known_bbox[0]) ** 2 + \
                  (self.last_known_bbox[3] - self.last_known_bbox[1]) ** 2
        diag = max(diag_sq, 1.0) ** 0.5
        center_dist_norm = center_dist / diag

        last_area = max(_bbox_area(self.last_known_bbox), 1.0)
        size_diff = abs(_bbox_area(bbox) - last_area) / last_area
        return center_dist_norm + size_diff

    def _best_by_histogram(self, frame, detections):
        best_det, best_score = None, -1.0
        for det in detections:
            hist = _hsv_histogram(_crop(frame, det["bbox"]))
            if hist is None:
                continue
            distance = cv2.compareHist(self.last_known_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            score = 1.0 - float(distance)
            if score > best_score:
                best_det, best_score = det, score
        if best_det is not None and best_score >= self.histogram_similarity_min:
            return best_det
        return None

    def _best_by_osnet(self, frame, detections):
        crops = [_crop(frame, det["bbox"]) for det in detections]
        embeddings = self.osnet_verifier.extract_batch(crops)
        best_det, best_score = None, -1.0
        for det, emb in zip(detections, embeddings):
            score = self.osnet_verifier.compare(self.last_known_embedding, emb)
            if score > best_score:
                best_det, best_score = det, score
        if best_det is not None and best_score >= self.osnet_similarity_min:
            return best_det
        return None

    def _acquire(self, frame, det: Dict[str, Any]) -> None:
        self.locked_track_id = det["track_id"]
        self.last_known_bbox = det["bbox"]
        self.miss_streak = 0
        # Appearance signature cached ONCE at acquisition, not refreshed every held frame — the
        # cost only needs to be paid at (rare) re-acquisition events, not every frame, which is
        # what keeps this cheap enough to matter for Raspberry Pi 5 budgeting.
        if self.method == "histogram":
            self.last_known_hist = _hsv_histogram(_crop(frame, det["bbox"]))
        elif self.method == "osnet":
            self.last_known_embedding = self.osnet_verifier.extract(_crop(frame, det["bbox"]))
