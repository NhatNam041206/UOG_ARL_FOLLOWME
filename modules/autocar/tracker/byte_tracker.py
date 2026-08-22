"""
ByteTrack: two-stage association tracker.

Core idea (what makes it "Byte" over plain SORT): low-confidence detections
(likely occluded people, not background) are *not* thrown away - they only
get a second chance to match already-tracked people, never to spawn a new
track. This keeps IDs alive through brief occlusion instead of dropping and
re-creating them.

A track that stops matching becomes "lost" but is kept (and still predicted
forward by its Kalman filter) for TRACK_BUFFER frames - if the same person
reappears nearby within that window, they get their old track_id back
instead of a new one. That's short-range, motion-based re-identification;
long-range re-identification (person left and re-entered minutes later, or
from a different spot) needs appearance matching, which is a separate stage
to add later.
"""
from typing import List

import numpy as np

import config
from tracker.base import Tracker
from tracker.kalman_filter import KalmanFilter
from tracker.matching import iou_distance, linear_assignment
from utils.types import Detection, TrackedObject


class TrackState:
    TRACKED = "tracked"
    LOST = "lost"


class STrack:
    """One tracked (or candidate) person: Kalman state + latest detection payload."""

    _next_id = 1

    def __init__(self, bbox: np.ndarray, score: float, keypoints=None):
        self._tlwh = self._xyxy_to_tlwh(np.asarray(bbox, dtype=float))
        self.score = score
        self.keypoints = keypoints

        self.kalman_filter = None
        self.mean = None
        self.covariance = None

        self.track_id = 0
        self.state = TrackState.TRACKED
        self.is_activated = False
        self.time_since_update = 0

    @staticmethod
    def _xyxy_to_tlwh(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @classmethod
    def _next_track_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id - 1

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self._next_track_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._tlwh_to_xyah(self._tlwh))
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.time_since_update = 0

    def re_activate(self, new_track: "STrack", frame_id: int):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track._tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.time_since_update = 0
        self.score = new_track.score
        self.keypoints = new_track.keypoints

    def predict(self):
        if self.mean is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0  # freeze height-velocity for lost tracks - stops the box shrinking to 0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)
        self.time_since_update += 1

    def update(self, new_track: "STrack", frame_id: int):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track._tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.time_since_update = 0
        self.score = new_track.score
        self.keypoints = new_track.keypoints

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]      # aspect * height -> width
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class BYTETracker(Tracker):
    def __init__(self, high_thresh=None, low_thresh=None, new_track_thresh=None,
                 match_thresh=None, low_match_thresh=None, track_buffer=None):
        self.high_thresh = config.TRACK_HIGH_THRESH if high_thresh is None else high_thresh
        self.low_thresh = config.TRACK_LOW_THRESH if low_thresh is None else low_thresh
        self.new_track_thresh = config.NEW_TRACK_THRESH if new_track_thresh is None else new_track_thresh
        self.match_thresh = config.MATCH_THRESH if match_thresh is None else match_thresh
        self.low_match_thresh = config.LOW_MATCH_THRESH if low_match_thresh is None else low_match_thresh
        self.max_time_lost = config.TRACK_BUFFER if track_buffer is None else track_buffer

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        self.frame_id += 1

        if detections:
            scores = np.array([d.score for d in detections])
        else:
            scores = np.empty((0,))

        high_idx = np.where(scores >= self.high_thresh)[0]
        low_idx = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]

        det_high = [STrack(detections[i].bbox, detections[i].score, detections[i].keypoints) for i in high_idx]
        det_low = [STrack(detections[i].bbox, detections[i].score, detections[i].keypoints) for i in low_idx]

        track_pool = self.tracked_tracks + self.lost_tracks
        for t in track_pool:
            t.predict()

        # --- stage 1: high-score detections vs every existing track (tracked + lost) ---
        dists = iou_distance(track_pool, det_high)
        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)

        matched: List[STrack] = []
        for itrack, idet in matches:
            track, det = track_pool[itrack], det_high[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
            else:
                track.re_activate(det, self.frame_id)
            matched.append(track)

        still_tracked = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.TRACKED]
        still_lost = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.LOST]
        unmatched_high = [det_high[i] for i in u_det]

        # --- stage 2: low-score detections only get a shot at currently-tracked tracks ---
        dists_low = iou_distance(still_tracked, det_low)
        matches_low, u_track2, _ = linear_assignment(dists_low, thresh=self.low_match_thresh)

        for itrack, idet in matches_low:
            track, det = still_tracked[itrack], det_low[idet]
            track.update(det, self.frame_id)
            matched.append(track)

        newly_lost = [still_tracked[i] for i in u_track2] + still_lost
        for t in newly_lost:
            t.state = TrackState.LOST

        # --- stage 3: strong leftover high-score detections become brand-new tracks ---
        new_tracks = []
        for det in unmatched_high:
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                new_tracks.append(det)

        self.tracked_tracks = [t for t in matched if t.state == TrackState.TRACKED] + new_tracks
        self.lost_tracks = [t for t in newly_lost if t.time_since_update <= self.max_time_lost]

        return [
            TrackedObject(track_id=t.track_id, bbox=t.xyxy, score=t.score, keypoints=t.keypoints)
            for t in self.tracked_tracks
            if t.is_activated
        ]
