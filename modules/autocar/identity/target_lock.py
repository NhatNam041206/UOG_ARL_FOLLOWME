"""
Picks ONE enrolled person's track back out of everyone ByteTrack is currently tracking, and
keeps it locked onto the same track_id across frames. Two distinct phases:

ACQUIRING (self.locked_track_id is None - covers both first startup and re-searching after a
lock was dropped): scores everyone via _score_track() (see below) over REID_ACQUIRE_ROUNDS rounds
(spaced >= REID_ACQUIRE_COOLDOWN_SEC apart, by wall clock - smooths over single-frame noise
without stalling for long). Each sample is stored as a MARGIN (similarity - that sample's own
threshold, since front-face and back-of-head scoring use different models on different scales -
see _score_track), so margins from either kind can be averaged together safely: positive means
"this sample would have passed on its own". The track_id with the highest average margin locks in,
provided that average is still >= 0. If nobody clears it, the round counter resets and acquisition
starts over on the next frame.

LOCKED (self.locked_track_id is set): trusts ByteTrack's track_id continuity completely as long as
the locked track_id is still present in the tracker's output - ZERO required model calls, no
matter how many other people are nearby or overlapping it. This matches how the tracker actually
behaves during an overlap: empirically, whichever person is in FRONT keeps a stable track_id
throughout, and it's the occluded person's track_id that disappears - there's no observed case of
ByteTrack silently handing one person's id to a different physical person mid-overlap, so there's
nothing to verify while the locked id is still being reported.

While present, it ALSO opportunistically builds a live back-of-head dataset
(identity/session_back_dataset.py) - see _maybe_capture_back_sample(). Real footage from this run
progressively replaces reference_back_head_embedding, since a person's actual lighting/distance/
angle on the day tends to match live footage far better than the one enrollment session did.

The only spending happens when the locked track_id goes MISSING from the tracker's output (the
target got occluded and ByteTrack has nothing to report for them) and needs to be found again:
every track_id that's brand-new this frame (wasn't present last frame - i.e. not some long-tracked
bystander, but someone who just appeared) gets scored via _score_track(). The best one that still
clears its own threshold (if any) IS the target reappearing - having had no detection while
hidden, ByteTrack can't re-match their reappearance to the old (now-stale) track_id and hands them
a fresh one instead, so this is how the lock gets reclaimed onto it. If nothing matches (or
nothing new appeared at all), the lock actually drops, falling back to ACQUIRING to search again.

_score_track() is the one place appearance is actually compared, for both phases - HEAD REGION
ONLY (identity/face_region.py's crop_head_lower, the top of the bbox down to the shoulder line;
lower-body appearance is never used, since it's dominated by clothing and breaks the moment the
person changes outfit), using whichever of two models applies:
  - FRONT (a real face is found in that crop): identity/face_recognizer.py's YuNet+SFace - an
    actual face detector + face recognition embedding, not a keypoint guess. Compared against the
    enrolled front-face reference at FACE_SIMILARITY_THRESHOLD.
  - BACK (no face found - facing away): identity/osnet_embedder.py's general appearance model on
    that same head-region crop, compared against the enrolled back-of-head reference (if this
    profile has one - see scripts/enroll_person.py) at REID_BACK_HEAD_SIMILARITY_THRESHOLD.
Either reference missing from the profile (older enrollment, or the back-of-head phase was never
done) means that case just can't be scored - returns None, same as a track with no face and no
back-of-head reference to fall back on.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from identity import face_region
from identity.face_recognizer import FaceRecognizer
from identity.osnet_embedder import OSNetEmbedder
from identity.session_back_dataset import SessionBackDataset
from identity.target_profile import load_target_profile
from utils.types import TrackedObject


class TargetLock:
    def __init__(self, profile_path: str, osnet_embedder: Optional[OSNetEmbedder] = None,
                 face_recognizer: Optional[FaceRecognizer] = None,
                 face_similarity_threshold: float = None, back_head_similarity_threshold: float = None,
                 acquire_rounds: int = None, acquire_cooldown_sec: float = None, device: str = "cpu"):
        profile = load_target_profile(profile_path)
        self.reference_face_embedding: Optional[np.ndarray] = profile["face_embedding"]
        self.reference_back_head_embedding: Optional[np.ndarray] = profile["back_head_embedding"]

        self.osnet_embedder = osnet_embedder or OSNetEmbedder(config.REID_MODEL_PATH, device=device)
        self.face_recognizer = face_recognizer or FaceRecognizer(
            config.FACE_DETECTOR_MODEL_PATH, config.FACE_RECOGNIZER_MODEL_PATH,
            score_threshold=config.FACE_DETECT_SCORE_THRESHOLD,
            nms_threshold=config.FACE_DETECT_NMS_THRESHOLD, top_k=config.FACE_DETECT_TOP_K,
        )
        self.face_similarity_threshold = (config.FACE_SIMILARITY_THRESHOLD if face_similarity_threshold is None
                                           else face_similarity_threshold)
        self.back_head_similarity_threshold = (config.REID_BACK_HEAD_SIMILARITY_THRESHOLD
                                                if back_head_similarity_threshold is None
                                                else back_head_similarity_threshold)
        self.acquire_rounds = config.REID_ACQUIRE_ROUNDS if acquire_rounds is None else acquire_rounds
        self.acquire_cooldown_sec = (config.REID_ACQUIRE_COOLDOWN_SEC if acquire_cooldown_sec is None
                                      else acquire_cooldown_sec)
        self._session_back_dataset = SessionBackDataset(
            self.osnet_embedder, config.SESSION_BACK_DATASET_DIR,
            config.SESSION_BACK_CAPTURE_INTERVAL_SEC, config.SESSION_BACK_MAX_SAMPLES,
        )

        self.locked_track_id: Optional[int] = None
        self._prev_track_ids: set = set()  # track_ids seen last frame, to spot brand-new arrivals
        # Re-identify confidence from the most recent real check (the acquiring average that
        # formed the current lock, or the most recent new-arrival reclaim while locked) - NOT
        # updated every frame, since most locked frames don't run a check at all. Exposed for the
        # "TARGET" label (main.py/utils/draw.py) so it shows re-id confidence, not detection.
        self.last_verify_score: Optional[float] = None
        # track_id -> running average of every raw similarity sample so far this acquisition
        # cycle (front or back, whichever applied each round) - the SAME average that decides who
        # locks in, for display on the "searching" candidate boxes so the number on screen means
        # what it looks like it means, instead of just the latest (possibly noisy) round.
        self.candidate_scores: Dict[int, float] = {}
        self._reset_acquisition()

    def _reset_acquisition(self):
        self._acquire_round = 0
        self._acquire_last_round_time: Optional[float] = None
        self._acquire_margins: Dict[int, List[float]] = {}
        self._acquire_sims: Dict[int, List[float]] = {}

    def update(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        if self.locked_track_id is None:
            return self._update_acquiring(tracks, frame)
        return self._update_locked(tracks, frame)

    # --- ACQUIRING: sampled over a few rounds spaced out in real time ---
    def _update_acquiring(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        now = time.time()
        if (self._acquire_last_round_time is not None
                and now - self._acquire_last_round_time < self.acquire_cooldown_sec):
            return None  # waiting out the cooldown before the next sampling round

        for t in tracks:
            result = self._score_track(t, frame)
            if result is None:
                continue
            sim, threshold = result
            self._acquire_margins.setdefault(t.track_id, []).append(sim - threshold)
            sims = self._acquire_sims.setdefault(t.track_id, [])
            sims.append(sim)
            self.candidate_scores[t.track_id] = sum(sims) / len(sims)  # running average, not just this round

        self._acquire_round += 1
        self._acquire_last_round_time = now
        if self._acquire_round < self.acquire_rounds:
            return None  # more rounds still to go

        averages = {tid: sum(margins) / len(margins) for tid, margins in self._acquire_margins.items()}
        self._reset_acquisition()
        if not averages:
            return None  # nobody scoreable this whole cycle - try again from scratch

        best_tid = max(averages, key=averages.get)
        if averages[best_tid] < 0:
            return None  # best candidate still isn't a good enough match - try again from scratch

        self.locked_track_id = best_tid
        # Seed with everyone visible right now, so nobody already in frame spuriously counts as a
        # "new arrival" on the very next locked frame.
        self._prev_track_ids = {t.track_id for t in tracks}
        self.last_verify_score = self.candidate_scores.get(best_tid)
        return self.locked_track_id

    # --- LOCKED: trust ByteTrack while the id is present; only act once it goes missing ---
    def _update_locked(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        current_ids = {t.track_id for t in tracks}
        new_ids = current_ids - self._prev_track_ids
        self._prev_track_ids = current_ids

        if self.locked_track_id in current_ids:
            # Still being tracked - trust it completely, no model calls REQUIRED, regardless of
            # who else is nearby or overlapping. Opportunistically build up a live back-of-head
            # dataset from this certainty (see _maybe_capture_back_sample).
            self._maybe_capture_back_sample(tracks, frame)
            return self.locked_track_id

        # The locked track_id vanished - most likely occluded behind someone else. Check every
        # brand-new arrival before giving up on the lock.
        reclaim_id, reclaim_score = self._check_new_arrivals(tracks, new_ids, frame)
        if reclaim_id is not None:
            self.locked_track_id = reclaim_id
            self.last_verify_score = reclaim_score
            return self.locked_track_id

        self._drop_lock()
        return None

    def _check_new_arrivals(self, tracks: List[TrackedObject], new_ids: set,
                             frame: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Among track_ids that just appeared this frame, find the best one matching the enrolled
        profile, if any - this is how a target hidden behind someone else and reassigned a fresh
        track_id on reappearing gets reclaimed. See the module docstring for why."""
        if not new_ids:
            return None, None
        best_id, best_margin, best_score = None, None, None
        for t in tracks:
            if t.track_id not in new_ids:
                continue
            result = self._score_track(t, frame)
            if result is None:
                continue
            sim, threshold = result
            margin = sim - threshold
            if margin < 0:
                continue
            if best_margin is None or margin > best_margin:
                best_id, best_margin, best_score = t.track_id, margin, sim
        return best_id, best_score

    def _maybe_capture_back_sample(self, tracks: List[TrackedObject], frame: np.ndarray) -> None:
        """While LOCKED and the target's track_id is still present, this is the one moment
        identity/session_back_dataset.py can trust a crop without spending a real check: the
        track_id is certain to be the target (see the module docstring), so if their face happens
        not to be visible right now, that crop IS a genuine back-of-head sample. Rate-limited via
        SessionBackDataset.due() so this stays cheap - most LOCKED frames still make zero model
        calls, only checking in about once every SESSION_BACK_CAPTURE_INTERVAL_SEC."""
        if self._session_back_dataset is None or not self._session_back_dataset.due():
            return
        target = next((t for t in tracks if t.track_id == self.locked_track_id), None)
        if target is None:
            return
        frame_h, frame_w = frame.shape[:2]
        head_crop, _ = face_region.crop_head_lower(frame, target.bbox, target.keypoints, frame_w, frame_h)
        if head_crop.size == 0:
            return
        if self.face_recognizer.detect_best_face(head_crop) is not None:
            return  # facing the camera right now - not a back-of-head moment
        self._session_back_dataset.capture(head_crop)
        self.reference_back_head_embedding = self._session_back_dataset.reference

    def _score_track(self, t: TrackedObject, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Head-region appearance similarity for one track against the enrolled profile, returned
        as (similarity, threshold_that_applies_to_it) since front-face and back-of-head scoring
        use two different models on two different similarity scales - see the module docstring.
        Returns None if neither is usable for this track right now (no face and no back-of-head
        reference, or a degenerate/empty crop)."""
        frame_h, frame_w = frame.shape[:2]
        head_crop, _ = face_region.crop_head_lower(frame, t.bbox, t.keypoints, frame_w, frame_h)
        if head_crop.size == 0:
            return None

        face_row = self.face_recognizer.detect_best_face(head_crop)
        if face_row is not None:
            if self.reference_face_embedding is None:
                return None  # a face is visible, but this profile has no front-face reference
            face_embedding = self.face_recognizer.extract(head_crop, face_row)
            sim = self.face_recognizer.compare(face_embedding, self.reference_face_embedding)
            return sim, self.face_similarity_threshold

        if self.reference_back_head_embedding is None:
            return None  # no face found (facing away), and no back-of-head reference either
        head_embedding = self.osnet_embedder.extract(head_crop)
        sim = self.osnet_embedder.compare(head_embedding, self.reference_back_head_embedding)
        return sim, self.back_head_similarity_threshold

    def _drop_lock(self):
        self.locked_track_id = None
        self._prev_track_ids = set()
        self._reset_acquisition()
