"""
Picks ONE enrolled person's track back out of everyone ByteTrack is currently tracking, and
keeps it locked onto the same track_id across frames. Two distinct phases:

ACQUIRING (self.locked_track_id is None - covers both first startup and re-searching after a
lock was dropped): scores FACE ONLY. The lower-body/aspect-ratio signal is deliberately not
used here - it needs the full body in frame, which someone standing close to the camera (or
just walking in) may not have yet. Every track with a visible face gets one appearance sample
per round; after REID_ACQUIRE_ROUNDS rounds (spaced >= REID_ACQUIRE_COOLDOWN_SEC apart, by wall
clock - smooths over single-frame noise without stalling for long), each track_id's samples are
averaged and the highest average locks in, provided it clears REID_SIMILARITY_THRESHOLD. If
nobody clears it, the round counter resets and acquisition starts over on the next frame.

LOCKED (self.locked_track_id is set): trusts ByteTrack's track_id continuity completely as long as
the locked track_id is still present in the tracker's output - ZERO OSNet calls, no matter how
many other people are nearby or overlapping it. This matches how the tracker actually behaves
during an overlap: empirically, whichever person is in FRONT keeps a stable track_id throughout,
and it's the occluded person's track_id that disappears - there's no observed case of ByteTrack
silently handing one person's id to a different physical person mid-overlap, so there's nothing to
verify while the locked id is still being reported.

The only OSNet spending happens when the locked track_id goes MISSING from the tracker's output
(the target got occluded and ByteTrack has nothing to report for them) and needs to be found
again: every track_id that's brand-new this frame (wasn't present last frame - i.e. not some
long-tracked bystander, but someone who just appeared) gets checked against the enrolled profile.
The best match that clears REID_SIMILARITY_THRESHOLD (if any) IS the target reappearing - having
had no detection while hidden, ByteTrack can't re-match their reappearance to the old (now-stale)
track_id and hands them a fresh one instead, so this is how the lock gets reclaimed onto it. If
nothing matches (or nothing new appeared at all), the lock actually drops, falling back to
ACQUIRING to search again from scratch.

HEAD REGION ONLY (identity/face_region.py) for every appearance check, in both phases - lower-body
appearance is deliberately never used, since it's dominated by clothing color/texture and breaks
the moment the person changes outfit. Within the head region: if the face is visible (nose+eyes
confidently detected), it's compared against the enrolled FRONT-face reference. If the face isn't
visible, it falls back to the enrolled BACK-of-head reference instead (scripts/enroll_person.py
has the person turn around during enrollment to capture this) - so someone walking with their back
to the camera can still be identified, both while first acquiring a lock and while being reclaimed
after an occlusion. Profiles enrolled before this existed have no back-of-head reference
(profile["back_head_embedding"] is None); for those, a track with no visible face simply can't be
scored at all, same as before this existed.
"""
import time
from typing import Dict, List, Optional

import numpy as np

import config
from identity import face_region
from identity.osnet_embedder import OSNetEmbedder
from identity.target_profile import load_target_profile
from utils.types import TrackedObject


class TargetLock:
    def __init__(self, profile_path: str, embedder: Optional[OSNetEmbedder] = None,
                 similarity_threshold: float = None, acquire_rounds: int = None,
                 acquire_cooldown_sec: float = None, device: str = "cpu"):
        profile = load_target_profile(profile_path)
        self.reference_head_embedding: np.ndarray = profile["head_embedding"]
        self.reference_back_head_embedding: Optional[np.ndarray] = profile["back_head_embedding"]

        self.embedder = embedder or OSNetEmbedder(config.REID_MODEL_PATH, device=device)
        self.similarity_threshold = config.REID_SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
        self.acquire_rounds = config.REID_ACQUIRE_ROUNDS if acquire_rounds is None else acquire_rounds
        self.acquire_cooldown_sec = (config.REID_ACQUIRE_COOLDOWN_SEC if acquire_cooldown_sec is None
                                      else acquire_cooldown_sec)

        self.locked_track_id: Optional[int] = None
        self._prev_track_ids: set = set()  # track_ids seen last frame, to spot brand-new arrivals
        # Re-identify confidence from the most recent real check (the acquiring average that
        # formed the current lock, or the most recent new-arrival reclaim while locked) - NOT
        # updated every frame, since most locked frames don't run a check at all. Exposed for the
        # "TARGET" label (main.py/utils/draw.py) so it shows re-id confidence, not detection.
        self.last_verify_score: Optional[float] = None
        # track_id -> most recent face-only similarity-to-target sample, for display on the
        # "searching" candidate boxes while still acquiring.
        self.candidate_scores: Dict[int, float] = {}
        self._reset_acquisition()

    def _reset_acquisition(self):
        self._acquire_round = 0
        self._acquire_last_round_time: Optional[float] = None
        self._acquire_scores: Dict[int, List[float]] = {}

    def update(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        if self.locked_track_id is None:
            return self._update_acquiring(tracks, frame)
        return self._update_locked(tracks, frame)

    # --- ACQUIRING: face-only, sampled over a few rounds spaced out in real time ---
    def _update_acquiring(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        now = time.time()
        if (self._acquire_last_round_time is not None
                and now - self._acquire_last_round_time < self.acquire_cooldown_sec):
            return None  # waiting out the cooldown before the next sampling round

        for t in tracks:
            sim_head = self._score_track(t, frame)
            if sim_head is None:
                continue
            self._acquire_scores.setdefault(t.track_id, []).append(sim_head)
            self.candidate_scores[t.track_id] = sim_head

        self._acquire_round += 1
        self._acquire_last_round_time = now
        if self._acquire_round < self.acquire_rounds:
            return None  # more rounds still to go

        averages = {tid: sum(scores) / len(scores) for tid, scores in self._acquire_scores.items()}
        self._reset_acquisition()
        if not averages:
            return None  # nobody with a visible face this whole cycle - try again from scratch

        best_tid = max(averages, key=averages.get)
        if averages[best_tid] < self.similarity_threshold:
            return None  # best candidate still isn't a good enough match - try again from scratch

        self.locked_track_id = best_tid
        # Seed with everyone visible right now, so nobody already in frame spuriously counts as a
        # "new arrival" on the very next locked frame.
        self._prev_track_ids = {t.track_id for t in tracks}
        self.last_verify_score = averages[best_tid]
        return self.locked_track_id

    # --- LOCKED: trust ByteTrack while the id is present; only act once it goes missing ---
    def _update_locked(self, tracks: List[TrackedObject], frame: np.ndarray) -> Optional[int]:
        current_ids = {t.track_id for t in tracks}
        new_ids = current_ids - self._prev_track_ids
        self._prev_track_ids = current_ids

        if self.locked_track_id in current_ids:
            # Still being tracked - trust it completely, no OSNet calls at all, regardless of who
            # else is nearby or overlapping.
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
                             frame: np.ndarray) -> "tuple[Optional[int], Optional[float]]":
        """Among track_ids that just appeared this frame, find the best one matching the enrolled
        profile, if any - this is how a target hidden behind someone else and reassigned a fresh
        track_id on reappearing gets reclaimed. See the module docstring for why."""
        if not new_ids:
            return None, None
        best_id, best_score = None, -1.0
        for t in tracks:
            if t.track_id not in new_ids:
                continue
            sim = self._score_track(t, frame)
            if sim is not None and sim >= self.similarity_threshold and sim > best_score:
                best_id, best_score = t.track_id, sim
        return best_id, (best_score if best_id is not None else None)

    def _score_track(self, t: TrackedObject, frame: np.ndarray) -> Optional[float]:
        """Head-region appearance similarity for one track against the enrolled profile: the
        front-face reference if the face is visible, otherwise the back-of-head reference (if this
        profile has one enrolled) - so someone facing away from the camera can still be scored.
        Returns None if neither reference is usable for this track right now."""
        if face_region.is_face_visible(t.keypoints):
            reference = self.reference_head_embedding
        elif self.reference_back_head_embedding is not None:
            reference = self.reference_back_head_embedding
        else:
            return None

        frame_h, frame_w = frame.shape[:2]
        head_crop, _ = face_region.crop_head_lower(frame, t.bbox, t.keypoints, frame_w, frame_h)
        if head_crop.size == 0:
            return None
        head_embedding = self.embedder.extract(head_crop)
        return self.embedder.compare(head_embedding, reference)

    def _drop_lock(self):
        self.locked_track_id = None
        self._prev_track_ids = set()
        self._reset_acquisition()
