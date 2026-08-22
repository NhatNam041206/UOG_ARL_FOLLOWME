"""Exercise identity/target_lock.py's actual acquiring/locked state machine with synthetic
TrackedObjects and a fake (deterministic, no real model) embedder - proves the wiring/logic is
correct without needing a camera or the real ~20MB OSNet ONNX model. Wall-clock time (used for
the acquisition cooldown) is faked so this runs instantly instead of actually sleeping.
A separate real-model smoke check (loads the actual onnx, runs inference on a random crop)
follows at the end."""
import sys
import time as time_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from identity import face_region, pose_gate
from identity.target_lock import TargetLock
from identity.target_profile import load_target_profile, save_target_profile, sanitize_person_name
from utils.types import TrackedObject


def make_track(track_id, x=100, y=100, w=80, h=200, score=0.9, face_visible=True):
    """Synthetic COCO-17 keypoints: shoulders always confident (stable head/lower split line),
    nose+eyes confident only if face_visible."""
    kpts = np.zeros((17, 3), dtype=float)
    face_conf = 0.9 if face_visible else 0.0
    kpts[0] = [x + w * 0.5, y + h * 0.08, face_conf]   # nose
    kpts[1] = [x + w * 0.45, y + h * 0.06, face_conf]  # left eye
    kpts[2] = [x + w * 0.55, y + h * 0.06, face_conf]  # right eye
    kpts[5] = [x + w * 0.3, y + h * 0.2, 0.9]          # left shoulder
    kpts[6] = [x + w * 0.7, y + h * 0.2, 0.9]          # right shoulder
    return TrackedObject(track_id=track_id, bbox=np.array([x, y, x + w, y + h], dtype=float),
                          score=score, keypoints=kpts)


class FakeEmbedder:
    """Returns fixed vectors in call order - crop content is ignored."""
    def __init__(self, embedding_by_call):
        self._embeddings = iter(embedding_by_call)

    def extract(self, crop):
        return next(self._embeddings)

    @staticmethod
    def compare(a, b):
        return float(np.dot(a, b))


class ExplodingEmbedder:
    """Raises if touched at all - proves a code path did NOT call the embedder when it shouldn't."""
    def extract(self, crop):
        raise AssertionError("embedder.extract() should not have been called here")

    @staticmethod
    def compare(a, b):
        raise AssertionError("embedder.compare() should not have been called here")


def unit(vec):
    return vec / np.linalg.norm(vec)


def new_lock(ref_embedding, back_ref_embedding=None, acquire_rounds=3, acquire_cooldown_sec=0.5):
    lock = TargetLock.__new__(TargetLock)  # bypass __init__ (which loads a real profile + real onnx)
    lock.reference_head_embedding = ref_embedding
    lock.reference_back_head_embedding = back_ref_embedding
    lock.similarity_threshold = config.REID_SIMILARITY_THRESHOLD
    lock.acquire_rounds = acquire_rounds
    lock.acquire_cooldown_sec = acquire_cooldown_sec
    lock.locked_track_id = None
    lock._prev_track_ids = set()
    lock.last_verify_score = None
    lock.candidate_scores = {}
    lock._reset_acquisition()
    return lock


def locked(ref_embedding, prev_ids=frozenset({1, 2}), **kw):
    """prev_ids seeds _prev_track_ids so the usual target(1)/other(2) pair used throughout these
    tests doesn't spuriously count as a "new arrival" - tests that specifically want to simulate a
    new arrival use a track_id outside this set (e.g. 3)."""
    lock = new_lock(ref_embedding, **kw)
    lock.locked_track_id = 1
    lock._prev_track_ids = set(prev_ids)
    return lock


class FakeClock:
    """Lets acquisition-cooldown tests advance wall-clock time instantly instead of sleeping."""
    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, dt):
        self.now += dt


fake_clock = FakeClock()
_real_time = time_module.time
time_module.time = fake_clock.time  # patches the `time` module globally for this process

frame = np.zeros((480, 640, 3), dtype=np.uint8)
ref_embedding = unit(np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32))
close_match = unit(np.array([0.95, 0.05, 0.0] + [0.0] * 509, dtype=np.float32))
# Partial (not zero) similarity to ref_embedding - fails REID_SIMILARITY_THRESHOLD on its own,
# but still similar enough that averaging it with 2 close_match samples clears the threshold
# (see test 4a: one bad round out of 3 shouldn't disqualify a track).
far_match = unit(np.array([0.4, 0.9165, 0.0] + [0.0] * 509, dtype=np.float32))

# --- 1. pose_gate: bbox aspect-ratio helper sanity ---
assert abs(pose_gate.aspect_ratio_from_bbox(np.array([0.0, 0.0, 80.0, 200.0])) - 0.4) < 1e-9
print("pose_gate.aspect_ratio_from_bbox OK")

# --- 2. face_region: face-visibility gate + head/lower split ---
visible_track = make_track(1, face_visible=True)
hidden_track = make_track(2, face_visible=False)
assert face_region.is_face_visible(visible_track.keypoints)
assert not face_region.is_face_visible(hidden_track.keypoints)
assert not face_region.is_face_visible(None)

head_crop, lower_crop = face_region.crop_head_lower(frame, visible_track.bbox, visible_track.keypoints, 640, 480)
assert head_crop.size > 0 and lower_crop.size > 0, "expected both regions non-empty for a normal bbox"
print("face_region gate + split OK")

# --- 3. target_profile save/load round-trip (new head/lower format + legacy fallback) ---
assert sanitize_person_name("Nguyen Van A") == "Nguyen_Van_A"
tmp_path = "models/_smoketest_profile.npz"
head_ref = unit(np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32))
lower_ref = unit(np.array([0.0, 1.0, 0.0] + [0.0] * 509, dtype=np.float32))
save_target_profile(tmp_path, head_ref, lower_ref, aspect_ratio=0.4, sample_count=8)
loaded = load_target_profile(tmp_path)
assert loaded["sample_count"] == 8
assert abs(loaded["aspect_ratio"] - 0.4) < 1e-9
assert np.allclose(loaded["head_embedding"], head_ref)
assert np.allclose(loaded["lower_embedding"], lower_ref)
assert loaded["back_head_embedding"] is None, "expected no back-of-head reference when not given to save"
print("target_profile save/load (new format, no back-of-head) OK")

tmp_back_path = "models/_smoketest_profile_back.npz"
back_ref = unit(np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32))
save_target_profile(tmp_back_path, head_ref, lower_ref, aspect_ratio=0.4, sample_count=8,
                     back_head_embedding=back_ref)
loaded_back = load_target_profile(tmp_back_path)
assert loaded_back["back_head_embedding"] is not None
assert np.allclose(loaded_back["back_head_embedding"], back_ref)
print("target_profile save/load (with back-of-head reference) OK")

legacy_path = "models/_smoketest_profile_legacy.npz"
legacy_vec = unit(np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32))
np.savez(legacy_path, embedding=legacy_vec, aspect_ratio=np.array(0.5), sample_count=np.array(3),
          created_at=np.array("2020-01-01"), allow_pickle=False)
loaded_legacy = load_target_profile(legacy_path)
assert np.allclose(loaded_legacy["head_embedding"], legacy_vec)
assert np.allclose(loaded_legacy["lower_embedding"], legacy_vec)
assert loaded_legacy["back_head_embedding"] is None, "expected pre-back-of-head profiles to load with None"
print("target_profile load (legacy single-embedding format) OK")

import os
os.remove(tmp_path)
os.remove(tmp_back_path)
os.remove(legacy_path)

# --- 4. ACQUIRING: face-only, 3 rounds spaced by the cooldown, averaged ---

# 4a. One bad round out of 3 shouldn't disqualify a track - the AVERAGE still clears threshold.
lock = new_lock(ref_embedding)
lock.embedder = FakeEmbedder([far_match])  # round 1: one bad sample
target = lock.update([make_track(1, face_visible=True)], frame)
assert target is None, "expected still-acquiring after round 1"

fake_clock.advance(0.5)
lock.embedder = FakeEmbedder([close_match])  # round 2
target = lock.update([make_track(1, face_visible=True)], frame)
assert target is None, "expected still-acquiring after round 2"

fake_clock.advance(0.5)
lock.embedder = FakeEmbedder([close_match])  # round 3 -> average of [far, close, close] should pass
target = lock.update([make_track(1, face_visible=True)], frame)
assert target == 1, f"expected track 1 to lock after 3 rounds (averaging survives 1 bad round), got {target}"
print("acquisition: locks after 3 rounds, one bad round tolerated by averaging OK")

# 4b. Cooldown is enforced: calling update() again before the cooldown elapses must NOT consume
# a round (proven by an embedder that raises if touched).
lock2 = new_lock(ref_embedding)
lock2.embedder = FakeEmbedder([close_match])
target = lock2.update([make_track(1, face_visible=True)], frame)  # round 1
assert target is None
assert lock2._acquire_round == 1

lock2.embedder = ExplodingEmbedder()
target = lock2.update([make_track(1, face_visible=True)], frame)  # same instant, no clock advance
assert target is None
assert lock2._acquire_round == 1, "expected the round counter to NOT advance before the cooldown elapses"
print("acquisition: cooldown blocks a second round from happening too soon OK")

# 4c. Nobody has a visible face for all 3 rounds -> no lock, and the cycle resets (round counter
# back to 0) instead of getting stuck. The embedder must never even be touched.
lock3 = new_lock(ref_embedding)
lock3.embedder = ExplodingEmbedder()
for i in range(3):
    target = lock3.update([make_track(1, face_visible=False)], frame)
    assert target is None
    fake_clock.advance(0.5)
assert lock3._acquire_round == 0, "expected the acquisition cycle to reset after 3 empty rounds"
print("acquisition: resets and retries when nobody has a visible face OK")

# 4d. Face hidden, but this profile HAS a back-of-head reference and it matches -> still scored
# and can still lock (using the back-of-head reference instead of skipping the track entirely).
back_ref = unit(np.array([0.0, 1.0, 0.0] + [0.0] * 509, dtype=np.float32))
back_close_match = unit(np.array([0.05, 0.95, 0.0] + [0.0] * 509, dtype=np.float32))
lock4d = new_lock(ref_embedding, back_ref_embedding=back_ref)
lock4d.embedder = FakeEmbedder([back_close_match] * 3)
target = None
for i in range(3):
    target = lock4d.update([make_track(1, face_visible=False)], frame)
    fake_clock.advance(0.5)
assert target == 1, f"expected a back-of-head match to lock even with the face never visible, got {target}"
print("acquisition: face hidden the whole time still locks via the back-of-head reference OK")

# --- 5. LOCKED: maintaining an existing lock (jump straight into the locked state by setting
# locked_track_id directly - acquisition is already covered above and is independent of this) ---

# 5a. The locked track_id is still present, nobody else around -> ZERO OSNet calls, stays locked.
lock5a = locked(ref_embedding)
lock5a.embedder = ExplodingEmbedder()
far_other = make_track(2, x=500)  # unrelated, already-known bystander (id 2 is in prev_ids)
target = lock5a.update([make_track(1, x=100), far_other], frame)
assert target == 1, f"expected the lock to hold with the id still present, got {target}"
print("locked: target still present, nobody else around -> zero OSNet calls OK")

# 5b. The locked track_id is still present even while someone else's bbox is HEAVILY OVERLAPPING
# it -> still ZERO OSNet calls. Empirically ByteTrack keeps a stable id for whoever's in front
# during an overlap (only the occluded person's id disappears), so there's nothing to verify here.
lock5b = locked(ref_embedding)
lock5b.embedder = ExplodingEmbedder()
near_other = make_track(2, x=110)  # heavily overlapping the target's bbox at x=100
for i in range(5):
    target = lock5b.update([make_track(1, x=100), near_other], frame)
    assert target == 1, f"overlap frame {i}: expected still locked with zero calls, got {target}"
print("locked: target still present even while heavily overlapped -> zero OSNet calls OK")

# 5c. The locked track_id vanishes (occluded), and a brand-new track_id appears that MATCHES the
# target's face -> the lock is reclaimed onto it. This is how the real target comes back after
# being fully hidden: no detection while occluded means ByteTrack can't re-match their
# reappearance to the old track_id and hands them a fresh one instead.
lock5c = locked(ref_embedding, prev_ids={1, 2})
lock5c.embedder = FakeEmbedder([close_match])  # only the new arrival (id 3) gets checked
reappeared = make_track(3, x=140, face_visible=True)
target = lock5c.update([reappeared], frame)  # id 1 is nowhere in this frame's tracks at all
assert target == 3, f"expected the lock to reclaim onto the reappeared target's new id, got {target}"
assert lock5c.locked_track_id == 3
print("locked: target reappearing under a brand-new track_id (fully hidden, now back) gets reclaimed OK")

# 5d. The locked track_id vanishes, and the only new arrival does NOT match the target's face ->
# the lock actually drops, falling back to ACQUIRING.
lock5d = locked(ref_embedding, prev_ids={1, 2})
lock5d.embedder = FakeEmbedder([far_match])  # new arrival (id 3) checked, fails
stranger = make_track(3, x=140, face_visible=True)
target = lock5d.update([stranger], frame)
assert target is None, f"expected the lock to drop when the only new arrival doesn't match, got {target}"
assert lock5d.locked_track_id is None, "expected the state machine to fall back to ACQUIRING (search mode)"
print("locked: target vanishes, non-matching new arrival -> lock drops, falls back to acquiring OK")

# 5e. The locked track_id vanishes, and NOTHING new appears at all (just an already-known
# bystander) -> drops immediately, no embedder call possible (no new arrivals to check).
lock5e = locked(ref_embedding, prev_ids={1, 2})
lock5e.embedder = ExplodingEmbedder()
target = lock5e.update([far_other], frame)  # id 2 is already-known, not new; id 1 is gone
assert target is None, f"expected an immediate drop when nothing new appears to check, got {target}"
print("locked: target vanishes with no new arrivals at all -> drops immediately, zero calls OK")

# 5f. Multiple new arrivals appear when the target vanishes - only the one that matches (and best,
# if more than one clears the threshold) gets reclaimed.
lock5f = locked(ref_embedding, prev_ids={1, 2})
lock5f.embedder = FakeEmbedder([far_match, close_match])  # id 3 fails, id 4 matches
decoy = make_track(3, x=140, face_visible=True)
reappeared2 = make_track(4, x=300, face_visible=True)
target = lock5f.update([decoy, reappeared2], frame)
assert target == 4, f"expected the lock to reclaim onto the one matching new arrival, got {target}"
print("locked: among multiple new arrivals, only the matching one gets reclaimed OK")

# 5g. A brand-new track_id appears while the locked target is STILL PRESENT -> ignored entirely,
# ZERO embedder calls. There's no ambiguity to resolve while the confirmed target is right there,
# so new arrivals elsewhere in frame are never even looked at.
lock5g = locked(ref_embedding, prev_ids={1, 2})
lock5g.embedder = ExplodingEmbedder()
newcomer = make_track(3, x=500, face_visible=True)  # brand-new id, target(1) still present
target = lock5g.update([make_track(1, x=100), newcomer], frame)
assert target == 1, f"expected the new arrival to be ignored while the target is present, got {target}"
print("locked: a new arrival is ignored while the locked target is still present, zero calls OK")

# 5h. After a reclaim, the newly-locked id is correctly seeded into _prev_track_ids - it doesn't
# spuriously look "new" again (and thus get needlessly re-checked) on the very next frame.
lock5h = locked(ref_embedding, prev_ids={1, 2})
lock5h.embedder = FakeEmbedder([close_match])
reappeared3 = make_track(3, x=140, face_visible=True)
lock5h.update([reappeared3], frame)
assert lock5h.locked_track_id == 3
lock5h.embedder = ExplodingEmbedder()
target = lock5h.update([reappeared3], frame)  # same id 3 again, still present
assert target == 3, f"expected the reclaimed lock to stay trusted with zero further calls, got {target}"
print("locked: a reclaimed id is trusted with zero calls on the following frame OK")

# 5i. The locked track_id vanishes, and a brand-new track_id appears FACING AWAY (no face visible)
# that matches the enrolled BACK-of-head reference -> still reclaimed. This is the walking-away
# case: the target left with their back to the camera and this profile has a back-of-head sample.
lock5i = locked(ref_embedding, prev_ids={1, 2}, back_ref_embedding=back_ref)
lock5i.embedder = FakeEmbedder([back_close_match])  # new arrival (id 3), face hidden, back matches
walking_away = make_track(3, x=140, face_visible=False)
target = lock5i.update([walking_away], frame)
assert target == 3, f"expected a back-of-head match to reclaim the lock, got {target}"
print("locked: target reappearing facing away gets reclaimed via the back-of-head reference OK")

time_module.time = _real_time
print("ALL identity smoke tests (synthetic) OK")

# --- 6. Real ONNX model load + inference shape/range sanity (no camera needed) ---
from identity.osnet_embedder import OSNetEmbedder

embedder = OSNetEmbedder(config.REID_MODEL_PATH, device="cpu")
random_crop = np.random.randint(0, 255, (300, 150, 3), dtype=np.uint8)
emb = embedder.extract(random_crop)
assert emb.shape == (512,), f"expected 512-dim embedding, got {emb.shape}"
assert abs(np.linalg.norm(emb) - 1.0) < 1e-4, f"expected L2-normalized output, got norm={np.linalg.norm(emb)}"
self_similarity = embedder.compare(emb, emb)
assert abs(self_similarity - 1.0) < 1e-4, f"expected self-similarity ~1.0, got {self_similarity}"
print(f"real OSNet ONNX model OK: embedding shape={emb.shape}, self-similarity={self_similarity:.4f}")

print("ALL identity smoke tests OK")
