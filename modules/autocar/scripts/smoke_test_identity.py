"""Exercise identity/target_lock.py's actual acquiring/locked state machine with synthetic
TrackedObjects and fake (deterministic, no real model) doubles for both the face recognizer
(YuNet+SFace) and the OSNet embedder - proves the wiring/logic is correct without needing a
camera or the real model files. Wall-clock time (used for the acquisition cooldown) is faked so
this runs instantly instead of actually sleeping.
A separate real-model smoke check (loads the actual onnx files, sanity-checks the API) follows at
the end."""
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


def make_track(track_id, x=100, y=100, w=80, h=200, score=0.9):
    """Synthetic COCO-17 keypoints: shoulders always confident, so crop_head_lower() has a
    stable head/lower split line regardless of which face/appearance path a test exercises."""
    kpts = np.zeros((17, 3), dtype=float)
    kpts[5] = [x + w * 0.3, y + h * 0.2, 0.9]  # left shoulder
    kpts[6] = [x + w * 0.7, y + h * 0.2, 0.9]  # right shoulder
    return TrackedObject(track_id=track_id, bbox=np.array([x, y, x + w, y + h], dtype=float),
                          score=score, keypoints=kpts)


class FakeEmbedder:
    """OSNet double (identity/osnet_embedder.py) - returns fixed vectors in call order, crop
    content is ignored."""
    def __init__(self, embedding_by_call):
        self._embeddings = iter(embedding_by_call)

    def extract(self, crop):
        return next(self._embeddings)

    @staticmethod
    def compare(a, b):
        return float(np.dot(a, b))


class ExplodingEmbedder:
    """Raises if touched at all - proves a code path did NOT call the OSNet embedder when it
    shouldn't (e.g. a track's face was found, so the back-of-head path must be skipped)."""
    def extract(self, crop):
        raise AssertionError("osnet_embedder.extract() should not have been called here")

    @staticmethod
    def compare(a, b):
        raise AssertionError("osnet_embedder.compare() should not have been called here")


# Sentinel "a face was found here" return value for FakeFaceRecognizer.detect_best_face() - real
# YuNet returns a 15-value (bbox+landmarks+score) row; the exact content doesn't matter to the
# fake, only whether it's None (no face) or not.
_FACE = np.zeros(15, dtype=np.float32)


class FakeFaceRecognizer:
    """identity/face_recognizer.py's FaceRecognizer double. detect_best_face() returns items from
    `detections_by_call` in order (each either _FACE - "a face was found" - or None - "no face").
    extract() returns items from `embeddings_by_call` in order (only consumed on calls following a
    _FACE detection with a usable reference - see identity/target_lock.py's _score_track)."""
    def __init__(self, detections_by_call, embeddings_by_call=()):
        self._detections = iter(detections_by_call)
        self._embeddings = iter(embeddings_by_call)

    def detect_best_face(self, crop):
        return next(self._detections)

    def extract(self, crop, face_row):
        return next(self._embeddings)

    @staticmethod
    def compare(a, b):
        return float(np.dot(a, b))


class ExplodingFaceRecognizer:
    """Raises if touched at all - proves a code path did NOT run face detection when it
    shouldn't (e.g. the locked track_id is still present, so nothing needs checking)."""
    def detect_best_face(self, crop):
        raise AssertionError("face_recognizer.detect_best_face() should not have been called here")

    def extract(self, crop, face_row):
        raise AssertionError("face_recognizer.extract() should not have been called here")

    def compare(self, a, b):
        raise AssertionError("face_recognizer.compare() should not have been called here")


def unit(vec):
    return vec / np.linalg.norm(vec)


def new_lock(ref_embedding=None, face_ref_embedding=None, back_ref_embedding=None,
             acquire_rounds=3, acquire_cooldown_sec=0.5):
    lock = TargetLock.__new__(TargetLock)  # bypass __init__ (which loads a real profile + real models)
    lock.reference_face_embedding = face_ref_embedding
    lock.reference_back_head_embedding = back_ref_embedding
    lock.face_similarity_threshold = config.FACE_SIMILARITY_THRESHOLD
    lock.back_head_similarity_threshold = config.REID_BACK_HEAD_SIMILARITY_THRESHOLD
    lock.acquire_rounds = acquire_rounds
    lock.acquire_cooldown_sec = acquire_cooldown_sec
    lock.locked_track_id = None
    lock._prev_track_ids = set()
    lock.last_verify_score = None
    lock.candidate_scores = {}
    lock._session_back_dataset = None  # disabled by default - see the dedicated tests for it
    lock._reset_acquisition()
    return lock


def locked(prev_ids=frozenset({1, 2}), **kw):
    """prev_ids seeds _prev_track_ids so the usual target(1)/other(2) pair used throughout these
    tests doesn't spuriously count as a "new arrival" - tests that specifically want to simulate a
    new arrival use a track_id outside this set (e.g. 3)."""
    lock = new_lock(**kw)
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
close_match = unit(np.array([0.95, 0.05, 0.0] + [0.0] * 509, dtype=np.float32))  # ~0.999 similarity
# Below REID_BACK_HEAD_SIMILARITY_THRESHOLD (0.7ish) on its own, but averaging it with 2
# close_match samples still clears it - see test 4a (one bad round tolerated by averaging).
far_match = unit(np.array([0.4, 0.9165, 0.0] + [0.0] * 509, dtype=np.float32))  # 0.4 similarity
# Below FACE_SIMILARITY_THRESHOLD (0.363) on its own, for the same reason but on the face-
# recognition scale, which sits much lower than OSNet's (SFace cosine scores run lower overall).
front_far_match = unit(np.array([0.15, 0.9887, 0.0] + [0.0] * 509, dtype=np.float32))  # 0.15 similarity

# --- 1. pose_gate: bbox aspect-ratio helper sanity ---
assert abs(pose_gate.aspect_ratio_from_bbox(np.array([0.0, 0.0, 80.0, 200.0])) - 0.4) < 1e-9
print("pose_gate.aspect_ratio_from_bbox OK")

# --- 2. face_region: head/lower split ---
track1 = make_track(1)
head_crop, lower_crop = face_region.crop_head_lower(frame, track1.bbox, track1.keypoints, 640, 480)
assert head_crop.size > 0 and lower_crop.size > 0, "expected both regions non-empty for a normal bbox"
print("face_region head/lower split OK")

# --- 3. target_profile save/load round-trip ---
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
assert loaded["face_embedding"] is None, "expected no face reference when not given to save"
print("target_profile save/load (base fields only) OK")

tmp_full_path = "models/_smoketest_profile_full.npz"
face_ref = unit(np.array([0.0, 0.0, 1.0] + [0.0] * 125, dtype=np.float32))
back_ref = unit(np.array([0.0, 1.0, 0.0] + [0.0] * 509, dtype=np.float32))
save_target_profile(tmp_full_path, head_ref, lower_ref, aspect_ratio=0.4, sample_count=8,
                     back_head_embedding=back_ref, face_embedding=face_ref)
loaded_full = load_target_profile(tmp_full_path)
assert np.allclose(loaded_full["back_head_embedding"], back_ref)
assert np.allclose(loaded_full["face_embedding"], face_ref)
print("target_profile save/load (with face + back-of-head references) OK")

legacy_path = "models/_smoketest_profile_legacy.npz"
legacy_vec = unit(np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32))
np.savez(legacy_path, embedding=legacy_vec, aspect_ratio=np.array(0.5), sample_count=np.array(3),
          created_at=np.array("2020-01-01"), allow_pickle=False)
loaded_legacy = load_target_profile(legacy_path)
assert np.allclose(loaded_legacy["head_embedding"], legacy_vec)
assert np.allclose(loaded_legacy["lower_embedding"], legacy_vec)
assert loaded_legacy["back_head_embedding"] is None
assert loaded_legacy["face_embedding"] is None, "expected pre-face-recognition profiles to load with None"
print("target_profile load (legacy single-embedding format) OK")

import os
os.remove(tmp_path)
os.remove(tmp_full_path)
os.remove(legacy_path)

# --- 4. ACQUIRING: sampled over a few rounds spaced by the cooldown, averaged as MARGINS
# (similarity - that sample's own threshold) so front-face and back-of-head samples - two
# different models on two different similarity scales - can be averaged together safely ---

# 4a. One bad round out of 3 shouldn't disqualify a track - the average MARGIN still clears 0.
# Front-face path throughout (a face is "found" every round).
lock = new_lock(face_ref_embedding=ref_embedding)
lock.face_recognizer = FakeFaceRecognizer([_FACE], [front_far_match])  # round 1: bad sample
target = lock.update([make_track(1)], frame)
assert target is None, "expected still-acquiring after round 1"

fake_clock.advance(0.5)
lock.face_recognizer = FakeFaceRecognizer([_FACE], [close_match])  # round 2
target = lock.update([make_track(1)], frame)
assert target is None, "expected still-acquiring after round 2"

fake_clock.advance(0.5)
lock.face_recognizer = FakeFaceRecognizer([_FACE], [close_match])  # round 3
target = lock.update([make_track(1)], frame)
assert target == 1, f"expected track 1 to lock after 3 rounds (averaging survives 1 bad round), got {target}"
print("acquisition: locks after 3 rounds, one bad round tolerated by averaging OK")

# 4b. Cooldown is enforced: calling update() again before the cooldown elapses must NOT consume
# a round (proven by a face recognizer that raises if touched).
lock2 = new_lock(face_ref_embedding=ref_embedding)
lock2.face_recognizer = FakeFaceRecognizer([_FACE], [close_match])
target = lock2.update([make_track(1)], frame)  # round 1
assert target is None
assert lock2._acquire_round == 1

lock2.face_recognizer = ExplodingFaceRecognizer()
target = lock2.update([make_track(1)], frame)  # same instant, no clock advance
assert target is None
assert lock2._acquire_round == 1, "expected the round counter to NOT advance before the cooldown elapses"
print("acquisition: cooldown blocks a second round from happening too soon OK")

# 4c. Nobody scoreable for all 3 rounds (no face detected, and no back-of-head reference either)
# -> no lock, and the cycle resets (round counter back to 0) instead of getting stuck. The OSNet
# embedder must never even be touched.
lock3 = new_lock(face_ref_embedding=ref_embedding)  # has a face reference, but no back reference
lock3.osnet_embedder = ExplodingEmbedder()
for i in range(3):
    lock3.face_recognizer = FakeFaceRecognizer([None])  # no face detected this round
    target = lock3.update([make_track(1)], frame)
    assert target is None
    fake_clock.advance(0.5)
assert lock3._acquire_round == 0, "expected the acquisition cycle to reset after 3 empty rounds"
print("acquisition: resets and retries when nobody is scoreable OK")

# 4d. No face ever detected, but this profile HAS a back-of-head reference and it matches -> still
# scored (via OSNet on the back-of-head reference) and can still lock.
back_ref = unit(np.array([0.0, 1.0, 0.0] + [0.0] * 509, dtype=np.float32))
back_close_match = unit(np.array([0.05, 0.95, 0.0] + [0.0] * 509, dtype=np.float32))
lock4d = new_lock(back_ref_embedding=back_ref)  # no face_ref_embedding
target = None
for i in range(3):
    lock4d.face_recognizer = FakeFaceRecognizer([None])  # no face detected
    lock4d.osnet_embedder = FakeEmbedder([back_close_match])
    target = lock4d.update([make_track(1)], frame)
    fake_clock.advance(0.5)
assert target == 1, f"expected a back-of-head match to lock even with no face ever detected, got {target}"
print("acquisition: no face ever detected still locks via the back-of-head reference OK")

# 4e. A profile with NEITHER a face reference NOR a back-of-head reference (e.g. a very old
# enrollment) can never lock - detection still runs (that's unavoidable - a face might be there),
# but extract()/compare() on either model must never be touched since there's no reference for
# either case.
lock4e = new_lock()  # no face_ref_embedding, no back_ref_embedding
lock4e.face_recognizer = FakeFaceRecognizer([_FACE])  # a face IS found, but no reference for it
lock4e.osnet_embedder = ExplodingEmbedder()
target = lock4e.update([make_track(1)], frame)
assert target is None
print("acquisition: a profile with no usable reference at all never locks OK")

# --- 5. LOCKED: maintaining an existing lock (jump straight into the locked state by setting
# locked_track_id directly - acquisition is already covered above and is independent of this) ---

# 5a. The locked track_id is still present, nobody else around -> ZERO model calls, stays locked.
lock5a = locked(face_ref_embedding=ref_embedding)
lock5a.face_recognizer = ExplodingFaceRecognizer()
lock5a.osnet_embedder = ExplodingEmbedder()
far_other = make_track(2, x=500)  # unrelated, already-known bystander (id 2 is in prev_ids)
target = lock5a.update([make_track(1, x=100), far_other], frame)
assert target == 1, f"expected the lock to hold with the id still present, got {target}"
print("locked: target still present, nobody else around -> zero model calls OK")

# 5b. The locked track_id is still present even while someone else's bbox is HEAVILY OVERLAPPING
# it -> still ZERO model calls. Empirically ByteTrack keeps a stable id for whoever's in front
# during an overlap (only the occluded person's id disappears), so there's nothing to verify here.
lock5b = locked(face_ref_embedding=ref_embedding)
lock5b.face_recognizer = ExplodingFaceRecognizer()
lock5b.osnet_embedder = ExplodingEmbedder()
near_other = make_track(2, x=110)  # heavily overlapping the target's bbox at x=100
for i in range(5):
    target = lock5b.update([make_track(1, x=100), near_other], frame)
    assert target == 1, f"overlap frame {i}: expected still locked with zero calls, got {target}"
print("locked: target still present even while heavily overlapped -> zero model calls OK")

# 5c. The locked track_id vanishes (occluded), and a brand-new track_id appears whose FACE
# matches the front-face reference -> the lock is reclaimed onto it. This is how the real target
# comes back after being fully hidden: no detection while occluded means ByteTrack can't re-match
# their reappearance to the old track_id and hands them a fresh one instead.
lock5c = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5c.face_recognizer = FakeFaceRecognizer([_FACE], [close_match])  # new arrival's face matches
lock5c.osnet_embedder = ExplodingEmbedder()
reappeared = make_track(3, x=140)
target = lock5c.update([reappeared], frame)  # id 1 is nowhere in this frame's tracks at all
assert target == 3, f"expected the lock to reclaim onto the reappeared target's new id, got {target}"
assert lock5c.locked_track_id == 3
print("locked: target reappearing under a brand-new track_id (fully hidden, now back) gets reclaimed OK")

# 5d. The locked track_id vanishes, and the only new arrival's face does NOT match -> the lock
# actually drops, falling back to ACQUIRING.
lock5d = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5d.face_recognizer = FakeFaceRecognizer([_FACE], [front_far_match])  # new arrival checked, fails
lock5d.osnet_embedder = ExplodingEmbedder()
stranger = make_track(3, x=140)
target = lock5d.update([stranger], frame)
assert target is None, f"expected the lock to drop when the only new arrival doesn't match, got {target}"
assert lock5d.locked_track_id is None, "expected the state machine to fall back to ACQUIRING (search mode)"
print("locked: target vanishes, non-matching new arrival -> lock drops, falls back to acquiring OK")

# 5e. The locked track_id vanishes, and NOTHING new appears at all (just an already-known
# bystander) -> drops immediately, no model call possible (no new arrivals to check).
lock5e = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5e.face_recognizer = ExplodingFaceRecognizer()
lock5e.osnet_embedder = ExplodingEmbedder()
target = lock5e.update([far_other], frame)  # id 2 is already-known, not new; id 1 is gone
assert target is None, f"expected an immediate drop when nothing new appears to check, got {target}"
print("locked: target vanishes with no new arrivals at all -> drops immediately, zero calls OK")

# 5f. Multiple new arrivals appear when the target vanishes - only the one whose face matches (and
# best, if more than one clears the threshold) gets reclaimed.
lock5f = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5f.face_recognizer = FakeFaceRecognizer([_FACE, _FACE], [front_far_match, close_match])  # id3 fails, id4 matches
lock5f.osnet_embedder = ExplodingEmbedder()
decoy = make_track(3, x=140)
reappeared2 = make_track(4, x=300)
target = lock5f.update([decoy, reappeared2], frame)
assert target == 4, f"expected the lock to reclaim onto the one matching new arrival, got {target}"
print("locked: among multiple new arrivals, only the matching one gets reclaimed OK")

# 5g. A brand-new track_id appears while the locked target is STILL PRESENT -> ignored entirely,
# ZERO model calls. There's no ambiguity to resolve while the confirmed target is right there, so
# new arrivals elsewhere in frame are never even looked at.
lock5g = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5g.face_recognizer = ExplodingFaceRecognizer()
lock5g.osnet_embedder = ExplodingEmbedder()
newcomer = make_track(3, x=500)  # brand-new id, target(1) still present
target = lock5g.update([make_track(1, x=100), newcomer], frame)
assert target == 1, f"expected the new arrival to be ignored while the target is present, got {target}"
print("locked: a new arrival is ignored while the locked target is still present, zero calls OK")

# 5h. After a reclaim, the newly-locked id is correctly seeded into _prev_track_ids - it doesn't
# spuriously look "new" again (and thus get needlessly re-checked) on the very next frame.
lock5h = locked(prev_ids={1, 2}, face_ref_embedding=ref_embedding)
lock5h.face_recognizer = FakeFaceRecognizer([_FACE], [close_match])
reappeared3 = make_track(3, x=140)
lock5h.update([reappeared3], frame)
assert lock5h.locked_track_id == 3
lock5h.face_recognizer = ExplodingFaceRecognizer()
lock5h.osnet_embedder = ExplodingEmbedder()
target = lock5h.update([reappeared3], frame)  # same id 3 again, still present
assert target == 3, f"expected the reclaimed lock to stay trusted with zero further calls, got {target}"
print("locked: a reclaimed id is trusted with zero calls on the following frame OK")

# 5i. The locked track_id vanishes, and a brand-new track_id appears FACING AWAY (no face
# detected) whose back-of-head appearance matches -> still reclaimed. This is the walking-away
# case: the target left with their back to the camera and this profile has a back-of-head sample.
lock5i = locked(prev_ids={1, 2}, back_ref_embedding=back_ref)
lock5i.face_recognizer = FakeFaceRecognizer([None])  # new arrival (id 3): no face detected
lock5i.osnet_embedder = FakeEmbedder([back_close_match])  # falls back to back-of-head, matches
walking_away = make_track(3, x=140)
target = lock5i.update([walking_away], frame)
assert target == 3, f"expected a back-of-head match to reclaim the lock, got {target}"
print("locked: target reappearing facing away gets reclaimed via the back-of-head reference OK")

# 5j. While LOCKED with the target's track_id still present, a moment with no face visible gets
# opportunistically captured into the live SessionBackDataset, which then REPLACES
# reference_back_head_embedding with the freshly-built running average - real session footage
# overriding whatever came from enrollment (or filling in if there was none at all).
import shutil
from identity.session_back_dataset import SessionBackDataset

tmp_session_dir = "models/_smoketest_session_back"
lock5j = locked()  # no back_ref_embedding at all - session capture must supply one from scratch
lock5j._session_back_dataset = SessionBackDataset(
    FakeEmbedder([back_close_match]), tmp_session_dir, capture_interval_sec=0.0, max_samples=5,
)
lock5j.face_recognizer = FakeFaceRecognizer([None])  # no face right now - a genuine back moment
target = lock5j.update([make_track(1, x=100), far_other], frame)
assert target == 1, f"expected the lock to stay put while capturing, got {target}"
assert lock5j.reference_back_head_embedding is not None, "expected a session capture to have happened"
assert np.allclose(lock5j.reference_back_head_embedding, back_close_match), \
    "expected the session reference to be the (unit-norm) captured embedding"
shutil.rmtree(tmp_session_dir, ignore_errors=True)
print("locked: a no-face moment while present gets captured, building/overriding the back-of-head reference OK")

time_module.time = _real_time
print("ALL identity smoke tests (synthetic) OK")

# --- SessionBackDataset: due()/capture()/reference in isolation (no TargetLock involved) ---
tmp_dataset_dir = "models/_smoketest_dataset"
ds = SessionBackDataset(FakeEmbedder([close_match, far_match]), tmp_dataset_dir,
                         capture_interval_sec=100.0, max_samples=2)
assert ds.reference is None, "expected no reference before any capture"
assert ds.due(), "expected due() to be True before the first capture"

fake_crop = np.zeros((40, 40, 3), dtype=np.uint8)
ds.capture(fake_crop)
assert ds.reference is not None
assert np.allclose(ds.reference, close_match), "expected the reference to be the single captured embedding"
assert not ds.due(), "expected due() to be False right after a capture (cooldown not elapsed)"

ds.capture(fake_crop)  # capture() itself doesn't check due() - caller's responsibility (as used above)
expected_avg = close_match + far_match
expected_avg = expected_avg / np.linalg.norm(expected_avg)
assert np.allclose(ds.reference, expected_avg), "expected the reference to average both captures"
assert not ds.due(), "expected due() to be False once max_samples is reached"
shutil.rmtree(tmp_dataset_dir, ignore_errors=True)
print("SessionBackDataset: due()/capture()/reference behave correctly in isolation OK")

# --- 6. Real ONNX model load + inference shape/range sanity (no camera needed) ---
from identity.face_recognizer import FaceRecognizer
from identity.osnet_embedder import OSNetEmbedder

embedder = OSNetEmbedder(config.REID_MODEL_PATH, device="cpu")
random_crop = np.random.randint(0, 255, (300, 150, 3), dtype=np.uint8)
emb = embedder.extract(random_crop)
assert emb.shape == (512,), f"expected 512-dim embedding, got {emb.shape}"
assert abs(np.linalg.norm(emb) - 1.0) < 1e-4, f"expected L2-normalized output, got norm={np.linalg.norm(emb)}"
self_similarity = embedder.compare(emb, emb)
assert abs(self_similarity - 1.0) < 1e-4, f"expected self-similarity ~1.0, got {self_similarity}"
print(f"real OSNet ONNX model OK: embedding shape={emb.shape}, self-similarity={self_similarity:.4f}")

face_recognizer = FaceRecognizer(
    config.FACE_DETECTOR_MODEL_PATH, config.FACE_RECOGNIZER_MODEL_PATH,
    score_threshold=config.FACE_DETECT_SCORE_THRESHOLD,
    nms_threshold=config.FACE_DETECT_NMS_THRESHOLD, top_k=config.FACE_DETECT_TOP_K,
)
no_face_row = face_recognizer.detect_best_face(random_crop)
assert no_face_row is None, "expected no face detected in random noise"
print("real YuNet+SFace models OK: loaded, and correctly find no face in random noise")

print("ALL identity smoke tests OK")
