"""All tunable constants in one place."""

# --- Detector ---
POSE_MODEL_PATH = "yolov8n-pose.pt"
DETECT_CONF = 0.4          # low-ish on purpose: ByteTrack itself filters by score in two tiers
DETECT_IMGSZ = 300

# --- Tracker (ByteTrack) ---
TRACK_HIGH_THRESH = 0.6     # detections >= this are "high score" (stage 1 matching)
TRACK_LOW_THRESH = 0.1      # detections in [LOW, HIGH) are "low score" (stage 2 matching only)
NEW_TRACK_THRESH = 0.7      # unmatched high-score detections need >= this to spawn a new track
MATCH_THRESH = 0.8          # IoU distance threshold for stage 1 (1 - IoU, so lower = stricter)
LOW_MATCH_THRESH = 0.5      # IoU distance threshold for stage 2
TRACK_BUFFER = 30           # frames a lost track is kept before being dropped for good

# --- Keypoints (COCO 17) ---
KEYPOINT_CONF_THRESH = 0.65  # below this, a keypoint is not drawn / not considered visible.
# Higher than the textbook 0.5: at close range (face/shoulders filling the
# frame, limbs off-screen), the pose model still guesses positions for the
# invisible keypoints with middling confidence, producing crossed-looking
# skeleton lines. This filters out more of those low-quality guesses.

# --- Target re-identification (identity/, scripts/enroll_person.py, main.py --target) ---
# Off by default (no --target given). Enrolls ONE specific person's appearance so the live
# pipeline can pick their track back out among everyone else detected, instead of ByteTrack's
# generic (unlabeled) track_ids. Two distinct phases:
#
# ACQUIRING (no one locked yet - covers both first startup and re-searching after a lost lock):
# scores HEAD REGION ONLY (OSNet similarity vs. the enrolled head embedding) - deliberately ignores
# the lower-body/aspect-ratio signal here, so someone whose legs aren't even in frame yet can still
# be picked up. If the face is visible (nose+eyes), scores against the FRONT-face reference;
# otherwise falls back to the BACK-of-head reference if this profile has one enrolled (see
# scripts/enroll_person.py) - so someone walking with their back to the camera can still be
# acquired. Samples everyone scoreable over REID_ACQUIRE_ROUNDS rounds, spaced >=
# REID_ACQUIRE_COOLDOWN_SEC apart (wall-clock, not frames - smooths out single-frame noise/pose
# without waiting an excessive amount of real time), averages each track_id's samples, and locks
# onto the highest average IF it clears REID_SIMILARITY_THRESHOLD. If nobody clears it, the round
# counter resets and it tries again from scratch on the next frame.
#
# LOCKED (maintaining an existing lock): trusts ByteTrack's track_id continuity completely as long
# as the locked track_id is still present in the tracker's output - ZERO OSNet calls, no matter
# how many other people are nearby or overlapping it. Empirically, ByteTrack keeps a stable id for
# whoever's IN FRONT during an overlap and it's the OCCLUDED person's id that disappears, not a
# silent hand-off to the wrong person - so there's nothing to verify while the locked id is still
# being reported.
# The only OSNet spending happens when the locked track_id goes MISSING (the target got occluded):
# every track_id that's brand-new this frame (wasn't present last frame) gets checked against the
# enrolled profile - front-face or back-of-head, same rule as ACQUIRING. The best match that clears
# REID_SIMILARITY_THRESHOLD (if any) IS the target reappearing - having had no detection while
# hidden, ByteTrack can't re-match their reappearance to the old (stale) track_id and hands them a
# fresh one instead, so this is how the lock gets reclaimed onto it. Only if nothing matches (or
# nothing new appeared) does the lock actually drop, falling back to ACQUIRING to search again.
REID_MODEL_PATH = "models/osnet_x1_0_msmt17.onnx"  # ONNX, CPU-only (onnxruntime), static [1,3,256,128] input
REID_SIMILARITY_THRESHOLD = 0.75  # PLACEHOLDER - cosine similarity cutoff, not yet calibrated on real data
REID_FACE_MIN_KEYPOINT_CONF = 0.5  # nose+eye keypoint confidence floor to count as "face visible"
REID_HEAD_SPLIT_FALLBACK_FRACTION = 0.35  # head-region height as a fraction of bbox height, used only
# when shoulder keypoints aren't confident enough to place the head/lower split line precisely
REID_ACQUIRE_ROUNDS = 3            # how many face-only sampling rounds before picking a target
REID_ACQUIRE_COOLDOWN_SEC = 0.5    # minimum wall-clock gap between acquisition rounds

# --- Enrollment (scripts/enroll_person.py) ---
ENROLL_ROI_PERCENT = (0.30, 0.0, 0.72, 1.0)  # (x1,y1,x2,y2) as a fraction of frame size - stand inside this box
ENROLL_COUNTDOWN_SEC = 3
ENROLL_DURATION_SEC = 8
ENROLL_SAMPLE_INTERVAL_FRAMES = 5
ENROLL_MIN_SAMPLES = 5
