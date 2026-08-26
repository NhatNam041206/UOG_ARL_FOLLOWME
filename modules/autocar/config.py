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
# generic (unlabeled) track_ids. Every appearance check happens on the HEAD REGION ONLY
# (identity/face_region.py's crop_head_lower - the top of the bbox down to the shoulder line) -
# lower-body appearance is never used, since it's dominated by clothing and breaks the moment the
# person changes outfit. Within that head region, two different models cover the two cases:
#   - FRONT (face visible): a real face detector, not a keypoint guess - YuNet
#     (identity/face_recognizer.py) finds the actual face inside the head-region crop, and SFace
#     turns it into an embedding compared against the enrolled FRONT-face reference via
#     FACE_SIMILARITY_THRESHOLD. This replaced running the general-purpose OSNet appearance model
#     on the whole head-region rectangle, whose lower portion (spanning the full bbox width down
#     to the shoulders) routinely included collar/shoulder clothing and let outfit changes corrupt
#     matches - a real face detector only ever looks at the face itself.
#   - BACK (no face detected in the head-region crop - the person is facing away): falls back to
#     OSNet (identity/osnet_embedder.py) on that same head-region crop, compared against the
#     enrolled BACK-of-head reference via REID_BACK_HEAD_SIMILARITY_THRESHOLD - there's no face
#     for a face detector to find here, so the older general-appearance approach is still used,
#     just only for this one case. Requires scripts/enroll_person.py's back-of-head phase; profiles
#     enrolled before that existed have no back-of-head reference and simply can't be scored when
#     facing away.
#
# Two distinct phases built on top of that scoring:
# ACQUIRING (no one locked yet - covers both first startup and re-searching after a lost lock):
# samples everyone scoreable over REID_ACQUIRE_ROUNDS rounds, spaced >= REID_ACQUIRE_COOLDOWN_SEC
# apart (wall-clock, not frames - smooths out single-frame noise without waiting an excessive
# amount of real time), averages each track_id's samples, and locks onto the highest average IF it
# clears its threshold (FACE_SIMILARITY_THRESHOLD or REID_BACK_HEAD_SIMILARITY_THRESHOLD, matching
# whichever model produced it). If nobody clears it, the round counter resets and it tries again.
#
# LOCKED (maintaining an existing lock): trusts ByteTrack's track_id continuity completely as long
# as the locked track_id is still present in the tracker's output - ZERO model calls, no matter how
# many other people are nearby or overlapping it. Empirically, ByteTrack keeps a stable id for
# whoever's IN FRONT during an overlap and it's the OCCLUDED person's id that disappears, not a
# silent hand-off to the wrong person - so there's nothing to verify while the locked id is still
# being reported.
# The only spending happens when the locked track_id goes MISSING (the target got occluded): every
# track_id that's brand-new this frame (wasn't present last frame) gets checked, front-face or
# back-of-head same as ACQUIRING. The best match that clears its threshold (if any) IS the target
# reappearing - having had no detection while hidden, ByteTrack can't re-match their reappearance
# to the old (stale) track_id and hands them a fresh one instead, so this is how the lock gets
# reclaimed onto it. Only if nothing matches (or nothing new appeared) does the lock actually drop,
# falling back to ACQUIRING to search again.
FACE_DETECTOR_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"  # ONNX (YuNet), CPU-only via cv2.FaceDetectorYN
FACE_RECOGNIZER_MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"  # ONNX (SFace), CPU-only via cv2.FaceRecognizerSF
FACE_DETECT_SCORE_THRESHOLD = 0.6  # YuNet detection confidence floor for "a face is here"
FACE_DETECT_NMS_THRESHOLD = 0.3    # YuNet non-max-suppression IoU threshold (opencv_zoo demo default)
FACE_DETECT_TOP_K = 5000           # YuNet: max candidate boxes considered before NMS
FACE_SIMILARITY_THRESHOLD = 0.363  # SFace cosine similarity cutoff - opencv_zoo's own documented
# "same person" threshold for this exact model (github.com/opencv/opencv_zoo face_recognition_sface);
# a reasonable starting point, but still worth recalibrating against this project's real footage.
REID_MODEL_PATH = "models/osnet_x1_0_msmt17.onnx"  # ONNX (OSNet), CPU-only (onnxruntime) - BACK-of-head only now
REID_BACK_HEAD_SIMILARITY_THRESHOLD = 0.7  # PLACEHOLDER - cosine similarity cutoff, not yet calibrated on real data
REID_HEAD_SPLIT_FALLBACK_FRACTION = 0.35  # head-region height as a fraction of bbox height, used only
# when shoulder keypoints aren't confident enough to place the head/lower split line precisely
REID_HEAD_CROP_WIDTH_FRACTION = 0.7  # PLACEHOLDER - head-region width as a fraction of shoulder-to-
# shoulder distance (identity/face_region.py's _head_x_range) - narrows the head crop away from the
# full bbox width so it excludes more collar/shoulder clothing, mattering most for the BACK-of-head
# OSNet path which has no face detector to otherwise ignore the crop's edges
REID_ACQUIRE_ROUNDS = 3            # how many sampling rounds before picking a target
REID_ACQUIRE_COOLDOWN_SEC = 0.5    # minimum wall-clock gap between acquisition rounds

# --- Live session back-of-head dataset (identity/session_back_dataset.py) ---
# While LOCKED and the target's track_id is still being reported (identity/target_lock.py is
# certain it's really them), any moment with no face visible is captured as a fresh back-of-head
# sample - real footage from THIS run, which then REPLACES REID_BACK_HEAD_SIMILARITY_THRESHOLD's
# reference embedding (identity/target_lock.py's reference_back_head_embedding), overriding
# whatever came from the one-time scripts/enroll_person.py back-of-head phase (or filling in for
# it if that was skipped). Images are saved to SESSION_BACK_DATASET_DIR/session_<timestamp>/ for
# inspection - gitignored, not reloaded on a future run.
SESSION_BACK_DATASET_DIR = "temp_dataset"
SESSION_BACK_CAPTURE_INTERVAL_SEC = 1.0  # minimum wall-clock gap between opportunistic captures -
# keeps the per-frame face-detection check (needed to confirm "no face right now") rare, not constant
SESSION_BACK_MAX_SAMPLES = 50  # caps memory/disk use per run; running average, not a growing average

# --- Enrollment (scripts/enroll_person.py) ---
ENROLL_ROI_PERCENT = (0.30, 0.0, 0.72, 1.0)  # (x1,y1,x2,y2) as a fraction of frame size - stand inside this box
ENROLL_COUNTDOWN_SEC = 3
ENROLL_DURATION_SEC = 8
ENROLL_SAMPLE_INTERVAL_FRAMES = 5
ENROLL_MIN_SAMPLES = 5
