# Modules

Working principles, public contracts, and parameters for each module. Every module exposes
exactly one importable file (`interface.py`); everything else described here is a private
implementation detail, included for explanation, not as an external API. For calibration status
and tuning notes on every parameter, see [`docs/parameters.md`](parameters.md) — this file
explains what each parameter *does*, that one tracks whether it's *set correctly*.

---

## `emergency_stop`

**Pipeline position:** independent safety layer, not part of either main pipeline's data flow —
runs on the raw full frame on its own.

**Purpose:** the sole collision-avoidance layer (no non-CV sensor backstop exists yet). Governing
principle: *when uncertain, STOP* — `UNCERTAIN` is treated identically to `STOP` by every
consumer.

**Working principle:**
1. A fixed "runway" trapezoid is defined once per frame size as two straight lines
   (`runway_left_line`/`runway_right_line`, normalized frame-fraction endpoints) — frame-relative
   only, no bird's-eye transform, no steering-angle skew (accepted simplification: turning is
   expected to be slow/rare).
2. The frame is cropped to the trapezoid's bounding rectangle (+`roi_buffer_px` margin, which
   only avoids clipping detection bboxes — it never loosens the actual membership test below).
3. A **standalone** YOLO11n instance (all 80 COCO classes — generic obstacle detection, not
   person-only) + ByteTrack runs on that crop.
4. Any detection below `min_detection_confidence` escalates the **whole frame** to `UNCERTAIN`
   rather than being silently dropped (a dropped low-confidence detection could be a real
   obstacle — the confidence floor is a safety gate, not a filter).
5. Each remaining detection's **ground-contact point** (bbox bottom-center, not centroid) is
   tested for exact trapezoid membership, then classified into far/mid/near zone by a y-fraction
   split (`zone_far_boundary`/`zone_mid_boundary`). A size pre-filter
   (`size_prefilter_width_px`/`height_px`) skips objects too small to matter, as a fast-path only
   — it never overrides the zone logic for a large object correctly in the far zone.
6. **Near zone → immediate STOP.** **Mid zone → STOP only after `t_mid_seconds` of continuous
   per-track dwell** (resets, doesn't pause, if the track leaves mid zone or drops out of
   detection before then).
7. **Resume hysteresis:** once the runway is clear, `STOP`→`GO` only after `resume_buffer_seconds`
   of *uninterrupted* clear frames — any interruption resets the timer to zero. Cold start begins
   with no clear-streak history (never assumes `GO` on frame 1).

**Public contract** (`EStopOutput`): `decision` (`GO`/`STOP`/`UNCERTAIN`), `reason` (e.g.
`"near_zone_object"`, `"low_confidence_detection"`, `"resume_buffer_pending"`),
`triggering_track_id`, `zone`, `timestamp`. Plus `last_latency_ms` for the frame-time
benchmarking the spec requires before a latency budget can be safely set.

**Key parameters:** `runway_left_line`/`runway_right_line`, `roi_buffer_px`,
`size_prefilter_width_px`/`height_px`, `zone_far_boundary`/`zone_mid_boundary`, `t_mid_seconds`,
`min_detection_confidence`, `resume_buffer_seconds` — **all 10 required**, all currently `null`
(uncalibrated placeholders). See [parameters.md](parameters.md#emergency_stop).

**Known limitation (intentional):** no steering-angle-aware trapezoid skewing.

---

## `human_detection`

**Pipeline position:** first stage of the legacy `wave_facing`/`both` pipeline.

**Purpose:** fast, whole-frame, person-only detection + tracking, feeding one bbox+`track_id`
per person to `wave_facing_gate`.

**Working principle:** a standalone YOLO11n instance, `classes=[0]` (COCO person only, both a
correctness filter and a real speed win — skips postprocessing for 79 irrelevant classes) +
ByteTrack (`persist=True`). `track_id` continuity depends on calling `.track()` on a continuous
frame stream from the *same* instance — skipping frames or reinstantiating resets tracking.
Explicitly does **not** do pose estimation, gesture recognition, facing-camera checks, or
identity/Re-ID — `track_id` is motion-continuity only, never a verified identity.

**Public contract** (`PersonDetection`): `track_id`, `bbox` (x1,y1,x2,y2), `confidence`.

**Key parameters:** `confidence_threshold` (0.5, working default — not calibration-gated, no
spec requires it). See [parameters.md](parameters.md#human_detection-optional).

---

## `wave_facing_gate` (Gesture Method 1 — `condition`)

**Pipeline position:** consumes one person crop per call — from `human_detection` in the legacy
pipeline, or from `human_detection_roi` (via `main.py`'s adapter) in the face-first pipeline
(`--modules pretrigger`/`followme`).

**Purpose:** two independent, individually-debounced signals — `is_waving` and
`is_facing_camera` — from static pose geometry + short-window motion. Does **not** compute the
final trigger; the caller ANDs `registered_person AND is_waving AND is_facing_camera`.

**Working principle:**
1. Crop → preprocessed → a **standalone** MoveNet Lightning instance → 17 COCO keypoints
   (y, x, score), decoded to bbox-pixel space.
2. **`is_facing_camera` (raw, stateless):** a crude four-keypoint visibility proxy — passes iff
   left_eye, right_eye, left_shoulder, right_shoulder *all* clear `confidence_threshold_facing`.
   Not head-pose estimation; can't distinguish "facing camera" from "facing camera at a steep
   angle," accepted for MVP.
3. **Gate A (static pose, per-frame, no memory), evaluated independently per arm:** passes iff
   wrist/elbow/shoulder confidence all clear `confidence_threshold_pose`, AND wrist is above
   `wrist_height_fraction` of bbox height, AND wrist is above the elbow, AND **both**
   wrist→elbow and wrist→shoulder vectors are within `verticality_threshold_deg` of vertical
   (deliberate redundancy so one noisy keypoint can't singlehandedly pass the check).
4. **Gate B (motion, short rolling window), evaluated independently per arm and independently of
   Gate A** (accumulates every frame regardless of Gate A's pass/fail — the two gates share no
   state): buffers wrist samples over `motion_window_seconds`; computes displacement vectors
   between consecutive samples; drops any vector shorter than `motion_min_displacement_px` (the
   noise floor that stops MoveNet's own per-frame inference jitter on a *static* pose from
   registering as spurious direction changes — calibration-critical, unverified by default);
   counts direction changes ≥ `motion_direction_change_angle_deg` between consecutive
   significant vectors; passes iff that count ≥ `motion_min_direction_changes` AND the buffer has
   ≥ `motion_min_samples`.
5. An arm is "waving" this frame iff **both** Gate A and Gate B pass for it; the first arm found
   passing wins (`wave_arm`), and `wave_arm` persists as the *last* arm that won even on frames
   where neither currently passes (debug/display convenience).
6. Both raw booleans (`waving_raw`, `facing_raw`) feed **independent** RED/YELLOW/GREEN
   `ConfirmationTracker` instances — `is_waving`/`is_facing_camera` are `True` only at `GREEN`.

**Public contract** (`GestureFacingResult`): `is_waving`, `is_facing_camera`, `waving_state`,
`facing_state`, `wave_arm`, `facing_confidence_min`, `keypoints_raw`, plus
`draw_debug(frame)` (keypoints, skeleton, arm vectors, Gate A pass/fail).

**Key parameters:** `confidence_threshold_facing`, `confidence_threshold_pose`,
`wrist_height_fraction`, `verticality_threshold_deg`, `motion_window_seconds`,
`motion_confidence_threshold`, `motion_min_samples`, `motion_min_direction_changes`,
`motion_direction_change_angle_deg`, `motion_min_displacement_px`,
`confirmation_duration_seconds`. See [parameters.md](parameters.md#wave_facing-method-1--moduleswave_facing_gate).

---

## `face_identity`

**Pipeline position:** stage 1 of the face-first pipeline — runs on the full frame, first.

**Purpose:** detect every face in frame and match each against a registry of pre-enrolled
people, by name.

**Working principle:**
1. A standalone YuNet instance (`cv2.FaceDetectorYN`) detects all faces in the full frame,
   returning bbox + 5-point landmarks (eyes, nose, mouth corners) + score per face in one pass.
2. Faces below `face_detection_confidence_threshold` are dropped.
3. Each remaining face's landmarks are used to **align** it: `cv2.estimateAffinePartial2D`
   estimates a similarity transform (rotation + uniform scale + translation, no shear) from the
   5 landmarks to a fixed 112×112 reference template (the standard ArcFace-family alignment
   template), then `cv2.warpAffine` produces a canonical 112×112 crop. (Alignment always
   normalizes to this canonical output regardless of how tightly/loosely the *original* face was
   cropped, so registration-time crop padding doesn't affect embedding quality.)
4. The aligned crop is embedded by a standalone EdgeFace-XS ONNX session → a 512-D vector,
   L2-normalized.
5. **Matching:** since both the live embedding and every registry entry's embedding are already
   L2-normalized, cosine similarity reduces to a plain dot product. The registry entry with the
   highest dot product wins; it's a match iff that best score ≥ `similarity_threshold_face_match`.
   The best score is always returned (even on a non-match) for calibration visibility.
6. A single frame's match is **not** debounced over time by this module — that's the calling
   pipeline's job if needed (currently neither `pretrigger` nor `followme` debounce it either;
   identity is re-evaluated fresh every frame).

**Registry** (`FaceRegistry`, `.npz` per person, keyed by sanitized name): `save_person()` takes
N sample embeddings (from the two-phase capture flow below), stores every sample plus a mean
composite (re-normalized) as the entry actually matched against. `load_all()` skips unreadable
files with a warning rather than failing the whole registry.

**Two-phase registration** (separate from the live pipeline): `capture_face_images.py` (Phase 1)
saves *padded* face crops (60% padding beyond the detected bbox) to
`raw_captures/<person>/NNN.jpg`, letting you review/swap source photos; `build_face_registry.py`
(Phase 2) re-detects, aligns, embeds, and writes the registry entry from that folder. Splitting
these means changing which photos back a person doesn't require re-running live capture.

**Public contract** (`FaceIdentityResult`): `face_found`, `face_bbox` (full-frame),
`is_registered_match`, `matched_person_name`, `match_confidence`, `face_detection_confidence`.
`evaluate()` returns a `List[...]` — zero, one, or many faces; it never picks "the" person.

**Key parameters:** `similarity_threshold_face_match`, `face_detection_confidence_threshold` —
both required, both fail-closed. See [parameters.md](parameters.md#face_identity).

---

## `human_detection_roi`

**Pipeline position:** stage 2 of the face-first pipeline — only ever triggered once a face has
already matched a registered person; never runs independently.

**Purpose:** find that same person's full-body bbox, scoped to a region around their face rather
than scanning the whole frame — tighter, and less likely to pick up a different person's body in
a crowd.

**Working principle:**
1. `compute_roi()` derives a crop rectangle from the matched face bbox: height budget =
   `face_height × roi_expansion_factor`, split `roi_upward_fraction` above the face /
   the rest below (body extends down, not up — downward-biased by default); width =
   `face_width × roi_expansion_factor × roi_width_fraction`, centered on the face.
2. A **standalone** YOLO11n instance, person-class-only, runs a **single-frame `.predict()`**
   (deliberately *not* `.track()`/ByteTrack — see the isolation rule in
   [architecture.md](architecture.md#design-rules-apply-across-every-module) — this ROI shifts
   every frame, following wherever the face currently is, which isn't a stable coordinate frame
   for a tracker's motion model) on that crop.
3. If the ROI yields multiple person detections (e.g. a crowd), `_select_best_detection()`
   disambiguates: prefer whichever detection's bbox actually **contains the face bbox's center**
   (among those, highest confidence); if none contain it, fall back to whichever detection's
   center is closest to the face center.
4. If nothing is found in the ROI, reports `person_found=False` — it does **not** silently retry
   against the full frame (that would defeat the point of ROI-scoping).
5. The winning detection's bbox is converted back to full-frame coordinates before returning.

**Public contract** (`HumanDetectionResult`): `person_found`, `person_bbox` (x,y,w,h,
full-frame), `detection_confidence`. No persistent `track_id` — see point 2 above; downstream
modules key their own state off the matched person's name instead.

**Key parameters:** `roi_expansion_factor`, `detection_confidence_threshold` (required,
fail-closed) plus `roi_upward_fraction`/`roi_width_fraction` (optional overrides, working
defaults). See [parameters.md](parameters.md#human_detection_roi) for the tuning workflow.

---

## `gesture_hand_keypoint` (Gesture Method 2 — `hand_keypoint`)

**Pipeline position:** consumes one person crop + that person's full-frame bbox (needed for the
palm-height gate) per call.

**Purpose:** detect a specific hand-shape sequence — **OPEN → CLOSED → OPEN → CLOSED** — using
*only* MediaPipe hand landmark geometry. No motion, trajectory, or arm geometry of any kind (a
full redesign from an earlier motion-based version; nothing of that version remains).

**Working principle:**
1. A **standalone** MediaPipe `HandLandmarker` (Tasks API — the legacy `mp.solutions.hands` API
   doesn't exist in this MediaPipe version) detects up to 2 hands per crop, returning all 21
   fixed-layout landmarks + a handedness label/confidence per hand. All 21 landmarks are always
   computed in one model pass — there is no "compute fewer keypoints" option; `num_hands` (fixed
   at 2) is the actual speed/coverage knob if that's ever worth exposing.
2. **Per-hand shape classification** (`hand_shape.py`, stateless, per-frame):
   - The 4 non-thumb fingers: tip farther from wrist than PIP joint = "extended" (each finger
     independently).
   - Thumb: `distance(thumb_tip, pinky_MCP) / distance(wrist, middle_MCP)` (hand-scale-normalized)
     exceeding `thumb_extension_ratio_threshold` = "extended."
   - **OPEN** iff ≥ `min_fingers_extended_open` of the 4 non-thumb fingers are extended **and**
     the thumb is independently extended.
   - **CLOSED** iff ≥ `min_fingers_curled_closed` of the 4 non-thumb fingers are curled — the
     thumb is **not** checked for CLOSED. (A natural fist rests the thumb *over* the curled
     fingers rather than tucking it into the palm, which does not reliably read as "curled" by
     the distance-based thumb test — requiring it false-negatives on a completely normal fist.)
   - Otherwise **NEITHER** (ambiguous — doesn't advance the sequence, isn't a reset either).
3. **Palm-height gate:** the wrist landmark, converted from crop-local to full-frame pixels, must
   sit in the upper `palm_height_fraction` of the person's **full-frame** bbox height (not just
   the crop's own height — these can differ if the crop pipeline ever pads/resizes).
4. **Single-hand selection, side-agnostic:** each frame, of all hands clearing
   `confidence_threshold`, only those *also* clearing the palm-height gate are eligible; the
   highest-confidence eligible hand drives the sequence. **Left/Right side plays no role** — one
   shared sequence machine per track, not one per side, so switching which physical hand is
   raised mid-gesture doesn't reset progress, and a MediaPipe handedness mislabel can't split one
   gesture into two stalled sequences.
5. **Sequence state machine** (`WAITING_OPEN → WAITING_CLOSE_1 → WAITING_OPEN_2 → WAITING_CLOSE_2
   → CONFIRMED`): advances one stage per clean OPEN/CLOSED read matching what that stage expects;
   a `NEITHER` read or "no eligible hand this frame" simply doesn't advance (no reset); exceeding
   `max_transition_gap_seconds` since the last transition resets to `WAITING_OPEN` (no partial
   credit); **failing the palm-height gate is an immediate reset**, stricter than a merely
   non-advancing frame. Must start from OPEN — a sequence starting at CLOSED doesn't count.
6. `CONFIRMED` is a one-shot momentary event, reconciled with the (continuous-condition-shaped)
   RED/YELLOW/GREEN tracker by holding a synthetic "True" pulse for
   `2 × confirmation_duration_seconds` — long enough for the tracker to complete its normal
   promotion and then visibly dwell at GREEN rather than instantly reverting to RED.

**Public contract** (`GestureMethodResult`): `is_waving`, `waving_state`, `sequence_stage`,
`confidence_debug` (best hand's handedness confidence), `keypoints_raw`,
`palm_facing_camera_debug` (debug-only, does not gate anything — see `palm_orientation.py`), plus
`draw_debug(frame, person_bbox_full_frame=...)`: skeleton/keypoints colored **yellow = OPEN,
green = CLOSED, gray = NEITHER/uncalibrated**, per-finger green/red extended/curled highlight
edges, and a red dotted horizontal line at the `palm_height_fraction` calibration cutoff.

**Key parameters:** `confidence_threshold`, `min_fingers_extended_open`,
`min_fingers_curled_closed` (both scored out of 4 non-thumb fingers — see point 2),
`thumb_extension_ratio_threshold` (gates OPEN only), `palm_height_fraction`,
`max_transition_gap_seconds`, `confirmation_duration_seconds` — all required, all fail-closed.
See [parameters.md](parameters.md#gesture_hand_keypoint-method-2) for the full tuning workflow.

---

## `gesture_trajectory_verifier` (Gesture Method 3 — `trajectory_verifier`)

**Pipeline position:** consumes one person crop per call (does not need the full-frame bbox —
ignores `person_bbox_full_frame` if passed).

**Purpose:** match a live arm trajectory against a small, shared, **generic** (not per-person)
set of reference gesture trajectories via shape similarity.

**Working principle:**
1. A **standalone** MoveNet Lightning instance (independent from `wave_facing_gate`'s own
   instance — model reused, no shared code/state) → 17 keypoints per frame, decoded to bbox pixel
   space.
2. **Per arm** (both computed every cycle, best score wins), a rolling `TrajectoryBuffer`
   accumulates (wrist, elbow, shoulder) samples over `trajectory_window_seconds` — all three
   points, not wrist-only (a design correction: wrist-only loses arm-shape information, since a
   bent vs. straight arm can trace the same wrist path). A sample is only added if
   wrist/elbow/shoulder confidence all clear `confidence_threshold`.
3. **Normalization** (identical treatment for live and reference trajectories): translate each
   point-track to start at its own first sample (relative motion, not absolute frame position);
   scale by the bbox height *at capture time* (a stable body-scale reference — wrist-to-shoulder
   distance itself changes during the gesture and would distort the normalization).
4. **Resampling:** fixed-length, **time-based** linear interpolation to `resample_length` evenly
   spaced points across the buffer's own time span (chosen over arc-length-based resampling — a
   wave is roughly periodic, so non-uniform speed is a smaller risk; simpler; a well-scoped
   future upgrade if empirical testing later shows shape fidelity suffers).
5. **Similarity:** the resampled (wrist, elbow, shoulder) x,y sequence is flattened to one vector
   and compared to every reference trajectory via **cosine similarity** (not DTW — an explicit
   non-goal unless proven insufficient). The best (arm, reference, score) triple across both arms
   and the whole reference set wins.
6. **"Not ready" floor:** if the reference set has fewer than `MIN_REFERENCE_COUNT = 2` entries
   (a fixed structural constant, not a config key — 0 or 1 references offer no meaningful "best
   of set" comparison), `evaluate()` unconditionally returns `is_waving=False` with
   `confidence_debug`/`matched_reference_id` both `None` and the real `reference_count` — visibly
   distinguishable from a genuine non-match (which always has a real score), so this can't be
   misread as "evaluated and didn't match" during calibration.
7. The best score clearing `similarity_threshold` is the raw candidate signal, debounced through
   the same RED/YELLOW/GREEN pattern as the other two methods.

**Reference trajectories:** captured separately via `capture_reference_trajectory.py`, stored as
`.npz` under `reference_trajectories/` (flattened vector + `resample_length` + `arm`), loaded
fresh (`load_all()`) on every `evaluate()` call — not per-person, a small shared generic set.

**Public contract** (`GestureMethodResult`): `is_waving`, `waving_state`, `confidence_debug`,
`matched_reference_id`, `arm`, `reference_count`, `keypoints_raw`. **No `draw_debug()` method** —
unlike Methods 1 and 2, this method's own standalone `visualize_gesture_trajectory_verifier.py`
draws its debug overlay directly from `keypoints_raw` rather than through the result object;
`main.py`'s adapter no-ops gracefully when asked to debug-draw this method.

**Key parameters:** `confidence_threshold`, `trajectory_window_seconds`,
`min_samples_for_comparison`, `resample_length`, `similarity_threshold`,
`confirmation_duration_seconds` — all required, all currently `null` (uncalibrated). See
[parameters.md](parameters.md#gesture_trajectory_verifier-method-3).

---

---

## `appearance_verifier`

**Pipeline position:** shared dependency, not part of either main pipeline's own data flow —
consumed by `modules.target_tracking` (a periodic re-verification sanity check during active
tracking) and `modules.target_recovery` (a fallback re-acquisition path). Holds no per-caller
state; both callers may run their own independent usage of it simultaneously.

**Purpose:** answers one question — "does this new person crop look like the same person as this
earlier set of reference crops?" — an appearance-based identity check, distinct from and
complementary to `face_identity`'s face-based check.

**Working principle:**
1. A **standalone** OSNet (`osnet_x1_0`) instance, via `torchreid`'s official `FeatureExtractor`
   utility (its documented preprocessing exactly: resize to 256×128, ImageNet mean/std
   normalization — not hand-rolled). Weights: the REAL Market1501-trained checkpoint (94.2%
   rank-1, 82.6% mAP), auto-downloaded via `gdown` from its published Google Drive file id and
   cached locally on first use — notably **not** what `torchreid`'s own `pretrained=True`
   shortcut provides (that only fetches an ImageNet-classification backbone despite the name;
   discovered and corrected during this module's implementation, see `embedder.py`'s docstring
   for the full account).
2. `embed(crop_bgr)`: BGR→RGB swap, `FeatureExtractor` forward pass, L2-normalize the output
   (the extractor's own forward pass does not normalize it).
3. `build_reference_set(person_crops)`: embeds every provided crop **once** and stores the
   vectors — callers build this once per episode (e.g. a tracking module's RECORD phase) and
   reuse it across many comparisons, never re-embedding reference crops per call.
4. `verify(candidate_crop, reference_set)`: embeds the candidate once, compares against every
   stored reference embedding via cosine similarity (both sides L2-normalized, so this reduces to
   a plain dot product — same pattern as `face_identity`'s matching stage), returns the **best**
   score found. `match_found = best_score >= similarity_threshold`.
5. **"Not ready" floor:** an empty reference set (`reference_frame_count == 0`) reports
   `match_found=False` without attempting a meaningless comparison — `best_similarity_score`
   stays a real placeholder (`0.0`, never `None`/`NaN`) so the type stays a plain `float`;
   callers must check `reference_frame_count`, not the score itself, to distinguish "not ready"
   from a genuine non-match.

**Public contract** (`AppearanceVerifierResult`): `match_found`, `best_similarity_score` (always
a real number), `reference_frame_count`.

**Key parameters:** `similarity_threshold` — required, fail-closed, and treated as **especially**
uncalibrated (see Known Limitations below — a starting guess here is less trustworthy than usual
elsewhere in this project). `osnet_model_name` — working default (`osnet_x1_0`), not
calibration-gated. See [parameters.md](parameters.md#appearance_verifier).

**Known limitations** (both apply to every caller, not just one — kept deliberately separate):
1. **Similar-clothing confusion.** OSNet-based appearance matching struggles to distinguish
   people wearing similar-colored/styled clothing, since appearance embeddings lean heavily on
   clothing as a feature. Test explicitly for this during calibration — don't assume it away.
2. **Cross-domain generalization drop.** Published OSNet benchmarks show accuracy can drop
   sharply on footage meaningfully different from its training distribution (Market-1501-family
   datasets). This project's own campus footage/lighting/camera are an untested domain relative
   to that training data — a distinct risk from clothing confusion, not the same one.

Both `target_tracking`'s periodic re-verify and `target_recovery`'s Path B fallback inherit both
risks whole; each caller uses its own independently-tunable threshold key rather than this
module's `similarity_threshold`, precisely so each can be tuned to its own risk tolerance (a
false LOST during active tracking is a different cost than a false re-acquisition during search).

---

## `target_tracking` — SUPERSEDED, not in the live call path

> **`modules.followme_orchestrator` no longer calls this module.** It now drives
> `modules/autocar` (vendored tracking+recovery backbone) via `autocar_adapter.py` instead — see
> [`autocar` / `autocar_adapter`](#autocar--autocar_adapter-vendored-tracking--recovery-backbone)
> below and [architecture.md](architecture.md#post-trigger-flow-tracking-recovery--steering-plans05-08-backbone-replaced-since).
> This module and its own `test_*.py`/`visualize_*.py` still exist and still run standalone;
> kept until the replacement is fully confirmed, then deleted. The description below documents
> what it does/did, for reference.

**Pipeline position (historical):** took over once a gesture trigger is confirmed (`is_waving`
reaches `GREEN` in whichever gesture method is active) — driven by `modules.followme_orchestrator`
(`main.py --modules followme`), which composed this module together with the rest of the
pipeline. Also complete and independently testable on its own. Handed off to
`modules.target_recovery` on `LOST`.

**Purpose:** lock onto the triggering person as "the target," record a short appearance
reference set, track them frame-to-frame, and report how far off-center they are for downstream
steering — without doing any steering computation itself.

**Working principle:**
1. `start(initial_person_bbox, frame, timestamp)` records the desired target bbox and enters
   `RECORDING`. The actual `track_id` **lock** happens on the next `update()` call: this module's
   own standalone YOLO+ByteTrack instance (`tracker.py`, `persist=True` — unlike
   `human_detection_roi`'s deliberately stateless single-frame `.predict()`, this tracker follows
   the locked target continuously) reports every visible person; whichever detection best matches
   the desired bbox (center-containment first, closest-center fallback — the same disambiguation
   *style* `human_detection_roi._select_best_detection` uses, independently reimplemented) has
   its `track_id` adopted as the lock. Every later frame, only that `track_id` is followed.
2. **RECORDING**: every frame the locked target is seen, its crop is appended to a buffer.
   Elapsed time is measured via the `timestamp` argument, never an assumed frame rate. Once
   `record_duration_seconds` elapses: if fewer than `min_recording_crops` usable crops were
   collected (confirmed with the user), RECORDING **extends** — keeps what it has, gives itself
   more time — rather than building a fragile reference set; otherwise
   `appearance_verifier.build_reference_set()` is called once and the module transitions to
   `TRACKING`.
3. **TRACKING**: every frame, `compute_horizontal_offset()` derives a normalized `-1.0` (frame-left)
   to `+1.0` (frame-right) deviation from the bbox center vs. frame center — deliberately **not**
   a true angle; FOV-based angle conversion is the downstream steering layer's job, never this
   module's (`camera.fov_degrees` never appears in its config).
4. **Periodic appearance re-verification**, every `appearance_reverify_interval_seconds` (not
   every frame, for cost reasons): calls `appearance_verifier.verify()` on the current crop
   against the reference set, using this module's **own**
   `appearance_reverify_similarity_threshold` (never `appearance_verifier`'s own
   `similarity_threshold` — kept independently tunable by design). **Two consecutive failures**
   (confirmed with the user, over declaring `LOST` on the first) are required before transitioning
   to `LOST` — a single bad-lighting/occlusion frame doesn't end tracking.
5. **Track loss**: if the locked `track_id` is missing from the tracker's output for
   `track_loss_grace_period_seconds` of continuous wall-clock time, transition to `LOST`. This
   check applies uniformly during RECORDING too (an extension beyond the spec's literal wording,
   which addresses it only under TRACKING) — RECORDING needs the same tracker running
   continuously to keep capturing the *moving* target, so "is the locked ID still being seen" is
   the same question in both states.
6. **`LOST`**: `target_locked` becomes `False`; `reference_set` stays populated with the
   last-built set so the caller (or `target_recovery` directly) has what it needs to attempt
   re-acquisition. Nothing further happens until `reset(fresh_person_bbox, frame, timestamp)` is
   called (re-enters `RECORDING`, identical to a fresh `start()`).

**Public contract** (`TrackingResult`): `target_locked`, `horizontal_offset`, `person_bbox`
(x,y,w,h, full-frame), `state` (`RECORDING`/`TRACKING`/`LOST`), `reference_set`.

**Key parameters:** `record_duration_seconds`, `appearance_reverify_interval_seconds`,
`appearance_reverify_similarity_threshold`, `track_loss_grace_period_seconds` — all required,
fail-closed. `min_recording_crops` (3) and `appearance_reverify_consecutive_failures` (2) are
working defaults confirmed with the user. See [parameters.md](parameters.md#target_tracking).

**Known limitations:**
- ByteTrack's `track_id` continuity is motion-based, not identity-verified (same caveat
  documented for `human_detection`). In a crowd, ByteTrack can silently reassign the locked
  `track_id` to a different nearby person after an occlusion without ever reporting a track loss
  — the periodic appearance re-verification exists specifically to catch this failure mode, not
  as a general accuracy improvement.
- No true-angle/FOV-based steering computation, and no PID or control-loop logic of any kind —
  both are the downstream steering layer's job (`plans/08`, not yet built).
- `reset()`'s implemented signature (`fresh_person_bbox, frame, timestamp`) intentionally differs
  from `plans/06_target_tracking.md §0.3`'s literal draft, which omitted those parameters despite
  `reset()`'s own docstring describing a fresh bbox being handed back — an internal spec
  inconsistency, resolved to match the described behavior.

---

## `target_recovery` — SUPERSEDED, not in the live call path

> Same status as `target_tracking` above — `autocar_adapter`'s `TargetLock` folds recovery
> directly into its own tracking state machine, so there is no separate recovery module/call site
> anymore. Kept until the replacement is fully confirmed, then deleted.

**Pipeline position (historical):** took over once `modules.target_tracking` reported
`state == LOST`. Driven by `modules.followme_orchestrator` (`main.py --modules followme`) — see
the note on `target_tracking` above.

**Purpose:** re-acquire the same registered target by searching the *whole* frame (not the
narrow region tracking was using), via two paths of different strength and cost.

**Working principle:**
1. `start(reference_set, target_person_name, timestamp)` begins a search episode.
   `target_person_name` identifies *which* registered person this episode is for — added beyond
   `plans/07_target_recovery.md §0.3`'s literally drafted signature (confirmed with the user,
   flagged by the spec itself as a likely real gap): `face_identity.evaluate()` can return
   multiple registered people's matches in a crowd, and Path A needs to know which one is
   actually the target rather than accepting any registered match.
2. **Path A (primary, tried every frame):** `face_identity.evaluate(frame, registry)`, filtered
   to a result where `is_registered_match` and `matched_person_name == target_person_name`. A
   match resets the consecutive-failure counter (`face_search_fail_count`) unconditionally — the
   thing that counter tracks is "is the target's face even detectable/matchable," which this
   already answers "yes" to — then `human_detection_roi.evaluate()` (identical to the main
   pipeline's own Stage 1→2 handoff) gets a fresh body bbox. `person_found` → `REACQUIRED` via
   `"face_match"`. If the face matched but the body wasn't found this exact frame (rare: face
   visible, body occluded), the episode keeps searching without incrementing the failure count.
   No match → `face_search_fail_count += 1`.
3. **Path B (fallback, only once `face_search_fail_count >= face_search_grace_attempts`):** a
   **standalone** whole-frame person detector (own independent YOLO instance, per the spec's own
   default — never `human_detection`'s existing detection call) finds every visible person;
   `appearance_verifier.verify()` runs against each candidate using this module's **own**
   `appearance_fallback_threshold`. The best candidate clearing it → `REACQUIRED` via
   `"appearance_fallback"`, using that candidate's bbox **directly** — `human_detection_roi` is
   deliberately **not** re-run, since Path B already found a body bbox by a different mechanism;
   re-scoping a region already known to contain the target would be pure waste.
4. `face_search_grace_attempts` is a **COUNT** of consecutive Path-A-failure frames, not a time
   duration (deliberate, per the spec — see Known Limitations). `search_timeout_seconds` (a real
   time duration) is checked every frame regardless of which path is being tried; once elapsed
   with no `REACQUIRED`, the episode resolves to `TIMEOUT`.
5. `REACQUIRED` and `TIMEOUT` are terminal for a given episode — the caller (orchestration layer)
   is responsible for calling `target_tracking.reset()` on `REACQUIRED`, or propagating a stop
   signal on `TIMEOUT`. This module produces no robot commands of its own.

**Public contract** (`RecoveryResult`): `status` (`SEARCHING`/`REACQUIRED`/`TIMEOUT`),
`reacquired_person_bbox` (populated only on `REACQUIRED`), `reacquired_via`
(`"face_match"`/`"appearance_fallback"`), `face_search_fail_count`, `elapsed_search_seconds`.

**Key parameters:** `face_search_grace_attempts`, `appearance_fallback_threshold`,
`search_timeout_seconds` — all required, fail-closed. See
[parameters.md](parameters.md#target_recovery).

**Known limitations:**
- `face_search_grace_attempts` is a count, not a duration — deliberate: face detection (YuNet,
  full-frame) is variable-cost inference, so a time-based gate would give Path A an inconsistent
  number of real attempts depending on system load that cycle (unfair to Path A on a slow cycle,
  unnecessarily cautious on a fast one). A count ties the threshold to actual attempts made,
  independent of frame rate. Do not conflate this with `search_timeout_seconds`, which correctly
  remains time-based since it bounds total wall-clock search duration, not attempt count.
- Path B inherits **both** of `appearance_verifier`'s named risks whole (see that module's
  section above) — similar-clothing confusion and cross-domain generalization drop. Not
  re-explained here; both apply exactly as documented there.

---

## `autocar` / `autocar_adapter` (vendored tracking + recovery backbone)

**Pipeline position:** takes over once a gesture trigger is confirmed — driven by
`modules.followme_orchestrator`, which calls `autocar_adapter.start()`/`update()` exclusively
for the post-trigger phase (replacing `target_tracking`+`target_recovery` above).

**Purpose:** lock onto the triggering person, track them frame-to-frame, and automatically
re-acquire them after an occlusion — one state machine handles both jobs, unlike the
tracking/recovery split it replaced.

**Two distinct pieces, two distinct owners:**
- **`modules/autocar/`** — a teammate's already-built tracking+recovery engine
  (`vinhh9608-byte/Autocar`, commit `27ee33a`), pulled in via `git clone` and kept **completely
  unmodified** — not even a new file added to that directory. Contains their own `detector/`
  (`YOLOv8PoseTorch`, ultralytics YOLOv8n-pose), `tracker/` (`BYTETracker`, a from-scratch
  numpy+scipy ByteTrack implementation — no `lap`/`filterpy` dependency), `identity/`
  (`TargetLock` — the actual state machine, `OSNetEmbedder` — ONNX-via-onnxruntime re-id
  embedding, `face_region.py` — splits a bbox into head/lower regions using pose keypoints,
  `target_profile.py` — the `.npz` save/load format), and their own `config.py`.
- **`modules/followme_orchestrator/autocar_adapter.py`** — the ONLY file that imports from
  `modules/autocar/`. Everything project-specific lives here, not in the vendored tree.

**Working principle (`TargetLock`, theirs, unmodified):**
1. **`ACQUIRING`** (no lock held): scores every visible track's HEAD region only (front-face if
   visible, back-of-head otherwise) against the enrolled profile, over `reid_acquire_rounds`
   sampling rounds spaced `reid_acquire_cooldown_sec` apart; the highest-average track that clears
   `reid_similarity_threshold` gets locked.
2. **`LOCKED`**: trusts the tracker's `track_id` continuity completely while it's still being
   reported — **zero** re-id inference cost, no matter how many other people are nearby. Their own
   empirical finding: whichever person is in FRONT during an overlap keeps a stable `track_id`;
   it's the occluded person's id that disappears, never a silent hand-off to the wrong physical
   person — so there's nothing to verify while the id is still present.
3. **Reclaim on loss**: the moment the locked `track_id` vanishes, every track_id that's
   brand-new this frame (not present last frame) gets scored against the enrolled profile the
   same way as `ACQUIRING`; the best match above threshold reclaims the lock. Nothing matching →
   the lock drops, falling back to `ACQUIRING` to search again from scratch.

**What the adapter adds (not in their code at all):**
- **Force-lock at `start()`**: runs one `detect()`+`track()` pass on the trigger frame, IoU-matches
  the caller's bbox against the resulting tracks, and reaches directly into
  `TargetLock.locked_track_id`/`_prev_track_ids` to lock onto it immediately — **skipping their
  own `ACQUIRING` entirely**, since `face_identity` + the gesture trigger already proved identity
  and location more precisely than a few rounds of face-only sampling would (and with more than
  one person in frame, re-deriving it could momentarily lock onto the wrong one). The one
  documented, deliberate reach into their private state — no public method does this, and their
  file is never edited.
- **`horizontal_offset`**: computed from the tracked bbox's center vs. frame-center each frame —
  their code has no steering-related output at all.
- **`recovery_timeout_seconds`**: their reclaim search retries indefinitely with no timeout on its
  own; the adapter wraps it with a timeout mirroring `target_recovery`'s old
  `search_timeout_seconds` convention exactly (`is not None and elapsed >= timeout` — `None` means
  never times out).
- **bbox convention translation**: this project uses `(x, y, w, h)`; their code uses `xyxy`
  throughout. Converted only at this one seam.

**Public contract** (`autocar_adapter.TrackingResult`): `target_locked`, `horizontal_offset`,
`person_bbox` (x,y,w,h, full-frame), `state` (`TRACKING`/`SEARCHING`/`LOST`),
`just_reacquired` (True only on the exact frame a mid-episode reclaim succeeds — the caller's
signal to reset `SteeringController`'s PID state).

**Key parameters:** `config/thresholds.yaml`'s `autocar:` section — detector/tracker/re-id values
carried over as their own considered starting points (not blind guesses), plus
`recovery_timeout_seconds` (ours, `null`/uncalibrated by default). See
[parameters.md](parameters.md#autocar).

**Requires a pre-enrolled profile** (`modules/autocar/models/enrolled_<name>.npz`) per followable
person — unlike the old `target_tracking`, there is no on-the-fly reference capture. See
`register_person.py` / [architecture.md](architecture.md#registration-ui-register_personpy--a-second-composition-root).
Also requires `modules/autocar/models/osnet_x1_0_msmt17.onnx` to exist — see
[technologies.md](technologies.md) for how it's obtained.

**Known limitations:**
- Same ByteTrack motion-continuity caveat as every other tracker in this project — their own
  empirical claim about front-occludes-back during overlaps is trusted, not independently
  re-verified here.
- No true-angle/FOV-based steering computation or PID logic in `modules/autocar/` itself — same
  architectural boundary the old `target_tracking` had; `SteeringController` still owns that.
- `ACQUIRING`'s multi-round face-sampling logic is present in the vendored code but never actually
  exercised in normal operation, since `start()` always force-locks successfully unless the
  trigger frame's detector missed the person entirely (rare) — in that one fallback case,
  `TargetLock` does run its own real `ACQUIRING` search.

---

## `followme_orchestrator`

**Pipeline position:** the composition root for the ENTIRE pipeline — the only module permitted
to import across other modules' `interface.py` boundaries besides `main.py` itself (a deliberate,
documented isolation exception — see [architecture.md](architecture.md#design-rules-apply-across-every-module)
rule #2). Composes `face_identity`, `human_detection_roi`, all three gesture methods, and
`autocar_adapter` (see above) into one `step(frame, timestamp) -> FollowMeCommand` call.
`main.py --modules followme` is a thin CLI wrapper around this module (`configure()`/`step()`) —
the actual sequencing logic lives here, not duplicated in `main.py`. `main.py --modules
pretrigger` (the original `face_first` mode, renamed) still exists separately, stopping at
TRIGGER, for calibrating the pre-trigger stages in isolation.

**Purpose:** so a caller doesn't need to hand-wire seven modules together the way `main.py`'s
`pretrigger` mode does for the pre-trigger portion only — one call per frame, one command out,
covering trigger detection all the way through steering.

**Working principle:**
1. **Eager warmup, at `configure()` time** (before `step()` is ever called): every model this
   pipeline will use — `face_identity`, `human_detection_roi`, the chosen gesture method, and
   `autocar_adapter`'s YOLO-pose+OSNet (plus one throwaway inference through each, absorbing a
   backend's first-inference cost too) — is constructed right here, eagerly. `configure()` itself
   therefore takes a few seconds; the trade is deliberate (confirmed with the user) — that cost
   is paid once at startup, never live, and specifically never at the exact moment a gesture
   trigger fires (autocar_adapter's detector/embedder used to construct lazily inside `start()`,
   which is the worst possible moment for a multi-second stutter).
2. **Pre-trigger** (mirrors `main.py --modules pretrigger`'s own sequencing exactly, per the
   spec's own audit instruction to replicate rather than reinvent it): `face_identity.evaluate()` →
   filter to `is_registered_match` → `human_detection_roi.evaluate()` per matched face → crop →
   the chosen gesture method's `evaluate()`. The **first** registered person whose gesture
   reaches `GREEN` this frame becomes the locked target — only one follow-me episode can ever be
   active (`autocar_adapter` is itself a single-episode module-level singleton) — and
   `autocar_adapter.start(person_name, ...)` force-locks immediately (see above).
3. **Post-trigger**: `autocar_adapter.update()` every frame. While `TRACKING`, its
   `horizontal_offset` feeds `SteeringController.update()` (see below) to produce a real steering
   angle — unless the steering config is still uncalibrated, in which case `should_move` is
   forced `False` (fail-closed, consistent with every other module in this project) even though
   the target is genuinely still being tracked. If this frame is a mid-episode reclaim
   (`just_reacquired`), `SteeringController.reset()` clears stale PID state first.
4. **`SEARCHING`** reports `should_move=False` (the robot moves forward only while actively
   following). **`LOST`** (recovery timed out) stops the robot and **auto-resumes** watching for
   a brand-new trigger (confirmed with the user, over requiring an explicit external reset call)
   — the next `step()` call simply falls back into the pre-trigger sequence on its own. Unlike the
   old `target_tracking`/`target_recovery` split, a successful reclaim resumes `TRACKING` on the
   very same frame with a real `horizontal_offset` already available — no transitional
   angle-held-at-zero frame is needed anymore.

**Public contract** (`FollowMeCommand`): `should_move` (bool, no speed parameter — speed is a
separate downstream concern), `steering_angle_degrees` (signed, `None` when `should_move` is
`False`), `debug_state` (this module's own state-name choices — `WAITING_FOR_TRIGGER`,
`TRACKING_STARTED`, `TRACKING`, `TRACKING_STEERING_UNCALIBRATED`, `RECOVERING`, `STOPPED`).
`configure(gesture_method=...)` **must** be called before the first `step()` — unlike every other
module's `configure()`, there is no sensible default gesture method to lazily initialize with.
`draw_steering_arrow(frame, command)` draws the calculated steering direction as an arrow from
bottom-center of the frame (0° = ahead, +/- = right/left) — not gated by `--debug`, since it's the
actual robot command, not a per-module debug readout; no-ops while `should_move` is `False`.

**Key parameters:** none of its own beyond what it composes — see `camera.fov_degrees` and the
`steering` section below (owned by `SteeringController`).

**Known limitations:**
- Inherits every composed module's own known limitations wholesale (ByteTrack motion-only
  identity, `autocar_adapter`'s OSNet accuracy caveats, etc.) — not re-explained here.
- No speed/velocity control of any kind — `should_move` is a boolean only, by explicit project
  decision.
- No direct hardware/servo interface — this module outputs a `FollowMeCommand`; whatever
  consumes that and actually drives a servo is out of scope for this entire project as built.

---

## `SteeringController` (part of `followme_orchestrator`)

**Purpose:** converts `target_tracking`'s normalized `horizontal_offset` (-1.0..+1.0) into a real
steering angle and runs PID on it — the one place in this whole project where a normalized
deviation becomes an actual angle (`target_tracking` explicitly refuses to do this itself, see
that module's known limitations).

**Working principle:** `error_degrees = horizontal_offset * (fov_degrees / 2.0)`, then standard
PID (`kp * error + ki * integral + kd * derivative`), clamped to
`± max_steering_angle_degrees`. `dt` for the integral/derivative terms is computed from real
elapsed wall-clock time (`timestamp - last_timestamp`), never assumed from a fixed frame
interval. `reset()` clears accumulated integral/derivative state and is called by the
orchestrator at the start of every new episode (fresh trigger, or a recovery-driven resume) so
stale error history never bleeds across episodes.

**Deliberately its own class, not merged into the orchestrator or any CV module** — this is a
correctness requirement, not a style preference: CV pipeline latency varies frame-to-frame (a
slow face-match frame, a slow gesture-method frame), and if the PID's timing were driven by
whatever cadence a combined class happened to get called at, that latency variance would
silently corrupt the D-term and integral accumulation. Keeping `SteeringController` separate,
fed a real `timestamp` explicitly by the orchestrator each cycle, avoids this entirely.

**Key parameters:** `kp`, `ki`, `kd`, `max_steering_angle_degrees` (all required, fail-closed —
`is_calibrated()` gates every `update()` call) and `camera.fov_degrees` (a fixed physical lens
property, not an empirical tuning target, but still required before any angle conversion is
possible). See [parameters.md](parameters.md#steering-plans08) and
[parameters.md](parameters.md#camera).

**Known limitations:**
- Hand-implemented (no PID library dependency) — a standard PID loop is ~20 lines and this
  project generally prefers a small hand-rolled implementation over a new dependency where the
  algorithm itself is this simple (same reasoning as `face_identity`'s
  `cv2.estimateAffinePartial2D` choice over pulling in a third-party alignment package).
- No integral windup guard beyond the final output clamp — if this proves insufficient once
  tuned against real hardware, that's a natural follow-up, not built preemptively here.

---

## See also

- [`docs/architecture.md`](architecture.md) — how these modules are wired together, pipeline flow diagrams, cross-module design rules
- [`docs/technologies.md`](technologies.md) — the underlying models/libraries referenced throughout this file
- [`docs/parameters.md`](parameters.md) — calibration status (🔴/🟡/🟢) and tuning notes for every parameter named above
