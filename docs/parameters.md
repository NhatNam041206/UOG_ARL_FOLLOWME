# Parameter Reference

Every tunable value in `config/thresholds.yaml`, what it means, what it affects, and its current
status. This is a reference for whoever runs the empirical calibration pass — it does not itself
calibrate anything. For what each parameter's module actually *does* (the algorithm the
parameter is tuning), see [`docs/modules.md`](modules.md); for how modules wire together, see
[`docs/architecture.md`](architecture.md).

## Status legend

| Status | Meaning |
|---|---|
| 🔴 **uncalibrated** (`null`) | Module fails closed — see each module's own "fail-closed" note below. Never treat a `null` as "use some sane default"; it's a deliberate refusal to guess. |
| 🟡 **starting guess** | Has a real number so the module can be exercised end-to-end, but that number came from a spec's suggested range, a reference implementation's own default, or a rough estimate — not measured against this project's actual hardware/footage. Do not trust it past a first smoke test. |
| 🟢 **working default** | Not a calibration target — a file path, model handle, or similar. Fine as-is; override only if you have a reason to. |

Every module that has calibration-gated parameters (marked 🔴/🟡 below) fails closed: while any
required key is `null`, that module produces no positive signal at all (`GO`/`is_waving`/
`is_registered_match`/`person_found`/etc. all stay `False`/negative), rather than silently
guessing. This is deliberate and consistent across every module in this repo.

---

## `emergency_stop`

Sole collision-avoidance layer (no non-CV sensor backstop exists yet) — see
`modules/emergency_stop/interface.py`'s module docstring. All 10 keys below are required; while
any is `null`, `EmergencyStopModule` outputs `UNCERTAIN` (STOP-equivalent) on every frame.

| Parameter | 🔴/🟡/🟢 | Meaning | Tuning notes |
|---|---|---|---|
| `runway_left_line` | 🔴 | `[[x_top_frac, y_top_frac], [x_bottom_frac, y_bottom_frac]]` — left edge of the trapezoid ahead of the robot, normalized 0–1 frame fractions. | Walk the robot's camera view and mark where the left edge of its actual driving path falls near the top and bottom of frame. |
| `runway_right_line` | 🔴 | Same shape as above, right edge. | Same method, right edge. |
| `roi_buffer_px` | 🔴 | Pixels added to the trapezoid's bounding rect when cropping the detector's input ROI. Does **not** loosen the runway-membership test itself — only avoids clipping bboxes at the crop edge. | Start small (10–30px); increase only if you see bboxes truncated at the ROI edge in debug overlays. |
| `size_prefilter_width_px` / `size_prefilter_height_px` | 🔴 | Bbox size floor below which a detection is skipped outright (fast-path safe-object filter) — NOT a substitute for zone logic; a large object can still be safely in the far zone. | Set below the smallest real obstacle's on-screen size at the *farthest* relevant distance, so nothing real gets pre-filtered away. |
| `zone_far_boundary` / `zone_mid_boundary` | 🔴 | Y-fraction dividers (same normalized space as the runway lines) splitting the runway into far/mid/near zones. Near zone runs to the runway's own bottom edge — no third divider. | Set from real measured stopping distances at your robot's speed — mid-zone dwell and near-zone-immediate-stop should map to actual physical braking distance. |
| `t_mid_seconds` | 🔴 | How long an object must continuously dwell in the mid zone before it triggers STOP (near-zone objects trigger immediately, no dwell). | Balance false-stop annoyance (too short) against reaction time lost (too long). |
| `min_detection_confidence` | 🔴 | YOLO detection confidence floor. A detection below this escalates the **whole frame** to `UNCERTAIN` rather than being silently dropped (a dropped low-confidence detection could be a real obstacle). | Set from your detector's real precision/recall curve on representative footage, not a generic YOLO default — this gates a safety-critical path. |
| `resume_buffer_seconds` | 🔴 | Seconds of uninterrupted "runway clear" required before `STOP`→`GO` (any interruption resets the timer to zero, no partial credit). Spec suggests ~2–3s as a starting point. | Longer = safer but more sluggish resume; shorter = faster resume but more sensitive to detection flicker. |
| `yolo_model_path` | 🟢 | This module's own standalone YOLO weights file. Deliberately never shared with any other module's detector instance (safety isolation). | Default `models/yolo11n.onnx`; override only to use a different model. |

**Not yet a config key at all** (deliberately): a latency budget. Spec requires frame-time
(capture+inference+decision) to be benchmarked on target hardware before this can be set
responsibly, since stopping distance depends on reaction time. `EmergencyStopModule.last_latency_ms`
is already measured every frame — use it to gather that benchmark first.

---

## `human_detection`

Standalone detector, not calibration-gated — no spec mandates it, and this is a generic detector,
not a safety layer. Its only historical consumers (`wave_facing_gate`, `target_tracking`,
`target_recovery`) have all been removed — this module has no live caller left at all currently;
kept as a runnable standalone tool.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `confidence_threshold` | 0.5 | 🟡 working value | YOLO detection confidence floor, person class only. | Standard YOLO default; raise to cut false detections in clutter, lower if distant/partial people are missed. |
| `yolo_model_path` | 🟢 | Same `models/yolo11n.onnx` file as every other YOLO-based module, but always its own fresh instance. | Override only for a different model. |

---

## `face_identity` (plans/01) — first stage of the face-first pipeline

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `similarity_threshold_face_match` | 0.4 | 🟡 | EdgeFace cosine similarity floor (range roughly [-1, 1], but real same-person scores cluster much higher) — the best-scoring registry entry must clear this to count as a match. Value is EdgeFace's own reference demo's default, not measured against this project's own registered faces. | **Watch the console's `match_confidence` output** (both `visualize_face_identity.py` and the pipeline print it) across several genuine-match and different-person frames, then set the threshold between those two score clusters. |
| `face_detection_confidence_threshold` | 0.6 | 🟡 | YuNet per-face detection score floor — faces below this are dropped before matching is even attempted. | Standard YuNet starting point. Lower if faces at distance/angle aren't detected; raise if spurious low-confidence "faces" appear. |
| `yunet_model_path` / `edgeface_model_path` / `registry_dir` / `raw_captures_dir` | 🟢 | Model file locations and where the registry/raw capture images live. | Override only if you moved the model files or want per-environment registry locations. |

Fail-closed: while either threshold is `null`, `evaluate()` returns an empty list every frame
(reports zero faces, not "faces found but unmatched").

---

## `human_detection_roi` (plans/02) — second stage

Flow: **Face → ROI (derived from face bbox size) → Human detection inside that ROI**, to tighten
the crop fed to the detector. Human detection is only ever triggered by an already-matched face —
it never runs on its own. No persistent `track_id`: the ROI crop shifts every frame (it follows
wherever the face currently is), which isn't a stable coordinate frame for tracker-style
persistence — confirmed with the user, see `modules/human_detection_roi/detector.py`'s docstring.
Fails closed to `person_found=False` on missing keys or nothing found in the ROI (no automatic
full-frame fallback, by design).

**ROI geometry** (`modules/human_detection_roi/roi.py`) — one function, four inputs:

```
roi_height  = face_h * roi_expansion_factor
up          = roi_height * roi_upward_fraction     # extends above the face
down        = roi_height - up                       # extends below the face
roi_width   = face_w * roi_expansion_factor * roi_width_fraction
```

The ROI is centered horizontally on the face, extends `up` above the face's top edge and `down`
below its bottom edge, clipped to the frame.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `roi_expansion_factor` | 5.0 | 🟡 | The single overall size knob — total ROI height budget as a multiple of face bbox height. Spec suggests starting ~4–6x. | **Too small**: body gets clipped, especially for close-up faces where a small multiple still isn't many pixels. **Too large**: risks catching a different person's body in a crowd — defeats the point of ROI-scoping. Watch the yellow ROI box via `visualize_human_detection_roi.py` and check it comfortably contains the whole body without engulfing neighbors. |
| `roi_upward_fraction` | 0.15 (default, not in yaml unless uncommented) | 🟡 | Share of `roi_height` allocated **above** the face's top edge; the remaining `1 - roi_upward_fraction` goes below. | Raise if hats/raised hands above the head get clipped by the ROI top edge. Keep low — the body is below the face, not above it; too high wastes ROI budget on background. |
| `roi_width_fraction` | 0.6 (default, not in yaml unless uncommented) | 🟡 | ROI width as a fraction of `face_w * roi_expansion_factor`. | Raise if arms/shoulders get clipped at the ROI's left/right edge (e.g. someone waving with an arm extended sideways); lower to reduce the chance of catching a person standing close beside the registered person. |
| `detection_confidence_threshold` | 0.5 | 🟡 | YOLO person-class confidence floor, applied *within* the ROI crop. | A small, tight ROI crop may need a different confidence profile than full-frame detection — tune against real ROI crops, not generic YOLO defaults. |
| `yolo_model_path` | 🟢 | Own fresh YOLO instance (no tracking — see above), same weights file as every other module. | Override only for a different model. |

**How to tune the ROI ratios specifically:** run
`python -m modules.human_detection_roi.visualize_human_detection_roi --mode camera`, stand at a
few different distances from the camera, and watch the yellow box. If it's clipping the body
(top of head, feet, or arms sticking out the sides), that tells you exactly which knob to adjust
— `roi_upward_fraction` for head/hat clipping, `roi_expansion_factor` for feet/overall size,
`roi_width_fraction` for arms/shoulders. Then check a second person standing nearby doesn't fall
inside the box (over-generous settings will catch bystanders — this is what `roi_expansion_factor`
being too large mainly risks).

---

## `gesture_hand_keypoint` (plans/03) — the sole TRIGGER gesture method

Two alternatives — `wave_facing` (`modules.wave_facing_gate`) and `gesture_trajectory_verifier`
— used to exist; both were removed (confirmed with the user — this is the only TRIGGER gesture
method left).

**REDESIGNED — no longer motion-based.** Pure hand-shape sequence classification, MediaPipe
landmark geometry only (no wrist motion, no trajectory, no arm geometry of any kind). Valid
gesture = **OPEN → CLOSED → OPEN → CLOSED**, each transition within `max_transition_gap_seconds`
of the last transition, starting from OPEN (a sequence starting at CLOSED doesn't count).

```
WAITING_OPEN -> (open) -> WAITING_CLOSE_1 -> (closed) -> WAITING_OPEN_2 -> (open) ->
WAITING_CLOSE_2 -> (closed) -> CONFIRMED
```

Every OPEN/CLOSED read counted toward the sequence must ALSO independently clear the
`palm_height_fraction` gate (palm/wrist in the upper fraction of the person's FULL-FRAME bbox
from `modules.human_detection_roi` — not just the hand-crop). **Failing that gate is an
immediate reset to WAITING_OPEN**, confirmed with the user — stricter than a merely
non-advancing frame. CONFIRMED then feeds the same shared RED/YELLOW/GREEN confirmation tracker
used elsewhere in this project. All 7 keys hold 🟡 starting-guess values — not calibrated. Hand
detection/keypoints are visualizable via `draw_debug()` regardless of calibration state; only
the gated `is_waving` verdict is fail-closed while any key is `null`.

**Only ONE hand drives the sequence per frame.** Of all detected hands clearing
`confidence_threshold`, only those that ALSO clear `palm_height_fraction` are eligible; among
those, the highest-confidence one is picked. **Left/Right side is deliberately ignored** — a
single shared sequence machine per track, not one per hand side, so switching which physical
hand you raise mid-gesture doesn't reset progress, and a MediaPipe left/right mislabel can't
split one gesture into two stalled sequences. If a confident hand exists but none clears the
height gate, that's a gate-failure reset (same as before); if no hand is confident enough at
all, the frame is skipped (no reading, no reset).

**Hand-shape classification, thumb handling (calibrated against reference open-palm/fist
photos):** a natural fist rests the thumb OVER the curled fingers, not tucked into the palm — so
the thumb's distance-based extension test does NOT reliably read "curled" for a real fist.
`min_fingers_extended_open`/`min_fingers_curled_closed` are therefore scored out of the **4
non-thumb fingers only**. OPEN additionally requires the thumb to independently pass
`thumb_extension_ratio_threshold` (a real open palm does spread the thumb away from the hand).
CLOSED does **not** check the thumb at all.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `confidence_threshold` | 0.5 | 🟡 | MediaPipe handedness/detection confidence floor. Below this, a hand is excluded from being the sequence-driving candidate this frame — doesn't advance, doesn't reset. | MediaPipe's own confidence semantics — not comparable to any other module's confidence values. |
| `min_fingers_extended_open` | 4 | 🟡 | Of the 4 NON-THUMB fingers, how many must be "extended" to count toward OPEN — the thumb is checked separately (must also pass `thumb_extension_ratio_threshold`). Integer 0–4. | Lower (e.g. 3) is more permissive if a fully-open reading proves too strict in practice. |
| `min_fingers_curled_closed` | 4 | 🟡 | Of the 4 NON-THUMB fingers, how many must be "curled" to classify as CLOSED (fist). Integer 0–4. Thumb state is ignored entirely for CLOSED — see above. | Same tuning logic as above, mirrored for the fist side. |
| `thumb_extension_ratio_threshold` | 0.6 | 🟡 **⚠️ unverified** | Thumb-specific test, only gates OPEN (not CLOSED): `distance(thumb_tip, pinky_MCP) / distance(wrist, middle_MCP)` must exceed this ratio to count as extended. | Watch the thumb's green/red coloring in `draw_debug()`'s output (drawn as a line from thumb tip to wrist) across real open-hand gestures to find the right cutoff — no need to tune it against fist footage since CLOSED ignores the thumb. |
| `palm_height_fraction` | 0.5 | 🟡 | Palm (wrist landmark) must be in the upper fraction of the person's FULL-FRAME bbox height (from `modules.human_detection_roi`), checked on EVERY frame counted toward the sequence. Also acts as the selector for which hand drives the sequence when multiple hands are visible. | Lower (e.g. 0.3) demands the hand be raised higher; higher (e.g. 0.7) is more permissive. Remember: failing this is an immediate full sequence reset, not just a skipped frame — too strict a value will make the gesture hard to complete even with genuine intent. |
| `max_transition_gap_seconds` | 1.5 | 🟡 | Timeout between consecutive sequence transitions — exceeding it resets to WAITING_OPEN with no partial credit. Spec suggests starting ~1–2s. | Shorter demands a snappier open-close-open-close; longer tolerates a slower, more deliberate gesture. No timeout pressure while still sitting in WAITING_OPEN (nothing to time from yet). |
| `confirmation_duration_seconds` | 1.0 | 🟡 | RED→YELLOW→GREEN debounce duration, applied to the CONFIRMED completion event (see `pipeline.py`'s comments for exactly how a one-shot event is reconciled with the continuous-condition-style tracker — a judgment call, not literally specified). | — |
| `model_path` | 🟢 | `hand_landmarker.task` bundle location. | Override only if you moved the model file. |

**How to tune the classification specifically:** run
`python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode camera`, and
watch two things together — the on-screen `stage=` label (which of the 5 sequence stages you're
in) and the per-finger green/red coloring on the hand skeleton (which fingers currently read as
extended). If OPEN never triggers despite an obviously open hand, one or more fingers (check the
thumb specifically) is reading red when it should be green — adjust that finger's threshold. If
CLOSED never triggers on a real fist, same diagnosis in the other direction.

**Known risk to test for specifically** (not a parameter, but relevant to calibration): MediaPipe's
palm detector has shown degraded accuracy in low-light/low-resolution benchmarks in published
literature. Test this method's calibration specifically under the campus's actual lighting
conditions, not just good-lighting footage.

---

## `appearance_verifier` (plans/05)

OSNet Re-ID embedding + cosine-similarity matching. Was a shared dependency of the now-removed
`target_tracking`/`target_recovery` (each defined its OWN separate threshold key rather than this
one). No live caller currently — kept as a runnable standalone tool; `autocar` uses its own
independent OSNet embedder instead (see that section below), not this module.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `similarity_threshold` | `null` | 🔴 **⚠️ especially uncalibrated** | Cosine similarity floor `best_similarity_score` must clear for `match_found=True`. | Two documented accuracy risks make a casual "starting guess" here less trustworthy than usual — see below. Test explicitly against BOTH: (1) two DIFFERENT people in similar-colored/styled clothing (should score low — if it doesn't, that's the clothing-confusion risk manifesting), and (2) the SAME person under noticeably different lighting/distance (should score high — if it doesn't, that's the cross-domain risk manifesting). Don't set this from positive examples alone. |
| `osnet_model_name` | `osnet_x1_0` | 🟢 | Which `torchreid` OSNet variant to build. | Override only if evaluating a different OSNet size/variant. |

**Two named risks — read before calibrating, do not treat as one vague "may be inaccurate" note:**
1. **Similar-clothing confusion.** OSNet-based appearance matching struggles to distinguish
   people wearing similar-colored/styled clothing, since appearance embeddings lean heavily on
   clothing as a feature.
2. **Cross-domain generalization drop.** Published OSNet benchmarks show accuracy can drop
   sharply on footage meaningfully different from its training distribution (Market-1501-family
   datasets) — this project's own campus footage/lighting/camera are an untested domain.

**How to tune:** run
`python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir <folder> --mode camera`,
watch `best_similarity_score` printed per frame against a known reference set, and specifically
run the two test scenarios above before trusting any threshold value.

---

## `target_tracking` / `target_recovery` (plans/06, plans/07) — REMOVED 2026-08-26

`modules/target_tracking` and `modules/target_recovery` (and their `test_*.py`/`visualize_*.py`
tools, and their `config/thresholds.yaml` sections) were deleted — fully superseded by `autocar`
below once its replacement was confirmed working. Full parameter tables/tuning notes for both are
in git history (`git log -- docs/parameters.md`) if ever needed for reference.

**One capability has NO replacement in `autocar` — worth knowing before assuming parity:**
`target_tracking` ran a periodic appearance re-verify during active TRACKING
(`appearance_reverify_interval_seconds`/`appearance_reverify_similarity_threshold`) specifically
to catch ByteTrack silently reassigning the locked `track_id` to a different nearby person
**without ever reporting a loss**. Confirmed by reading
`modules/autocar/identity/target_lock.py`'s `TargetLock._update_locked()` directly: while the
locked `track_id` stays present in the tracker's own output, it is trusted with **zero
re-verification, by design** ("no model calls REQUIRED", their own comment) — `autocar` only
re-checks identity when the `track_id` itself disappears and a new one takes its place. This
protection no longer exists anywhere in the live pipeline. If it matters for your deployment,
it would need to be added to `autocar_adapter.py`, not restored from the deleted module (its
design assumed a locally-owned tracker instance `autocar_adapter` doesn't have the same shape of).

---

## `autocar` (vendored tracking + recovery backbone, replaces `target_tracking`/`target_recovery` above)

Vendored from vinhh9608-byte/Autocar, currently commit `8037862` (re-vendored from `27ee33a`).
Read only by `modules/followme_orchestrator/autocar_adapter.py` — never by
`modules/autocar/config.py` itself (their vendored file, never edited; these values are passed
into their classes' own constructor override parameters instead). Detector/tracker/re-id values
below are carried over AS-IS from their own `config.py` — their own considered starting points,
not blind guesses, so treated as 🟡 starting guesses rather than 🔴 uncalibrated.

**What changed in `8037862`** (worth knowing before touching the re-id keys below): FRONT-face
matching no longer uses OSNet on a keypoint-guessed head rectangle — it now runs a REAL face
detector (YuNet) + face-recognition embedding (SFace) on that same head-region crop
(`identity/face_recognizer.py`), replacing the old `reid_similarity_threshold` key with
`face_similarity_threshold` below. OSNet is now used ONLY for the BACK-of-head case (no face
detected = facing away), under `back_head_similarity_threshold`. Their old
`reid_face_min_keypoint_conf` key is gone entirely — "is a face visible" is now answered by
running the real detector, not by guessing from keypoint confidence.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `detect_conf` | 0.4 | 🟡 | YOLOv8-pose detection confidence floor. Low on purpose — their ByteTrack itself filters by score in two tiers. | Carried from their `DETECT_CONF`; recalibrate only if their own tracker's two-tier filtering proves wrong for this project's footage. |
| `detect_imgsz` | 320 | 🟢 | YOLOv8-pose inference resolution. Set to 320 directly (a multiple of the max stride 32) — was 300, which ultralytics silently auto-rounded up to 320 every run anyway; this just makes the config say what actually happens. | Any other multiple of 32 works too (e.g. 288, 352) if you want to trade accuracy for speed. |
| `track_high_thresh` / `track_low_thresh` / `new_track_thresh` / `match_thresh` / `low_match_thresh` / `track_buffer` | 0.6 / 0.1 / 0.7 / 0.8 / 0.5 / 30 | 🟡 each | Their ByteTrack's own tuning knobs — high/low score tiers, IoU match thresholds per stage, how many frames a lost track survives before being dropped. | Carried from their `config.py` unmodified; see `modules/autocar/tracker/byte_tracker.py`'s own docstring for what each does. |
| `face_similarity_threshold` | 0.363 | 🟡 | SFace cosine similarity cutoff for `TargetLock`'s FRONT-face match (`identity/face_recognizer.py` — real face detection + recognition, not OSNet, as of their commit 8037862). This is opencv_zoo's own documented "same person" cutoff for this exact model, a real published starting point — still worth recalibrating against this project's footage. | Trust more than a typical "starting guess" (it's a published reference value for this exact model), but still verify on real footage before relying on it. |
| `back_head_similarity_threshold` | 0.7 | 🟡, **especially uncalibrated** | Cosine similarity cutoff for `TargetLock`'s BACK-of-head match — OSNet, used ONLY for this case now (no face visible = facing away). Same OSNet accuracy caveats as `appearance_verifier.similarity_threshold` above — similar-clothing confusion, cross-domain generalization drop (this project's footage vs. MSMT17 training data). | Do not trust past a first smoke test; calibrate against this project's own footage before relying on recovery accuracy. |
| `reid_head_split_fallback_fraction` | 0.35 | 🟡 | Head-region height as a fraction of bbox height, used only when shoulder keypoints aren't confident enough to place the split line precisely. **Informational only — NOT actually overridable from this file.** `identity/face_region.py` reads this (and `reid_head_crop_width_fraction`) directly off the vendored `config.py` module's own globals, not through `TargetLock`'s constructor, so there's no seam for the adapter to pass an override through. Changing this key in `thresholds.yaml` currently has no effect — a known gap, not a bug, until/unless that seam is added. | Carried from their `REID_HEAD_SPLIT_FALLBACK_FRACTION`; edit their vendored `config.py` directly if this genuinely needs to change (breaks the "vendored tree stays untouched" rule — weigh that first). |
| `reid_acquire_rounds` / `reid_acquire_cooldown_sec` | 3 / 0.5 | 🟡 each | How many face-only sampling rounds (and the wall-clock gap between them) before `ACQUIRING` picks a target. Only exercised via the adapter's IoU-force-lock fallback path — normal triggers skip `ACQUIRING` entirely (see `docs/modules.md`). | Low-priority to calibrate given how rarely this path runs in practice. |
| `recovery_timeout_seconds` | `null` | 🔴 (**ours, not theirs**) | Their own reclaim search retries indefinitely with no timeout at all — this closes that gap, mirroring `target_recovery.search_timeout_seconds`'s exact convention. While `null`, a lost target is searched for forever, never reporting back to `followme_orchestrator`. | Spec-equivalent starting range to the old `target_recovery.search_timeout_seconds` (~1–2 minutes) is a reasonable starting point. |

**Known gap, no config key exists for this at all:** unlike the old (now-deleted) `target_tracking`
module, `TargetLock` never re-verifies identity while a locked `track_id` stays present in the
tracker's own output — see the `target_tracking`/`target_recovery` section above for the full
detail. Not something you can currently tune away by editing this file.
| `device` | `"cpu"` | 🟢 | `"cpu"` or `"cuda:0"`, passed straight through to their detector/embedder. | Override only if running on a machine with a usable CUDA GPU. |

**How to tune:** `cd modules/autocar && python main.py --source 0 --target
models/enrolled_<name>.npz` exercises their tracking+recovery engine directly (their own
`ACQUIRING`, not the adapter's force-lock) — good for isolating whether a bad result is the
engine/profile or the adapter's own logic. For the full composed pipeline, use
`modules.followme_orchestrator.visualize_followme_orchestrator`.

---

## `register_person` (`scripts/register_person.py` — not a per-frame pipeline module)

The ROI a detection must fall inside during capture to count as a sample — see
`registration_overlay.crop_to_roi()`. Exists so that if someone else walks through the background
(or stands next to the subject) during registration, their face/body doesn't get captured into
the subject's profile by mistake — the operator points the camera so the intended subject stands
inside this box; everyone else outside it is ignored. Two separate keys because the FRONT and BACK
phases use different detectors (face vs. person) and the subject may naturally stand slightly
differently in each. Not calibration-gated (no fail-closed behavior): both keys have reasonable
starting-box defaults so the tool works out of the box, adjust to match your actual camera
framing.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `front_roi_percent` | `[0.20, 0.05, 0.80, 0.95]` | 🟡 | `[x1,y1,x2,y2]`, frame-fraction box checked against the detected FACE bbox during FRONT capture (in the earlier ROI-gated design) / used to crop the saved RAW frame (current design — see `docs/architecture.md`'s Registration UI section). | Widen/narrow to match how far back the subject stands from the camera during registration. |
| `back_roi_percent` | `[0.15, 0.0, 0.85, 1.0]` | 🟡 | Same, for BACK capture — wider by default since a turned-around body silhouette varies more than a face. | Same tuning approach as `front_roi_percent`. |

**How to tune:** run `python -m scripts.register_person` (Tkinter UI) or `python -m
scripts.register_person <name>` (headless) and watch the drawn ROI box live — the Tkinter flow's post-crop pause
(`CaptureWindow`'s OK/Cancel dialog) is the actual point to check whether the box was sized
correctly, by opening `registration_captures/<name>/cropped/` and looking at the results.

---

## `camera`

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `camera_index` | 2 | 🟢 | OS camera device index used by `main.py --mode camera` when `--camera-index` isn't passed on the command line. Not a calibration value — just which physical camera to open. | Override via `--camera-index` or this key. |
| `fov_degrees` | 85.0 | 🟢 | Horizontal field of view, in degrees — this project's camera hardware datasheet value. A fixed PHYSICAL lens property, not empirically tuned like most values in this file, but it gates `followme_orchestrator.SteeringController` (plans/08): without it, `horizontal_offset` can never become a real steering angle at all, and `should_move` stays forced `False` while actively tracking. | Set from the actual camera hardware's own datasheet, never guessed or tuned like a threshold. |
| `lens_type` | `"hybrid"` | 🟢 | Informational only — documents the camera's physical lens type. Never consumed by any calculation. | Update if the hardware changes; no downstream effect either way. |
| `focus_type` | `"fixed"` | 🟢 | Informational only, same rationale as `lens_type`. Recorded specifically because a FIXED-focus camera is what makes treating `fov_degrees` as one constant safe — a variable-focus/zoom lens would need FOV to become a function of zoom level instead. | Not this project's hardware situation currently; flag prominently if it ever changes. |

---

## `steering` (plans/08 — `followme_orchestrator.SteeringController`)

PID conversion of `autocar_adapter`'s normalized `horizontal_offset` into a real steering angle
(via `camera.fov_degrees` above) for the Ackermann servo. 4 of the 5 keys below are required —
while any of `kp`/`ki`/`kd`/`max_steering_angle_degrees` is `null`,
`SteeringController.is_calibrated()` is `False` and the orchestrator forces `should_move=False`
even while a target is genuinely still being tracked (same fail-closed convention as every other
module in this project). `SteeringController.update()` returns the ABSOLUTE servo angle, ready to
write directly to the servo — `servo_center_degrees ± (clamped PID output)` — not a signed
error-around-0 value; nothing downstream needs to add an offset.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `kp` | 1.0 | 🟡 | Proportional gain — output contribution directly proportional to the current error (degrees off-center). | Starting value supplied by hardware owner, not yet validated against real Ackermann hardware — treat as a starting point, not a finished calibration. |
| `ki` | 0.5 | 🟡 | Integral gain — accumulates error over time, corrects small persistent offsets a proportional-only controller would leave uncorrected. | Same caveat as `kp` — watch for slow oscillation ("integral windup") once tested live. |
| `kd` | 0.5 | 🟡 | Derivative gain — dampens oscillation by reacting to the RATE of change of error, not just its current value. | Same caveat as `kp`/`ki`. |
| `max_steering_angle_degrees` | 45.0 | 🟢 | Hard clamp on the PID's error output BEFORE it's added to `servo_center_degrees` — an Ackermann/servo hardware limit, NOT a tuning target. With `servo_center_degrees=90`, this yields a final output range of `[45, 135]`. | Servo/Ackermann mechanism's own datasheet limit — confirmed with the user, not guessed. |
| `servo_center_degrees` | 90.0 | 🟢 | The servo's own straight-ahead pulse position — added to the PID's clamped error to produce the final absolute servo angle `SteeringController.update()` returns. Not calibration-gated (has this working default); override only if your servo's physical center differs from 90. | Confirmed with the user as this project's servo hardware convention. |

**How to tune:** run
`python -m modules.followme_orchestrator.visualize_followme_orchestrator --mode camera`,
trigger a follow episode, and watch the live `steering=` readout while adjusting `kp`/`ki`/`kd`
in `config/thresholds.yaml` between runs — classic PID tuning order (`kp` first, then `ki`, then
`kd`), against the real Ackermann hardware once available, not simulated/guessed values. The
current `kp`/`ki`/`kd` are the hardware owner's starting values, not yet live-tuned — expect to
revisit them once the actual servo/Ackermann rig is in hand.

---

## `mqtt_bridge` (`modules/mqtt_bridge`, `--mqtt` only)

Publishes each frame's `FollowMeCommand` over MQTT to a Pi 4 motor controller — see
`docs/modules.md#mqtt_bridge` and `docs/mqtt_handoff_pi4.md` for the wire contract. `broker_host`
and `publish_hz` are both required; while either is `null`, `publish()` always returns `False`
without attempting a broker connection (same fail-closed convention as every other module).
`servo_center_degrees` is NOT duplicated here — it's read from the `steering` section above.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `broker_host` | `null` | 🔴 | Pi 4's IP address on the shared network. | Set once the Pi 4's address on the shared network is known — not guessable/defaultable, since a wrong IP would silently fail to reach the right device. |
| `broker_port` | 1883 | 🟢 | Standard unencrypted MQTT broker port. | Override only if the broker is configured on a non-standard port. |
| `topic` | `"autobot/control/followme"` | 🟢 | MQTT topic both sides publish/subscribe to. | Override only if the Pi 4 subscriber is coded to expect a different topic string — keep both sides in sync (see `docs/mqtt_handoff_pi4.md`). |
| `publish_hz` | `null` | 🔴 | How often `publish()` actually sends a message (internally rate-limited in `interface.py`, not the transport layer). | Starting range discussed: 5–15 Hz. Must be empirically tuned against real servo response + network latency on actual hardware — do not assume a value. |

---

## How to actually calibrate

1. **Run the relevant `visualize_*.py` / `test_*.py` tool** for the module you're tuning (see each
   module's own script — `python -m modules.<name>.visualize_<name> --mode camera`).
2. **Watch the printed debug fields**, not just the pass/fail booleans — `match_confidence`,
   `confidence_debug`, `facing_confidence_min`, etc. are all exposed specifically so you can see
   the raw number before deciding a threshold, not just whether it currently passes.
3. **Collect both positive and negative examples** — a genuine wave AND a still/reaching arm; a
   registered face AND a stranger's face; good lighting AND the actual deployment lighting. A
   threshold set from positive examples alone will over-trigger.
4. **Update `config/thresholds.yaml`**, not the code — every threshold in this repo is designed
   to be tuned from that one file, never hardcoded.
5. **Re-run the same visualization tool** to confirm the new value behaves as expected before
   moving to the next parameter.
