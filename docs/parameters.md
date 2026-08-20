# Parameter Reference

Every tunable value in `config/thresholds.yaml`, what it means, what it affects, and its current
status. This is a reference for whoever runs the empirical calibration pass — it does not itself
calibrate anything.

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
| `yolo_model_path` | 🟢 | This module's own standalone YOLO weights file. Deliberately never shared with any other module's detector instance (safety isolation). | Default `yolo11n.onnx` at repo root; override only to use a different model. |

**Not yet a config key at all** (deliberately): a latency budget. Spec requires frame-time
(capture+inference+decision) to be benchmarked on target hardware before this can be set
responsibly, since stopping distance depends on reaction time. `EmergencyStopModule.last_latency_ms`
is already measured every frame — use it to gather that benchmark first.

---

## `wave_facing` (Method 1 — `modules.wave_facing_gate`)

MoveNet Lightning wrist/elbow/shoulder geometry + motion. Currently holds 🟡 starting-guess
values (spec's own suggested ranges) so it can be exercised end-to-end — **not calibrated**. If
any key is set back to `null`, `is_waving`/`is_facing_camera` both stay `False`.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `confidence_threshold_facing` | 0.3 | 🟡 | Min confidence on ALL of left_eye/right_eye/left_shoulder/right_shoulder before `is_facing_camera`'s raw check can pass. | Spec suggests 0.3–0.5. Raise if false "facing" triggers from partial occlusion; lower if a genuinely-facing person isn't detected. |
| `confidence_threshold_pose` | 0.3 | 🟡 | Min confidence on wrist/elbow/shoulder before Gate A (static pose) evaluates at all — fails closed if below. | Same MoveNet confidence semantics as above; tune together. |
| `wrist_height_fraction` | 0.5 | 🟡 | Wrist must be above (numerically less than) this fraction of bbox height to count as "raised." 0.5 = upper half. | Lower (e.g. 0.35) demands a higher raise; raise (e.g. 0.6) is more permissive. |
| `verticality_threshold_deg` | 25 | 🟡 | Max degrees from vertical for BOTH the wrist→elbow and wrist→shoulder vectors — rejects a horizontally-extended arm. Spec range ~20–30°. | Tighten to reject more "reaching" false positives; loosen if genuine waves at an angle get rejected. |
| `motion_window_seconds` | 1.2 | 🟡 | Rolling time window for the wrist-motion buffer (Gate B). Spec range ~1.0–1.5s. | Should roughly span one full wave cycle; too short truncates the gesture, too long dilutes it with stale motion. |
| `motion_confidence_threshold` | 0.3 | 🟡 | Confidence floor for accumulating a wrist sample into the motion buffer — independent of `confidence_threshold_pose`, deliberately (Gate B accumulates regardless of Gate A's pass/fail). | Can diverge from `confidence_threshold_pose` if motion tracking needs a different reliability bar than the posture check. |
| `motion_min_samples` | 5 | 🟡 | Minimum buffered samples before Gate B evaluates at all. | Depends on effective frame rate; needs enough samples to see multiple direction changes within `motion_window_seconds`. |
| `motion_min_direction_changes` | 2 | 🟡 | Minimum direction reversals (angle ≥ `motion_direction_change_angle_deg` between consecutive displacement vectors) required to count as oscillating motion. | 2 is the minimum for "back and forth"; raise for stricter multi-cycle requirement. |
| `motion_direction_change_angle_deg` | 90 | 🟡 | **Sensitivity knob #1.** How sharp a turn between consecutive wrist displacement vectors counts as a "direction change." | Lower = more sensitive to subtle direction shifts (more false positives from jitter); higher = only counts sharp reversals. |
| `motion_min_displacement_px` | 8 | 🟡 **⚠️ unverified noise floor** | **Sensitivity knob #2, and the primary defense against false positives.** Minimum frame-to-frame wrist movement (pixels, in bbox coordinate space) before it's even considered for a direction-change comparison. Below this, movement is treated as MoveNet's own inference jitter, not real motion. | **Must be measured** from real held-still-raised-arm footage on target hardware *before* trusting this value — see the module spec §4.4/§11. Too low and a person holding a still raised arm (stretching, reaching) can register spurious direction changes from model jitter alone and false-trigger as waving. |
| `confirmation_duration_seconds` | 1.0 | 🟡 | Seconds a raw per-frame condition must hold continuously (YELLOW) before promoting to GREEN/confirmed. Spec range ~1.0–2.0s. Applies independently to both `is_waving` and `is_facing_camera`. | Shorter = more responsive but more single-frame-flicker-prone; longer = steadier but feels less immediate. |
| `movenet_tfhub_handle` | (TF Hub URL) | 🟢 | Where to load MoveNet Lightning from — auto-downloads and caches. | Override only to point at a local SavedModel. |

---

## `human_detection`

Feeds per-person bboxes to `wave_facing` in the original (non-face-first) demo pipeline. Not
calibration-gated — no spec mandates it, and this is a generic detector, not a safety layer.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `confidence_threshold` | 0.5 | 🟡 working value | YOLO detection confidence floor, person class only. | Standard YOLO default; raise to cut false detections in clutter, lower if distant/partial people are missed. |
| `yolo_model_path` | 🟢 | Same `yolo11n.onnx` file as every other YOLO-based module, but always its own fresh instance. | Override only for a different model. |

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

## `gesture_hand_keypoint` (plans/03, Method 2)

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
| `confidence_threshold` | 0.5 | 🟡 | MediaPipe handedness/detection confidence floor. Below this, a hand is excluded from being the sequence-driving candidate this frame — doesn't advance, doesn't reset. | Independent from Method 1's confidence values — MediaPipe's confidence semantics don't transfer. |
| `min_fingers_extended_open` | 4 | 🟡 | Of the 4 NON-THUMB fingers, how many must be "extended" to count toward OPEN — the thumb is checked separately (must also pass `thumb_extension_ratio_threshold`). Integer 0–4. | Lower (e.g. 3) is more permissive if a fully-open reading proves too strict in practice. |
| `min_fingers_curled_closed` | 4 | 🟡 | Of the 4 NON-THUMB fingers, how many must be "curled" to classify as CLOSED (fist). Integer 0–4. Thumb state is ignored entirely for CLOSED — see above. | Same tuning logic as above, mirrored for the fist side. |
| `thumb_extension_ratio_threshold` | 0.6 | 🟡 **⚠️ unverified** | Thumb-specific test, only gates OPEN (not CLOSED): `distance(thumb_tip, pinky_MCP) / distance(wrist, middle_MCP)` must exceed this ratio to count as extended. | Watch the thumb's green/red coloring in `draw_debug()`'s output (drawn as a line from thumb tip to wrist) across real open-hand gestures to find the right cutoff — no need to tune it against fist footage since CLOSED ignores the thumb. |
| `palm_height_fraction` | 0.5 | 🟡 | Palm (wrist landmark) must be in the upper fraction of the person's FULL-FRAME bbox height (from `modules.human_detection_roi`), checked on EVERY frame counted toward the sequence. Also acts as the selector for which hand drives the sequence when multiple hands are visible. | Lower (e.g. 0.3) demands the hand be raised higher; higher (e.g. 0.7) is more permissive. Remember: failing this is an immediate full sequence reset, not just a skipped frame — too strict a value will make the gesture hard to complete even with genuine intent. |
| `max_transition_gap_seconds` | 1.5 | 🟡 | Timeout between consecutive sequence transitions — exceeding it resets to WAITING_OPEN with no partial credit. Spec suggests starting ~1–2s. | Shorter demands a snappier open-close-open-close; longer tolerates a slower, more deliberate gesture. No timeout pressure while still sitting in WAITING_OPEN (nothing to time from yet). |
| `confirmation_duration_seconds` | 1.0 | 🟡 | RED→YELLOW→GREEN debounce duration, applied to the CONFIRMED completion event (see `pipeline.py`'s comments for exactly how a one-shot event is reconciled with the continuous-condition-style tracker — a judgment call, not literally specified). | Independently tunable from Methods 1 and 3. |
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

## `gesture_trajectory_verifier` (plans/04, Method 3)

Wrist+elbow+shoulder trajectory shape-matching against a shared reference set (time-based
resampling + cosine similarity). Currently **partially** filled in — see status column.

| Parameter | Current | Status | Meaning | Tuning notes |
|---|---|---|---|---|
| `confidence_threshold` | 0.6 | 🟡 | MoveNet per-keypoint confidence floor — ALL of wrist/elbow/shoulder must clear it before a sample is added to the trajectory buffer. | Independent from Method 1's MoveNet confidence values, even though it's the same underlying model. |
| `trajectory_window_seconds` | `null` | 🔴 | Rolling window for the live trajectory buffer. | Independently tunable from Method 1's `motion_window_seconds` — don't assume the same span. |
| `min_samples_for_comparison` | 8 | 🟡 | Minimum buffered samples (per arm) before a trajectory comparison is even attempted. | Needs enough points to make resampling to `resample_length` meaningful — should generally be ≥ half of `resample_length`. |
| `resample_length` | 20 | 🟡 | Fixed number of points every trajectory (live and reference) is resampled to before comparison. Spec suggests starting ~20. | More points = finer shape fidelity but more sensitive to noise; fewer = smoother but coarser comparison. |
| `similarity_threshold` | `null` | 🔴 | Cosine similarity floor the best-scoring reference must clear for a waving candidate. Spec suggests starting ~0.7–0.85. | Watch `confidence_debug` (the best score seen) across real waves vs. non-waves via the visualization tool before setting this. |
| `confirmation_duration_seconds` | `null` | 🔴 | RED→YELLOW→GREEN debounce duration. | Independently tunable from Methods 1 and 2. |
| `movenet_tfhub_handle` / `reference_dir` | 🟢 | Model source and reference-trajectory storage location. | Override only if relocating either. |

**Fixed structural rule, not a config key by design:** the reference set needs **at least 2**
entries before this module will attempt a real comparison at all (`MIN_REFERENCE_COUNT = 2` in
`modules/gesture_trajectory_verifier/config.py`) — 0 or 1 both report a distinct "not ready"
signal (`reference_count < 2` in the result) rather than a misleadingly-computed low score. Use
`capture_reference_trajectory.py` to add reference waves; you need at least two before this
method can produce anything but "not ready."

---

## `camera`

| Parameter | Meaning |
|---|---|
| `camera_index` | OS camera device index used by `main.py --mode camera` when `--camera-index` isn't passed on the command line. Not a calibration value — just which physical camera to open. |

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
