# Tunable Parameters Reference

Every configurable value in [`config/settings.yaml`](../config/settings.yaml), grouped by
subsystem. For each: what it controls, what raising/lowering it trades off, and where it's
consumed in code. Values shown are the file's current values as of this writing — check the
live file for the current numbers.

**Placeholder values.** Anything flagged **PLACEHOLDER** below has no empirical basis yet — it's
either a reasonable starting guess or copied from an unrelated precedent elsewhere in the
project (noted where that's the case). None of them should be treated as validated defaults.

---

## Camera & Input

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `camera_index` | `2` | Which OS camera device `cv2.VideoCapture` opens. | Hardware-dependent, not an accuracy tunable — set to whichever index your webcam enumerates as. |
| `input_resolution` | `[640, 480]` | Every frame is resized to this before any processing (`main.py`, `demo_wave_trigger.py`, `pipeline.py`). | Higher = sharper detail for detection/pose/re-id (helps distant people, fine finger/wrist positions) but slower — every downstream model (YOLO, OSNet, MoveNet) processes a bigger tensor. Lower = faster, more Raspberry Pi 5 friendly, at the cost of accuracy on small/far subjects. |
| `camera_fov_horizontal_deg` | `60.0` | Used only in `pipeline.py`'s `angle_offset_deg` trig calculation (how far off-center the target is, in degrees). | Must match your actual webcam's horizontal field of view or the angle output is skewed. **Does not affect** `demo_wave_trigger.py`'s trigger boolean at all — only relevant if you later build steering logic on top of `AngleResult`. |
| `flip_horizontal` | `true` | Mirrors the frame horizontally after capture. | Purely cosmetic/correctness — set `true` if your webcam feed appears mirrored, no effect on any model's accuracy. |

## Detection (YOLO + ByteTrack)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `yolo_model_path` | `yolo11n.onnx` | Which YOLO11 weights file `YoloDetector` loads. | Swapping to a larger YOLO11 variant (s/m/l) trades detection accuracy for speed — `n` (nano) is already the fastest/smallest, likely the right choice for Pi 5. |
| `detection_imgsz` | *(empty → Ultralytics default 640)* | Inference resolution YOLO internally resizes to (independent of `input_resolution`). | Lowering to `416`/`320` cuts YOLO's own compute meaningfully (biggest single lever for Pi 5 detection speed) at the cost of recall and bbox precision for people far from the camera. **PLACEHOLDER** — left at default, not yet tuned for the target hardware. |

## Re-Identification (OSNet) — real mode and `--any-person` `method=osnet` only

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `osnet_variant` | `osnet_x1_0` | Which OSNet checkpoint size `OSNetVerifier` loads (`x1_0` → `x0_75` → `x0_5` → `x0_25`, decreasing size/accuracy). | Smaller variants are faster and lighter (better for Pi 5) at a documented accuracy cost — see the in-file comment for the exact Rank-1/mAP numbers per variant from the torchreid Model Zoo. |
| `similarity_threshold` | `0.80` | Minimum cosine similarity between a candidate's embedding and the registered reference to count as a match. | Higher = stricter (fewer false accepts of a different person, more false rejects of the real target under lighting/pose changes). Lower = the opposite. **PLACEHOLDER** — this value was carried over unchanged from an older MobileNetV3-based verifier and has not been recalibrated for OSNet's own similarity-score distribution, which is a different model with likely different score ranges. |
| `verify_every_n_frames` | `1` | How often a non-active track gets a *fresh* OSNet re-verification vs. reusing its last score (the currently sticky-locked target and any brand-new track are always verified fresh regardless of this setting). | `1` = verify every track every frame (current, most accurate, most expensive). Raising it cuts OSNet forward-pass cost proportionally for background tracks, at the cost of staler verification for people who aren't currently the target. |

## Registration (Stage 1 capture UX — not an accuracy tunable)

| Parameter | Value | Controls |
|---|---|---|
| `roi_percent` | `[0.302, 0.0, 0.715, 1.0]` | The on-screen framing guide box shown during registration capture (`src/registration.py`) — the rest of the preview is blurred, this region stays sharp, telling the person where to stand. Adjustable live via sliders in the registration UI. **Not used anywhere in real-time tracking** — capture-time only. |
| `registration_countdown_sec` | `3` | Seconds of on-screen countdown before raw capture starts. |
| `registration_duration_sec` | `8` | How long the raw capture phase runs. |
| `registration_sample_interval_frames` | `5` | Sample every Nth frame during capture, spreading samples across the capture window instead of using every consecutive frame. |
| `registration_min_samples` | `5` | Minimum valid samples required before a registration is accepted; below this, registration fails. |

## Dynamic ROI-constrained detection (real mode only, once a target is sticky-locked)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `roi_margin_percent` | `0.5` | How far `pipeline.py` expands the search region around the target's *previous* bbox, as a fraction of that bbox's own width/height, per side. | Larger = more tolerant of fast target motion between frames (bigger search window catches more movement) but the ROI-constrained detection approaches full-frame cost as it grows, and a bigger window is more likely to catch a nearby distractor. **PLACEHOLDER**. |
| `roi_failure_max_frames` | `5` | Consecutive ROI-detection misses tolerated before forcing a full-frame rescan. | Higher = more patient before falling back (good if ROI misses are usually transient) but a real target loss takes longer to notice. Lower = snappier full-frame recovery, more full-frame scans if the ROI is noisy. **PLACEHOLDER** — this is also the precedent value copied into `max_consecutive_bad_frames` and `lock_grace_frames` below. |

## Aspect Ratio Hard Gate (real mode only)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `aspect_ratio_tolerance_percent` | `0.30` | Allowed relative deviation between a candidate's current bbox aspect ratio and the reference aspect ratio captured at registration — independent veto on top of appearance similarity. | Lower = stricter (better protection against a differently-shaped impostor wearing similar-colored clothing) but more likely to reject the real target during pose changes (crouching, turning, arms raised). Higher = more forgiving of pose variation, weaker protection. **PLACEHOLDER**, deliberately wide as a starting point. |

## Temporal Smoothing (real mode only — applies only across identical track_ids)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `mode` | `ema` | Which smoothing algorithm (`ema` or `voting`) — mutually exclusive, must pick one. | EMA reacts continuously; voting is a discrete pass/fail window — see the two rows below. |
| `ema_alpha` | `0.3` | (EMA mode) Weight given to this frame's raw score vs. the running smoothed average. | Higher = reacts faster to genuine changes but noisier/more jumpy. Lower = smoother but slower to confirm a match or notice it's gone. **PLACEHOLDER**. |
| `voting_window_size` | `5` | (Voting mode) Sliding window length in frames. | Larger window = more stable vote ratio, slower to respond to real change. **PLACEHOLDER**. |
| `voting_ratio` | `0.6` | (Voting mode) Fraction of the window that must be "pass" for the overall vote to pass. | Higher = stricter confirmation (fewer false locks) but slower to lock on, more sensitive to a few bad frames. **PLACEHOLDER**. |
| `voting_min_ready_percent` | `0.6` | (Voting mode) Fraction of the window that must be filled before the vote ratio is trusted at all; below this, falls back to a raw per-frame threshold check (cold-start / new-track fallback). | Higher = waits longer before trusting the smoothed signal on a new track. **PLACEHOLDER**. |

---

## Wave + Facing Trigger Gate demo (`wave_trigger_demo` section, `demo_wave_trigger.py`)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `pose_model_url` | TF Hub MoveNet Lightning URL | Which pose model `MoveNetPoseEstimator` loads. | Fixed by spec (COCO 17-keypoint singlepose) — not really meant to be tuned, but technically swappable to another MoveNet variant/source. |
| `threshold_keypoint_conf_wave` | `0.1` | Minimum MoveNet confidence for a wrist/shoulder keypoint to count as reliably visible when evaluating the wave posture condition (`src/wave_detector.py`). | Lower = keeps evaluating gestures even on shaky pose estimates (useful in poor lighting/low resolution) but risks noisy false wave detections from unreliable keypoints. Higher = stricter, fewer false positives, but more frames get treated as "can't evaluate" and routed into the bad-frame-tolerance path instead. **PLACEHOLDER** — lowered from the spec's default `0.3` to `0.1`, presumably after observing MoveNet under-report confidence in your actual lighting/camera setup; worth confirming this isn't just masking a resolution/lighting problem upstream. |
| `threshold_keypoint_conf_facing` | `0.1` | Same idea, but gates the 4 keypoints (both eyes, both shoulders) used by the facing-camera proxy. Independently tunable from the wave threshold (`src/wave_detector.py`'s `WaveFacingGate`). | Lower = easier to satisfy "facing camera" (more lenient, may accept partial profile views). Higher = stricter, requires clear/frontal visibility of all 4 points. **PLACEHOLDER**. |
| `wave_buffer_size` | `20` frames | How many recent qualifying (arm-raised) frames are kept to analyze the wrist's oscillation (`src/wave_detector.py`). | Larger = requires a longer sustained wave before enough evidence accumulates; smooths out noise but adds latency and demands the person keep waving longer. Smaller = more responsive, but more sensitive to brief jitter looking like a wave. **PLACEHOLDER** — sized for ~0.6–1s at 20–30fps; re-measure against your actual achieved fps (which will be lower on a Pi 5). |
| `wave_direction_changes_min` | `3` | Minimum number of left-right sign changes in wrist-x movement within the buffer to call it a wave. | Higher = requires more clearly repetitive back-and-forth motion (fewer false positives from a single swipe or a reaching motion) but harder/slower to trigger. Lower = easier to trigger, more prone to firing on hand jitter. **PLACEHOLDER**, no empirical basis. |
| `wave_amplitude_norm_min` | `0.05` | Minimum swing width required, in MoveNet's normalized (letterboxed 192×192 input) coordinate space. | Higher = requires a bigger, more deliberate swing, filtering out tiny tremor. Lower = detects subtler waves but more prone to noise-driven false triggers. **PLACEHOLDER**, no empirical basis; also see `docs/implementation_audit.md` for the caveat that this coordinate space isn't exactly "the crop," it's the padded model input. |
| `max_consecutive_bad_frames` | `5` | How many consecutive low-confidence frames are tolerated before the wave buffer is wiped entirely (gesture-tracking-only "lost," independent of the identity tracker). | Higher = more tolerant of occlusion/motion-blur streaks without losing accumulated wave progress. Lower = resets more eagerly — safer against stale data lingering, but a short bout of bad luck costs more progress. **PLACEHOLDER**, copied from `roi_failure_max_frames`'s precedent value above. |
| `log_csv_path` | `logs/wave_trigger_demo_log.csv` | Where the per-frame CSV log (trigger state + per-module timing) is written. | Operational, not an accuracy tunable. |

### `any_person_tracking` (nested under `wave_trigger_demo`, `--any-person` mode only)

| Parameter | Value | Controls | Trade-off |
|---|---|---|---|
| `reacquisition_method` | `position` | Which of 4 strategies (`src/any_person_tracker.py`) picks the new primary once the sticky lock is broken: `largest_bbox` (free), `position` (free, closest to last known bbox), `histogram` (cheap, HSV color match), `osnet` (heaviest, re-id embedding match). | Cost and re-acquisition robustness both increase in that order. All 4 share the same sticky-lock behavior for the common case (target still detected) — this only affects what happens at the rarer moment of genuine re-acquisition. See `docs/implementation_audit.md` for the full comparison. Also settable per-run via `--reacquisition-method` without editing the config. |
| `lock_grace_frames` | `5` | Consecutive frames the locked `track_id` may be absent from detections before the lock is abandoned and re-acquisition runs. | Higher = more tolerant of brief detector misses/occlusion without an unnecessary re-acquisition — but a genuinely-departed target takes longer to notice is gone (no gesture is evaluated during that whole grace window). Lower = snappier to notice real loss, but more prone to unwanted re-acquisitions from a single missed detection. **PLACEHOLDER**, same precedent as `roi_failure_max_frames`/`max_consecutive_bad_frames`. |
| `histogram_similarity_min` | `0.5` | Minimum HSV-histogram similarity (0–1) required to accept a re-acquisition candidate when `method="histogram"`; below this, falls back to `largest_bbox` for that event. | Higher = stricter appearance matching (less likely to wrongly re-lock onto a different, similarly-colored person) but more likely to fall back to the naive largest-bbox pick when the real match's similarity dips (e.g. a lighting change). Lower = fewer fallbacks, higher chance of a wrong re-lock. **PLACEHOLDER**. |
| `osnet_similarity_min` | `0.5` | Same idea as above, but for OSNet cosine similarity when `method="osnet"`. | Same trade-off shape as `histogram_similarity_min`; OSNet's similarity distribution is likely different from the histogram method's, and — like `similarity_threshold` above — has not been empirically calibrated for this use case. **PLACEHOLDER**. |

---

## Quick guidance for Raspberry Pi 5 tuning

The parameters with the biggest expected impact on per-frame latency, roughly ranked:

1. `input_resolution` and `detection_imgsz` — shrink both first; everything downstream (YOLO, OSNet, MoveNet) pays for whatever size you hand it.
2. `osnet_variant` — drop to `osnet_x0_5` or `osnet_x0_25` if identity accuracy tolerates it (real mode / `method=osnet` only).
3. `any_person_tracking.reacquisition_method` — `largest_bbox`/`position` are effectively free; `osnet` pays a full re-id forward pass, but only at (rare) re-acquisition events, not every frame.
4. `verify_every_n_frames` — cheap way to cut OSNet cost for background (non-target) tracks in real mode.

Use `demo_wave_trigger.py`'s per-frame CSV log (`log_csv_path`) and the live overlay timing line to see exactly where your frame budget is going before changing any of these — don't guess.
