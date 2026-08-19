# Implementation Audit — Wave + Facing Trigger Gate Demo

Audit of the changes made to implement `docs/master_spec.md` against the actual repository
state. Covers what was added/changed, every deviation from a literal reading of the spec (with
reasoning), what was tested and how, and what's still unverified.

## Files added

| File | Purpose |
|---|---|
| `src/pose_estimator.py` | `MoveNetPoseEstimator` — MoveNet Lightning wrapper (spec §3). |
| `src/wave_detector.py` | `WaveFacingGate` / `GestureResult` — wave rule + facing proxy (spec §4–5). |
| `demo_wave_trigger.py` | Entry script — real mode + `--any-person` test mode (spec §2, §6). |
| `docs/*` | This tech report. |

## Files changed

| File | Change |
|---|---|
| `config/settings.yaml` | Added `wave_trigger_demo` section (all spec §7 placeholders, single location per the spec's "no scattered hard-coding" requirement). |
| `requirements.txt` | Added `tensorflow`, `tensorflow-hub`, `setuptools<81` (see `technologies.md`). |

## Files explicitly NOT touched (per spec §8 boundary)

`src/pipeline.py`, `src/detector.py`, `src/verifier.py`, `src/types.py`, `src/view_estimator.py`,
`src/registry.py`, `src/registration.py`, `src/person_selector.py`, `main.py`. Confirmed by
re-reading the final diffs — the demo layer only imports and reads from these, never edits them.

## Deviations from a literal reading of the spec, and why

1. **`CV_Verification_Handoff_Doc.md` does not exist.** Searched the entire
   `D:\GW_UNIVERSITY\AIS\AUTOBOT_V2` tree; not present under any name. The spec cites it for: the
   crop-coordinate ROI bug warning (§9.3), the `roi_failure_max_frames=5` precedent (§6), the
   single-target Follow-Me design assumption (§5.1), and the pipeline completion status (§8.1).
   Proceeded by reading `src/pipeline.py` directly instead — its own inline comments cover the
   ROI coordinate-conversion concern independently (see `process_frame()`'s detection block,
   which explicitly converts ROI-local bbox coordinates back to full-frame before returning), and
   its `roi_failure_max_frames` config default is directly visible in `config/settings.yaml`. The
   single-target assumption (§8, second bullet) was **verified against actual code** rather than
   assumed on faith from the missing doc — `process_frame()`'s sticky-target selection always
   narrows to at most one `target_track`, confirmed by reading the selection logic end to end.

2. **MoveNet runtime: TF Hub SavedModel, not a manually downloaded `.tflite`.** The spec offers
   either. `hub.load(url)` auto-downloads and caches the model on first use — the same
   auto-download-on-first-use pattern Ultralytics already uses for YOLO weights in this project
   (`src/detector.py`'s fallback download path), so it doesn't introduce a new operational
   pattern, just a new dependency. A `.tflite`-based approach (via `tflite-runtime` or the newer
   `ai-edge-litert`) was considered and rejected: those packages have inconsistent Windows/Python
   3.11 wheel availability, while full `tensorflow` has official Windows wheels and was verified
   to install and run cleanly on this machine.

3. **Wave-detector buffer semantics when the arm is legitimately lowered (not low-confidence).**
   Spec §4.3 only specifies fault tolerance for *low-confidence* frames (skip, don't reset).
   It's silent on what happens to the `wrist_x` buffer during a frame where confidence is fine
   but the arm just isn't raised (§4.1 posture condition false for a reason other than
   occlusion). Implemented as: **don't push, don't reset** — the buffer only ever accumulates
   samples from frames where the posture condition holds, and simply ages out old samples via
   its fixed-size deque. Reasoning: a normal "hi" wave keeps the arm raised throughout the
   oscillation, so this is the natural case; resetting on every momentary "arm below shoulder"
   reading would fight normal wave motion (which briefly dips near the bottom of each swing).

4. **`amplitude_norm` is normalized to the padded MoveNet input square, not the crop.** The spec
   says "normalized 0-1 coordinates of the bbox crop" for `threshold_amplitude_norm`. MoveNet's
   `resize_with_pad` preprocessing letterboxes a non-square crop into the 192×192 input with
   padding, so the model's raw normalized output is relative to that padded square, not the
   crop's own aspect ratio. Used the raw MoveNet-normalized value directly rather than adding a
   correction step, consistent with the spec's own framing of this threshold as a rough
   placeholder with "no empirical basis" — documented inline in `config/settings.yaml` so
   whoever calibrates this later knows the exact coordinate space it's measured in.

5. **`--any-person` testing mode (added after initial delivery, on request).** Not in the
   original spec, which gates everything on `registered_person`. Added as an opt-in flag that
   bypasses `FollowPipeline`/identity verification entirely (raw `YoloDetector.track()`, largest
   bbox picked as the test subject, `registered_person` forced `True`). Justification: the
   original demo requires a person to already be registered via the Tkinter selector before any
   gesture logic can be exercised at all, which blocked testing in an environment without a
   pre-existing registry entry. The bypass mode shares every downstream step (crop → MoveNet →
   gate → trigger → overlay) with the real mode via the same tuple-returning source functions —
   see `architecture.md` "Why the two modes share one downstream path" — so it cannot silently
   drift from the real gesture logic. It is visually flagged in the overlay
   ("ANY-PERSON MODE (identity check bypassed)") so it's never mistaken for the spec-compliant
   demo.

6. **`threshold_keypoint_conf` split into `threshold_keypoint_conf_wave` /
   `threshold_keypoint_conf_facing` (added after initial delivery, on request).** The spec (§5)
   explicitly anticipated this: "if split, use clear names ... so they don't get confused during
   later calibration" — it was delivered shared-by-default per the spec's own default, then split
   on request so the two gates can be tuned independently (e.g. a stricter facing bar without
   also making wave detection stricter). `WaveFacingGate.__init__` now takes both explicitly;
   `demo_wave_trigger.py`'s debug-overlay skeleton dots use `min()` of the two as a purely
   cosmetic display cutoff (show a point if it clears either gate's bar).

7. **Console log noise reduction (added after initial delivery, on request).** `src/pose_estimator.py`
   now sets `TF_CPP_MIN_LOG_LEVEL`/`GLOG_minloglevel` and filters the `pkg_resources` deprecation
   warning *before* importing `tensorflow`/`tensorflow_hub` (import-order matters — filtering
   after the import is too late for warnings raised during the import itself). One TF
   native-backend startup line (`oneDNN custom operations are on...`, preceded by an absl
   pre-init preamble) cannot be suppressed from Python — it's printed directly to stderr by the
   C++ layer before Python logging config takes effect. None of this was ever a functional
   problem; confirmed by the live run in the gap note below, which got past all of it.

## Testing performed

No camera or display is available in the implementation environment, and no person was
registered in `logs/registry/` (empty at time of writing) — so the live end-to-end demo has
**not** been run interactively. What was verified instead:

| Check | Method | Result |
|---|---|---|
| `tensorflow` / `tensorflow-hub` import | Direct import in venv | OK, after the `setuptools<81` fix |
| MoveNet Lightning loads + infers | `hub.load(...)` + dummy `[192,192,3]` input | Output shape `(1,1,17,3)` confirmed to match spec |
| `resize_with_pad` padding behavior | Empirical probe: marked pixels in a non-square test image, checked where they land in the 192×192 output | Confirmed **centered** (symmetric) padding, not top/left-aligned — this determined the inverse-mapping math in `demo_wave_trigger.py`'s skeleton overlay |
| All new/changed `.py` files | `py_compile` | No syntax errors |
| `demo_wave_trigger.py` imports (full stack: torchreid, YOLO, MoveNet) | Direct `import demo_wave_trigger` | Succeeds, no import-time errors |
| `WaveFacingGate` direction-change/amplitude math | Synthetic zigzag `wrist_x` sequence fed through `update()` | Direction changes and amplitude computed correctly; `is_waving` flips `True` exactly when the configured thresholds are crossed |
| `WaveFacingGate` bad-frame tolerance | One low-confidence frame injected mid-sequence | Buffer length unchanged (frame skipped, not reset), confirming §4.3 semantics |
| Pipeline→crop→MoveNet→gate wiring (real mode) | Synthetic `AngleResult` + fake `last_detections`, real `MoveNetPoseEstimator` + `WaveFacingGate` | Runs end-to-end with no errors, correct shapes |
| `--any-person` primary-detection selection | Fake detector returning two people, one larger | Correctly picks the larger bbox; correctly reports "no target" on an empty detection list |

## Known gaps / not yet done

- **Live run confirmed starting, gesture accuracy still unverified.** On 2026-08-18 the user ran
  `python demo_wave_trigger.py --any-person` on real hardware: YOLO loaded, MoveNet loaded from
  TF Hub, camera opened (`640x480@30fps`). This confirms the wiring works end-to-end outside the
  synthetic tests above. Not yet confirmed: whether real waves get detected reliably, whether the
  facing proxy triggers correctly at a real desk distance, and FPS with both OSNet and MoveNet
  running (only relevant to real mode, not `--any-person`, which skips OSNet).
- **All spec §7 placeholders are still placeholders** — `threshold_keypoint_conf_wave`,
  `threshold_keypoint_conf_facing`, `wave_buffer_size`, `wave_direction_changes_min`,
  `wave_amplitude_norm_min`, `max_consecutive_bad_frames` in `config/settings.yaml`'s
  `wave_trigger_demo` section. None have been calibrated against real data, per the spec's own
  instruction not to.
- **`CV_Verification_Handoff_Doc.md` is still missing.** If it exists somewhere outside the
  searched tree, it should be located and cross-checked against the assumptions in
  `docs/architecture.md` and this audit (particularly the single-target and ROI-coordinate
  points reconstructed from `pipeline.py` alone).
