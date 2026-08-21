# Commands Reference

Every way to run this project, from the full pipeline down to a single module in isolation.
Commands assume you're in the repo root with the project's venv active
(`.venv\Scripts\activate` on Windows / `source .venv/bin/activate` elsewhere) — or prefix each
`python` below with `.venv/Scripts/python.exe` (Windows) if you don't activate it.

All camera commands default to `--camera-index` from `config/thresholds.yaml`'s `camera:`
section (falls back to `0`) — pass `--camera-index N` to override. All video commands need a
real video file path; there is no bundled sample video in this repo.

---

## 1. Full pipeline (`main.py`)

```bash
# Emergency stop only
python main.py --mode camera --modules estop --show

# Original whole-frame wave+facing demo (no identity check)
python main.py --mode camera --modules wave_facing --show --debug

# Both of the above together, same frame
python main.py --mode camera --modules both --show --debug

# pretrigger: the face-first pipeline (plans/01-04), STOPPING at TRIGGER -- for calibrating the
# pre-trigger stages in isolation before trusting the full pipeline (was called face_first).
python main.py --mode camera --modules pretrigger --gesture-method hand_keypoint --show --debug
python main.py --mode camera --modules pretrigger --gesture-method condition --show --debug
python main.py --mode camera --modules pretrigger --gesture-method trajectory_verifier --show --debug

# followme: the FULL pipeline (plans/01-08) -- face match -> ROI body detect -> gesture ->
# tracking -> recovery -> PID steering, continuing PAST the trigger via modules.followme_orchestrator.
# --debug draws EVERY phase's overlay (face, ROI, gesture, tracking, recovery), not just the summary text.
python main.py --mode camera --modules followme --gesture-method hand_keypoint --show --debug

# Against a recorded video instead of a live camera
python main.py --mode video --video path/to/file.mp4 --modules followme --gesture-method hand_keypoint --show --debug
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--mode camera \| video` | always | Live webcam vs. recorded file (`--video` required for the latter) |
| `--camera-index N` | `--mode camera` | OS camera device index |
| `--modules` | always | `estop` \| `wave_facing` \| `both` \| `pretrigger` (stops at TRIGGER) \| `followme` (full pipeline through steering) |
| `--gesture-method` | `--modules pretrigger \| followme` | `condition` (Method 1) \| `hand_keypoint` (Method 2) \| `trajectory_verifier` (Method 3) — required |
| `--face-registry-dir` | `--modules pretrigger \| followme` | Defaults to `modules/face_identity/registry_data` |
| `--config` | `--modules followme` | Defaults to `config/thresholds.yaml`, passed to `followme_orchestrator.configure()` |
| `--show` | always | Opens a display window |
| `--debug` | always | Full per-phase debug overlay — only visible combined with `--show`. `pretrigger`: face bbox + ROI + gesture keypoints/skeleton/state. `followme`: all of that PLUS tracking bbox/center-line/reverify and recovery search status, via `modules.followme_orchestrator.draw_debug()`. |

`--modules followme` is a thin wrapper `main.py` puts around
`modules.followme_orchestrator.interface` — the actual trigger → tracking → recovery → steering
composition logic lives in that module, not in `main.py` (see
[`architecture.md`](architecture.md)'s isolation-exception note). For an even richer live debug
overlay than `main.py --modules followme --show` draws (frame-center line, per-state coloring,
steering readout), run `followme_orchestrator`'s own tool directly instead:

```bash
python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode camera [--camera-index 0]
python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode video --video path/to/file.mp4
```

---

## 2. Per-module standalone tools

Every module has a `test_*.py` (prints results, minimal/no window) and, where the module's spec
requires it, a `visualize_*.py` (draws a debug overlay). Run any of them with `-m` from the repo
root.

### `emergency_stop`

```bash
python -m modules.emergency_stop.test_estop path/to/video.mp4 --show
```
No `--mode camera` option — video file only. All 10 thresholds are `null` by default (see
[`parameters.md`](parameters.md#emergency_stop)), so this reports `UNCERTAIN` on every frame
until calibrated.

### `human_detection` (legacy whole-frame detector)

```bash
python -m modules.human_detection.test_human_detection path/to/video.mp4 --show
```

### `wave_facing_gate` (Gesture Method 1)

```bash
python -m modules.wave_facing_gate.test_wave_facing path/to/video.mp4 --show
```
Feeds the WHOLE frame as a single `track_id=1` crop (no upstream detector in this test) — point
the camera at one person.

### `face_identity`

```bash
# Test matching against the registry, from a video
python -m modules.face_identity.test_face_identity path/to/video.mp4 --registry-dir modules/face_identity/registry_data

# Live visualization (bbox + match status overlay)
python -m modules.face_identity.visualize_face_identity --mode camera [--camera-index 0]
python -m modules.face_identity.visualize_face_identity --mode video --video path/to/video.mp4
```

**Registering a new person (two phases):**
```bash
# Phase 1: capture padded face-crop photos from the webcam
python -m modules.face_identity.capture_face_images <person_name> [--camera-index 0] [--samples 5]

# Phase 2: build the registry entry (embeddings) from those photos
python -m modules.face_identity.build_face_registry <person_name>

# Phase 2 also works against ANY folder of your own photos, skipping Phase 1 entirely:
python -m modules.face_identity.build_face_registry <person_name> --images-dir path/to/your/photos
```

### `human_detection_roi`

```bash
# Test with a manually-supplied fixed face bbox (no face detector in this test)
python -m modules.human_detection_roi.test_human_detection_roi path/to/video.mp4 --face-bbox X Y W H

# Live visualization, chained with face_identity (face bbox in blue, ROI in yellow, person bbox in green)
python -m modules.human_detection_roi.visualize_human_detection_roi --mode camera [--camera-index 0]
python -m modules.human_detection_roi.visualize_human_detection_roi --mode video --video path/to/video.mp4
```

### `gesture_hand_keypoint` (Gesture Method 2)

```bash
python -m modules.gesture_hand_keypoint.test_gesture_hand_keypoint path/to/video.mp4

# Live visualization, chained with face_identity + human_detection_roi. Shows the hand skeleton
# colored yellow=OPEN/green=CLOSED/gray=NEITHER, per-finger extended/curled coloring, the red
# dotted palm_height_fraction calibration line, and the current sequence stage.
python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode camera [--camera-index 0]
python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode video --video path/to/video.mp4
```

### `gesture_trajectory_verifier` (Gesture Method 3)

```bash
python -m modules.gesture_trajectory_verifier.test_gesture_trajectory_verifier path/to/video.mp4

# Live visualization, chained with face_identity + human_detection_roi. Draws the live wrist
# path plus a normalized live-vs-best-reference inset panel.
python -m modules.gesture_trajectory_verifier.visualize_gesture_trajectory_verifier --mode camera [--camera-index 0]
python -m modules.gesture_trajectory_verifier.visualize_gesture_trajectory_verifier --mode video --video path/to/video.mp4
```

**Capturing a reference wave** (needed before this method can produce anything but "not ready" —
requires at least 2 references in the shared set):
```bash
python -m modules.gesture_trajectory_verifier.capture_reference_trajectory <reference_id> --mode camera [--camera-index 0]
python -m modules.gesture_trajectory_verifier.capture_reference_trajectory <reference_id> --mode video --video path/to/video.mp4
# Perform ONE clean wave, then press 'q' to finish and save.
```

### `appearance_verifier`

```bash
# Test: compare every frame of a video against a folder of reference images
python -m modules.appearance_verifier.test_appearance_verifier path/to/video.mp4 --reference-dir path/to/reference_images/

# Visualization: camera/video (continuous) or a single image
python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir path/to/reference_images/ --mode camera [--camera-index 0]
python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir path/to/reference_images/ --mode video --video path/to/video.mp4
python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir path/to/reference_images/ --mode image --image path/to/candidate.jpg
```
First run downloads the Market1501-pretrained OSNet checkpoint (~10MB, one-time, needs network
access to Google Drive) — see [`technologies.md`](technologies.md).

### `target_tracking`

```bash
# Test: exercises RECORDING -> TRACKING -> LOST using an auto-placed centered bbox (no interaction)
python -m modules.target_tracking.test_target_tracking path/to/video.mp4

# Visualization: click-and-drag a box around a person to start an episode; 'r' re-arms, 'q' quits
python -m modules.target_tracking.visualize_target_tracking --mode camera [--camera-index 0]
python -m modules.target_tracking.visualize_target_tracking --mode video --video path/to/video.mp4
```

### `target_recovery`

```bash
# Test: runs a search episode against a video until REACQUIRED or TIMEOUT
python -m modules.target_recovery.test_target_recovery path/to/video.mp4 --target-person-name <name> --reference-dir path/to/reference_images/

# Visualization: shows which path (A/B) is active, fail count, and the timeout countdown
python -m modules.target_recovery.visualize_target_recovery --target-person-name <name> --reference-dir path/to/reference_images/ --mode camera [--camera-index 0]
python -m modules.target_recovery.visualize_target_recovery --target-person-name <name> --reference-dir path/to/reference_images/ --mode video --video path/to/video.mp4
```
`<name>` must already exist in the face registry (see `face_identity`'s registration commands
above) — `target_recovery`'s Path A filters specifically for this name.

### `followme_orchestrator` (the FULL pipeline, all 8 plans composed together)

```bash
python -m modules.followme_orchestrator.test_followme_orchestrator path/to/video.mp4 --gesture-method hand_keypoint

# Live visualization: face-center line, tracked/reacquired bbox, debug_state, should_move, steering angle
python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode camera [--camera-index 0]
python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode video --video path/to/video.mp4
```
Requires `--gesture-method` (no default, same requirement as `main.py --modules pretrigger/followme`).
Steering output (`should_move`/`steering_angle_degrees`) stays fail-closed (`should_move=False`)
until `camera.fov_degrees` and the `steering` section's `kp`/`ki`/`kd`/`max_steering_angle_degrees`
are calibrated (see [`parameters.md`](parameters.md#steering-plans08)) — trigger detection,
tracking, and recovery all work regardless of steering calibration state.

---

## 3. Recommended order for a first end-to-end smoke test

1. `capture_face_images` + `build_face_registry` — register yourself.
2. `visualize_face_identity` — confirm you're matched live.
3. `visualize_human_detection_roi` — confirm your body bbox tracks correctly off your face.
4. Whichever gesture method's `visualize_*` you plan to use — confirm the gesture triggers GREEN.
5. `main.py --modules pretrigger --gesture-method <method> --show --debug` — the pre-trigger
   pipeline, all four stages together (stops at TRIGGER).
6. `appearance_verifier` / `target_tracking` / `target_recovery`'s own tools, independently —
   confirm each post-trigger module works in isolation before trusting the composed version.
7. `main.py --modules followme --gesture-method <method> --show --debug` — the FULL pipeline in
   one command: trigger → tracking → recovery → steering, all composed, with every phase's debug
   overlay visible. (`modules.followme_orchestrator.visualize_followme_orchestrator` runs the
   same composed pipeline with its own dedicated status readout instead of `main.py`'s.)

Every threshold referenced above starts uncalibrated (`null`) — see
[`parameters.md`](parameters.md) for what each command's module needs filled in before it'll
produce a positive result instead of failing closed.
