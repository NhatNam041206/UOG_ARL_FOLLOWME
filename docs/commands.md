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
# Simplest form: --modules defaults to "register", opening the Tkinter registration UI
# (the UI's "Follow Me" button works out of the box — no extra flag needed)
python main.py

# Register ONE person headlessly, no UI
python main.py --modules register --person-name Nam --camera-index 0

# Register, then immediately chain into followme mode once registration succeeds
python main.py --modules register --person-name Nam --then-followme --show

# pretrigger: the face-first pipeline (plans/01-04), STOPPING at TRIGGER -- for calibrating the
# pre-trigger stages in isolation before trusting the full pipeline (was called face_first).
# Uses modules.gesture_hand_keypoint as the TRIGGER gesture method (the only one left — two
# others, "condition"/wave_facing_gate and "trajectory_verifier"/gesture_trajectory_verifier,
# were removed).
python main.py --mode camera --modules pretrigger --show --debug

# followme: the FULL pipeline (plans/01-08) -- face match -> ROI body detect -> gesture ->
# tracking+recovery (modules.autocar, via autocar_adapter) -> PID steering, continuing PAST the
# trigger via modules.followme_orchestrator. --debug draws EVERY phase's overlay; the
# steering-direction arrow is drawn separately, whenever --show is on.
python main.py --mode camera --modules followme --show --debug

# Against a recorded video instead of a live camera
python main.py --mode video --video path/to/file.mp4 --modules followme --show --debug

# Headless (no --show) over SSH, still fully reviewable afterward: every pretrigger/followme run
# ALWAYS writes runs/<timestamp>_<mode>/{meta.json,decisions.jsonl}; add --save-video to also
# save the annotated overlay as debug.avi, even with no window open. See "Reviewing a run" below.
python main.py --mode camera --modules followme --save-video
python main.py --mode camera --modules followme --save-video --log-dir /home/pi/followme_runs

# Watch a headless run LIVE instead of only reviewing it afterward — see "Watching a run live" below
python main.py --mode camera --modules followme --stream

# Register headlessly over SSH too (registration's capture window is no longer forced-on — see
# --show below); --stream shows the live ROI/person-count overlay the same way
python main.py --modules register --person-name Nam --stream

# Also publish each frame's FollowMeCommand over MQTT to a Pi 4 motor controller (see
# docs/mqtt_handoff_pi4.md) -- requires config/thresholds.yaml's mqtt_bridge.broker_host and
# publish_hz to be set (both start null/fail-closed; publish() no-ops until then).
python main.py --mode camera --modules followme --show --mqtt
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--mode camera \| video` | `pretrigger`/`followme` | Live webcam vs. recorded file (`--video` required for the latter); required for these two, ignored for `register` |
| `--camera-index N` | `--mode camera`, `register` | OS camera device index |
| `--modules` | always | `pretrigger` (stops at TRIGGER) \| `followme` (full pipeline through steering) \| `register` (**default**) — hands off to `register_person`, not a per-frame pipeline |
| `--face-registry-dir` | `pretrigger \| followme` | Defaults to `modules/face_identity/registry_data` |
| `--config` | `followme`, `register` | Defaults to `config/thresholds.yaml`, passed to `followme_orchestrator.configure()` or `register_person.run()`; also reused by `mqtt_bridge.configure()` when `--mqtt` is set |
| `--mqtt` | `followme` | Publishes each frame's `FollowMeCommand` over MQTT via `modules.mqtt_bridge` — see `docs/mqtt_handoff_pi4.md`. Off by default; `mqtt_bridge`/`paho-mqtt` are never imported unless this is set. Fails closed (no-ops) while `mqtt_bridge.broker_host`/`publish_hz` are unset in `thresholds.yaml`. |
| `--person-name` | `register` | Headless single-person registration, no UI. Omit to open the Tkinter UI. |
| `--front-samples` / `--back-samples` | `register` | Sample counts per capture phase (default 15 each) |
| `--then-followme` | `register --person-name` | On success, fall through into the same `followme` camera loop |
| `--show` | always | Opens a display window. `register`'s capture window used to ignore this flag entirely and always show (a real gap — a true SSH session with no display would hang here); both `main.py --modules register --person-name` and `register_person.py`'s own standalone CLI now gate it the same as `pretrigger`/`followme`, off by default. |
| `--debug` | `pretrigger \| followme` | Full per-phase debug overlay — only visible combined with `--show` (or saved via `--save-video`/streamed via `--stream`, see below). `pretrigger`: face bbox + ROI + gesture keypoints/skeleton/state. `followme`: all of that PLUS `autocar_adapter`'s tracked bbox/center-line/state readout, via `modules.followme_orchestrator.draw_debug()`. |
| `--save-video` | `pretrigger \| followme` | Saves the annotated debug overlay to `runs/<run_id>/debug.avi` (MJPG) — independent of `--show`, works headlessly over SSH. The overlay is drawn even without `--show` whenever this is set, so the saved video matches what `--debug`/`--show` would have displayed live. |
| `--log-dir` | `pretrigger \| followme \| register --person-name` | Where per-run structured logs are written (default `runs`). Always on for these — see "Reviewing a run" below. |
| `--stream` | `pretrigger \| followme \| register --person-name \| register --interactive` | Publishes the live debug overlay (or, for `register`, the ROI/person-count capture overlay) over HTTP at `127.0.0.1:8080` — see "Watching a run live" below. Independent of `--show`/`--save-video`; any combination works. **With `--modules register` and neither `--person-name` nor `--interactive` given, auto-selects `--interactive`** (prints a note explaining why) rather than erroring or silently falling through to the Tkinter UI, which has no streaming equivalent. |
| `--interactive` | `register` | Opens the interactive registration console instead of the headless single-person path or the Tkinter UI — see "Registration console" below. Mutually exclusive with `--person-name` and `--then-followme` (use the console's own `follow <name>` command instead of `--then-followme` to pick who to follow). |

`--modules followme` is a thin wrapper `main.py` puts around
`modules.followme_orchestrator.interface` — the actual trigger → tracking → recovery → steering
composition logic lives in that module, not in `main.py` (see
[`architecture.md`](architecture.md)'s isolation-exception note). For an even richer live debug
overlay than `main.py --modules followme --show` draws (frame-center line, per-state coloring,
steering readout), run `followme_orchestrator`'s own tool directly instead:

```bash
python -m modules.followme_orchestrator.visualize_followme_orchestrator --mode camera [--camera-index 0]
python -m modules.followme_orchestrator.visualize_followme_orchestrator --mode video --video path/to/file.mp4
```

`followme_orchestrator.configure()` (called once, before the frame loop, by any of the paths
above) eagerly loads every model the pipeline will use — takes a few seconds up front so nothing
cold-starts later, mid-session, at the exact moment a gesture trigger fires.

### Reviewing a run (`runs/<timestamp>_<mode>/` — plans/10_debug_logging_observability.md)

Every `--modules pretrigger`/`followme` run, and every `register --person-name` headless capture
session (see `plans/11_registration_interactive_console.md` chunk 7 — the Tkinter UI still
doesn't write one), writes a timestamped, self-contained folder, printed at startup as
`logging to runs/...`:

```
runs/20260826T210455Z_followme/
    meta.json         # git commit, argv, full resolved thresholds.yaml snapshot at run start,
                       # start/end time, frame count, how the run ended, video info if saved
    decisions.jsonl     # one structured JSON record per frame — face match, gesture sequence
                        # progress, tracking state, steering command
    debug.avi           # only if --save-video was passed
```

This is what makes a headless SSH test run (no `--show`) fully reviewable afterward — pull the
whole folder down and inspect it locally instead of needing a monitor on the Pi:

```bash
scp -r pi@<host>:~/UOG_AIS_FOLLOWME/runs/20260826T210455Z_followme .
python -c "import json; print(json.load(open('20260826T210455Z_followme/meta.json'))['exit_reason'])"
python tail_log.py --latest             # pretty-printed live tail, on the Pi itself — see below
```
`runs/` is gitignored — these are per-run artifacts, not project source.

### Watching a run live (`--stream`, `debug_stream.py`)

`--save-video`/`decisions.jsonl` are for reviewing a run **afterward** — `--stream` is for
watching one **live**, as an alternative to `--show` when no display is attached (the normal case
over SSH). It's dev-tooling only, deliberately not wired into any module's core pipeline path —
see [`debug_stream.py`](../debug_stream.py)'s own docstring. Binds `127.0.0.1` only, never exposed
on the network — view it through an SSH local port-forward:

```bash
# on your own machine, in a separate terminal, while the Pi run is active:
ssh -L 8080:localhost:8080 pi@<host>
# then open http://127.0.0.1:8080/ in a browser on your machine
```

**Stopping a `--stream`-only run** (no `--show`, so there's no `q`-keypress window to click into):
`Ctrl+C` in the terminal running `main.py` is the legitimate way to stop it — it exits cleanly
(camera released, stream/logger closed, `meta.json` records `exit_reason="user_quit"`), not as a
raw crash/traceback. For `register_person.py --interactive` specifically, `Ctrl+C` is
context-sensitive: pressed while a `register <name>` command is actively capturing, it cancels
just that command and drops you back at the `>>>` prompt; pressed while idle at the prompt, it
exits the whole console.

Works for `pretrigger`/`followme` (the same annotated overlay `--save-video` would have written)
and for `register --person-name` (the live ROI box + green/yellow person-count gate during
capture — the same overlay `--show` would have displayed). Throttled by design (published at a
reduced rate/JPEG quality, not full-rate) so it doesn't compete with the inference loop for CPU on
a Pi; combine freely with `--show`/`--save-video`/`--log-dir` — none of them exclude each other.

**`--modules register --stream` with neither `--person-name` nor `--interactive` auto-selects
`--interactive`** — the Tkinter CRUD UI (registration's list/pick/re-capture/delete flow) already
has its own live window and was deliberately never given a streaming equivalent, so it's the one
`register` path `--stream` can never apply to (see `plans/10_debug_logging_observability.md`
chunk 6). Rather than requiring `--interactive` to also be typed explicitly, `main.py` picks it
for you and says so out loud:
```
Note: --stream needs --person-name or --interactive under --modules register; opening the
interactive console (pass --interactive explicitly to skip this note).
```

### Registering people (`register_person.py` / `--modules register`)

```bash
# Tkinter CRUD app: register new people, or pick/re-capture/delete existing ones
python register_person.py

# Headless, one person, no UI — --show off by default here too now (see the --show row above);
# add --show to open a local window
python register_person.py Nam --camera-index 0 [--front-samples 15] [--back-samples 15]

# Same, but also/instead streamed over HTTP — see "Watching a run live" above
python register_person.py Nam --stream
```

### Registration console (`--interactive`)

An interactive alternative to both of the above for a headless SSH session — list who's
registered, register/re-register a person, or delete one, all from a plain text prompt, no
display needed at all (plans/11_registration_interactive_console.md). Available both through
`main.py` and through `register_person.py`'s own standalone CLI:

```bash
python main.py --modules register --interactive
python main.py --modules register --interactive --stream   # + watch the live capture overlay in a browser
python main.py --modules register --interactive --log-dir /home/pi/register_runs

python register_person.py --interactive              # equivalent, standalone form
python register_person.py --interactive --stream
```
`--interactive` is mutually exclusive with `--person-name` (both entry points reject the
combination outright) and with `--then-followme` — `--then-followme` means "whoever was just
registered", which doesn't resolve to a single person in a console session that can register
zero, one, or many. Use the console's own `follow <name>` command instead — pick exactly who to
follow, from the same prompt:

```
Registration console. Commands: list, register <name>, delete <name>, follow <name>, quit
>>> list
  Nam      front= 15 back= 15  ready=True
>>> register Alice
  FRONT: face the camera, stand inside the box. Need 15 samples — 'q' to stop early.
    saved '...' (1/15)
    ...
  OK
>>> delete Nam
  Delete 'Nam'? [y/N] y
  deleted
>>> follow Alice
  Selected 'Alice' — exiting console to start followme mode.

'Alice' selected — continuing into followme mode.
```
`follow <name>` requires the person to already be fully registered (both a face registry entry
and a target profile — `list`'s `ready=` column shows this); on an unready name it prints why and
stays in the console instead of exiting. Only reaches followme mode through `main.py` — through
`register_person.py`'s own standalone CLI, `follow` still selects the name and reports it, but
that script has no camera-loop machinery of its own to continue into, same limitation as the
`--person-name` path's own `--then-followme` — it tells you to relaunch via `main.py` instead.

`--front-samples`/`--back-samples`/`--camera-index`/`--config` all apply the same way they do to
the headless `<person_name>` form above — one fixed sample count for every `register <name>`
command in the session, not overridable per-command. `Ctrl+C` mid-`register` cancels just that
command and returns to the `>>>` prompt; `Ctrl+C` at an idle prompt exits the whole console.
Logs the same way the headless form does — `runs/<timestamp>_register/decisions.jsonl` — but
never prints them to this terminal; watch them live from a SECOND terminal/SSH session instead:

```bash
python tail_log.py --latest                       # auto-picks the run just started above
python tail_log.py runs/20260827T120000Z_register/decisions.jsonl   # or point at one directly
python tail_log.py --latest --lines 0               # skip history, only show new records
```
```
Tailing runs/20260827T120000Z_register/decisions.jsonl — Ctrl+C to stop.
[12:00:03.104] stage=capture person_name=Alice phase=front saved=1 samples_needed=15 person_count=1
[12:00:04.110] stage=capture person_name=Alice phase=front saved=2 samples_needed=15 person_count=1
```
Works against ANY mode's log, not just `register` — `main.py --modules followme --log-dir ...`'s
`decisions.jsonl` tails the same way. `Ctrl+C` stops watching cleanly (doesn't touch the run
itself, which keeps going in its own terminal).

Two capture phases per person — FRONT (face the camera) then BACK (turn around) — each saved as
RAW frames first, then cropped to a configurable ROI (`config/thresholds.yaml`'s
`register_person:` section) as its own separate, inspectable file
(`registration_captures/<name>/{raw,cropped}/{front,back}/*.jpg`) before anything gets built. The
Tkinter flow pauses after cropping so you can check the crops before continuing. Builds BOTH
`modules/face_identity/registry_data/<name>.npz` (face match) and
`modules/autocar/models/enrolled_<name>.npz` (re-id profile) from the same session.

While capturing, a live person-count check (`registration_data.LiveSubjectDetector`) only accepts
a frame when exactly 1 person is inside the ROI — the box shows green when accepted, yellow when
0 or 2+ people are in view. Person-count only, not identity — face/pose identity detection still
happens later, only during the build step, on the cropped images.

Requires `modules/autocar/models/osnet_x1_0_msmt17.onnx` to exist — see
[`technologies.md`](technologies.md) for how it's obtained (not part of the vendored repo).

### Testing the vendored tracking+recovery backbone alone

`modules/autocar/main.py` is their own standalone demo — entirely independent of
`autocar_adapter`/`followme_orchestrator`, useful for validating a specific enrolled profile in
isolation. Must be run with `modules/autocar/` as the working directory (their internal paths —
e.g. the OSNet weights path — are relative to it, not this project's repo root):

```bash
cd modules/autocar
python main.py --source 0 --target models/enrolled_Nam.npz --device cpu
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

### `gesture_hand_keypoint` (the TRIGGER gesture method)

Two other gesture methods — `wave_facing_gate` ("condition") and `gesture_trajectory_verifier`
("trajectory_verifier") — used to exist as interchangeable alternatives; both were removed
(confirmed with the user — hand_keypoint is the only TRIGGER gesture method now).

```bash
python -m modules.gesture_hand_keypoint.test_gesture_hand_keypoint path/to/video.mp4

# Live visualization, chained with face_identity + human_detection_roi. Shows the hand skeleton
# colored yellow=OPEN/green=CLOSED/gray=NEITHER, per-finger extended/curled coloring, the red
# dotted palm_height_fraction calibration line, and the current sequence stage.
python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode camera [--camera-index 0]
python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode video --video path/to/video.mp4
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

`modules/target_tracking` and `modules/target_recovery` — REMOVED (2026-08-26). `followme_orchestrator`
drives `modules/autocar` (via `autocar_adapter.py`) for tracking+recovery instead — see
[`architecture.md`](architecture.md)'s Post-trigger flow section and "Testing the vendored
tracking+recovery backbone alone" above. See [`parameters.md`](parameters.md#target_tracking--target_recovery-plans06-plans07--removed-2026-08-26)
for the one capability (periodic re-verify against silent ByteTrack ID reassignment) that has no
replacement in `autocar`.

### `followme_orchestrator` (the FULL pipeline, all 8 plans composed together)

```bash
python -m modules.followme_orchestrator.test_followme_orchestrator path/to/video.mp4

# Live visualization: face-center line, tracked/reacquired bbox, debug_state, should_move, steering angle
python -m modules.followme_orchestrator.visualize_followme_orchestrator --mode camera [--camera-index 0]
python -m modules.followme_orchestrator.visualize_followme_orchestrator --mode video --video path/to/video.mp4
```
Steering output (`should_move`/`steering_angle_degrees`) stays fail-closed (`should_move=False`)
until `camera.fov_degrees` and the `steering` section's `kp`/`ki`/`kd`/`max_steering_angle_degrees`
are calibrated (see [`parameters.md`](parameters.md#steering-plans08)) — trigger detection,
tracking, and recovery all work regardless of steering calibration state.

---

## 3. Recommended order for a first end-to-end smoke test

1. `register_person.py` (or `main.py`, default) — register yourself: FRONT + BACK capture,
   builds both the face registry entry and the `modules/autocar` re-id profile in one session.
2. `visualize_face_identity` — confirm you're matched live.
3. `visualize_human_detection_roi` — confirm your body bbox tracks correctly off your face.
4. `gesture_hand_keypoint`'s `visualize_*` — confirm the gesture triggers GREEN.
5. `main.py --modules pretrigger --show --debug` — the pre-trigger pipeline, all four stages
   together (stops at TRIGGER).
6. `cd modules/autocar && python main.py --source 0 --target models/enrolled_<you>.npz` — confirm
   the vendored tracking+recovery backbone locks onto and re-acquires you in isolation, using the
   profile step 1 just built.
7. `main.py --modules followme --show --debug` — the FULL pipeline in
   one command: trigger → tracking+recovery (`autocar_adapter`) → steering, all composed, with
   every phase's debug overlay AND the steering-direction arrow visible.
   (`modules.followme_orchestrator.visualize_followme_orchestrator` runs the same composed
   pipeline with its own dedicated status readout instead of `main.py`'s.)

Every threshold referenced above starts uncalibrated (`null`) — see
[`parameters.md`](parameters.md) for what each command's module needs filled in before it'll
produce a positive result instead of failing closed.
