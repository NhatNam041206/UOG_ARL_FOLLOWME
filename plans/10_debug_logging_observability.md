# Debug/Observability Logging — Plan & Status

Living doc for the run-logging/debug-observability work, tracked chunk by chunk so each piece can
be verified independently as it lands. Updated after each chunk completes — re-read this file to
see current status rather than scrolling chat history.

See also [plans/11_registration_interactive_console.md](11_registration_interactive_console.md)
— the follow-on plan for operating (not just watching) `register_person.py` over SSH. It reuses
chunk 6 (`--stream`) unchanged and will extend chunk 4's `RunLogger` usage to `register` mode;
its own §5 tracks exactly what that will change here, once implemented.

## Why

Deployment target: this project lives in git, gets `git pull`ed onto a Raspberry Pi, and is
exercised remotely over SSH (no monitor attached to the Pi) during the testing phase, then later
gets called as a module from a larger pipeline. The existing `print()`-per-frame output is
ephemeral (lost when the SSH session closes), unparseable, and missing the fields that actually
explain *why* a decision was made (face match confidence, gesture sequence progress, emergency-stop
reasoning, recovery search state). Goal: every run produces a self-contained, `scp`-able folder
with a structured decision log, a run manifest, and optionally an annotated video — without
needing `--show` or physical access to the Pi's screen.

## Folder structure

```
runs/
  <UTC_ISO8601>_<mode>/          e.g. 20260826T210455Z_followme/
    meta.json                     # run manifest — written at start, finalized at close()
    decisions.jsonl                # one structured JSON record per frame
    debug.avi                      # only when --save-video is passed (chunk 5)
```
`runs/` is gitignored — these are run artifacts, not project source.

## `meta.json` schema

```json
{
  "mode": "followme", "source": "camera:0", "argv": ["--modules", "followme", "--debug"],
  "git_commit": "0af78e7...", "git_dirty": false,
  "thresholds_snapshot": { "...full resolved config/thresholds.yaml at run start..." },
  "start_ts": 1735200000.0, "end_ts": 1735200240.5,
  "frame_count": 4821, "exit_reason": "user_quit",
  "video_saved": true, "video_path": "debug.avi", "video_fps": 24.1, "video_resolution": [640, 480],
  "stream_enabled": false, "stream_url": null
}
```

## `decisions.jsonl` per-frame schema (target — see chunk status for what's wired in so far)

```json
{
  "ts": 1735200000.123, "frame": 4821,
  "face_identity": {"face_found": true, "matched_person_name": "alice", "match_confidence": 0.87},
  "human_detection_roi": {"person_found": true, "detection_confidence": 0.91},
  "gesture": {
    "waving_state": "GREEN", "sequence_stage": "WAITING_OPEN",
    "open_count": 2, "close_count": 2, "total_confirmed_count_session": 3
  },
  "target_tracking": {"state": "TRACKING", "horizontal_offset": 0.12,
                       "last_reverify_score": 0.91, "last_reverify_pass": true},
  "target_recovery": {"status": null, "face_search_fail_count": 0, "elapsed_search_seconds": 0.0},
  "emergency_stop": {"decision": "GO", "reason": null, "zone": null},
  "steering": {"should_move": true, "steering_angle_degrees": 94.3},
  "debug_state": "TRACKING"
}
```

## Chunk breakdown & status

### Chunk 1 — `run_logging.py` (RunLogger) — ✅ DONE

New file [run_logging.py](../run_logging.py) — cross-cutting utility, not a CV module (no model
loading, no own-instance-isolation concerns; same category as `argparse` itself — shared
infrastructure every mode of `main.py` uses).

`RunLogger`:
- `start(mode, argv, source_desc, thresholds_config_path) -> run_dir` — creates
  `runs/<ts>_<mode>/`, snapshots the resolved `thresholds.yaml` into `meta.json`, opens
  `decisions.jsonl` for append.
- `log_frame(**fields)` — appends one JSON line, flushed immediately (crash/SSH-disconnect safety
  — a run folder is readable even if the process is killed mid-run).
- `set_video_info(...)` / `set_stream_info(...)` — record optional artifact info into `meta.json`.
- `close(frame_count, exit_reason)` — finalizes `meta.json`, closes the file handle.

**Tests**: [test_run_logging.py](../test_run_logging.py) — 7 pytest unit tests, no CV models or
video files required (pure filesystem/JSON logic). All passing.

**Verify yourself**:
```bash
python -m pytest test_run_logging.py -v
```
Expect `7 passed`.

### Chunk 2 — gesture sequence open/close counts — ✅ DONE

`open_count`/`close_count` are a pure lookup on `sequence_stage` — no new state machine fields
needed, since the fixed OPEN→CLOSE→OPEN→CLOSE progression already determines them:

| stage | open_count | close_count |
|---|---|---|
| `WAITING_OPEN` | 0 | 0 |
| `WAITING_CLOSE_1` | 1 | 0 |
| `WAITING_OPEN_2` | 1 | 1 |
| `WAITING_CLOSE_2` | 2 | 1 |
| `CONFIRMED` | 2 | 2 |

Added as `STAGE_COUNTS` in
[modules/gesture_hand_keypoint/sequence_state_machine.py](../modules/gesture_hand_keypoint/sequence_state_machine.py).
Also added `total_confirmed_count` — a per-`track_id` cumulative counter (session lifetime,
reset only by `release_track()`, never by a sequence reset/timeout) incremented in
[modules/gesture_hand_keypoint/pipeline.py](../modules/gesture_hand_keypoint/pipeline.py) whenever
a sequence reaches `CONFIRMED`. Both are threaded through `PipelineResult` and out through the
public `GestureMethodResult` in
[modules/gesture_hand_keypoint/interface.py](../modules/gesture_hand_keypoint/interface.py).

**Tests**:
[modules/gesture_hand_keypoint/test_sequence_counts.py](../modules/gesture_hand_keypoint/test_sequence_counts.py)
— 6 pytest unit tests driving a real `SequenceStateMachine` with synthetic `OPEN`/`CLOSED` shape
values (no MediaPipe model or video needed, since the state machine itself has no CV dependency).
Covers: full-sequence progression, wrong-start-shape non-advance, height-gate-failure reset,
timeout reset. All passing.

**Verify yourself**:
```bash
python -m pytest modules/gesture_hand_keypoint/test_sequence_counts.py -v
```
Expect `6 passed`.

Note: `total_confirmed_count`'s increment and the full `GestureHandKeypointPipeline.evaluate()`
path (which needs the real MediaPipe hand detector) aren't covered by an automated test here —
this dev environment doesn't have `mediapipe` installed (pre-existing, unrelated to this change;
confirmed via `python -c "import mediapipe"` failing before any of these edits). All touched files
were verified with `python -m py_compile` (no syntax errors). To verify `total_confirmed_count`
end-to-end, run the module's existing video-based smoke test
(`python -m modules.gesture_hand_keypoint.test_gesture_hand_keypoint <video>`) on hardware with
mediapipe installed and watch the printed line across multiple full waves — new fields aren't
printed by that script yet, so watching `sequence_stage` cycle back to `WAITING_OPEN` repeatedly is
the current proxy until chunk 4 wires the new JSONL logger in.

### Chunk 3 — orchestrator debug-info plumbing — ✅ DONE (with a schema correction — read this)

**Correction to the plan above**: while wiring this, I found the original `decisions.jsonl`
schema was wrong about the pipeline's actual shape. `emergency_stop` is **never called** anywhere
in `main.py` or `modules/followme_orchestrator` — it only has its own standalone test script
([modules/emergency_stop/test_estop.py](../modules/emergency_stop/test_estop.py)), confirmed by
grepping for `emergency_stop` across both. Logging it would mean writing a permanently-null block
that looks like real telemetry but isn't. Separately, `target_tracking`/`target_recovery` (the two
modules those field names came from) also aren't what `followme_orchestrator` actually uses — its
own docstring says so directly: *"autocar_adapter's TargetLock folds tracking AND recovery into
one state machine... No separate recovery module/call site exists anymore."* The real live object
is `autocar_adapter.TrackingResult` (`target_locked`, `state` ∈ `TRACKING`/`SEARCHING`/`LOST`,
`horizontal_offset`, `just_reacquired`) — no `face_search_fail_count`/`elapsed_search_seconds`/
`last_reverify_score` exist on the path this pipeline actually runs.

**Corrected per-frame schema** (supersedes the draft at the top of this doc):
```json
{
  "face_identity": {"face_found": true, "matched_person_name": "alice", "match_confidence": 0.87},
  "human_detection_roi": {"person_found": true, "detection_confidence": 0.91},
  "gesture": {"waving_state": "GREEN", "sequence_stage": "WAITING_OPEN",
              "open_count": 2, "close_count": 2, "total_confirmed_count_session": 3},
  "tracking": {"target_locked": true, "state": "TRACKING", "horizontal_offset": 0.12,
               "just_reacquired": false}
}
```
Each block is `null` when that phase didn't run this particular frame (mirrors `draw_debug()`'s
own "only whichever phase(s) actually ran" convention) — e.g. `face_identity`/`human_detection_roi`/
`gesture` are all `null` while a tracking episode is active, since the pre-trigger phase doesn't
run then.

**If you actually want `emergency_stop` in the loop**, that's a separate, real integration gap —
the module exists and works standalone but nothing calls it from the live pipeline. Worth a
dedicated task rather than folding into this logging work; flag if you want that scoped.

**Implementation**: new file
[modules/followme_orchestrator/debug_snapshot.py](../modules/followme_orchestrator/debug_snapshot.py)
— a pure function `build_debug_snapshot(face_result, person_result, gesture_result,
tracking_result)`, deliberately isolated with **zero imports from any CV module**, which is what
makes it unit-testable without mediapipe/onnxruntime installed (the rest of this package cannot be
imported in this dev environment — see chunk 2's note). `pipeline.py` gained a `debug_snapshot()`
method that calls it with the frame's already-computed `_debug_pretrigger`/`gesture_adapter`/
`_debug_tracking_result` state; `gesture_adapter.py` gained a `last_result` property (the full
`GestureMethodResult`, not just the narrowed `(bool, str)` `evaluate()` already returns) — guarded
so a stale result from a previous frame/face is never reported as current (see the code comment in
`pipeline.py`'s `debug_snapshot()`). Exposed publicly as
`modules.followme_orchestrator.interface.debug_snapshot()`.

**Tests**:
[modules/followme_orchestrator/test_debug_snapshot.py](../modules/followme_orchestrator/test_debug_snapshot.py)
— 5 pytest unit tests using plain duck-typed fake result objects (no CV deps needed at all, unlike
the rest of this package). All passing.

**Verify yourself**:
```bash
python -m pytest modules/followme_orchestrator/test_debug_snapshot.py -v
```
Expect `5 passed`. All touched files also verified with `python -m py_compile` (no syntax errors);
end-to-end `debug_snapshot()` output against a real video needs mediapipe/onnxruntime installed,
which this dev environment doesn't have — verify on the Pi/dev machine that has them via
`modules.followme_orchestrator.test_followme_orchestrator` once chunk 4 prints/logs it.

### Chunk 4 — `main.py` wiring (RunLogger into both pipeline modes, `--log-dir` flag) — ✅ DONE

Both `run_pretrigger_pipeline()` and `run_followme_pipeline()` now take a `logger: RunLogger`
parameter. `main()` creates one `RunLogger(log_root=args.log_dir)` (new `--log-dir` flag, default
`"runs"`) right before dispatching, calls `.start(args.modules, sys.argv[1:], source_desc,
args.config)`, and prints the resulting run directory — so a live SSH session immediately shows
where its logs are landing, not just the existing mode banner.

- **followme**: logs `frame`, `ts`, `debug_state`, `should_move`, `steering_angle_degrees`, plus
  `**debug_snapshot()` spread in directly (chunk 3's `face_identity`/`human_detection_roi`/
  `gesture`/`tracking` blocks) — one `logger.log_frame(...)` call per frame, right after the
  existing `print(line)`.
- **pretrigger**: logs `frame`, `ts`, `mode="pretrigger"`, and a `people` list — one entry per
  registered face evaluated that frame (unlike followme, pretrigger is explicitly multi-person per
  spec §1, so this doesn't collapse to a single block). Needed a `last_result` property added to
  `main.py`'s own `_GestureMethodAdapter` (mirrors chunk 3's addition to the orchestrator's
  adapter) so `sequence_stage`/`open_count`/`close_count`/`total_confirmed_count_session` are
  available structured, not just baked into the printed `extra` string.
- **Exit tracking**: both loops now track `exit_reason` (`"completed"` | `"user_quit"` on the `q`
  keypress or `Ctrl+C` | `"error"` on an uncaught exception, re-raised after recording) and call
  `logger.close(frame_count, exit_reason)` in `finally`, so `meta.json` always reflects how the run
  actually ended.

**Correction, added later (prompted by "are there any legitimate ways to exit when --stream is
enabled")**: the claim above — "even after Ctrl+C-style interruption" — was WRONG as originally
written. `except Exception:` does **not** catch `KeyboardInterrupt` (it's a `BaseException`, not
an `Exception`) — the `finally` cleanup (camera release, `stream.stop()`, `logger.close()`) did
always run correctly, but the exception itself still propagated all the way out as a raw
traceback, with no clean message and no `"user_quit"` recorded. This mattered in practice
specifically for `--stream`-only headless runs (`--mode camera`, no `--show`): with no `--show`,
`cv2.waitKey()` is never even called, so there is no `q`-keypress path at all — `Ctrl+C` was the
**only** way to stop such a run early, and it produced a stack-trace dump instead of a clean exit.
Fixed: added an explicit `except KeyboardInterrupt:` clause (before `except Exception:`) to both
`run_pretrigger_pipeline()` and `run_followme_pipeline()` — prints `"Interrupted, stopping."` and
records `exit_reason="user_quit"`, same as the `q`-keypress path, instead of letting it propagate.

**Tests**: none new for this chunk specifically — it's wiring, not new logic, and covered
transitively by chunks 1/3's own tests (the pieces being wired together). Verified with
`python -m py_compile main.py` and `python -c "import main"` (main.py's own top-level imports are
just `cv2`/`yaml`/`run_logging` — the heavy CV imports are deferred inside each pipeline function,
so this import check passes even without mediapipe/onnxruntime installed) and
`python main.py --help` to confirm the new `--log-dir` flag is wired into argparse correctly. The
`KeyboardInterrupt` fix specifically could not be given an automated test in this dev environment
(both pipeline functions need the full mediapipe/onnxruntime stack to even construct) — verified
by code inspection only; see `test_register_person_keyboard_interrupt.py`
(plans/11_registration_interactive_console.md) for the equivalent fix in `register_person.run()`,
which COULD be fully tested since that function's camera/detector are injectable.

**Verify yourself** (needs a real camera/video + all CV deps installed, which this dev environment
lacks):
```bash
python main.py --mode video --video <path> --modules followme --show --debug
```
then check `runs/<newest>/meta.json` and `runs/<newest>/decisions.jsonl` were created and populated.

### Chunk 5 — `--save-video` (MJPG/avi writer) — ✅ DONE

New `--save-video` flag. Two things had to change together, not just "add a VideoWriter":

1. **The overlay-drawing was gated on `args.show` alone** — meaning a headless SSH run (no
   `--show`) never drew anything onto `frame` at all, so saving that frame to video would have
   produced a blank/unannotated clip. Fixed by introducing `want_overlay = args.show or
   args.save_video` and switching every overlay-drawing `if args.show:` to `if want_overlay:` in
   both pipelines, while keeping `cv2.imshow()`/`waitKey()` gated on `args.show` alone (still no
   window pops up unless you actually asked for one).
2. New `open_debug_video_writer(cap, run_dir)` in `main.py` — opens `runs/<run_id>/debug.avi` with
   the `MJPG` fourcc (not `mp4v`/H.264 — deliberate, see the function's own docstring: no hardware
   encoder on the Pi, software H.264 is meaningfully heavier than Motion-JPEG). Falls back to
   `20.0` fps / `640x480` when `cap.get(...)` reports `0` (common for webcams) rather than
   producing an unplayable 0-fps file. Both pipelines open it once (if `--save-video`), call
   `logger.set_video_info(video_path, fps, resolution)` so `meta.json` records what was actually
   saved, write every frame after overlays are drawn, and release it in `finally`.

**Tests**: [test_main_video_writer.py](../test_main_video_writer.py) — 4 pytest unit tests against
a real `cv2.VideoWriter` (cv2 itself has no heavy-model dependency, unlike mediapipe/onnxruntime)
using a fake `cap` object that only implements `.get()`. Covers: reported fps/resolution used
correctly, fps-unreported fallback, resolution-unreported fallback, frames actually written to
disk. All passing.

**Verify yourself**:
```bash
python -m pytest test_main_video_writer.py -v
```
Expect `4 passed`. End-to-end (`--save-video` producing a real annotated clip from the live
pipeline) needs the same real-camera/full-deps environment as chunk 4's verification.

### Chunk 6 — `--stream` (dev-only, throttled MJPEG-over-HTTP) — ✅ DONE

Scope grew beyond the original followme-only plan: registration (`register_person.py`) needed the
same capability, and along the way I found its "headless" CLI path (`register_person.run()`,
used by `main.py --modules register --person-name`) wasn't actually headless at all —
`_capture_phase_cli()` called `cv2.imshow`/`cv2.waitKey` unconditionally, no `--show` gate
existed. That's a real gap for the "test over SSH" goal, not just missing streaming, so it got
fixed as part of this chunk rather than left for later.

**New file** [debug_stream.py](../debug_stream.py) — stdlib-only (`http.server.ThreadingHTTPServer`,
no new dependency — matches `docs/technologies.md`'s "hand-implemented over a library dependency"
preference). `DebugStreamServer`: `start(port=8080) -> url` (binds `127.0.0.1` only, background
daemon thread), `update_frame(frame)` (throttled — only every Nth call, default every 3rd, actually
re-encodes/publishes, so it doesn't compete with inference for CPU), `stop()`. Serves
`multipart/x-mixed-replace` JPEG at `/stream.mjpg` plus a trivial `/` viewer page.

**`register_person.py`**: `_capture_phase_cli()`/`run()` now take `show`/`stream` params —
`want_overlay = show or stream is not None`, so the capture loop can run with zero display
attached. `run()`'s own default (`show=True`) preserves the standalone CLI's historical
always-show behavior unchanged; only `main.py`'s dispatch passes `show=args.show` explicitly,
aligning `register` with `pretrigger`/`followme`'s existing `--show` convention *within main.py
specifically* — a deliberate, documented behavior change (see `docs/commands.md`'s `--show` row).
Added `--stream` to `register_person.py`'s own `parse_args()` too.

**`main.py`**: new `--stream` flag, applies to `pretrigger`/`followme`/`register --person-name`.
`main()` starts one `DebugStreamServer` (when passed) right alongside the existing `RunLogger`,
prints the URL the same way `logging to runs/...` prints, stops it in `finally`. Both pipeline
functions gained a `stream` parameter; `want_overlay` now also fires on `stream is not None` (not
just `--show`/`--save-video`), and `stream.update_frame(frame)` is pushed alongside
`video_writer.write(frame)` — one drawn buffer, all three consumers (`--show`/`--save-video`/
`--stream`) read from it.

**Tests**: [test_debug_stream.py](../test_debug_stream.py) — 7 pytest unit tests against a real
`DebugStreamServer` bound to an OS-assigned port (`port=0`), using real HTTP requests
(`urllib.request`) — no camera/model dependency. Covers: localhost-only binding, index page,
404 on unknown path, a pushed frame actually arriving as a real JPEG over the MJPEG stream,
throttling skipping non-multiple pushes, `update_frame()`/`stop()` being safe to call before
`start()`/twice. All passing.

**Verify yourself**:
```bash
python -m pytest test_debug_stream.py -v
```
Expect `7 passed`. `register_person.py` and `main.py` both verified with `python -m py_compile`
and a plain `import` (no missing-dependency errors — `register_person.py` imports cleanly in this
dev environment, unlike anything touching mediapipe/onnxruntime). End-to-end (an actual browser
watching a live camera run over an SSH port-forward) needs a real camera + full deps, same
caveat as chunks 4-5.

**Two bugs found and fixed after the initial chunk 6 landing, both from live testing (`python
main.py --stream` still opening a window):**
1. `register_person.py`'s own standalone `main()` never passed `show=` to `run()`, so it always
   fell back to `run()`'s old `show=True` default regardless of `--stream` — added `--show` to
   `register_person.py`'s own argparse, flipped `run()`'s default to `show=False` (both real call
   sites now pass it explicitly, so the old default was a dangling footgun).
2. `--stream` had no effect at all — silently — when `--modules register` was used without
   `--person-name`, since that combination opens the Tkinter `RegistrationApp` instead, which
   never reads `args.stream`. Worse than a silent no-op: a genuinely headless SSH session (no X11
   forwarding — the exact case `--stream` exists for) would then fail/hang on
   `RegistrationApp.mainloop()` needing a real display. Fixed as a fail-fast `parser.error()` at
   argument-parsing time in `main.py`, before any Tkinter code runs, rather than a silent ignore.

## Running everything already implemented (chunks 1-6) at once

```bash
python -m pytest test_run_logging.py test_main_video_writer.py test_debug_stream.py \
    modules/gesture_hand_keypoint/test_sequence_counts.py \
    modules/followme_orchestrator/test_debug_snapshot.py -v
```
Expect `29 passed`.
