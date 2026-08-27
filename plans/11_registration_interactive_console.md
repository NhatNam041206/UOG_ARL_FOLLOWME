# Registration Interactive Console — Plan & Architecture

Companion to [plans/10_debug_logging_observability.md](10_debug_logging_observability.md) (the
run-logging/`--stream` work) — this plan covers the piece that came after: giving
`register_person.py` a way to be **operated** (list/register/delete), not just **watched**, over
a plain SSH session with no display attached. Written before implementation, per the same
propose-then-build rhythm as plans/10 — status/verification per chunk gets filled in as each
chunk lands, so re-read this file (not just chat history) to see current progress.

## 1. Background — why this exists

The Raspberry Pi deployment target is SSH-only, headless: no monitor attached, tested by `git
pull`ing onto the Pi and driving it remotely. `register_person.RegistrationApp` (the Tkinter CRUD
app — list/register/re-capture/delete people) **requires a real display** and simply cannot run
there at all. The existing headless path (`register_person.run()`, `--person-name`) covers
Create/Update (registering one named person) but has no equivalent for Read (list who's
registered) or Delete — those only exist inside the Tkinter app today. This plan closes that gap
for the headless case specifically, without touching how registration works when a display *is*
available (the Tkinter app is untouched, still the recommended path when you have one).

## 2. Design history — why this ended up as two CLIs instead of a web app

Recorded here deliberately, not just in chat, so a future change to this feature starts from the
same reasoning rather than re-deriving it (or re-proposing something already tried and rejected).

**v0 — already built (plans/10 chunk 6): `--stream`, a read-only MJPEG viewer.** Solves "watch
the pipeline live over SSH" for `pretrigger`/`followme`/`register --person-name`. Does **not**
solve "control the pipeline over SSH" — it's purely a video feed, no input capability at all.
This piece is reused unchanged by everything below — see §3.1.

**v1 — considered and rejected: an interactive local web app.** The natural next thought was
"add buttons to the stream's web page" — a small HTTP server with `POST /api/register`,
`POST /api/delete`, etc., replacing the Tkinter UI with a browser UI reachable via an SSH
port-forward. Analysis surfaced real problems, not just extra work:
- `cv2.VideoCapture` isn't safe to read from multiple threads, so a dedicated worker thread would
  need to own the camera, with HTTP request threads only ever reading shared state — a command
  queue, a busy-flag, and a `/api/status` polling endpoint all become necessary just to coordinate
  that.
- Mutating actions (register/delete) *must* be `POST`, never `GET` — a classic, well-known web
  mistake (a `GET`-triggered delete is trivially fired by accident: prefetch, a bookmark, a stray
  `<img src>`).
- No authentication at all — safety would rest entirely on the server binding to `127.0.0.1` and
  nothing else, which has to be hard-coded (no `--host` override ever), not just a default.
- `multipart/x-mixed-replace` MJPEG streaming is inconsistently supported on mobile Safari.
- The web app's lifecycle (start once, run indefinitely, handle many actions across one session)
  doesn't match `main.py`'s "run once and exit" shape used everywhere else — it would need its
  own long-running entry point, more like the existing Tkinter branch (`app.mainloop()`, blocks
  until closed) than the headless one-shot `run()` path.

**v2 — chosen: reuse SSH itself as the control channel.** The key realization: SSH already gives
an authenticated, single-user, inherently *sequential* I/O channel — a REPL's `input()` call
naturally serializes commands the same way a web server's concurrent request threads do not. This
removes nearly the entire v1 problem list for free:
- No worker thread or command queue — each REPL command runs `run()` to completion synchronously
  before the next command is even read, exactly like the CLI already does for one registration
  today. "Only one thing touches the camera at a time" is true by construction, not by a lock.
- No `POST`/`GET` question — there is no HTTP mutation surface at all for control.
- No auth story to build — SSH already is the boundary.
- `--stream` keeps its *exact* existing role, completely unmodified — it's still pure output, now
  just fed by a REPL command instead of a startup flag. See §3.1.

## 3. Final architecture

### 3.1 Three independent channels — pick the right one, don't blur them

| Channel | What it is | Can mutate state? | Status |
|---|---|---|---|
| **Control** | CLI 1 — `register_person.run_interactive()`, a REPL on stdin/stdout | **Yes — the only one that can** | Planned, §5 |
| **Visual feedback** | The existing `DebugStreamServer` (`--stream`, plans/10 chunk 6) | No — read-only video | Already built, reused unchanged |
| **Audit / log trail** | CLI 2 — new `tail_log.py`, tails `runs/<id>/decisions.jsonl` | No — read-only text | Planned, §5 |

Each of the three can run independently — CLI 1 works with no `--stream` and no `tail_log.py`
running; `--stream` and `tail_log.py` are purely optional observers of whatever CLI 1 does.

### 3.2 Why three separate channels instead of one merged terminal

A live-scrolling log or video feed sharing a terminal with an `input()` prompt is bad UX on its
own merits (a typed command gets visually interrupted mid-keystroke by async output landing in
the same stream) — but the deeper reason is the same read/write separation this codebase already
enforces elsewhere (own-instance isolation, fail-closed-by-default, `debug_snapshot()` being
explicitly NOT part of the typed `FollowMeCommand` contract in plans/10 chunk 3): a new feature
should have one clearly-owned writer, and everything else should be a strictly observational
reader. Two terminals (or one SSH connection + `tmux`) enforce that split for free, in the
transport itself, not just in code discipline.

### 3.3 Properties / invariants — read this before extending the feature

These are the rules that must stay true for future changes to this feature to compose safely,
not just a status report of what chunk did what:

1. **CLI 1 is the ONLY writer.** `--stream` and `tail_log.py` never mutate registry state, ever.
   If a future change needs the stream or the log tailer to trigger an action, that's a sign the
   architecture needs to change deliberately (back toward something like the rejected v1), not a
   thing to bolt on quietly to a "read-only" component.
2. **One `run()` call at a time, enforced by the REPL's own synchronous structure, not a lock.**
   Do not parallelize registration (e.g. "register two people at once") without redesigning
   camera ownership from scratch — `cv2.VideoCapture` is not safe to read from multiple threads,
   full stop (this is why v1 needed a worker thread + queue at all).
3. **CLI 1 must call the SAME `register_person.run()` the headless `--person-name` path already
   calls — never a second implementation of the capture state machine.** Any future capture-logic
   change (new sample-count knob, new gate) then applies to both paths automatically. This mirrors
   how "Update" (re-capture) was never its own code path either — see `registration_data.py`'s
   own CRUD-mapping docstring.
4. **`stream=None` must always be a safe no-op everywhere in this design** — already true of
   `DebugStreamServer.update_frame()`'s existing contract (plans/10 chunk 6); nothing new to
   enforce here, just don't break it.
5. **`runs/<id>/` is now a THREE-consumer convention** (`pretrigger`, `followme`, and — once
   chunk 7 below lands — `register --interactive`), not a followme-specific thing. Any future
   4th log-producing mode should call `RunLogger.start(mode, argv, source_desc, config_path)` the
   same way, not invent a parallel convention.
6. **`tail_log.py` must stay schema-agnostic.** `decisions.jsonl` content differs by mode
   (`pretrigger` logs `people: [...]`, `followme` logs `face_identity`/`gesture`/`tracking`
   blocks, `register --interactive` will log yet another shape — see §5 chunk 7). The tailer
   pretty-prints whatever keys a record has generically; it must never hard-code field names tied
   to one specific mode, or it silently breaks the moment a different mode's file is pointed at
   it.
7. **MVP scope, deliberately deferred, not forgotten** (mirrors plans/10's own chunk-status
   habit of naming what's NOT done, not just what is):
   - No per-command sample-count override in the REPL — uses the session-wide
     `--front-samples`/`--back-samples` the console was started with.
   - No `stop`/cancel REPL command in v1 — a REPL command already blocks synchronously, so
     `Ctrl+C` is the natural cancel; verify it's caught and produces a clean message, not a raw
     stack trace, when implementing chunk 8.
   - No camera auto-reconnect on failure.
   - No confirmation beyond a plain `y`/`N` prompt before delete (no undo, no soft-delete) — same
     reasoning as the rejected web design's "no auth beyond localhost binding": acceptable because
     there's no multi-user boundary to protect here at all.
8. **`follow <name>` is the console's ONLY way to select a followme target — `--then-followme`
   deliberately never gets one** (added post-chunk-10, see the fixes section below §4). The
   `RegistrationApp._on_follow_me()`/`args.person_name`-with-`--then-followme` pattern (validate
   `ready_for_followme`, hand a name back to the caller, caller falls through into the shared
   followme camera loop) is the ONE selection mechanism this whole file uses across all three
   register paths — any future 4th path should reuse that exact shape (a `chosen_name` value the
   caller reads after the fact), not invent a new one. `run_interactive()`'s return contract is
   therefore `(exit_code, chosen_name)`, not a plain `int` — a real, load-bearing part of its
   signature now, not an implementation detail.

## 4. Chunk breakdown & status

Same "each by each, verify before moving on" rhythm as plans/10.

### Chunk 7 — wire `RunLogger` into `register` mode — ✅ DONE (for `--person-name`; `--interactive` lands with chunk 8)

Scope as decided: wired into the headless `--person-name` path (both `main.py`'s dispatch and
`register_person.py`'s own standalone CLI) — the Tkinter path stays untouched, per §1's own
boundary. `--interactive`'s own record shapes (chunk 8) will reuse the same `RunLogger`/schema
conventions established here, not invent new ones.

**Implementation**: `_capture_phase_cli()` now takes `frame_idx`/`logger` and returns
`(saved, frame_idx)` — `frame_idx` threads across BOTH phase calls (front then back) inside
`run()`, so `decisions.jsonl` frame numbers are globally monotonic across one registration, not
reset per phase (mirrors `main.py`'s own `frame_idx` convention in its two pipeline loops). Each
logged record: `frame`, `ts`, `mode="register"`, `stage` (`"countdown"`/`"capture"`),
`person_name`, `phase` (`"front"`/`"back"`), `saved`, `samples_needed`, `person_count` (`None`
during countdown, since the person-count detector doesn't run until the capture loop). `run()`
itself now tracks `exit_reason` (`"completed"`/`"stopped_early"`/`"error"`) and calls
`logger.close(...)` in its own `finally` — mirrors `run_pretrigger_pipeline`/
`run_followme_pipeline`'s exact pattern from plans/10 chunk 4. Both call sites
(`main.py`'s dispatch, `register_person.py`'s own `main()`) create+`start()` the `RunLogger`
themselves, matching how they already create+start `DebugStreamServer` right next to it — `run()`
only closes what it's handed, never starts its own.

Also added `--log-dir` to `register_person.py`'s own standalone `parse_args()` (didn't exist
before — only `main.py` had it), and fixed the two staleness spots flagged in §5 below
(`main.py`'s `--log-dir` help text, `docs/commands.md`'s `--log-dir` row and "Reviewing a run"
intro line) — both now correctly say `register --person-name` is a third `RunLogger` consumer.

**Tests**: [test_register_person_logging.py](../test_register_person_logging.py) — 4 pytest unit
tests using a fake `cv2.VideoCapture`-alike and a fake person-count detector (no real camera or
model needed — `registration_data.LiveSubjectDetector`'s real YOLO-pose model is never
constructed). `registration_data.CAPTURES_DIR` monkeypatched to a temp dir so nothing touches the
real `registration_captures/`; all three capture-timing constants monkeypatched to `0.0` so every
eligible frame saves immediately — deterministic, frame-count-driven, no wall-clock dependency.
Covers: exact save count + record shape, `frame_idx` threading correctly across two phase calls
(not reset per phase), `logger=None` being a safe no-op, a camera that never produces a frame not
crashing the logging path. All passing.

**Verify yourself**:
```bash
python -m pytest test_register_person_logging.py -v
```
Expect `4 passed`. `register_person.py`/`main.py` also verified with `python -m py_compile` and a
plain `import register_person` (imports cleanly in this dev environment — no mediapipe/
onnxruntime dependency touched by this chunk). End-to-end (`python main.py --modules register
--person-name X --log-dir ...` against a real camera, checking the resulting `runs/<id>/` folder)
needs a real camera + full deps, same caveat as plans/10's own chunks 4-6.

### Chunk 8 — `register_person.run_interactive()` — ✅ DONE

The REPL itself: `list` / `register <name>` / `delete <name>` / `quit`/`exit`/Ctrl+D, per §3's
architecture. Also wired into `register_person.py`'s own standalone CLI as `--interactive`
(mutually exclusive with the positional `person_name` argument — checked explicitly in `main()`,
errors out rather than picking one silently), consistent with chunk 7's precedent of wiring both
the standalone CLI and (eventually, chunk 10) `main.py`'s dispatch in the same pass.

**A real design conflict surfaced and got fixed while implementing this**: `run()` (chunk 7)
closes whatever `RunLogger` it's given in its own `finally`, correct for the one-shot
`--person-name` case but wrong here — the REPL shares ONE logger/`decisions.jsonl` across MANY
`register <name>` commands in a session, so the first command's `run()` call would have closed
the file out from under every subsequent command. Fixed with a new `close_logger_on_exit: bool =
True` parameter on `run()` (default preserves the one-shot callers' existing behavior unchanged);
`run_interactive()` passes `close_logger_on_exit=False` and closes the shared logger itself, once,
when the REPL session ends — ownership matches invariant §3.3.1 ("CLI 1 is the only writer") and
§3.3.5 (`runs/<id>/` convention) exactly: the REPL, not any individual command, owns the session's
log lifecycle.

**Other decisions**: `exit_reason` defaults to `"user_quit"` (not `"completed"`) since every
normal way this loop ends — `quit`/`exit`/EOF/Ctrl+C — genuinely is a user-initiated stop, unlike
`pretrigger`/`followme` where "the video ran out" is a distinct, real `"completed"` case. `list`/
`delete` log their own lightweight records (`stage="list"`/`"delete"`, a REPL-level
`command_index` counter — deliberately NOT reusing the video-oriented `frame` key for
non-camera events); `register <name>` does NOT get an extra wrapper record — `run()`'s own
internal per-frame `countdown`/`capture` records (chunk 7) already fully cover what happened,
logging a duplicate summary on top would add noise without new information. Per §3.3.7's MVP
scope: each `register <name>` command's own frame numbers restart at 0 (not threaded globally
across a whole REPL session) — accepted deliberately, not an oversight (see the function's own
docstring for why this is still unambiguous: `ts` + `person_name`/`phase`/`command_index` fully
disambiguate regardless).

**Tests**: [test_register_person_interactive.py](../test_register_person_interactive.py) — 10
pytest unit tests, scripting `input()` via `unittest.mock.patch` and monkeypatching
`registration_data.list_people`/`delete_person` and `register_person.run` itself — no real
camera/model touched anywhere. Covers: `list` (with and without people, logging), `register`
(OK/FAILED reporting, confirms `close_logger_on_exit=False` is actually passed), `delete`
(confirmed and declined paths, logging), unknown command, EOF, `Ctrl+C` exiting cleanly with no
raised traceback and the correct `exit_reason` in `meta.json`, plain `quit` also recording
`exit_reason="user_quit"`. All passing.

**Verify yourself**:
```bash
python -m pytest test_register_person_interactive.py -v
```
Expect `10 passed`. `register_person.py` also reverified with `python -m py_compile` and a plain
`import` (clean, no missing-dependency errors) and `--help` (confirms `--interactive` is wired
into argparse correctly). End-to-end (an actual SSH session typing `list`/`register X`/`delete X`
against a real camera) needs a real camera + full deps, same caveat as every other chunk in this
plan and in plans/10.

### Post-chunk-8 fixes — prompted by two direct questions, both answered by checking, not assuming

**1. "Are there any legitimate ways to exit the program when `--stream` is enabled?"** Audited and
found a real gap, shared with `plans/10` (see that file's own Chunk 4 correction — the same bug
existed in `run_pretrigger_pipeline()`/`run_followme_pipeline()`): `run()`'s `except Exception:`
clause never caught `KeyboardInterrupt`, so `Ctrl+C` during a `--stream`-only headless capture
(`show=False`, so no `q`-keypress path exists at all) propagated as a raw traceback instead of
exiting cleanly — even though the camera/logger cleanup in `finally` always ran correctly either
way. Fixed with an explicit `except KeyboardInterrupt:` clause in `run()`, positioned so it also
resolves a second question for free: **where** it's caught controls **what** Ctrl+C cancels.
Caught inside `run()` itself (not duplicated in `run_interactive()`) means:
- `Ctrl+C` during an active `register <name>` capture → cancels just that registration, returns
  `1`, records `exit_reason="stopped_early"`. For the one-shot CLI path there's nothing left to
  do afterward anyway (program just ends); for the REPL, `run_interactive()`'s loop simply prints
  `FAILED` and returns to its own `>>>` prompt — the session keeps running.
- `Ctrl+C` while just sitting idle at the REPL's `>>>` prompt (no registration in progress) → a
  SEPARATE catch, already in `run_interactive()` itself (chunk 8) — exits the whole console.

Same keystroke, different meaning, entirely by design: whichever operation is actually running
gets cancelled, nothing more. Verified with
[test_register_person_keyboard_interrupt.py](../test_register_person_keyboard_interrupt.py) — 2
tests, a fake camera that raises `KeyboardInterrupt` mid-capture, confirming `run()` returns `1`
(not a raised exception), the camera's `.release()` still gets called, and `meta.json` records
`exit_reason="stopped_early"`.

**2. "Does the interactive CLI actually work with `--stream` enabled, and does it keep logs OUT of
its own terminal (leaving `tail_log.py`, chunk 9, as the real way to watch them)?"** Both verified
directly rather than asserted — see
[test_register_person_interactive_streaming.py](../test_register_person_interactive_streaming.py):
- `test_register_command_in_repl_actually_streams_frames` drives `run_interactive()` with a REAL
  `DebugStreamServer` (only the camera/detector are faked) and confirms a JPEG frame actually
  lands in the stream's buffer during a `register <name>` command typed into the REPL — not just
  asserted from reading the code, actually exercised end-to-end (minus the camera/model
  themselves, per this dev environment's usual constraint).
- `test_repl_stdout_never_contains_raw_jsonl_log_lines` confirms no `decisions.jsonl` record ever
  gets printed to the REPL's own stdout (searches captured output for the JSON keys every logged
  record has) — while separately confirming the records genuinely were written to the log FILE.
  Architecturally this was already true by construction (§3.1/§3.2: the REPL only ever calls
  `print()` for user-facing messages, `logger.log_frame()` writes silently to disk, nothing wires
  one into the other) — this test exists to make that guarantee checked, not just asserted, so a
  future change that accidentally added a debug `print(record)` next to a `log_frame()` call would
  fail a test instead of silently regressing.

**Not yet true, worth stating plainly**: `main.py --modules register --interactive` does **not**
exist yet — only `register_person.py --interactive` (its own standalone CLI) has this today.
Wiring `--interactive` into `main.py`'s own dispatch is chunk 10, still pending.

### Chunk 9 — `tail_log.py` — ✅ DONE

The log-tailing CLI, as scoped: opens a `decisions.jsonl` (or its containing run directory —
`_resolve_path()` accepts either), prints the last `--lines` existing records (default 10, `0` =
skip straight to following), then polls for and prints new ones as they arrive. `--latest`
auto-picks the run whose `decisions.jsonl` was **most recently written to**, not the most recently
created — ranked by the file's own mtime, not the containing directory's, because on POSIX a
directory's mtime does NOT update when a file inside it is merely appended to (only on
create/delete/rename); ranking by directory mtime would have silently misidentified an old,
finished run as "latest" over a newer one still actively being written. `Ctrl+C` caught cleanly
(`"Stopped watching."`, not a raised traceback) — same convention as everywhere else in this
project.

**Schema-agnostic per invariant §3.3.6, actually enforced by the implementation, not just
promised**: `format_record()` makes zero assumptions about which keys exist — it renders `ts` (if
present) as a leading timestamp, then every other top-level key generically as `key=value`,
compacting nested dicts/lists to single-line JSON rather than Python's `repr()` (kept
copy-pasteable back into a JSON tool). Verified against three genuinely different shapes in
tests: flat `followme`-style fields, a nested `tracking` dict, and a `pretrigger`-style `people`
list — the same function handles all three without any mode-specific branch.

**Tests**: [test_tail_log.py](../test_tail_log.py) — 13 pytest unit tests, pure filesystem/string
logic, no camera/model/network dependency. Covers: `format_record()` across the three shapes
above (with/without a timestamp), `find_latest_run()`'s mtime-of-the-file-not-the-directory
ranking (including a run directory that never got a `decisions.jsonl` at all — e.g. a crash
before `RunLogger.start()` finished — being correctly skipped rather than crashing or winning by
default), `_resolve_path()`'s directory-vs-file forms, a missing path returning a clean error
(not an exception), the actual follow loop picking up a genuinely-appended line
(`time.sleep` monkeypatched as the deterministic test hook — see the test's own comment for why),
`--lines 0` skipping existing content, and a malformed/torn line being shown raw instead of
crashing the tailer. All passing.

**Verify yourself**:
```bash
python -m pytest test_tail_log.py -v
```
Expect `13 passed`. Also manually verified end-to-end in this dev environment specifically (no
mediapipe/onnxruntime dependency exists anywhere in this file, unlike most other verification
caveats in this plan and in plans/10) — generated a real run via `RunLogger` directly, then ran
`python tail_log.py --latest --log-dir <dir> --lines 5` against it and confirmed the correct run
was found and both records printed with correct formatting (timestamp prefix, nested dict
compacted, generic `key=value` rendering) before the process was killed.

## Running everything already implemented (chunks 7-9 + the fixes above, plus plans/10's chunks 1-6) at once

```bash
python -m pytest test_run_logging.py test_main_video_writer.py test_debug_stream.py \
    test_register_person_logging.py test_register_person_interactive.py \
    test_register_person_keyboard_interrupt.py test_register_person_interactive_streaming.py \
    test_tail_log.py \
    modules/gesture_hand_keypoint/test_sequence_counts.py \
    modules/followme_orchestrator/test_debug_snapshot.py -v
```
Expect `60 passed`.

### Chunk 10 — `main.py` wiring + docs — ✅ DONE

New `--interactive` flag under `--modules register`, third shape alongside the existing
`--person-name`/Tkinter branches — confirms §2's v1-vs-v2 lifecycle note in practice: it returns
directly after `run_interactive()` finishes, same as the Tkinter branch's `if not
app.chosen_name: return 0` case, never falling through into the `args.mode="camera";
args.modules="followme"` code the other two register paths share (a console session has no
single `chosen_name` to hand off with) — verified by a dedicated test, not just asserted.

**Two new fail-fast validations**, same "catch it in argument-parsing, not after opening a
window/camera" convention as chunk 6's original `--stream`-without-`--person-name` check:
- `--interactive` + `--person-name` → rejected (ambiguous: which path should run?).
- `--interactive` + `--then-followme` → rejected, with a reason, not just "invalid combination":
  a console session can register zero, one, or many people, so there is no single "the person who
  just registered" to chain into `followme` mode with — unlike the `--person-name` path, which
  registers exactly one and unambiguously means that one.
- The existing chunk-6 `--stream`-without-`--person-name` check was widened (not replaced) to
  also accept `--interactive` as a second valid streaming-capable path — covered by a regression
  test specifically checking the OLD rejection case still rejects, not just that the new
  acceptance case is accepted.

**A real, unrelated bug found and fixed while getting this chunk's tests to pass reliably**:
`debug_stream.py`'s `_stream_mjpeg()` only wrapped its streaming loop in the client-disconnect
`try/except`, not the initial `send_response()`/`send_header()`/`end_headers()` calls before it —
a client disconnecting during that brief window (rare in real use, but common enough across many
back-to-back automated test runs) raised unguarded, surfacing as an intermittent
`PytestUnhandledThreadExceptionWarning` in full-suite test runs. Fixed by wrapping the whole
method, and widening the caught exception from three named subclasses
(`BrokenPipeError`/`ConnectionResetError`/`ConnectionAbortedError`) to their common base
`OSError` — any socket-level failure there means the same thing (the client is gone), no reason
to enumerate every subclass by name.

**A second, related flakiness bug found in this plan's OWN test suite, not the product code**:
the original `test_tail_file_shows_initial_lines_then_follows_new_ones` (chunk 9) monkeypatched
`tail_log.time.sleep` — but `tail_log.time` IS the real, process-wide `time` module, not a
private copy, so that patch affected every OTHER thread in the process during the test's window
too, including a lingering `DebugStreamServer` handler thread from an earlier test still calling
`time.sleep()` internally, which desynced the test's own call-counting logic. Fixed by adding a
non-invasive test-only seam to `tail_file()` (`_max_idle_polls`, module-private, never exposed via
the CLI) instead of patching shared global state — see that test's own updated comment, and
`tail_file()`'s own docstring, for the full story. Confirmed fixed by running the full suite 5
times in a row with zero flakiness (previously failed roughly 1 in 3 runs).

**Tests**: [test_main_register_interactive.py](../test_main_register_interactive.py) — 6 pytest
unit tests, monkeypatching `sys.argv` and `register_person.run_interactive` — no real camera/model
touched. Covers: both new mutual-exclusion rejections, the widened `--stream` guard's OLD
rejection case still rejecting (regression check) and NEW acceptance case actually working,
`run_interactive()`'s arguments and return value both propagated correctly (a distinctive return
value proves it's not hardcoded), and the no-fall-through-into-followme behavior specifically.
All passing.

**Verify yourself**:
```bash
python -m pytest test_main_register_interactive.py -v
```
Expect `6 passed`. Also manually verified end-to-end: ran `main.py --modules register
--interactive --stream --camera-index 999` for real (bad camera index deliberately, since no real
camera exists in this dev environment) and confirmed the streaming server started, the console
prompt appeared, and both validation-rejection cases produced clean `parser.error()` messages
with no traceback.

### Post-chunk-10 fixes — two real usability gaps, both from direct user feedback

**1. `--stream` without `--person-name`/`--interactive` used to hard-error — now auto-selects
`--interactive` instead.** Raised directly: *"if stream required the person to be given, why
dont we just assign interactive flag into stream while person name is not chosen?"* Correct call
— the original chunk-10 `parser.error()` forced the operator to type `--interactive` by hand just
to unblock `--stream`, when `--interactive` is unambiguously the right (and only remaining)
headless-capable choice at that point anyway. Changed the guard in `main.py` from an error to
`args.interactive = True`, with a printed note (not silent) explaining what happened and how to
skip the note next time.

**2. The interactive console had no way to actually CHOOSE a followme target — a real design gap,
not just a missing nice-to-have.** Raised directly: *"the interactive doesnt have the option to
return the chosen person... I cant see any option I want to choose to be the target of follow me
mode."* Chunk 10's original `--interactive`+`--then-followme` rejection was correct AS FAR AS IT
WENT (`--then-followme` genuinely has no single-target meaning in a multi-command console
session) — but it stopped there instead of asking what SHOULD replace it, leaving the console with
strictly less capability than the Tkinter UI it was meant to be a headless equivalent of (Tkinter
has "Follow Me"; the console had nothing).

Fixed by adding a fourth REPL command, **`follow <name>`** — the console's own answer to
`RegistrationApp._on_follow_me()`, not a new invention: validates `ready_for_followme` (same
check the Tkinter button makes), selects the name and exits the console if ready, or explains why
and stays in the console if not. This required changing `run_interactive()`'s own return contract
from a plain `int` to `(exit_code, chosen_name)` — a real, deliberate breaking change to a
function added only one chunk ago (chunk 8), made now rather than worked around, since nothing
outside this session's own code depended on the old signature yet. `main.py`'s `--interactive`
dispatch now reads `chosen_name` the exact same way it already reads `RegistrationApp.chosen_name`
and `args.person_name` (with `--then-followme`) — one shared fall-through into the followme camera
loop, three ways to reach it. `register_person.py`'s own standalone CLI (no camera-loop machinery
of its own, same limitation the `--person-name` path already has there) reports the chosen name
and tells the operator to relaunch through `main.py` to actually act on it.

**Tests**: 3 new (`test_follow_command_selects_ready_person_and_exits`,
`test_follow_command_rejects_not_ready_person_and_stays_in_console` in
[test_register_person_interactive.py](../test_register_person_interactive.py); `test_
interactive_dispatch_falls_through_to_followme_when_follow_used` in
[test_main_register_interactive.py](../test_main_register_interactive.py) — this last one is the
one that actually proves the fix: it fakes `open_capture()` and `run_followme_pipeline()` and
asserts the fall-through genuinely reaches the followme dispatch, not just that `chosen_name` was
returned). Plus a renamed/rewritten test for the auto-select-`--interactive` behavior
(`test_stream_alone_auto_selects_interactive_instead_of_erroring`, replacing the old
hard-rejection test it superseded) and every existing `run_interactive()` caller/test updated for
the new two-value return. All three affected test files re-run 3x in a row with zero flakiness.

**Verify yourself**:
```bash
python -m pytest test_register_person_interactive.py test_register_person_interactive_streaming.py test_main_register_interactive.py -v
```
Expect `21 passed` (12 + 2 + 7 — supersedes chunk 8's "10 passed" and chunk 10's "6 passed" counts
above, which were correct at the time but are now stale; left as-is above as historical record,
per this doc's own convention — see `plans/10`'s own chunk 4 correction for the same convention
applied there).

Also manually verified end-to-end against real data: `follow Nam` (an already fully-registered
person from real prior use, confirmed via `list` first) correctly selected and printed `'Nam'
selected — continuing into followme mode.`, reaching `main.py`'s real `open_capture()` call (which
then failed only because a deliberately-invalid camera index was used for the test, not because
of anything in this fix).

## Running everything already implemented (chunks 7-10 + all fixes above, plus plans/10's chunks 1-6) at once

```bash
python -m pytest test_run_logging.py test_main_video_writer.py test_debug_stream.py \
    test_register_person_logging.py test_register_person_interactive.py \
    test_register_person_keyboard_interrupt.py test_register_person_interactive_streaming.py \
    test_tail_log.py test_main_register_interactive.py \
    modules/gesture_hand_keypoint/test_sequence_counts.py \
    modules/followme_orchestrator/test_debug_snapshot.py -v
```
Expect `69 passed` (66 + the 3 new `follow`/auto-select tests above). All 10 chunks of this plan
are now complete, plus the two post-chunk-10 fixes above.

## 5. Audit of plans/10 — what this invalidates or extends

- **✅ Done with chunk 7**: `docs/commands.md`'s `--log-dir` row and "Reviewing a run" intro line,
  and `main.py`'s own `--log-dir` argparse help text, all updated to list `register --person-name`
  as a third `RunLogger`-using mode alongside `pretrigger`/`followme`. `plans/10`'s own chunk 4
  section text was deliberately left as originally written (it accurately describes what chunk 4
  itself did at the time — `register` logging is this plan's own chunk 7, not a retroactive edit
  to chunk 4's history).
- **Chunk 6 (`--stream`) needs NO changes** — its own scope note already explicitly listed
  `register --person-name` as a caller, and `--interactive`'s `register <name>` command reuses
  that exact same call path (§3.3.3), so chunk 6's existing description stays accurate as-is.
- **No other chunk in plans/10 is affected** — chunks 1/2/3/5 (JSONL schema, gesture counts,
  `debug_snapshot()`, `--save-video`) are `pretrigger`/`followme`-specific and untouched by this
  plan.
