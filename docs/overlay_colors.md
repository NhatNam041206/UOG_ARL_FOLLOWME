# Debug Overlay Reference

What every color in this project's debug overlays means, what each overlay draws, and where it
comes from. Companion to [`commands.md`](commands.md) (how to launch these overlays) and
[`architecture.md`](architecture.md#debug-visualization-architecture) (how the drawing calls are
wired together). All color tuples below are **BGR** (OpenCV's convention, not RGB) — e.g.
`(0, 0, 255)` is red, not blue.

This file is itself an audit: every color constant below was read directly from the current
source (`grep`'d, not recalled), file-by-file, so it stays accurate as the actual overlays
change — if a color here ever looks wrong, the code is the source of truth, not this doc.

---

## The universal RED / YELLOW / GREEN convention

Every debounced confirmation signal in this project (`is_waving`, `is_facing_camera`, gesture
completion, search status) reduces to this same three-color state, drawn as a bbox/label color
almost everywhere:

| Color | BGR | Meaning |
|---|---|---|
| 🔴 Red | `(0, 0, 255)` | Raw condition currently failing / not held long enough. |
| 🟡 Yellow | `(0, 220, 255)` | Condition passing, still accumulating `confirmation_duration_seconds` before it's trusted. |
| 🟢 Green | `(0, 200, 0)` | Condition held continuously long enough — confirmed, trusted. |

This exact triple (`RED`/`YELLOW`/`GREEN` → those three BGR values) is independently
re-declared in `main.py` (`_WAVE_STATE_COLOR`) and
`modules/gesture_hand_keypoint/visualize_gesture_hand_keypoint.py` — same isolation convention as
the `ConfirmationTracker` classes themselves (independently reimplemented per module, not shared,
but deliberately kept visually identical so the meaning transfers across every screen you look at).

**Caveat worth knowing before reading composited overlays** (`--modules followme`): yellow and
green are *reused* for unrelated meanings in other layers drawn on the same frame — e.g. yellow
also means "this is the ROI search region" (`human_detection_roi`) and "currently in RECORDING
phase" (`target_tracking`), and green also means "person bbox found" and "face matched." When
every phase's overlay is composited together, a yellow box is not always a YELLOW confirmation
state — check *which* box/line it is (bbox vs. thin ROI rectangle vs. center line) and the
accompanying text label, not color alone.

---

## Pre-trigger phase (`--modules pretrigger`, and the first half of `--modules followme`)

Layered in this drawing order (later layers drawn on top):

### 1. `face_identity` — `FaceIdentityResult.draw_debug(frame)`

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 200, 0)` | Face bbox — matched a registered person. Label shows `{name} ({score})`. |
| 🔴 Red | `(0, 0, 255)` | Face bbox — detected but did NOT match any registered person. Label shows `no match ({score})`. |

Draws: one rectangle + one text label per detected face, in **full-frame** coordinates. Only
called at all (by `main.py`/`followme_orchestrator`) for faces that already passed
`is_registered_match` upstream, so in practice you will only ever see the green case from those
callers — the red case is reachable from `visualize_face_identity.py`, which draws every
detected face regardless of match.

### 2. `human_detection_roi` — `HumanDetectionResult.draw_debug(frame, matched_face_bbox)`

| Color | BGR | Meaning |
|---|---|---|
| 🟡 Yellow | `(0, 220, 255)` | The **ROI search region** — the area `compute_roi()` derived from the face bbox and scoped the body detector to. Drawn whenever `roi_expansion_factor` is calibrated, regardless of whether a person was actually found inside it. |
| 🟢 Green | `(0, 200, 0)` | The **detected person bbox** — only drawn when `person_found` is `True`. |

Also (in `main.py`'s `pretrigger` mode and `visualize_human_detection_roi.py` specifically, not
in the reusable `draw_debug()` itself): a separate blue-ish `(255, 100, 0)` rectangle for the
**input face bbox**, drawn by the calling script itself, one layer below.

### 3. Gesture method (`gesture_hand_keypoint`, the sole TRIGGER gesture method) — `GestureMethodResult.draw_debug(crop, ...)`

Two alternatives — `wave_facing_gate` ("condition") and `gesture_trajectory_verifier`
("trajectory_verifier") — used to exist and each had their own distinct overlay; both were
removed (confirmed with the user). Draws onto the **person crop** (not full-frame coordinates) —
a numpy view into `frame`, so drawing on it updates `frame` in place.

| Color | BGR | Meaning |
|---|---|---|
| 🟡 Yellow | `(0, 255, 255)` | Whole-hand skeleton/keypoints — this hand currently classifies as **OPEN**. |
| 🟢 Green | `(0, 200, 0)` | Whole-hand skeleton/keypoints — this hand currently classifies as **CLOSED** (fist). |
| ⚪ Gray | `(200, 200, 200)` | Whole-hand skeleton/keypoints — **NEITHER** (ambiguous), or classification thresholds aren't calibrated yet (falls back here rather than crashing). |
| 🟢 Green | `(0, 255, 0)` | Per-finger edge highlight — hand_shape.py counts this specific finger as **extended**. A more granular diagnostic than the whole-hand color above; drawn on top of it. |
| 🔴 Red | `(0, 0, 255)` | Per-finger edge highlight — this specific finger counts as **curled**. Also used for the red **dotted horizontal line** at the `palm_height_fraction` calibration cutoff (a distinct element, not a finger). |

Two independent colorings are layered on the same hand: the whole-hand shape color (yellow/
green/gray, on the skeleton+keypoints) and the per-finger extended/curled color (green/red, on
each finger's tip↔PIP edge and the thumb's tip↔wrist edge) — they can disagree on a single frame
(e.g. 4 fingers read green/extended but the whole hand is still gray if the 5th finger doesn't
agree), which is expected, not a bug.

### 4. Pipeline-level bbox + label (`main.py`, `--show` alone — not gated by `--debug`)

Drawn *after* every phase overlay above, so it isn't occluded:

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 200, 0)` | `TRIGGER = True` this frame (`is_waving` reached GREEN). Label appends `"  TRIGGER!"`. |
| 🟡 Yellow | `(0, 220, 255)` | `waving_state == "YELLOW"` — confirmation still building. |
| 🔴 Red | `(0, 0, 255)` | Neither of the above. |

---

## Post-trigger phase (`--modules followme` only — not reachable from `pretrigger`)

Drawn on top of everything above via `modules.followme_orchestrator.draw_debug()`, once a
tracking episode is active (the pre-trigger overlays above stop being drawn at that point — only
one "mode" of overlay is active per frame, matching `debug_state`).

### `autocar_adapter` — `TrackingResult.draw_debug(frame)` (replaces `target_tracking`/`target_recovery` below)

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 200, 0)` | Tracked bbox + status text — state is **TRACKING**. |
| 🟠 Orange | `(0, 160, 255)` | Status text only (no bbox — nothing is currently locked) — state is **SEARCHING**. |
| 🔴 Red | `(0, 0, 255)` | Status text only — state is **LOST**. |
| ⚫ Gray-ish | `(120, 120, 120)` | The vertical frame-center reference line — always drawn, doesn't change with state. |

Also draws a text line with the current `state`/`horizontal_offset`. Unlike the old
`target_tracking`+`target_recovery` pair, there is only ever ONE overlay here — tracking and
recovery are the same state machine now, not two separate results handed off mid-frame.

### `followme_orchestrator` summary overlay (`main.py`'s own layer, `--show` alone)

`main.py --modules followme` draws a final summary layer on top of everything above, using its
own copy of the state→color mapping (`_FOLLOWME_STATE_COLOR`, matched to
`modules/followme_orchestrator/visualize_followme_orchestrator.py`'s identical table):

| `debug_state` | Color | BGR |
|---|---|---|
| `WAITING_FOR_TRIGGER` | Gray | `(180, 180, 180)` |
| `TRACKING_STARTED` | Yellow | `(0, 220, 255)` |
| `TRACKING` | Green | `(0, 200, 0)` |
| `TRACKING_STEERING_UNCALIBRATED` | Orange | `(0, 160, 255)` |
| `RECOVERING` | Orange-red | `(0, 140, 255)` |
| `STOPPED` | Red | `(0, 0, 255)` |

(`RECORDING`, `RECORDING_STEERING_UNCALIBRATED`, and `REACQUIRED_RESUMING` no longer exist as
states — the old `target_tracking` design had a separate RECORDING phase and a one-frame
transitional reacquire state; `autocar_adapter` force-locks immediately at trigger time and
resumes `TRACKING` with a real offset the same frame it reclaims, so neither is needed anymore.)

`TRACKING_STEERING_UNCALIBRATED` and `RECOVERING` each get their own distinct orange-family shade
specifically so they're visually distinguishable from plain TRACKING at a glance — "the target is
genuinely being tracked/recovering, but `should_move` is being held `False`" is a meaningfully
different situation from a clean green TRACKING frame, worth a different color rather than
reusing yellow or red.

### Steering-direction arrow — `draw_steering_arrow(frame, command)` (`--show` alone, NOT gated by `--debug`)

| Color | BGR | Meaning |
|---|---|---|
| 🟡 Cyan-yellow | `(0, 255, 255)` | The arrow + angle text — the CALCULATED direction the robot is actually being commanded to move this frame. |

An arrow from bottom-center of the frame: 0° points straight up (ahead), positive angles tilt
right, negative left — the same sign convention `SteeringController.update()` uses internally.
Distinct from every other overlay on this page in one way: it's not a per-module debug readout,
it's the literal output being acted on, so it draws whenever `--show` is on regardless of
`--debug`, and draws nothing at all while `should_move` is `False` (there is no "calculated
direction" to show when the robot isn't being told to move).

---

## Legacy pipeline — REMOVED from `main.py` (`--modules estop`/`wave_facing`/`both` no longer exist)

`estop`/`wave_facing`/`both` are gone from `main.py --modules`'s choices — the face-first
pipelines fully superseded this whole-frame, no-identity-check demo. `emergency_stop` and
`human_detection` themselves still exist and are still independently runnable/testable (see
below); only `main.py`'s dedicated modes for exercising them together are gone.

### `emergency_stop` (`main.py`'s `_ESTOP_COLOR`)

| `EStopDecision` | Color | BGR |
|---|---|---|
| `GO` | Green | `(0, 200, 0)` |
| `STOP` | Red | `(0, 0, 255)` |
| `UNCERTAIN` | Yellow | `(0, 220, 255)` |

**`UNCERTAIN` is drawn as yellow but must never be treated as a softer state than `STOP`** — see
[`modules.md`](modules.md#emergency_stop). The color exists only to distinguish "confidently
clear" from "not confident enough to say GO" for a human reading the overlay; every consumer of
the decision itself treats `UNCERTAIN` identically to `STOP`.

### `human_detection` (legacy whole-frame detector, `test_human_detection.py`)

Always plain green (`(0, 200, 0)`) — this module has no confirmation/debounce logic of its own
(it's a raw detector), so its bbox color never changes; only presence/absence of a box is
meaningful.

---

## Standalone-only overlays (not reachable from `main.py`)

### `appearance_verifier` (`visualize_appearance_verifier.py`)

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 200, 0)` | `match_found = True`. |
| 🔴 Red | `(0, 0, 255)` | `match_found = False` (compared, genuinely didn't match). |
| ⚪ Gray | `(200, 200, 200)` | `reference_frame_count == 0` — "not ready," never attempted a real comparison. Distinguish this from red: gray means no comparison happened at all, red means one happened and failed. |

### `target_tracking`'s click-and-drag selection box (`visualize_target_tracking.py`, SUPERSEDED module)

A fixed orange-blue `(255, 200, 0)`, 1px — purely a UI affordance while dragging out the initial
bbox before an episode starts; not a status indicator.

---

## See also

- [`modules.md`](modules.md) — what each `draw_debug()` call's underlying module actually
  computes (the logic behind what gets colored).
- [`commands.md`](commands.md) — exact commands to launch each overlay.
- [`architecture.md`](architecture.md#debug-visualization-architecture) — how `draw_debug()`
  calls are composed across modules (the wiring, not the color meanings).
