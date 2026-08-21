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
re-declared in `main.py` (`_WAVE_STATE_COLOR`), `modules/gesture_hand_keypoint/visualize_gesture_hand_keypoint.py`,
and `modules/gesture_trajectory_verifier/visualize_gesture_trajectory_verifier.py` — same
isolation convention as the `ConfirmationTracker` classes themselves (independently
reimplemented per module, not shared, but deliberately kept visually identical so the meaning
transfers across every screen you look at).

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

### 3. Gesture method (one of three, `--gesture-method`) — `GestureMethodResult.draw_debug(crop, ...)`

All three draw onto the **person crop** (not full-frame coordinates) — a numpy view into `frame`,
so drawing on it updates `frame` in place.

**Method 1 — `condition` (`wave_facing_gate`):**

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 255, 0)` | Keypoint circle — MoveNet confidence > 0.5 ("high confidence"). |
| 🟠 Orange | `(0, 165, 255)` | Keypoint circle — confidence between the drawing floor (0.2) and 0.5 ("low confidence but shown"). |
| ⚫ Dark gray | `(100, 100, 100)` | Skeleton connection lines between keypoints. |
| 🔵 Blue | `(255, 0, 0)` | Left-arm wrist→elbow→shoulder vector. |
| 🔴 Red | `(0, 0, 255)` | Right-arm wrist→elbow→shoulder vector. |
| ⚪ Light gray | `(200, 200, 200)` | (Available but not currently drawn by `draw_debug()` — reserved for a bbox outline.) |

Arm-vector **color is side (left/right), not pass/fail** — Gate A pass/fail is instead encoded
as **line thickness**: 3px if Gate A currently passes for that arm, 1px if it doesn't. Don't read
arm-vector color as a verdict; read its thickness.

**Method 2 — `hand_keypoint` (`gesture_hand_keypoint`):**

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

**Method 3 — `trajectory_verifier` (`gesture_trajectory_verifier`):**

| Color | BGR | Meaning |
|---|---|---|
| 🟢 Green | `(0, 255, 0)` | Keypoint circles (all 17 MoveNet points, confidence ≥ 0.2). |
| ⚫ Gray | `(100, 100, 100)` | Skeleton connection lines. |
| 🟠 Orange | `(0, 200, 255)` | Highlighted wrist→elbow→shoulder vector for whichever arm (`self.arm`) produced the best similarity score this frame — drawn 3px thick, on top of the plain skeleton. |

Unlike Method 1, this highlight doesn't encode pass/fail via thickness — it just marks *which*
arm is currently the best-scoring candidate. (Its own standalone `visualize_gesture_trajectory_verifier.py`
tool additionally draws a small inset panel comparing the live wrist path (`(255, 150, 0)`, a
blue-ish orange) against the best-matching reference trajectory's path (`(0, 200, 255)`, orange)
— that inset is specific to that one script, not part of the reusable `draw_debug()`.)

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

### `target_tracking` — `TrackingResult.draw_debug(frame)`

| Color | BGR | Meaning |
|---|---|---|
| 🟡 Yellow | `(0, 220, 255)` | Tracked bbox + status text — state is **RECORDING**. |
| 🟢 Green | `(0, 200, 0)` | Tracked bbox + status text — state is **TRACKING**. |
| 🔴 Red | `(0, 0, 255)` | Tracked bbox + status text — state is **LOST** (drawn only in the single transitional frame this result is still visible before recovery takes over the overlay). |
| ⚫ Gray-ish | `(120, 120, 120)` | The vertical frame-center reference line — always drawn, doesn't change with state. |

Also draws a text line with the current `state`/`horizontal_offset`, and — only when at least
one periodic appearance re-verify has run this episode — a second line with the last
`reverify score`/`pass`, in the same state-color.

### `target_recovery` — `RecoveryResult.draw_debug(frame)`

| Color | BGR | Meaning |
|---|---|---|
| 🟡 Yellow | `(0, 220, 255)` | Status text — **SEARCHING**. |
| 🟢 Green | `(0, 200, 0)` | Status text + reacquired bbox (3px) + `"REACQUIRED via {path}"` label — **REACQUIRED**. |
| 🔴 Red | `(0, 0, 255)` | Status text — **TIMEOUT**. |

Status text always shows `face_search_fail_count` and `elapsed_search_seconds` alongside the
color, regardless of status.

### `followme_orchestrator` summary overlay (`main.py`'s own layer, `--show` alone)

`main.py --modules followme` draws a final summary layer on top of everything above, using its
own copy of the state→color mapping (`_FOLLOWME_STATE_COLOR`, matched to
`modules/followme_orchestrator/visualize_followme_orchestrator.py`'s identical table):

| `debug_state` | Color | BGR |
|---|---|---|
| `WAITING_FOR_TRIGGER` | Gray | `(180, 180, 180)` |
| `TRACKING_STARTED` | Yellow | `(0, 220, 255)` |
| `RECORDING` | Yellow | `(0, 220, 255)` |
| `TRACKING` | Green | `(0, 200, 0)` |
| `RECORDING_STEERING_UNCALIBRATED` | Orange | `(0, 160, 255)` |
| `TRACKING_STEERING_UNCALIBRATED` | Orange | `(0, 160, 255)` |
| `RECOVERING` | Orange-red | `(0, 140, 255)` |
| `REACQUIRED_RESUMING` | Green | `(0, 200, 0)` |
| `STOPPED` | Red | `(0, 0, 255)` |

The two `_STEERING_UNCALIBRATED` and `RECOVERING` states each get their own distinct
orange-family shade specifically so they're visually distinguishable from plain
RECORDING/TRACKING/REACQUIRED at a glance — "the target is genuinely being tracked/recovering,
but `should_move` is being held `False`" is a meaningfully different situation from a clean
green TRACKING frame, worth a different color rather than reusing yellow or red.

---

## Legacy pipeline (`--modules estop` / `wave_facing` / `both`)

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

### `wave_facing` legacy pipeline bbox (`main.py`'s `wave_bbox_color()`, reusing `_WAVE_STATE_COLOR`)

Same RED/YELLOW/GREEN triple as the universal convention above, computed from BOTH
`is_waving`/`is_facing_camera` together: green only once *both* are GREEN, yellow if *either* is
still YELLOW, red otherwise. The per-person debug overlay under it (when `--debug` is set) is
`wave_facing_gate`'s own `draw_debug()`, described under Method 1 above.

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

### `target_tracking`'s click-and-drag selection box (`visualize_target_tracking.py`)

A fixed orange-blue `(255, 200, 0)`, 1px — purely a UI affordance while dragging out the initial
bbox before an episode starts; not a status indicator.

---

## See also

- [`modules.md`](modules.md) — what each `draw_debug()` call's underlying module actually
  computes (the logic behind what gets colored).
- [`commands.md`](commands.md) — exact commands to launch each overlay.
- [`architecture.md`](architecture.md#debug-visualization-architecture) — how `draw_debug()`
  calls are composed across modules (the wiring, not the color meanings).
