# Feature Spec: Target Recording + Tracking (`modules/target_tracking/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `docs/architecture.md`, `docs/modules.md`, `docs/parameters.md`,
  `Agent_Instruction_Framework.md`, `modules/appearance_verifier/interface.py` (already built —
  see §0.3 for how this module is allowed to use it), `modules/human_detection/interface.py`
  (for the existing YOLO+ByteTrack pattern reference — see §0.3's isolation note on what that
  means and doesn't mean).
- MAY NOT use: any external planning conversation, chat history, or document not physically
  present in the repository.

### §0.2 Structural Placement
`modules/target_tracking/`, with `interface.py`, private implementation, `test_target_tracking.py`,
and a standalone `visualize_target_tracking.py` (§7).

### §0.3 Interface & Isolation Contract

```python
@dataclass
class TrackingResult:
    target_locked: bool           # True while in RECORD or TRACKING; False once lost/handed to search
    horizontal_offset: float | None  # normalized -1.0 (frame-left edge) to +1.0 (frame-right edge),
                                       # 0.0 = target bbox center exactly on frame's vertical centerline.
                                       # None if target_locked is False.
    person_bbox: tuple[int,int,int,int] | None  # full-frame pixel space, current tracked bbox
    state: Literal["RECORDING", "TRACKING", "LOST"]
    reference_set: object | None  # the built ReferenceEmbeddingSet from appearance_verifier,
                                    # once RECORDING completes — the caller passes this forward
                                    # into the recovery/search module when state becomes LOST

def start(initial_person_bbox: tuple[int,int,int,int], frame: np.ndarray, timestamp: float) -> None:
    """Called once, when the gesture trigger first goes GREEN. Locks onto initial_person_bbox
    as the target and begins the RECORDING phase."""

def update(frame: np.ndarray, timestamp: float) -> TrackingResult:
    """Called once per frame while this module owns the active follow-me episode (i.e. from
    the moment start() is called until state becomes LOST and the caller has taken over)."""

def reset() -> None:
    """Called by the caller once recovery re-acquires the target (from the recovery module,
    specced separately) and hands a fresh bbox back — re-enters RECORDING with that new bbox,
    same as a fresh start()."""
```

**Isolation statement — name the specific temptation:** this module needs its OWN, independent
YOLO+ByteTrack instance (`model.track(..., persist=True)`), separate from `human_detection`'s,
`emergency_stop`'s, and `human_detection_roi`'s existing instances — per this repo's own
"own-instance isolation" design rule (`docs/architecture.md`, rule #2). Do not import or reuse
any of those three modules' tracker objects, even though they load the identical `yolo11n.onnx`
weights file. This module's tracker also runs on a DIFFERENT calling pattern than
`human_detection_roi`'s (which is deliberately stateless/single-frame `.predict()`, per rule #6)
— this module's ByteTrack call is meant to persist continuously frame-to-frame while a target is
locked, closer in spirit to `human_detection`'s usage, but still its own separate instance.

This module MAY call `appearance_verifier.verify()` (already built, §0.1) for the periodic
re-verification described in §3 — that's the intended, sanctioned use of that module's public
interface, not a violation of isolation (isolation forbids sharing *state/instances*, not
calling another module's public `interface.py` contract; the same pattern `human_detection_roi`
uses when it's called from `main.py`, or how gesture methods are called with a crop). Do NOT
reach into `appearance_verifier`'s private implementation files — only its `interface.py`.

### §0.4 Ambiguity Handling
If the exact ByteTrack calling convention needed to persist a single locked target (as opposed
to `human_detection`'s "track everyone in frame" usage) is unclear, or if there's ambiguity
about how to keep a `track_id` "locked" once assigned vs. ByteTrack's normal multi-target
behavior, STOP and ask rather than guess at ByteTrack API usage not already demonstrated
elsewhere in this repo.

### §0.5 Mandatory Pre-Implementation Audit
1. Read `modules/human_detection/`'s existing ByteTrack usage in full before writing any new
   tracking code — match its calling conventions where they transfer, but confirm with the user
   before assuming any convention transfers to this "lock onto one specific target" use case,
   which is functionally different from `human_detection`'s "track everyone" use case.
2. Search for naming collisions with `TrackingResult`, `target_tracking`, or similar.
3. Report findings before proceeding.

---

## §2 Purpose & Context

Once a gesture trigger is confirmed (`is_waving` reaches `GREEN` in whichever gesture method is
active), this module takes over: locks onto that person as "the target," records a short set of
reference appearance frames, then tracks them continuously frame-to-frame, reporting how far
off-center they are (for downstream steering) every frame. If tracking is lost, this module hands
off to a separate recovery/search module (specced separately) rather than handling recovery
itself — this module's job ends at declaring `LOST` and providing the `reference_set` the
recovery module needs.

```
gesture trigger GREEN -> start(initial_bbox, ...)
        |
        v
   RECORDING  (S3)
        | (record_duration_seconds elapsed)
        v
   TRACKING   (S4)
        | (track lost, per track_loss_grace_period_seconds)
        v
   LOST -> caller hands off to recovery module (out of scope here)
        | (recovery re-acquires, calls reset() with fresh bbox)
        v
   RECORDING again (loop)
```

---

## §3 RECORDING Phase

On `start()`, begin capturing cropped `person_bbox` frames (BGR crops, same "numpy view into
`frame`" convention used throughout this codebase per `docs/architecture.md` rule #5) for
`record_duration_seconds` (config, time-based per project convention — NOT a fixed frame count,
since actual frame count collected will vary with real fps). Track elapsed wall-clock time via
the `timestamp` argument passed into `update()`, not an assumed frame rate.

Once the duration elapses, call `appearance_verifier.build_reference_set()` (§0.3) on the
collected crops to produce the `ReferenceEmbeddingSet`, store it as this module's `reference_set`,
and transition to `TRACKING`.

**Stop-and-ask item:** what should happen if fewer than some minimum number of usable crops were
collected during `record_duration_seconds` (e.g. the detector momentarily lost the bbox mid-
recording)? Present this to the user rather than picking a silent minimum-count floor —
`appearance_verifier`'s own "not ready" floor pattern (§7 of its spec) may be the right model to
follow here, but confirm before implementing.

---

## §4 TRACKING Phase

### §4.1 Motion continuity
Run this module's own isolated YOLO+ByteTrack instance (§0.3) to follow the locked target
frame-to-frame. Confirm with the user (per §0.4) the exact mechanism for "locking" onto one
specific `track_id` at `RECORDING`->`TRACKING` transition time and continuing to follow only that
ID, since this differs from `human_detection`'s existing "report everyone" pattern.

### §4.2 Horizontal deviation calculation
```python
def compute_horizontal_offset(person_bbox: tuple[int,int,int,int], frame_width: int) -> float:
    bbox_center_x = person_bbox[0] + person_bbox[2] / 2
    frame_center_x = frame_width / 2
    pixel_offset = bbox_center_x - frame_center_x
    return pixel_offset / (frame_width / 2)  # normalized -1.0 to +1.0
```
This is deliberately a **normalized offset, not a true angle in degrees**. Converting to a real
angle (using the camera's FOV) is explicitly NOT this module's job — that conversion happens
downstream in the steering/control layer (specced separately), which is why `camera.fov_degrees`
does not appear anywhere in this module's config (§6) even though it exists elsewhere in
`config/thresholds.yaml`. Do not add FOV-based angle computation here; if tempted to, stop and
ask first — this boundary was a deliberate architecture decision, not an oversight.

### §4.3 Periodic appearance re-verification (silent ID-switch protection)
ByteTrack's `track_id` continuity is motion-based, not identity-verified — per
`docs/modules.md`'s own documented caveat on `human_detection`, "`track_id` is motion-continuity
only, never a verified identity." In a crowd, ByteTrack can silently reassign the locked
`track_id` to a *different* nearby person after an occlusion, without ever reporting a track
loss. To catch this: every `appearance_reverify_interval_seconds` (config, time-based — NOT
every frame, for cost reasons), call `appearance_verifier.verify()` (§0.3) on the current tracked
crop against `reference_set`.

**Stop-and-ask item:** what should happen when a periodic re-verify FAILS (low similarity to the
reference set)? Options to present to the user, not decide unilaterally: (a) immediately treat
this as equivalent to a track loss -> transition to `LOST`, or (b) something softer (e.g. require
two consecutive failed re-verifies before declaring loss, to avoid a single bad-lighting frame
causing an unnecessary full recovery cycle). This directly affects false-positive/false-negative
tradeoffs and should be confirmed, not assumed.

### §4.4 Track loss detection
If the locked `track_id` is missing from ByteTrack's output continuously for
`track_loss_grace_period_seconds` (config, **time-based**, converted against real elapsed time
via the `timestamp` argument — not a fixed frame count, since fps isn't assumed constant
anywhere else in this codebase either), transition to `LOST`.

---

## §5 LOST State — Handoff Only

This module does NOT implement recovery/search logic — that's a separate module. On declaring
`LOST`, `TrackingResult.target_locked` becomes `False` and `TrackingResult.reference_set` is
still populated with the last-built reference set, so the caller (orchestration layer, or the
recovery module directly) has what it needs to attempt re-acquisition. This module then does
nothing further until its `reset()` is called with a freshly re-acquired bbox.

---

## §6 Configuration

```yaml
target_tracking:
  record_duration_seconds: null       # TODO: calibrate, red - e.g. start ~1-2s as a yellow guess
  appearance_reverify_interval_seconds: null  # TODO: calibrate, red
  track_loss_grace_period_seconds: null       # TODO: calibrate, red
  # appearance re-verify similarity threshold - deliberately its OWN key, NOT shared with
  # appearance_verifier.similarity_threshold or the recovery module's fallback threshold
  # (see appearance_verifier spec S4 for why these three must stay independently tunable)
  appearance_reverify_similarity_threshold: null  # TODO: calibrate, red
  yolo_model_path: yolo11n.onnx        # green, own independent instance per S0.3
```

Note: `camera.fov_degrees`/`lens_type`/`focus_type` are NOT this module's config — they belong
to the steering layer, specced separately. Do not add them here even though the deviation
calculation is conceptually related — see §4.2's explicit boundary note.

---

## §7 Visualization Requirement (mandatory for this module)

Standalone `visualize_target_tracking.py`:
- Runs the full RECORDING -> TRACKING -> LOST cycle against a webcam feed or video file, given a
  manually-specified or clicked initial bbox to simulate the gesture-trigger handoff (confirm
  with the user which input method fits their testing workflow better before implementing).
- Draws the tracked bbox, a vertical line at frame-center, and a visible readout of the current
  `horizontal_offset` value and `state`.
- Displays the periodic re-verify's last similarity score and pass/fail status when it runs
  (not every frame — only visible on the frames where §4.3 actually executes).
- Logs `TrackingResult` fields to console per frame.

---

## §8 Documentation Requirement (mandatory)

Update `docs/modules.md`, `docs/parameters.md`, `docs/architecture.md` (repo layout + a new
pipeline-flow addition showing the post-trigger flow: trigger -> `target_tracking` -> LOST ->
recovery module -> back), following the exact existing format/depth conventions used for every
other module, same as required in `modules/appearance_verifier`'s spec §7. Include this module's
own **known limitations** subsection in `docs/modules.md` covering: ByteTrack's motion-only
`track_id` semantics (and why periodic re-verify exists to compensate, referencing §4.3), and the
explicit non-goal of angle conversion (§4.2) so a future reader doesn't wonder why FOV isn't
handled here.

---

## §9 Explicit Non-Goals

- No recovery/search logic (`SEARCHING` state) — separate module.
- No true-angle/FOV-based steering computation — downstream steering layer's job.
- No PID or any control-loop logic whatsoever.
- No sharing of tracker instances with any other module (§0.3).
