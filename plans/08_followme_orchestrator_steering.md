# Feature Spec: FollowMe Orchestrator + Steering Controller (`modules/followme_orchestrator/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `docs/architecture.md`, `docs/modules.md`, `docs/parameters.md`,
  `Agent_Instruction_Framework.md`, and the `interface.py` of every module this orchestrator
  composes: `face_identity`, `human_detection_roi`, the three gesture methods (via `main.py`'s
  existing `_GestureMethodAdapter` pattern — reuse that adapter concept rather than inventing a
  new one), `modules/target_tracking/interface.py`, `modules/target_recovery/interface.py`.
- MAY NOT use: any external planning conversation, chat history, or document not physically
  present in the repository.

### §0.2 Structural Placement
`modules/followme_orchestrator/`, with `interface.py` as the sole public entry point, and — per
this repo's convention that `main.py` is normally "the only file that imports across module
boundaries" (`docs/architecture.md`) — an explicit, documented EXCEPTION note (§1) explaining why
this module is allowed to import across `face_identity` / `human_detection_roi` / gesture methods
/ `target_tracking` / `target_recovery` boundaries, unlike every other module in this repo.
Include `test_followme_orchestrator.py` and a standalone `visualize_followme_orchestrator.py`
(§5) that runs the ENTIRE post-trigger pipeline end-to-end for demo/debug purposes.

The `SteeringController` (PID) lives in its own file within this same module
(`steering_controller.py`), NOT merged into the orchestrator class itself — see §3 for why they
must stay separate classes despite being composed together.

### §0.3 Interface & Isolation Contract

```python
@dataclass
class FollowMeCommand:
    should_move: bool          # True = move forward, False = stop. NO speed parameter --
                                 # per explicit project decision, speed is not this pipeline's
                                 # concern; downstream robot control handles that separately.
    steering_angle_degrees: float | None  # signed angle for the Ackermann servo; None when
                                            # should_move is False (no meaningful steering
                                            # target when stopped)
    debug_state: str           # current high-level pipeline state, for logging/visualization
                                 # (e.g. "SEARCHING_FOR_PERSON", "GESTURE_PENDING", "TRACKING",
                                 # "RECOVERING", "STOPPED") -- exact state names are this
                                 # module's own design choice, not fixed by this spec

def step(frame: np.ndarray, timestamp: float) -> FollowMeCommand:
    """The single method the rest of the system (or a human operator's test harness) calls,
    once per frame, to get the current robot command. Internally sequences: face_identity ->
    human_detection_roi -> gesture method -> (if triggered or already tracking) target_tracking
    -> (if LOST) target_recovery -> SteeringController.update(offset, real_dt)."""
```

**Isolation EXCEPTION, stated explicitly, not silently overridden:** this module is the one
deliberate exception to "own-instance isolation, no module imports another's `interface.py`
except through `main.py`" — it exists specifically to be the reusable, importable version of what
`main.py`'s script-level code currently does ad hoc for the `face_first` pipeline. This is a
conscious architecture decision (composition root, not a violation) — document it as such in
`docs/architecture.md` rather than leaving it looking like an unexplained rule-break.

This module still MUST NOT reach into any composed module's PRIVATE implementation files — only
their `interface.py` contracts, exactly as `main.py` already does today.

### §0.4 Ambiguity Handling
The exact internal state machine names/transitions for `debug_state` are left to the
implementing agent's judgment (not fully specified in §0.3) — but the SEQUENCING described in §1
is not optional and must be followed exactly. If any step in that sequencing is unclear or seems
to conflict with how the composed modules' interfaces actually work once you inspect them, stop
and ask rather than reconcile the conflict silently.

### §0.5 Mandatory Pre-Implementation Audit
1. Read `main.py`'s existing `face_first` pipeline code in full, including `_GestureMethodAdapter`
   — this orchestrator should closely mirror that existing sequencing for the pre-trigger portion
   (Stage 1 → 2 → gesture), not reinvent it, since that logic is already working and tested.
2. Confirm exactly how `main.py` currently decides when a gesture trigger has occurred (reading
   `is_waving`/`waving_state` off whichever `GestureMethodResult` the active method returns) and
   replicate that exact check here, rather than approximating it.
3. Search for naming collisions with `FollowMeCommand`, `followme_orchestrator`,
   `SteeringController`, or similar.
4. Report findings before proceeding.

---

## §1 Purpose & Context: Full Sequencing

This module composes the entire face-first pipeline PLUS the new post-trigger tracking/recovery
modules into one steppable unit, so a caller doesn't need to hand-wire seven modules together
(the way `main.py`'s script code currently does for the pre-trigger portion only).

```
step(frame, timestamp):

  IF not currently tracking (no active target_tracking episode):
      run existing face_first pre-trigger sequence, exactly as main.py already does:
        face_identity.evaluate(frame, registry)
          -> filter to is_registered_match
          -> human_detection_roi.evaluate(frame, matched_face_bbox)
          -> crop = frame[...]
          -> gesture_method.evaluate(track_id, crop, timestamp, person_bbox)
          -> is_waving reaches GREEN?
               YES -> modules.target_tracking.start(person_bbox, frame, timestamp)
                      (this is the ONLY place target_tracking.start() is ever called
                      fresh, as opposed to .reset() -- see below)
               NO  -> should_move=False, steering_angle_degrees=None,
                      debug_state reflects "no confirmed trigger yet"

  ELSE (a target_tracking episode is active):
      result = modules.target_tracking.update(frame, timestamp)
      IF result.state in ("RECORDING", "TRACKING"):
          feed result.horizontal_offset into SteeringController.update() (S3)
          -> should_move=True, steering_angle_degrees=<PID output>
      IF result.state == "LOST":
          IF no recovery episode active yet:
              modules.target_recovery.start(result.reference_set, timestamp)
          recovery_result = modules.target_recovery.update(frame, registry, timestamp)
          IF recovery_result.status == "REACQUIRED":
              modules.target_tracking.reset()  # per target_tracking's own interface --
                                                  # confirm exact bbox-passing mechanism
                                                  # during audit S0.5, since reset()'s
                                                  # signature there doesn't currently
                                                  # take a bbox argument -- FLAG this
                                                  # inconsistency and resolve it with the
                                                  # user rather than picking a fix silently
              should_move=True (resume steering next cycle)
          IF recovery_result.status == "TIMEOUT":
              should_move=False, steering_angle_degrees=None,
              debug_state="STOPPED", and this follow-me episode is fully over --
              confirm with the user whether the orchestrator should then reset entirely
              back to "waiting for a fresh gesture trigger" or require an external reset call
          IF recovery_result.status == "SEARCHING":
              should_move=False (per project description: robot moves forward only while
              follow_me is actively true; searching is not an actively-following state)
```

**Flagged inconsistency to resolve, not silently patch:** `modules/target_tracking/interface.py`'s
`reset()` as specced takes no bbox argument, but the recovery hand-off clearly needs to pass the
freshly re-acquired bbox back in. Confirm with the user whether `target_tracking`'s spec should
be amended (its `reset()` signature updated to accept the bbox) or whether the orchestrator should
call something else — do not invent a workaround (e.g. a private/back-door attribute set) to
paper over the mismatch.

---

## §2 Camera Configuration (new, required)

Add a `camera` config section extension — the existing `camera.camera_index` key already exists
in `config/thresholds.yaml`; ADD to that same section, do not create a duplicate `camera:` block:

```yaml
camera:
  camera_index: 0          # existing key, unchanged
  fov_degrees: null         # TODO: 🟢 once set -- e.g. 85 for this project's known hybrid-lens,
                              # fixed-focus camera, per the hardware's own datasheet. Marked as a
                              # working-default-once-filled (🟢) rather than a calibration target
                              # (🔴/🟡), since horizontal FOV is a fixed physical property of the
                              # lens, not something tuned empirically like the other thresholds
                              # in this file -- but it must still be set correctly from the
                              # actual hardware's spec, not guessed.
  lens_type: null           # TODO: 🟢, e.g. "hybrid" -- informational/documentation value,
                              # not currently consumed by any calculation, but kept alongside
                              # fov_degrees/focus_type so the camera's physical properties are
                              # recorded in one place for future reference.
  focus_type: null          # TODO: 🟢, e.g. "fixed" -- same rationale as lens_type. Relevant
                              # specifically because a FIXED-focus camera means fov_degrees is a
                              # single constant, safe to treat as one config value; if this
                              # project ever used a variable-focus/zoom lens, fov_degrees would
                              # need to become a function of zoom level instead of a constant --
                              # flag this explicitly in docs/parameters.md's tuning notes so a
                              # future reader understands why a single scalar is sufficient here.
```

**Stop-and-ask item:** the user provided FOV=85°, lens type="Hybrid", focus="Fixed" for the
actual project hardware — set these as the working defaults, but confirm the exact string values
expected for `lens_type` (free text vs. an enum) with the user before finalizing, since this
spec doesn't prescribe a fixed vocabulary for that field.

---

## §3 SteeringController (PID) — Separate Class, Own Timing

```python
class SteeringController:
    """
    Deliberately NOT merged into FollowMeOrchestrator or any CV module -- see rationale below.
    Owns its own real wall-clock timing for dt, independent of whatever cadence step() happens
    to be called at (CV pipeline latency varies frame to frame; the PID's D-term and any
    integral windup handling need a real, accurately-measured dt, not an assumed constant tick).
    """
    def __init__(self, kp: float, ki: float, kd: float, max_steering_angle_degrees: float):
        ...

    def update(self, horizontal_offset: float, timestamp: float) -> float:
        """
        horizontal_offset: -1.0 to +1.0 normalized error signal from modules.target_tracking.
        timestamp: real wall-clock time (same clock/units as everywhere else in this pipeline)
        -- dt is computed internally as (timestamp - self._last_update_timestamp), NOT assumed.

        Converts horizontal_offset into a true steering angle using config.camera.fov_degrees
        (S2) -- THIS is where that FOV-based conversion belongs, not in target_tracking, per
        the explicit architecture boundary established in that module's spec S4.2.

        Returns a signed angle in degrees, clamped to +/- max_steering_angle_degrees (an
        Ackermann/servo hardware limit, not a calibration target -- confirm this value's
        source with the user, likely a servo datasheet limit rather than something tuned).
        """
```

**Rationale to preserve in code comments and docs, not just this spec (per the earlier
architecture discussion this spec is derived from):** PID timing correctness depends on `dt`
being real elapsed time, not an assumed frame interval. Embedding the PID inside a class that
ALSO does CV inference would let CV latency variance (a slow face-match frame, a slow gesture-
method frame) silently corrupt PID timing. Keeping `SteeringController` a separate class with its
own `update()` call, fed a real `timestamp` each cycle by the orchestrator, avoids this — this is
a correctness requirement, not just a style preference.

**Stop-and-ask item:** `kp`/`ki`/`kd` gain values and `max_steering_angle_degrees` are NOT
specified in this spec — these require either a servo datasheet value (for the max angle) or
empirical tuning against the actual Ackermann hardware (for the PID gains) that cannot be
guessed. Add them to `config/thresholds.yaml` under a new `steering` section, all `null`/🔴,
following the same fail-closed convention as every other calibration-gated value in this project
— confirm with the user whether "fail closed" for an ungated PID means `should_move` should be
forced `False` entirely while these are unset, mirroring how every other module in this project
refuses to produce a positive signal while required config is `null`.

---

## §4 Configuration Summary

```yaml
camera:
  camera_index: 0       # existing
  fov_degrees: null      # new, see S2
  lens_type: null         # new, see S2
  focus_type: null         # new, see S2

steering:
  kp: null                # TODO: calibrate empirically against real hardware, 🔴
  ki: null                # TODO: calibrate empirically, 🔴
  kd: null                # TODO: calibrate empirically, 🔴
  max_steering_angle_degrees: null  # TODO: 🟢 once set -- servo/Ackermann hardware limit
```

---

## §5 Visualization Requirement (mandatory for this module)

Standalone `visualize_followme_orchestrator.py`:
- Runs the FULL pipeline end-to-end (face → ROI → gesture → tracking → recovery → steering)
  against a webcam feed or video file — the only visualization tool in this whole feature set
  that exercises everything together, since every other module's visualizer is deliberately
  scoped to just that one module.
- Displays `debug_state`, `should_move`, `steering_angle_degrees`, and a visible frame-center
  reference line alongside the tracked bbox, so a human watching the demo can see the steering
  decision in context.
- Logs `FollowMeCommand` fields to console per frame.

---

## §6 Documentation Requirement (mandatory)

Update `docs/architecture.md` (add this orchestrator to the repository layout, document the
isolation EXCEPTION explicitly per §0.3, and add a new top-level pipeline-flow diagram showing
the complete trigger→tracking→recovery→steering loop, cross-referencing rather than duplicating
the more detailed diagrams already added by `target_tracking`'s and `target_recovery`'s own
documentation work), `docs/modules.md` (add sections for both `followme_orchestrator` and
`SteeringController`, including the PID-timing rationale from §3 in the known-limitations-style
section), `docs/parameters.md` (add the `camera` section additions and the new `steering` section,
exact table format matching every other section), and `docs/technologies.md` if a PID library
dependency is added rather than hand-implemented (confirm with the user which approach is
preferred before choosing).

---

## §7 Explicit Non-Goals

- No speed/velocity control of any kind — `should_move` is a boolean only, per explicit project
  decision that speed is handled entirely by a different, downstream flow.
- No direct hardware/servo interface code — this module outputs a `FollowMeCommand`; whatever
  consumes that and actually drives the servo is out of scope here.
- No modification to any of the six composed modules' own internal logic — this module only
  calls their existing public interfaces in sequence.
