# Feature Spec: Target Recovery / Search (`modules/target_recovery/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `docs/architecture.md`, `docs/modules.md`, `docs/parameters.md`,
  `Agent_Instruction_Framework.md`, `modules/face_identity/interface.py` (already built),
  `modules/human_detection_roi/interface.py` (already built),
  `modules/appearance_verifier/interface.py` (already built),
  `modules/target_tracking/interface.py` (already built — this module receives its output).
- MAY NOT use: any external planning conversation, chat history, or document not physically
  present in the repository.

### §0.2 Structural Placement
`modules/target_recovery/`, with `interface.py`, private implementation, `test_target_recovery.py`,
and a standalone `visualize_target_recovery.py` (§6).

### §0.3 Interface & Isolation Contract

```python
@dataclass
class RecoveryResult:
    status: Literal["SEARCHING", "REACQUIRED", "TIMEOUT"]
    reacquired_person_bbox: tuple[int,int,int,int] | None  # full-frame pixel space, populated
                                                              # ONLY when status == REACQUIRED
    reacquired_via: Literal["face_match", "appearance_fallback"] | None  # which path succeeded,
                                                              # for debugging/calibration visibility
    face_search_fail_count: int   # current consecutive Path-A-failure count, for debug/visualization
    elapsed_search_seconds: float # for debug/visualization and the visible search-timeout countdown

def start(reference_set: object, timestamp: float) -> None:
    """Called once when modules.target_tracking reports state == LOST. reference_set is the
    TrackingResult.reference_set handed off from that module (built by
    modules.appearance_verifier.build_reference_set() during the RECORDING phase)."""

def update(frame: np.ndarray, registry: object, timestamp: float) -> RecoveryResult:
    """Called once per frame while searching. registry is the FaceRegistry needed by
    face_identity.evaluate() — same object the main face_first pipeline already loads and
    passes to face_identity elsewhere; do not build a second registry instance."""
```

**Isolation statement — name the specific temptation:** this module orchestrates calls to THREE
existing modules' public interfaces (`face_identity`, `human_detection_roi`, `appearance_verifier`)
— this is the intended, sanctioned pattern (calling public `interface.py` contracts is not a
violation of isolation; sharing live state/instances IS). This module must instantiate its OWN
`appearance_verifier` usage pattern independently from `modules/target_tracking`'s own periodic-
reverify usage of the same module — per `appearance_verifier`'s own spec §4, this module's
fallback-path threshold (`appearance_fallback_threshold`, §5 below) is a SEPARATE, independently
tunable config value from `target_tracking`'s `appearance_reverify_similarity_threshold` — do not
collapse them into one shared value.

This module must NOT reach into `face_identity`'s or `human_detection_roi`'s private
implementation — only their `interface.py` contracts, exactly as `main.py`'s existing `face_first`
pipeline already does when calling them in sequence (§1's flow mirrors that existing Stage 1→2
call pattern for the Path A branch specifically).

### §0.4 Ambiguity Handling
If anything about how `face_identity.evaluate()`'s existing multi-face-return behavior (it
returns `List[FaceIdentityResult]`, not a single result — see `docs/modules.md`) should be
filtered/selected for THIS module's single-target-reacquisition purpose is unclear, stop and ask
rather than assume "just take the first match" is correct — there could be a specific
`matched_person_name` this module should be filtering for, and that detail matters for
correctness in a crowd with multiple registered people.

### §0.5 Mandatory Pre-Implementation Audit
1. Confirm how the calling pipeline identifies WHICH registered person this recovery episode is
   for (i.e. does `start()` need a `target_person_name` parameter to filter
   `face_identity.evaluate()`'s multi-face results correctly, rather than accepting any
   registered match?) — this is likely a real gap in §0.3's interface as drafted; flag it and
   confirm the correct fix with the user before implementing, rather than silently adding a
   parameter or silently ignoring the multi-person case.
2. Search for naming collisions with `RecoveryResult`, `target_recovery`, or similar.
3. Report findings before proceeding.

---

## §2 Purpose & Context

When `modules.target_tracking` reports `state == LOST`, this module takes over: searches the full
frame (ROI expanded from whatever narrow region tracking was using) to re-acquire the same
target, using two paths of different strength and cost.

```
target_tracking reports LOST, hands off reference_set
        |
        v
   start(reference_set, timestamp) -- resets search timer and fail counter
        |
        v
   SEARCHING loop, each frame via update():
        |
        +-- Path A (primary, ALWAYS checked first, every frame):
        |     face_identity.evaluate(frame, registry) -> is_registered_match (for the
        |     correct target person -- see S0.5 audit item #1)?
        |       YES -> face_search_fail_count resets to 0.
        |              human_detection_roi.evaluate(frame, matched_face_bbox)
        |              -> fresh person_bbox -> status=REACQUIRED, reacquired_via="face_match"
        |       NO  -> face_search_fail_count += 1
        |
        +-- Path B (fallback, ONLY attempted when face_search_fail_count >=
        |     face_search_grace_attempts -- a COUNT, not a time duration; see S4 for why):
        |     whole-frame human detection -> candidate bboxes
        |     -> appearance_verifier.verify() against reference_set for each candidate
        |     -> best score >= appearance_fallback_threshold?
        |       YES -> status=REACQUIRED, reacquired_via="appearance_fallback",
        |              reacquired_person_bbox = the matched candidate bbox DIRECTLY
        |              (do NOT call human_detection_roi for this path -- the body bbox
        |              was already found by the detection step above; re-running ROI
        |              detection on it would be pure waste, see S4.2)
        |
        +-- elapsed_search_seconds >= search_timeout_seconds (checked every frame,
              regardless of which path was tried) -> status=TIMEOUT
```

`REACQUIRED` and `TIMEOUT` are terminal for a given search episode — the caller (orchestration
layer) is responsible for calling `modules.target_tracking.reset()` with the reacquired bbox on
`REACQUIRED`, or propagating `follow_me=False`/"STOP" on `TIMEOUT`. This module does not call
`target_tracking` itself — no cross-module instance sharing, per isolation rule.

---

## §3 Path A — Face-Based Re-Acquisition (Primary)

Reuses `face_identity.evaluate(frame, registry)` exactly as the main `face_first` pipeline's
Stage 1 already does — full frame in, no ROI narrowing (recovery search intentionally covers the
whole frame, matching the "expand ROI to full frame" requirement). On a match for the correct
target person (see §0.5 audit item), immediately follow with `human_detection_roi.evaluate(frame,
matched_face_bbox)` — identical to the normal pipeline's Stage 1→2 handoff — to get a clean,
freshly-detected `person_bbox` before reporting `REACQUIRED`.

This path is treated as **sufficient on its own** — a single successful face match immediately
ends the search, no corroboration from Path B is required or checked.

---

## §4 Path B — Appearance-Based Fallback

### §4.1 Why this exists, and why it's gated
Path A can fail not because the target isn't present, but because their face isn't visible or
matchable (facing away, occluded, too far for reliable detection) — a common, expected scenario
during active following, not an edge case. Path B exists specifically for this situation. It is
deliberately WEAKER evidence (no face confirmation at all) and is therefore only consulted as a
fallback, never checked in parallel with/instead of Path A.

### §4.2 The gating value is a COUNT, not a time duration — this is deliberate, not a stylistic
choice
`face_search_grace_attempts` must be an integer count of consecutive Path-A-failure frames, NOT
a time duration. Rationale (do not silently "simplify" this back to time-based): face detection
(YuNet, full-frame) is variable-cost inference: a fixed time window could complete a very
different number of real detection attempts depending on system load that cycle, meaning a
time-based gate would give Path A an inconsistent, load-dependent number of real tries before
falling back — unfair to Path A on a slow cycle, and unnecessarily cautious on a fast one. A
count-based gate ties the threshold to actual attempts made, independent of frame rate. Contrast
with `search_timeout_seconds` (§5), which correctly remains time-based, since it bounds total
wall-clock search duration, not attempt count — do not conflate the two config values or their
units.

### §4.3 Re-acquisition without a face bbox
On a Path B match, the matched candidate bbox (from whatever whole-frame human detection this
module runs for the fallback search) is used DIRECTLY as `reacquired_person_bbox` — do not run
`human_detection_roi` afterward. That module exists to scope a search from a face bbox down to a
body; Path B has already found a body bbox directly via a different mechanism, so re-running ROI
detection on it would be redundant, wasted computation on a region already known to contain the
target.

**Stop-and-ask item:** what whole-frame human detector should Path B use for its candidate
bboxes — a fresh, independent YOLO instance (own-instance isolation, consistent with every other
module's own detector), or is there a case for reusing `human_detection`'s existing whole-frame
detection call if it happens to already be running elsewhere in the pipeline? Confirm with the
user; default to a fresh independent instance per the established isolation rule unless told
otherwise.

---

## §5 Search Timeout — Overall Abandonment Clock

`search_timeout_seconds` (config, time-based, e.g. ~1-2 minutes per the original design range) is
checked every frame regardless of which path is currently being attempted. Once
`elapsed_search_seconds >= search_timeout_seconds` with no `REACQUIRED` from either path, report
`status=TIMEOUT`. The caller is responsible for translating this into `follow_me=False` and a
"STOP" signal — this module itself does not produce robot commands, only the search-episode
status.

---

## §6 Visualization Requirement (mandatory for this module)

Standalone `visualize_target_recovery.py`:
- Simulates a search episode against a webcam feed or video file, given a pre-built or
  synthetically-constructed `reference_set` (confirm with the user the best way to supply test
  reference data without requiring a live `target_tracking` RECORDING phase to have just run).
- Displays which path is currently active/being attempted each frame, the running
  `face_search_fail_count`, and a visible countdown of `elapsed_search_seconds` against
  `search_timeout_seconds`.
- On `REACQUIRED`, clearly shows which path succeeded (`reacquired_via`) and draws the resulting
  bbox.
- Logs `RecoveryResult` fields to console per frame.

---

## §7 Documentation Requirement (mandatory)

Update `docs/modules.md`, `docs/parameters.md`, `docs/architecture.md`, following the exact
existing format/depth conventions, same as required in the other two new modules' specs. In
`docs/architecture.md`, extend the post-trigger pipeline-flow diagram (added by
`modules/target_tracking`'s documentation work) to show the full loop: trigger → tracking →
LOST → recovery (this module) → REACQUIRED → back to tracking, or → TIMEOUT → STOP. In
`docs/modules.md`'s **known limitations** subsection for this module, explicitly document: (a)
why `face_search_grace_attempts` is a count and not a time value (§4.2's rationale, in condensed
form), and (b) that Path B inherits both of `appearance_verifier`'s named risks (clothing-color
confusion, cross-domain generalization) — cross-reference rather than re-explain them in full.

---

## §8 Explicit Non-Goals

- No robot command generation (`follow_me`/"STOP" translation) — caller's job.
- No steering/angle computation of any kind.
- Does not itself call `modules.target_tracking.reset()` — the orchestration layer does that on
  seeing `status == REACQUIRED`, per §2's stated handoff boundary.
- No corroboration/combination logic between Path A and Path B beyond the strict primary/fallback
  ordering specified in §2-§4 — do not add a "both agree" mode or similar without asking first.
