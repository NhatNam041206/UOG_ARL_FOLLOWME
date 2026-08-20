# Feature Spec: Gesture Detection — Method 2, Hand Keypoint (`modules/gesture_hand_keypoint/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `Module_Architecture.md`, `Agent_Instruction_Framework.md`, this spec, `modules/human_detection_roi/interface.py`, and — for the SHAPE of the shared gesture-method interface only, not its internal logic — `modules/gesture_condition/interface.py` (Method 1, already built).
- MAY NOT use: any external planning conversation, chat history, or document not physically present in the repository. MAY NOT copy Method 1's internal Gate A/Gate B logic into this module — see isolation statement below.

### §0.2 Structural Placement
Lives at `modules/gesture_hand_keypoint/`, with `interface.py`, private implementation, `test_gesture_hand_keypoint.py`, and a visualization entry point (§6).

### §0.3 Interface & Isolation Contract

This module is one of **three interchangeable gesture-detection methods** competing to be empirically selected as the final Follow-Me gesture detector (project decision: alternatives, not ensemble — pick the best one after real-data testing). All three methods MUST expose the same shared output contract so they are swappable and comparable:

```python
@dataclass
class GestureMethodResult:
    track_id: int
    is_waving: bool          # debounced, confirmed result (see §4 confirmation state machine)
    waving_state: Literal["RED", "YELLOW", "GREEN"]
    method_name: str = "hand_keypoint"   # fixed for this module
    confidence_debug: Optional[float] = None   # method-specific debug metric, for calibration only
    keypoints_raw: Optional[object] = None     # raw MediaPipe Hands output, for debugging

def evaluate(track_id: int, person_crop_bgr: np.ndarray, timestamp: float) -> GestureMethodResult:
    """
    Input: a person bbox crop (from modules/human_detection_roi), already face-matched
    and ROI-scoped upstream. This module does its own hand detection/localization within
    that crop — it does not receive a pre-cropped hand image.
    """
```

**Isolation statement — name the specific temptation:** This module MUST NOT import, call, or share any state (motion buffers, confirmation trackers, config objects) with `modules/gesture_condition/` (Method 1) or `modules/gesture_trajectory_verifier/` (Method 3), even though all three solve the same problem and will look structurally similar. Each method must work correctly if the other two are deleted from the repo entirely. If you find yourself wanting to reuse Method 1's `ConfirmationTracker` class or similar, implement an equivalent one locally in this module instead — do not share the class or instance across modules. This duplication is deliberate, not an oversight: it keeps empirical comparison fair (each method's real, standalone cost/complexity) and keeps the modules genuinely swappable per `Module_Architecture.md`.

### §0.4 Ambiguity Handling
This entire module's core detection logic (what counts as "waving" using hand landmarks) has NOT been designed yet — only the model choice (MediaPipe Hands) has been decided. Section §3 below sets boundaries and open questions but deliberately does not fully specify the wave-detection rule. The agent must treat §3's open items as mandatory stop-and-ask points, not a license to invent a detection rule freely. Propose a concrete rule to the user and get confirmation before implementing, don't just implement your first idea.

### §0.5 Mandatory Pre-Implementation Audit
1. Confirm MediaPipe is installable in the project's environment (check for conflicts with existing dependencies, e.g. TensorFlow/PyTorch version constraints already in use for MoveNet/YOLO/OSNet).
2. Search for any existing hand-tracking or gesture-classification code in the repo.
3. Report findings, including any dependency conflicts, before proceeding.

---

## §1 Purpose & Context

Third stage of the exploratory face-first Follow-Me pipeline, one of three interchangeable gesture-detection implementations:

```
full frame → face detect+match → human detection/ROI → [THIS MODULE or Method 1 or Method 3] → is_waving
```

Method 1 (condition-based geometry: wrist/elbow/shoulder angle + motion) is already built at `modules/gesture_condition/`. This module is Method 2 — using MediaPipe Hands' 21-point hand landmark model instead of MoveNet's coarse wrist/elbow/shoulder pose keypoints, to test whether finer-grained hand geometry (finger position, palm orientation) improves wave detection accuracy over Method 1's arm-level approach.

---

## §2 Model: MediaPipe Hands

- 21 3D hand landmarks per detected hand (fingertips, knuckles, wrist), via the two-stage BlazePalm + landmark pipeline.
- Runs CPU-only, no GPU required — confirmed suitable for this project's edge hardware tier.
- **Known limitation to carry into calibration testing, not solve in code:** MediaPipe's palm detector has shown degraded accuracy in some low-light/low-resolution benchmarks (as low as ~58% in one clinical low-light study, vs. much higher in good lighting). This is directly relevant to a crowded, variably-lit campus environment. This is not something to "fix" in this module — it's a real risk to measure empirically against Methods 1 and 3 during calibration (§7). Document it in code comments so it isn't forgotten.
- Runs on the person crop from `modules/human_detection_roi`, doing its own palm detection within that crop (MediaPipe's own two-stage pipeline, not a separate module).

---

## §3 Wave Detection Logic — OPEN, REQUIRES STOP-AND-ASK BEFORE IMPLEMENTING

Unlike Method 1 (which has fully specified geometric conditions), this module's detection rule has only been scoped at a high level, not designed. The agent must propose a concrete design and get user confirmation before writing detection logic. Considerations to raise with the user, not resolve unilaterally:

1. **What hand-landmark-derived signal indicates waving?** Candidate directions to present to the user (not to choose from unilaterally):
   - Finger-spread state (open palm vs. closed) combined with wrist motion, similar in spirit to Method 1's motion gate but using the hand's own landmark centroid instead of the coarse MoveNet wrist point.
   - Palm orientation (facing camera vs. facing away) as an additional condition, since MediaPipe's landmarks can support this and Method 1's coarser keypoints cannot.
   - Whether to require a specific hand shape (open palm) at all, versus any raised moving hand regardless of finger configuration — this changes what "waving" means for this method and should be an explicit, discussed choice, not an assumption.
2. **Confidence gating** — MediaPipe returns per-landmark and per-hand confidence; establish a threshold analogous to Method 1's `confidence_threshold_pose`/`confidence_threshold_wave`, but as a new, independent config value (§5), not copied from Method 1's calibrated numbers (which won't transfer, since it's a different model with different confidence semantics).
3. **Temporal confirmation** — this module still uses the shared RED/YELLOW/GREEN confirmation pattern (§4) for debouncing across frames; the open question is only about the *per-frame* raw signal, not the debouncing mechanism, which is fixed.

**Agent instruction:** draft a specific proposed rule addressing points 1-2 above, present it to the user, and wait for confirmation before implementing. Do not implement a first guess and treat it as final.

---

## §4 Confirmation State Machine (shared pattern, fixed — not open for redesign)

Same RED/YELLOW/GREEN debounce pattern used elsewhere in this project: a raw per-frame `is_waving` candidate must hold continuously for `confirmation_duration_seconds` before `GestureMethodResult.is_waving` reports True. Implement this locally in this module (do not import Method 1's implementation — see §0.3 isolation statement), using the same transition rules:

```
RED    --[pass]-->                          YELLOW (start timer)
YELLOW --[pass, timer >= duration]-->        GREEN
YELLOW --[fail]-->                            RED (discard timer)
GREEN  --[fail]-->                            RED (discard timer)
GREEN  --[pass]-->                            GREEN (stays)
```

One `ConfirmationTracker` instance per `track_id`, reset on track loss (caller calls a `release_track(track_id)` equivalent).

---

## §5 Configuration

```yaml
gesture_hand_keypoint:
  confidence_threshold: null  # TODO: calibrate, MediaPipe-specific, not shared with Method 1
  confirmation_duration_seconds: null  # TODO: calibrate — may end up matching Method 1's value,
                                         # but must be independently tunable
  # additional keys depend on the wave-detection rule finalized per §3 — do not add keys for
  # a rule that hasn't been confirmed with the user yet
```

---

## §6 Visualization Requirement (mandatory for this module)

Standalone visualization entry point (e.g. `visualize_gesture_hand_keypoint.py`):

- Takes a person-crop stream as input (from `modules/human_detection_roi`, live-chained or pre-recorded — confirm which with the user).
- Draws all 21 MediaPipe hand landmarks on each detected hand, plus a bbox-colored indicator (red/yellow/green) reflecting the current confirmation state.
- Displays the raw per-frame candidate value and confidence alongside the debounced state, so the difference between "instant signal" and "confirmed result" is visible during testing.
- Logs `GestureMethodResult` fields to console per frame.

---

## §7 Empirical Calibration Note

Per project convention, every threshold above is a placeholder. In addition, this method should be tested specifically against the low-light/partial-occlusion conditions flagged in §2, since that is its most likely failure mode relative to Methods 1 and 3 — call this out explicitly in the test plan, not just generic accuracy testing.

---

## §8 Explicit Non-Goals

- No sharing of code, state, or config with Method 1 or Method 3 (§0.3).
- No identity verification, human detection, or ROI logic — all handled upstream.
- No decision-making about which of the three methods "wins" — that's a downstream empirical comparison the user performs after all three are built and tested, not something this module decides.
