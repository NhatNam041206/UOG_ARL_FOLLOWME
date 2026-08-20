# Feature Spec: Gesture Detection — Method 3, Trajectory Verifier (`modules/gesture_trajectory_verifier/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `Module_Architecture.md`, `Agent_Instruction_Framework.md`, this spec, `modules/human_detection_roi/interface.py`, and — for interface SHAPE only — `modules/gesture_condition/interface.py` and `modules/gesture_hand_keypoint/interface.py`.
- MAY NOT use: any external planning conversation, chat history, or document not physically present in the repository. MAY NOT import Method 1 or Method 2's internal detection logic — see isolation statement below.

### §0.2 Structural Placement
Lives at `modules/gesture_trajectory_verifier/`, with `interface.py`, private implementation, `test_gesture_trajectory_verifier.py`, a `reference_trajectories/` data directory (§4), and a visualization entry point (§7).

### §0.3 Interface & Isolation Contract

Third of three interchangeable gesture-detection methods (see Method 1 and Method 2 specs for the same shared-contract pattern):

```python
@dataclass
class GestureMethodResult:
    track_id: int
    is_waving: bool
    waving_state: Literal["RED", "YELLOW", "GREEN"]
    method_name: str = "trajectory_verifier"   # fixed for this module
    confidence_debug: Optional[float] = None    # best similarity score against reference set
    matched_reference_id: Optional[str] = None  # which reference trajectory scored best, for debugging
    keypoints_raw: Optional[object] = None

def evaluate(track_id: int, person_crop_bgr: np.ndarray, timestamp: float) -> GestureMethodResult:
    """
    Input: person crop from modules/human_detection_roi, same as Methods 1 and 2.
    """
```

**Isolation statement — name the specific temptation:** This module MUST NOT import or share the motion-buffer logic, keypoint extraction wrapper, or confirmation tracker from `modules/gesture_condition/` (Method 1) or `modules/gesture_hand_keypoint/` (Method 2) — even though Method 1's Gate B (motion buffer) and this module's trajectory buffer are conceptually similar (both track wrist position over time). Implement this module's buffer independently. This module DOES reuse MoveNet Lightning as its pose-keypoint source (same model as Method 1, since this method needs wrist/elbow/shoulder points, not hand landmarks) — reusing the *model* is fine (same underlying pretrained weights, loaded as a fresh instance), but reusing Method 1's *code/classes* that operate on that model's output is not. If unsure whether something counts as "the model" (fine to reuse) vs. "Method 1's logic" (must reimplement), stop and ask.

### §0.4 Ambiguity Handling
Two design points remain genuinely open from the original design discussion and must be confirmed with the user before implementation, not decided silently:
1. Time-based vs. arc-length resampling for the fixed-length trajectory (§2.3).
2. Behavior when zero or one reference trajectories are available (§4.3).

### §0.5 Mandatory Pre-Implementation Audit
1. Confirm MoveNet Lightning is accessible as a reusable model instance (per §0.3, loading a fresh instance is fine).
2. Search for any existing trajectory-comparison or DTW/similarity-matching code in the repo.
3. Report findings before proceeding.

---

## §1 Purpose & Context

Third gesture-detection method for the exploratory face-first Follow-Me pipeline:

```
full frame → face detect+match → human detection/ROI → [THIS MODULE or Method 1 or Method 2] → is_waving
```

This method captures the live motion path (trajectory) of the wrist, elbow, and shoulder over a short time window, and compares its *shape* against a set of stored reference wave trajectories using fixed-length resampling + cosine similarity (decided over DTW for simplicity, consistent with this project's "engineered/lightweight over heavier approach unless proven insufficient" principle).

**Confirmed design correction from the original single-point design:** tracking the wrist alone was found insufficient — it loses arm-shape information (bent vs. straight arm, elbow-driven vs. shoulder-driven motion can look similar as a wrist-only path). This module tracks **three points — wrist, elbow, shoulder — for ONE arm at a time**, where "which arm" is determined by this module's own comparison logic (whichever arm's trajectory produces the best similarity score against the reference set), not inherited from Method 1's Gate A. Both arms' trajectories should be computed and compared independently each evaluation cycle; report whichever scores higher.

**Confirmed design decision:** reference trajectories are a **shared, generic set** — not captured per registered person. One-time setup, reused for everyone.

---

## §2 Trajectory Capture & Processing

### §2.1 Live trajectory buffer (per arm, per track_id)

```python
@dataclass
class _TrajectorySample:
    timestamp: float
    wrist: tuple[float, float]     # (x, y), bbox-pixel space
    elbow: tuple[float, float]
    shoulder: tuple[float, float]

TRAJECTORY_WINDOW_SECONDS = <placeholder, likely similar range to Method 1's motion window, but
                              independently tunable — do not hardcode equal to Method 1's value>
```

Buffer accumulates one `_TrajectorySample` per frame per arm, subject to a confidence gate on all three keypoints (wrist, elbow, shoulder) — reuse the same confidence-gating *pattern* as Method 1 (gate closed if any keypoint is below threshold), implemented independently per §0.3.

### §2.2 Normalization (applied identically to live and reference trajectories)

Each of the three point-tracks (wrist, elbow, shoulder) is normalized the same way:
1. **Translate**: subtract the first sample's position, so the trajectory represents relative motion from its own start, not absolute frame position.
2. **Scale**: divide by a stable body-scale reference (e.g. shoulder-to-hip distance, or bbox height at capture time) — NOT wrist-to-shoulder distance alone, since that itself changes during a wave and would distort the normalization. Confirm this scale-reference choice with the user if a clearly better option exists once real data is available; a reasonable default to propose is bbox height, since it is the most stable measurement across a waving motion.

### §2.3 Fixed-length resampling — STOP AND ASK BEFORE IMPLEMENTING

```python
TRAJECTORY_RESAMPLE_LENGTH = <placeholder, e.g. start ~20 points>
```

Two resampling strategies were identified and NOT resolved:
- **Time-based**: sample at evenly spaced time intervals across the window.
- **Arc-length-based**: sample at evenly spaced distances along the path itself, which can better preserve shape fidelity for motion that speeds up/slows down mid-gesture.

Present this tradeoff to the user (arc-length is likely more robust to non-uniform wave speed but is more complex to implement correctly) and get an explicit choice before implementing. Do not default to one silently.

### §2.4 Similarity comparison

```python
def flatten_trajectory(samples: list[_TrajectorySample]) -> np.ndarray:
    # concatenate normalized (wrist, elbow, shoulder) x,y across all resampled points
    # into one flat vector for cosine similarity comparison
    ...

def trajectory_similarity(live_vec: np.ndarray, reference_vec: np.ndarray) -> float:
    # cosine similarity between the two flattened vectors
    ...

SIMILARITY_THRESHOLD = <placeholder, e.g. start ~0.7-0.85, calibrate empirically>
```

---

## §3 Combining Both Arms

```python
def gesture_candidate_this_frame(left_buffer, right_buffer, reference_set, config) -> tuple[bool, str | None, float]:
    """
    Compute similarity for whichever arm has enough buffered samples (see §2.1 minimum-
    samples gate, same pattern as Method 1's motion_min_samples), against every reference
    trajectory in reference_set. Return the best (arm, reference_id, score) triple.
    is_waving_candidate = best_score >= config.similarity_threshold.
    """
```

---

## §4 Reference Trajectory Set (build from scratch)

### §4.1 Storage
Store as a small set of pre-normalized, pre-resampled reference trajectories — e.g. in `modules/gesture_trajectory_verifier/reference_trajectories/*.npz` or similar, following whatever storage convention the §0.5 audit finds elsewhere in the project (mirror it if one exists; ask if none does).

### §4.2 Capture process
Since this is a **shared generic set** (not per-person), build a small standalone capture script (e.g. `capture_reference_trajectory.py`) that:
- Records a person performing a clean wave (webcam or pre-recorded video)
- Extracts the wrist/elbow/shoulder trajectory using the same MoveNet-based extraction as live inference
- Normalizes and resamples it per §2.2-2.3
- Saves it as a new reference entry with a unique ID

Recommend capturing multiple reference waves (different people, slightly different speeds/styles) to build a small but diverse reference set — but don't over-engineer this into a large dataset-collection tool; a simple script producing a handful of references is sufficient for initial testing.

### §4.3 Zero/one reference handling — STOP AND ASK BEFORE IMPLEMENTING
If the reference set is empty or has only one entry when this module runs, decide: should `evaluate()` return a clear "not ready" signal (e.g. `is_waving=False` with a distinct debug flag) rather than silently producing a low-confidence but technically-computed result? Present this to the user rather than deciding unilaterally — the risk of silently returning `False` from an empty reference set is that it looks identical to "genuinely evaluated and didn't match," which would be misleading during calibration.

---

## §5 Confirmation State Machine (shared pattern, fixed)

Same RED/YELLOW/GREEN debounce as Methods 1 and 2, implemented independently per §0.3 isolation statement — do not import another module's `ConfirmationTracker`.

---

## §6 Configuration

```yaml
gesture_trajectory_verifier:
  trajectory_window_seconds: null       # TODO: calibrate
  min_samples_for_comparison: null      # TODO: calibrate
  resample_length: null                 # TODO: calibrate, pending §2.3 resolution
  similarity_threshold: null            # TODO: calibrate
  confirmation_duration_seconds: null   # TODO: calibrate, independently tunable from Methods 1/2
```

---

## §7 Visualization Requirement (mandatory for this module)

Standalone visualization entry point (e.g. `visualize_gesture_trajectory_verifier.py`):

- Takes a person-crop stream as input, live-chained or pre-recorded.
- Draws the live wrist/elbow/shoulder trajectory (both arms) as it accumulates, and separately renders the best-matching reference trajectory (normalized/resampled) for visual side-by-side comparison — this is the most valuable debug view for this method specifically, since "does the shape look similar" is the whole point.
- Displays the current best similarity score and which reference/arm produced it.
- Logs `GestureMethodResult` fields to console per frame.

---

## §8 Explicit Non-Goals

- No sharing of code/state with Method 1 or Method 2 (§0.3), though reusing the MoveNet model instance itself is fine.
- No per-person reference trajectory capture — shared generic set only (§4), per confirmed project decision.
- No DTW implementation — fixed-length resampling + cosine similarity only, per confirmed project decision. If empirical testing later shows this insufficient, that is a future stop-and-ask conversation, not something to preemptively build now.
- No decision-making about which of the three methods "wins" empirically.
