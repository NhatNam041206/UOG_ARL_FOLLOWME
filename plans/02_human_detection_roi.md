# Feature Spec: Human Detection (ROI-Scoped) (`modules/human_detection_roi/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase, `Module_Architecture.md`, `Agent_Instruction_Framework.md`, this spec, and `modules/face_identity/interface.py` (to know exactly what input this module receives).
- MAY NOT use: any external planning conversation, chat history, or document not physically present in the repository.

### §0.2 Structural Placement
Per `Module_Architecture.md`: lives at `modules/human_detection_roi/`, with `interface.py`, private implementation, `test_human_detection_roi.py`, and a visualization entry point (§5).

### §0.3 Interface & Isolation Contract

**Typed input/output contract:**

```python
@dataclass
class HumanDetectionResult:
    person_found: bool
    person_bbox: Optional[tuple[int, int, int, int]]  # (x, y, w, h), FULL FRAME pixel space
    detection_confidence: Optional[float]

def evaluate(frame: np.ndarray, matched_face_bbox: tuple[int, int, int, int]) -> HumanDetectionResult:
    """
    Input: the full frame, PLUS the matched face bbox from modules/face_identity (only a
    face that already matched a registered person should reach this module — see §1).

    This module runs person/body detection SCOPED to a region around the given face bbox,
    not on the whole frame. It does not re-verify identity — identity confirmation already
    happened in the face_identity module. This module's only job is: given "the person is
    here, roughly, because their face is here," find their full body bbox.
    """
```

**Isolation statement — name the specific temptation:** This module MUST NOT call or share state with the teammate's existing detection/tracking pipeline, even though both do "person detection." That pipeline serves the authoritative Stage 2 trigger; this module serves the separate exploratory face-first pipeline (see project decision: face-based matching stays independent of OSNet Re-ID). Do not import from it. Also do not reach back into `modules/face_identity/` internals — only its public `FaceIdentityResult`/bbox output, already passed in as this module's input, may be used.

### §0.4 Ambiguity Handling
If the ROI-scoping strategy (how large a region around the face bbox to search — see §2) is unclear or you're tempted to guess a multiplier, STOP and ask rather than pick a number silently.

### §0.5 Mandatory Pre-Implementation Audit
Before writing new code:
1. Search for any existing person/human detector already available in the project (the project's memory indicates YOLO is already in the stack for the teammate's pipeline — check if it's accessible/reusable here, or if a separate instance is required per the isolation rule in §0.3).
2. Search for naming collisions with `HumanDetectionResult`, `human_detection_roi`, or similar.
3. Report findings and ask before proceeding — in particular, ask explicitly whether reusing the *same YOLO model weights* (loading a fresh, separate instance — not sharing a live object/session) is acceptable, since that's different from sharing state, and the isolation rule in §0.3 is about not sharing *live pipeline state*, not about being forbidden from using the same underlying model file.

---

## §1 Purpose & Context

Second stage of the exploratory face-first Follow-Me pipeline:

```
full frame → face detect+match → [THIS MODULE: human detection, ROI-scoped] → gesture method 1/2/3
```

Per the confirmed pipeline design: once a face has been detected AND matched to a registered person (by `modules/face_identity/`), this module finds that same person's full-body bounding box — but restricts its search to a region around the matched face, not the whole frame. This is a deliberate scoping optimization (cheaper than full-frame detection) and also a correctness measure (reduces the chance of picking up a different person's body in a crowd).

This module does **not** do identity verification of any kind — that already happened upstream. If the face bbox it receives wasn't actually a match, that's a face_identity module concern, not this module's.

---

## §2 ROI Scoping Strategy

Given `matched_face_bbox`, define a search region for the body detector that is larger than the face bbox alone (a body extends well below and around a face) but not the full frame.

```python
ROI_EXPANSION_FACTOR = <placeholder, e.g. start ~4-6x the face bbox height, downward-biased
                         since the body extends mostly below the face, not above>
```

**Stop-and-ask item, not a default to fill in:** the exact expansion strategy (a fixed multiplier on face bbox size vs. a fixed pixel margin vs. something proportional to expected person height at that distance in frame) is genuinely ambiguous without seeing real footage. Present the tradeoffs to the user and get a decision rather than picking one. A reasonable starting proposal to offer: expand primarily downward and laterally from the face bbox, since the body is below the face, not above it — but confirm this framing with the user before implementing.

```python
def compute_roi(face_bbox: tuple[int,int,int,int], frame_shape: tuple[int,int], config) -> tuple[int,int,int,int]:
    """
    Returns an expanded region (clipped to frame boundaries) within which the body
    detector will search. See stop-and-ask note above before finalizing the expansion logic.
    """
```

**Fallback behavior:** if the body detector finds nothing within the scoped ROI, this module should NOT silently fall back to searching the full frame without being told to — that would defeat the purpose of ROI-scoping and could pick up the wrong person. Report `person_found = False` and let the caller decide whether to retry with a larger ROI on a subsequent frame. Flag this as a stop-and-ask item if the user wants different fallback behavior.

---

## §3 Detection Model

Use the project's existing YOLO integration if the §0.5 audit confirms one is accessible and reusable without violating the isolation rule (i.e., loading a fresh model instance is fine; importing the teammate's live pipeline object is not). If no reusable YOLO integration is found, stop and ask before adding a new object-detection dependency.

---

## §4 Output Contract Detail

`person_bbox` returned by this module must be in **full-frame pixel coordinates**, not ROI-relative — downstream gesture methods expect to crop directly from the original frame using this bbox. Do not return ROI-relative coordinates; convert back to full-frame space before returning.

---

## §5 Visualization Requirement (mandatory for this module)

Standalone visualization entry point (e.g. `visualize_human_detection_roi.py`):

- Takes a webcam feed or video/image input, matching the pattern established in `modules/face_identity`'s visualization tool if the agent has access to it, or the existing root-level main file's conventions.
- For each frame, draws:
  - The face bbox (input to this module) in one color
  - The computed ROI search region (§2) in a second, distinct color
  - The final detected person bbox (output) in a third color
- Must be runnable in isolation, taking `modules/face_identity`'s output as its input (either by chaining live, or by accepting pre-recorded `FaceIdentityResult` data for testing without a live camera) — confirm with the user which mode is wanted if not obvious.
- Logs `HumanDetectionResult` fields to console for inspection without watching video.

---

## §6 Configuration

```yaml
human_detection_roi:
  roi_expansion_factor: null  # TODO: calibrate empirically, pending stop-and-ask resolution in §2
  detection_confidence_threshold: null  # TODO: calibrate empirically
```

---

## §7 Explicit Non-Goals

- No identity verification — that's `modules/face_identity`'s job, already done upstream.
- No gesture detection of any kind.
- No sharing of live state/instances with the teammate's Re-ID/tracking pipeline (§0.3).
- No automatic full-frame fallback search without explicit confirmation (§2).
