# Feature Spec: Face Detection + Face Matching (`modules/face_identity/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase (files, comments, structure) at the project root, `Module_Architecture.md`, `Agent_Instruction_Framework.md`, and this spec.
- MAY NOT use: any external planning conversation, chat history, or document not physically present in the repository — even if such material is referenced elsewhere. If something in this spec seems to assume context you don't have (e.g. exact repo layout, existing config file names), stop and ask rather than guess.

### §0.2 Structural Placement
Follow `Module_Architecture.md` for the overall module shape. This feature's module lives at `modules/face_identity/`, containing:
- `interface.py` — the only file other modules or `main.py` may import from
- private implementation files (face detection wrapper, embedding wrapper, matching logic, registry I/O)
- `test_face_identity.py`
- a `visualize_face_identity.py` (or equivalent) entry point — see §5 Visualization Requirement below; this is required for this module, not optional tooling

### §0.3 Interface & Isolation Contract

**Typed input/output contract:**

```python
@dataclass
class FaceIdentityResult:
    face_found: bool
    face_bbox: Optional[tuple[int, int, int, int]]  # (x, y, w, h) in FULL FRAME pixel space
    is_registered_match: bool
    matched_person_id: Optional[str]
    match_confidence: Optional[float]     # embedding similarity score, for debugging/calibration
    face_detection_confidence: Optional[float]

def evaluate(frame: np.ndarray, registry: FaceRegistry) -> list[FaceIdentityResult]:
    """
    Input: the FULL raw frame (not a crop — this module owns face detection from scratch,
    per the confirmed pipeline: face detection runs first, before any other cropping).

    Returns a list because a frame may contain zero, one, or multiple faces. The caller
    (pipeline orchestration / next stage) decides what to do with multiple matches — this
    module does not pick "the" person, it reports everything it found.
    """
```

**Isolation statement — name the specific temptation:** This module MUST NOT reuse or share state/instances with the teammate's OSNet-based Re-ID/tracking pipeline, even though both systems are conceptually "identity verification." They are separate, parallel identity-matching systems per explicit project decision — this module's face-based match is exploratory and NOT the authoritative identity signal Stage 2 relies on. Do not import, call, or share embeddings/registries with the OSNet verifier module. Do not attempt to "reconcile" or cross-validate against it unless separately instructed. If you find yourself wanting to import anything from the teammate's Re-ID module, stop and ask first.

Also do not share any state/instances with `modules/gesture_condition/` (Method 1) or any other gesture-detection module — this module's only job is producing `FaceIdentityResult`; it does not know about waving, gestures, or Stage 2 trigger logic at all.

### §0.4 Ambiguity Handling
If any design or implementation decision in this spec is unclear, underspecified, or you find yourself about to make an assumption to fill a gap — STOP and ask the user. Do not self-decide. This applies especially to model hyperparameters, file formats, and anything marked "TODO — placeholder" below.

### §0.5 Mandatory Pre-Implementation Audit
Before writing any new code:
1. Search the existing codebase for any existing face detection, face embedding, or registration mechanism — even partial or unused.
2. Search for naming collisions: does anything already use `FaceIdentityResult`, `face_identity`, `registry`, or similar names?
3. Report findings to the user before proceeding. Ask whether to merge with, replace, or build fully separate from anything found. Do not decide this unilaterally.

---

## §1 Purpose & Context

This module is the FIRST stage of a larger exploratory Follow-Me gesture pipeline:

```
full frame → [THIS MODULE: face detect + match] → [human detection/ROI] → [gesture method 1/2/3] → is_waving
```

Given a full camera frame, this module:
1. Detects all faces present (no identity yet).
2. For each detected face, extracts an embedding and compares it against a registered-person face registry.
3. Reports which face(s), if any, match a registered person.

This is a **standalone, exploratory pipeline** — separate from and not integrated with the teammate's OSNet-based Re-ID/tracking system (see §0.3 isolation statement). Its output feeds only into this pipeline's own downstream stages (human detection, gesture methods), not into the authoritative Stage 2 trigger.

There is currently **no existing face registration system** — this must be built from scratch as part of this module (see §4).

---

## §2 Models

### §2.1 Face Detection: YuNet
- Use YuNet via OpenCV (`cv2.FaceDetectorYN`) — already bundled in OpenCV, no new dependency required.
- Confirm the OpenCV version in the project already includes `FaceDetectorYN` before assuming it's available; if not, flag the version requirement to the user rather than silently downgrading to a different detector.
- Output: face bounding box(es) in full-frame pixel coordinates, plus a detection confidence score per face.

### §2.2 Face Embedding: MODEL CHOICE PENDING — STOP AND ASK

**Do not pick a default here.** Two candidates were researched, each with a real tradeoff the user has not yet resolved:

- **ArcFace (via InsightFace)** — field-standard accuracy (~99.86% LFW), but InsightFace's pretrained models are licensed for non-commercial research use by default; commercial use requires separate licensing. For a university project this is very likely fine, but has not been explicitly confirmed by the user.
- **EdgeFace or MobileFaceNet** — lighter-weight alternatives (EdgeFace: 1.77M params, 99.73% LFW) without the InsightFace licensing question, at a modest accuracy cost.

**Agent instruction:** before implementing this section, present both options to the user (accuracy/weight/licensing tradeoff as summarized above) and get an explicit choice. Do not default to either. This is a stop-and-ask item, not a placeholder to fill with your best guess.

---

## §3 Face Matching Logic

```python
SIMILARITY_THRESHOLD_FACE_MATCH = <placeholder, pending calibration, see config/thresholds.yaml>

def match_face(face_crop: np.ndarray, registry: FaceRegistry) -> tuple[bool, Optional[str], Optional[float]]:
    """
    1. Extract embedding from face_crop using the chosen embedding model (§2.2).
    2. Compare against every registered person's stored embedding(s) in the registry
       (cosine similarity or the embedding model's standard distance metric — use
       whatever InsightFace/EdgeFace's own documentation recommends, don't invent a
       new metric).
    3. Return the best match if its similarity clears SIMILARITY_THRESHOLD_FACE_MATCH,
       else (False, None, best_score_seen) so the caller can still see how close the
       nearest miss was, for calibration/debugging purposes.
    """
```

**Known limitation to document, not solve here:** a single frame's face match is not debounced/confirmed over time in this module — that temporal confirmation (if needed) is a decision for the pipeline orchestration layer that calls this module repeatedly, not this module's job to implement internally. State this explicitly in code comments so it isn't silently assumed to already exist.

---

## §4 Registration System (build from scratch)

No prior face registration exists in this project. This module must include:

1. **Capture:** a way to capture face crop(s) for a person being registered (webcam capture, or load from a folder of images — ask the user which capture method they want if not obvious from existing patterns elsewhere in the repo, e.g. the existing body-registration system's capture pattern in `registration.py`, if the agent finds it during the §0.5 audit).
2. **Embedding + storage:** extract the embedding for each captured face and store it, associated with a person ID/name. Mirror the existing project's storage pattern if the audit (§0.5) finds one (e.g. the `.npz` per-person registry pattern used for OSNet) — do not invent an incompatible new format without checking first.
3. **Multiple samples per person:** capture more than one face sample per person (multiple angles/expressions) if practical, since a single reference face is fragile — but confirm this against what the audit finds for the existing registration pattern; don't assume without checking.

---

## §5 Visualization Requirement (mandatory for this module)

The user needs to visually verify this module's behavior independently of the rest of the pipeline. Build a standalone visualization entry point (e.g. `visualize_face_identity.py`) that:

- Accepts a webcam feed or a video file / image folder as input (whichever is more practical given the existing repo's conventions — check for an existing pattern in the root-level main file the user referenced before inventing a new one).
- For each frame, draws:
  - A bounding box around every detected face (regardless of match status)
  - A visually distinct marker/color for faces that match a registered person vs. faces that don't
  - The matched person's ID/name and similarity score as on-screen text, for any match
- Runs standalone — this module must be testable and visually verifiable without any other pipeline stage (human detection, gesture methods) being implemented yet.
- Prints or logs `FaceIdentityResult` fields to console/log for frames where a face is detected, so numeric output is inspectable even without watching the video.

---

## §6 Configuration

All placeholder thresholds go in `config/thresholds.yaml` (or wherever the existing project convention places them — check during §0.5 audit), never hardcoded inline:

```yaml
face_identity:
  similarity_threshold_face_match: null  # TODO: calibrate empirically
  face_detection_confidence_threshold: null  # TODO: calibrate empirically
```

---

## §7 Explicit Non-Goals

- This module does NOT do human/body detection — that's the next pipeline stage (`modules/human_detection_roi/`), separately specced.
- This module does NOT do gesture detection of any kind.
- This module does NOT reconcile with or feed into the teammate's OSNet Re-ID pipeline (§0.3).
- This module does NOT implement temporal confirmation/debouncing of matches across frames — single-frame match only, per §3's documented limitation.
