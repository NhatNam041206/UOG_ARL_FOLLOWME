# Feature Spec: Appearance Verifier — OSNet Re-ID (`modules/appearance_verifier/`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase (`main.py`, `Module_Architecture.md`/`docs/architecture.md`,
  `docs/modules.md`, `docs/parameters.md`, `Agent_Instruction_Framework.md`), and this spec.
- MAY NOT use: any external planning conversation, chat history, or document not physically
  present in the repository. If something here assumes context you don't have, stop and ask.

### §0.2 Structural Placement
Per this repo's established convention (see every existing `modules/*/` directory): lives at
`modules/appearance_verifier/`, with `interface.py` as the only importable file, private
implementation files, `test_appearance_verifier.py`, and a standalone
`visualize_appearance_verifier.py` (§6 — required, not optional tooling).

### §0.3 Interface & Isolation Contract

```python
@dataclass
class AppearanceVerifierResult:
    match_found: bool
    best_similarity_score: float          # ALWAYS populated, even on no-match — for
                                            # calibration visibility, same pattern as
                                            # face_identity's match_confidence
    reference_frame_count: int             # how many reference frames were actually compared
                                            # against — visible so "0 references" and "compared,
                                            # didn't match" are never confused (same principle as
                                            # gesture_trajectory_verifier's MIN_REFERENCE_COUNT
                                            # not-ready floor)

def build_reference_set(person_crops: list[np.ndarray]) -> ReferenceEmbeddingSet:
    """
    Takes a list of already-cropped person-bbox images (BGR) — e.g. the frames captured
    during the calling pipeline's RECORD phase — and returns an embedded reference set
    this module can later compare against. Embedding happens once here, not per-comparison.
    """

def verify(candidate_crop: np.ndarray, reference_set: ReferenceEmbeddingSet) -> AppearanceVerifierResult:
    """
    Embeds candidate_crop and compares against every embedding in reference_set via
    cosine similarity. Returns the BEST score found, and match_found = best_score >=
    config.similarity_threshold.
    """
```

**Isolation statement — name the specific temptation:** this module MUST NOT import, call, or
share any state/instance with the teammate's separate OSNet-based Re-ID pipeline
(`UOG_ARL_FOLLOWME`, a different git repository per this project's own `docs/technologies.md`
note on the face-registry storage format), even though both use OSNet and solve a conceptually
similar problem. Load your own independent OSNet weights/session — do not attempt to import,
call, or wrap anything from that other repository. If you find a way to technically reach that
codebase (e.g. it happens to be on the Python path, or reachable via a relative import), stop
and ask before using it — reusing their trained weights file itself (not their code) may be
acceptable, but confirm with the user first rather than assuming.

This module also MUST NOT share state with any of the three existing gesture-method modules,
`face_identity`, or `human_detection_roi` — per this repo's own established "own-instance
isolation" design rule (see `docs/architecture.md`, design rule #2). This module will be
consumed by two DIFFERENT callers (a tracking module and a recovery/search module, specced
separately) — it must not assume or hardcode which caller is using it, and must not hold any
state that would make it unsafe for both to use their own independent instances of this module
simultaneously.

### §0.4 Ambiguity Handling
If any decision below is unclear — model input preprocessing details, embedding normalization,
exact OSNet variant/weights source — STOP and ask rather than guess. This is a new model
integration in this codebase; getting the preprocessing pipeline wrong silently produces a
module that runs without errors but never matches correctly, which is worse than an explicit
failure.

### §0.5 Mandatory Pre-Implementation Audit
1. Confirm what OSNet implementation/weights source is available and installable in this
   project's environment (e.g. `torchreid`, an ONNX export, or another packaging) — check for
   conflicts with the existing `torch`/`torchvision` versions already pinned in
   `requirements.txt` for `ultralytics`.
2. Search the repo for any existing appearance-embedding or Re-ID code.
3. Search for naming collisions with `AppearanceVerifierResult`, `appearance_verifier`, or
   similar.
4. Report findings, including any dependency version conflicts, before proceeding.

---

## §1 Purpose & Context

This module answers one question: **"does this new person crop look like the same person as
this earlier set of reference crops?"** — an appearance-based identity check, distinct from and
complementary to `face_identity`'s face-based check. It exists because two other modules (specced
separately) need this same capability:

1. A tracking module's periodic sanity check — catching a motion tracker silently switching to
   track a different nearby person.
2. A recovery/search module's fallback re-acquisition path — used when the primary face-based
   re-acquisition signal has failed repeatedly (the person's face isn't visible/matchable).

Both callers are out of scope for this spec — this module only provides the shared capability.

---

## §2 Model: OSNet

Use OSNet (Omni-Scale Network) for person re-identification embeddings — chosen over a generic
ResNet backbone because OSNet is purpose-built and trained specifically for the same-person-
across-views matching task, not repurposed from generic image classification.

**Two known, documented risks — both MUST be written into `docs/modules.md` and
`docs/parameters.md` as explicit calibration cautions (§7), not just code comments:**

1. **Similar-clothing confusion** — OSNet-based appearance matching struggles to distinguish
   people wearing similar-colored/styled clothing, since appearance embeddings lean heavily on
   clothing as a feature. This is a known, previously-documented limitation in this project's
   broader context (the teammate's separate Re-ID pipeline hit the same issue).
2. **Cross-domain generalization drop** — published benchmarks show OSNet's accuracy can drop
   sharply when deployed on footage meaningfully different from its training distribution
   (Market-1501-family datasets) — this project's own campus footage, lighting, and camera are
   an untested domain relative to that training data. This is a distinct risk from clothing
   confusion and must be documented separately, not folded into the same caution.

Because of both risks, `similarity_threshold` (the config value gating `match_found`) must be
treated as especially uncalibrated (🔴 in `docs/parameters.md`'s status legend) — do not set a
🟡 "starting guess" default without flagging that the guess is more likely wrong than usual for
this module, given the two named risks above.

---

## §3 Embedding & Comparison Logic

```python
def embed(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess crop_bgr per OSNet's expected input format (resize, normalization — follow
    OSNet's own documented preprocessing exactly, do not invent a preprocessing pipeline)
    and return an L2-normalized embedding vector.
    """

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Since both embeddings are L2-normalized, this reduces to a plain dot product —
    same pattern already used in modules/face_identity's matching stage. Reuse that
    mathematical pattern (not the code — own-instance isolation still applies), don't
    invent a different distance metric without asking."""
```

`build_reference_set()` embeds every provided reference crop once and stores the resulting
vectors — do not re-embed reference crops on every `verify()` call, that's wasted repeated work
for data that doesn't change during a single tracking/search episode.

`verify()` compares the candidate embedding against every stored reference embedding and returns
the single best (highest-similarity) result — same "best of set" pattern as
`gesture_trajectory_verifier`'s reference-set comparison.

---

## §4 Configuration

```yaml
appearance_verifier:
  similarity_threshold: null  # TODO: calibrate — 🔴, see §2's two named risks before trusting
                                # any starting guess here more than usual
  osnet_model_path: null      # TODO: 🟢 once set — where to load OSNet weights from
```

Both callers of this module (tracking, recovery) will define their OWN threshold config keys
in their own module sections (e.g. `appearance_reverify_similarity_threshold` for the tracking
module's periodic check, `appearance_fallback_threshold` for the recovery module's fallback
path) — they are NOT required to reuse `appearance_verifier.similarity_threshold` directly,
since the two callers may reasonably want different strictness for different purposes (a
periodic sanity check during confident tracking vs. a full re-acquisition decision with no other
corroborating evidence). Do not collapse these into one shared threshold without asking — that
would be a real behavioral coupling between two otherwise-independent callers.

---

## §5 Output Contract Detail

`best_similarity_score` must ALWAYS be populated with a real number (never `None`/`NaN` when
`reference_frame_count > 0`), even when `match_found` is `False` — this is required for
calibration visibility, exactly matching the existing pattern in `face_identity.match_confidence`
and `gesture_trajectory_verifier.confidence_debug`. Only when `reference_frame_count == 0` should
the module report a distinct "not ready" condition (mirroring `gesture_trajectory_verifier`'s
`MIN_REFERENCE_COUNT` floor pattern) rather than attempting a meaningless comparison against an
empty set.

---

## §6 Visualization Requirement (mandatory for this module)

Standalone `visualize_appearance_verifier.py`:
- Accepts two inputs: a small folder/set of "reference" images and a "candidate" image or live
  feed, so the matching behavior can be inspected in complete isolation from any tracking or
  recovery logic.
- Displays the best similarity score prominently, plus a clear match/no-match indicator based on
  the configured threshold.
- Logs `AppearanceVerifierResult` fields to console per comparison.
- Include a documented way to test the two named risk scenarios specifically (§2) — e.g. a
  suggested test procedure in the script's docstring: compare two DIFFERENT people wearing
  similar-colored clothing (should ideally score low, but may not — that's the point of testing
  it), and compare the SAME person across noticeably different lighting/distance (should score
  high).

---

## §7 Documentation Requirement (mandatory, not optional)

After implementation, update the existing project docs to reflect this new module, following
their established format exactly (match the style of existing entries for other modules):

- **`docs/modules.md`**: add a full section for `appearance_verifier`, matching the depth/format
  of existing module sections — pipeline position (describe it as a shared dependency used by
  two other modules, name them once those are built, or state "consumed by modules specced
  separately" if built first), purpose, working principle, public contract, key parameters, and
  a **known limitations** subsection explicitly covering BOTH risks from §2 — do not merge them
  into a single vague "may have accuracy issues" note; they are two distinct, separately-worth-
  testing-for risks.
- **`docs/parameters.md`**: add the `appearance_verifier` config section, following the exact
  table format (Parameter / Current / Status 🔴🟡🟢 / Meaning / Tuning notes) used by every
  other module's section. Mark `similarity_threshold` 🔴 with tuning notes that explicitly
  reference needing to test against both named risk scenarios, not just generic positive/negative
  examples.
- **`docs/technologies.md`**: add OSNet to the Models table, following the exact row format of
  existing entries (model name, format, used by, role) — and add a line under "Why these
  choices" explaining OSNet was chosen over a generic ResNet backbone specifically because it's
  purpose-trained for re-identification, not repurposed classification features, while noting
  the two accepted risks as the tradeoff.
- **`docs/architecture.md`**: add `appearance_verifier` to the repository layout listing.

Do not skip this section — this project's documentation set is explicitly maintained as source
material, not an afterthought, per `docs/README.md`'s own stated purpose.

---

## §8 Explicit Non-Goals

- No tracking logic, no recovery/search state machine — this module only answers "does this
  crop match this reference set," nothing about when/why to call it.
- No decision-making about thresholds for its two different callers' specific use cases (§4) —
  those belong in the calling modules' own config sections.
- No sharing of code/state with the teammate's separate OSNet-based Re-ID pipeline (§0.3).
