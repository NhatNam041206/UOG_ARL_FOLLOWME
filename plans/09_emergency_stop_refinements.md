# Feature Spec: Emergency Stop Refinements + Follow-Me Composition
(`modules/emergency_stop/`, `modules/followme_orchestrator/`, `main.py`)

## §0 Instructions for the Implementing Agent

### §0.1 Reference Scope
- MAY use: the actual codebase (`modules/emergency_stop/*.py`,
  `modules/followme_orchestrator/*.py`, `main.py`), `docs/architecture.md`, `docs/modules.md`,
  `docs/parameters.md`, `docs/overlay_colors.md`, `docs/commands.md`.
- MAY NOT use: any external planning conversation, chat history, or document not physically
  present in the repository — this file is the durable record of that conversation; treat it as
  self-contained.
- Unlike `plans/01-08`, this is NOT a from-scratch module spec — `emergency_stop` and
  `followme_orchestrator` already exist and already work. This spec describes REFINEMENTS to
  existing, working code. Read the existing files fully before touching anything; do not assume
  the descriptions below fully substitute for reading `modules/emergency_stop/pipeline.py`,
  `detection.py`, `config.py`, `zones.py` and `modules/followme_orchestrator/interface.py`,
  `pipeline.py` directly.

### §0.2 Structural Placement
No new module directory. New files: `modules/emergency_stop/frame_anomaly.py`. Modified files:
`modules/emergency_stop/{config.py,detection.py,pipeline.py}`,
`modules/followme_orchestrator/{interface.py,pipeline.py}`,
`modules/followme_orchestrator/visualize_followme_orchestrator.py`, `main.py`,
`config/thresholds.yaml`.

### §0.3 Interface & Isolation Contract

**`modules/emergency_stop/config.py` — two new keys:**
```python
excluded_classes: List[str] = field(default_factory=list)   # NOT in REQUIRED_KEYS — see §2
frame_anomaly_min_stddev: Optional[float] = None             # ADDED to REQUIRED_KEYS — see §3
```

**`modules/emergency_stop/interface.py` — `EStopOutput` gains no new fields.** The new anomaly
check reports through the EXISTING `decision`/`reason` fields (`decision="UNCERTAIN"`,
`reason="frame_anomaly_low_variance"`) — do not add a separate boolean or a new `EStopDecision`
member. `EStopOutput` also gains a `draw_debug(frame)` method (it currently has none — every
other module's per-frame result type already follows this convention; `main.py`'s OLD legacy
pipeline hand-drew this overlay inline instead, which is being cleaned up as part of this work,
not preserved).

**`modules/followme_orchestrator/interface.py` — rename, not a new field:**
```python
@dataclass
class FollowMeCommand:
    is_move: bool                            # RENAMED from should_move — same meaning, same type
    steering_angle_degrees: Optional[float]
    debug_state: str
```
Same rename in `pipeline.py`'s internal `PipelineResult` NamedTuple. This is a pure rename — do
not change what the field represents (follow-me's own tracking/steering-calibration intent). The
AND-with-e-stop composition described in §5 happens OUTSIDE this module, in `main.py` — do not
add e-stop awareness inside `followme_orchestrator` itself (see §0.3's isolation note below).

**Isolation statement — name the specific temptation:** it will be tempting to have
`followme_orchestrator` import `emergency_stop` directly and fold the AND into `FollowMeCommand`
itself, since that would make `main.py`'s job trivial. **Do not do this.** `followme_orchestrator`
already composes `face_identity`, `human_detection_roi`, the gesture methods, and
`autocar_adapter` — that is its documented scope (`docs/architecture.md`'s isolation-exception
note). `emergency_stop` is a genuinely independent safety layer with its own instance, its own
calibration state, and its own failure modes — composing it belongs at `main.py`'s level, the
same place the OLD legacy `both` mode composed `emergency_stop` + `wave_facing_gate`
independently. `FollowMeCommand.is_move` continues to mean ONLY "follow-me's own intent" —
`main.py` computes the final gated value itself, does not ask `followme_orchestrator` to know
about e-stop.

### §0.4 Ambiguity Handling
- If `excluded_classes` contains a name not present in the loaded YOLO model's `model.names`
  table, log a clear warning naming the unrecognized entry and continue with the rest of the list
  — do not raise/crash the whole module over a typo in a non-required config list.
- If unsure whether the frame-anomaly check should run on the FULL frame or the runway ROI crop,
  it's FULL FRAME (deliberate — see §3.1's reasoning; the ROI crop is a narrower region that
  could still show "clear" even with the camera partially obstructed outside the trapezoid).
- If unsure whether `is_move` in `main.py`'s printed/displayed output should be follow-me's own
  value or the final e-stop-gated value, it is the FINAL gated value — that's the actual answer
  to "is the robot moving," which is the whole point of this composition (§5).

### §0.5 Mandatory Pre-Implementation Audit
1. Re-read `modules/emergency_stop/detection.py`'s current docstring ("NO class filter... this
   module detects any COCO object... confirmed with the user") — the denylist in §2 below
   REPLACES this specific prior decision with a refinement, not a reversal; update the docstring
   to describe the new denylist capability while preserving the "class-agnostic BY DEFAULT" framing
   (empty `excluded_classes` == the exact previous behavior, unchanged).
2. Confirm `ultralytics.YOLO.track()`'s `classes=` parameter accepts an explicit id list (it
   does — `human_detection`'s own person-only filter already uses this exact mechanism,
   `classes=[0]`) before relying on it for §2's implementation.
3. Confirm no other file already defines a `frame_anomaly` name, module, or similar, before
   adding `modules/emergency_stop/frame_anomaly.py`.
4. Report findings before proceeding, same as every other plan in this directory.

---

## §1 Purpose & Context

Three related but separable pieces of work, arising from the same conversation:

1. **`emergency_stop` currently treats every detected object identically regardless of class** —
   deliberate (see `detection.py`'s docstring), but produces no way to quiet chronically-irrelevant
   COCO classes that would never physically appear as a floor-level obstacle.
2. **`emergency_stop` has no defense against its own detector going blind or misleading** — a
   covered lens or a collision-imminent frame (an object filling the entire FOV at point-blank
   range) can structurally fail to produce a normal, confident YOLO detection at all, since the
   detector needs a recognizable object *shape* with edges/context — the single most dangerous
   scenario is exactly the one this module currently has no direct check for.
3. **`emergency_stop` isn't wired into `main.py`'s `followme` mode at all today** — `should_move`
   (being renamed `is_move`) is currently controlled ENTIRELY by follow-me's own tracking/steering
   state, with no safety-layer veto. This composes `emergency_stop`'s decision into the actual
   displayed/reported movement decision.

---

## §2 Class Denylist (`excluded_classes`)

### §2.1 Why a denylist, not an allowlist
An allowlist (e.g. "only person/table/chair") was considered and rejected during planning: it
would make any object type not on the curated list — a mop bucket, a backpack on the floor, a
traffic cone, construction debris, spilled material — invisible to the safety system, even though
each is a completely real physical hazard. `emergency_stop`'s whole reason for existing (per its
own `docs/modules.md` entry) is being the SOLE collision-avoidance layer with no other sensor
backstop — narrowing its detection scope to a preset category list is a real safety regression,
not a refinement. A denylist keeps "unrecognized/uncategorized object → still trips the safety
net" as the default, while allowing specific, deliberately-chosen classes to be excluded.

### §2.2 Config
`config/thresholds.yaml`'s `emergency_stop:` section gains:
```yaml
excluded_classes: []  # e.g. ["kite", "frisbee", "sports ball"] — COCO class NAMES, not ids.
```
NOT added to `REQUIRED_KEYS` — an empty list is a fully valid, safe default that reproduces
today's exact behavior unchanged. This is additive capability; introducing it must not silently
narrow the safety net for anyone who doesn't explicitly opt in. Leave the shipped default EMPTY
— do not pre-populate it with the `kite`/`frisbee`/`sports ball` example classes; that decision
belongs to whoever calibrates this module for the actual deployment environment.

### §2.3 Implementation
`modules/emergency_stop/detection.py`'s `EStopDetector.__init__` resolves `excluded_classes`
(names) against `self.model.names` (the loaded model's own `{id: name}` table) into the
COMPLEMENT set — i.e. build `included_class_ids = [i for i in model.names if name_of(i) not in
excluded_classes]` once, at construction time — and pass that as `model.track(...,
classes=included_class_ids)`. Use `ultralytics`' own native class-filtering argument (the same
mechanism `human_detection`'s person-only filter already uses via `classes=[0]`) rather than
post-filtering the returned detections — cheaper (the model itself skips those classes) and
reuses an existing pattern instead of inventing a new one.

Unrecognized names in `excluded_classes` (typos, classes not in this model's 80-class COCO set):
log a warning naming the specific unrecognized entry, skip it, continue with the rest — never
raise/crash the whole module over one bad config entry (same "plumbing tolerates the environment"
convention every other module in this project follows).

---

## §3 Frame-Level Anomaly Check (camera obstructed / point-blank collision)

### §3.1 Why global variance, not a floor-color baseline
A "compare against the floor's expected color" approach was considered and rejected: floor color
and lighting vary by room across a school environment, so a fixed reference color would need
either hand-calibration per deployment location or a continuously-adapting rolling baseline
(added statefulness/complexity, and a real risk of the baseline drifting to "normal" while slowly
approaching an obstruction). Instead: convert the FULL FRAME (not the narrower runway ROI crop —
an obstruction could sit outside the trapezoid while the crop still looks clear) to grayscale and
compute its standard deviation. A normal scene — visible floor, walls, distant background, edges
between surfaces — has real spatial variance. Both target scenarios collapse that variance toward
zero: a covered lens is one uniform color/texture; an object filling the entire FOV at point-blank
range is one continuous surface with no visible depth/edges anywhere in frame. This needs no
history, no per-room recalibration, and is cheap enough to run before the YOLO detector.

**Accepted limitation, stated explicitly (do not "fix" this without asking first):** this check
cannot distinguish "camera covered" from "collision imminent" from each other. Both are reported
identically. This is intentional, not an oversight — both situations correctly resolve to the
same safe action (refuse to confirm the path is clear), and this module already treats multiple
structurally-different situations identically this way (`low_confidence_detection`,
`uncalibrated_config`, and `invalid_frame` are all `UNCERTAIN` too, for the same reason).

### §3.2 Config
`frame_anomaly_min_stddev: null` — ADDED to `REQUIRED_KEYS` (now 11 required keys, not 10). Fails
closed exactly like every other threshold in this module: while `null`, `missing_keys()` already
catches it before this check is ever reached (the module is already fully `UNCERTAIN` until ALL
required keys are calibrated, so this new key doesn't change behavior in isolation — it only
takes effect once the module is otherwise fully calibrated).

### §3.3 Implementation
New file `modules/emergency_stop/frame_anomaly.py`:
```python
def is_anomalous(frame: np.ndarray, min_stddev: float) -> bool:
    """True if `frame` is suspiciously uniform (grayscale stddev below min_stddev) — see
    plans/09 §3.1 for why this catches both a covered lens and a point-blank collision without
    needing a floor-color baseline."""
```
Called from `EmergencyStopPipeline.process_frame()` — placement: immediately after the existing
`frame is None`/`invalid_frame` check, BEFORE the ROI crop and `self.detector.track(...)` call.
On a positive hit: `return self._decide("UNCERTAIN", "frame_anomaly_low_variance", None, None,
now)` — same shape as every other early-return in this method, no new return-tuple shape needed.
Running this before detection also means a genuinely obstructed frame skips the (wasted,
potentially misleading) YOLO inference entirely.

---

## §4 `is_move` Rename (`modules/followme_orchestrator/`)

Pure rename, `should_move` → `is_move`, in:
- `modules/followme_orchestrator/interface.py`'s `FollowMeCommand` dataclass field.
- `modules/followme_orchestrator/pipeline.py`'s internal `PipelineResult` NamedTuple field, and
  every place inside that file constructing one.
- `modules/followme_orchestrator/visualize_followme_orchestrator.py`'s references to
  `result.should_move`.
- `main.py`'s `run_followme_pipeline`'s references to `command.should_move`.
- `modules/followme_orchestrator/interface.py`'s `draw_steering_arrow()`, which currently checks
  `command.should_move` to decide whether to draw anything.

No behavioral change from the rename alone — see §5 for the actual new behavior (e-stop gating).

---

## §5 Composing `emergency_stop` into `main.py`'s `followme` Mode

### §5.1 Where the composition lives
`main.py`'s `run_followme_pipeline`, NOT inside `followme_orchestrator` (see §0.3's isolation
note). Construct one `EmergencyStopModule()` instance (own-instance isolation — its own YOLO,
never shared with `human_detection_roi`'s or `autocar_adapter`'s own instances, exactly the
pattern every other module in this project already follows) before the frame loop starts, same
place `followme_orchestrator.configure()` is already called.

### §5.2 Per-frame flow
Each iteration, independently (neither depends on the other's output, same as the old legacy
`both` mode ran `emergency_stop` + `wave_facing_gate` on the same frame):
```python
estop_output = estop.process_frame(frame)
command = step(frame, timestamp)  # followme_orchestrator, unchanged internally
final_is_move = command.is_move and estop_output.decision == EStopDecision.GO
```
`final_is_move` is the ACTUAL answer to "is the robot moving this frame" — e-stop can force it
`False` even when follow-me has a locked target and a calibrated steering command; follow-me
cannot override e-stop. `command.is_move` itself is untouched by this — it continues to mean
only "follow-me's own intent," per §0.3.

### §5.3 Known, intended consequence — state clearly in code comments where this is wired up
`emergency_stop` has 11 required config keys, ALL `null` by default (10 existing + the new
`frame_anomaly_min_stddev` from §3.2) — meaning it reports `UNCERTAIN` on every single frame until
fully calibrated. The moment this composition lands, `followme` mode's real `final_is_move` will
default to ALWAYS `False`, regardless of tracking state, until someone calibrates
`emergency_stop`'s section of `config/thresholds.yaml` too. This is the intended fail-closed
behavior (consistent with this project's convention everywhere else), not a bug — but it IS a
real change to `followme` mode's default observable output, and should be called out plainly in
the commit/PR description, not left for someone to discover by confusion.

### §5.4 `--modules pretrigger` is explicitly OUT of scope
`pretrigger` produces no movement command at all today — there is no natural `is_move`-equivalent
value for `emergency_stop` to gate there. Do not add `emergency_stop` to `pretrigger` as part of
this work without a separate, explicit decision.

---

## §6 Overlay/Display Requirements (mandatory, `--show` on `followme` mode)

- `EStopOutput.draw_debug(frame)` (new method, §0.3) — same color scheme the OLD legacy pipeline
  used inline (`GO`=green, `STOP`=red, `UNCERTAIN`=yellow — see `docs/overlay_colors.md`'s
  existing `emergency_stop` entry for the exact BGR values already documented), plus `reason` and
  `zone` text. Drawn every frame `--show` is on, not gated by `--debug` — e-stop's decision is a
  safety-relevant status, not a per-module debug readout (same treatment `draw_steering_arrow`
  already gets).
- A clearly separate on-screen line for the FINAL `final_is_move` (§5.2) — distinct from
  follow-me's own `command.is_move`/`debug_state` line, so a viewer can see both "does follow-me
  want to move" and "is e-stop allowing it" and the resulting final decision, not just one merged
  boolean.
- `draw_steering_arrow(frame, command)` — call this ONLY when `final_is_move` is `True`, not
  merely `command.is_move`. Do not change `draw_steering_arrow`'s own internal check (§4 already
  scoped it to `command.is_move`, which is correct for `followme_orchestrator`'s own isolated
  concern) — instead, gate the CALL SITE in `main.py` on `final_is_move`, so the arrow never
  implies movement that e-stop is actually blocking.
- Console print line (both `--show` and headless) includes e-stop's `decision`/`reason` and the
  final `is_move`, alongside follow-me's existing fields.

---

## §7 Documentation Requirement (mandatory)

Update, following the exact existing format/depth conventions already established in each file:
- `docs/modules.md`: `emergency_stop`'s section — document `excluded_classes` and the frame-anomaly
  check in "Working principle," update "Public contract"/"Key parameters" (11 required keys, not
  10), add both to "Known limitations" (the denylist's "unknown object still trips it" tradeoff
  already documented as intentional; the anomaly check's covered-vs-collision ambiguity from
  §3.1). `followme_orchestrator`'s section — `is_move` rename, the new e-stop composition.
- `docs/parameters.md`: `emergency_stop` section — add `excluded_classes` (🟢 working default,
  empty) and `frame_anomaly_min_stddev` (🔴 uncalibrated) rows.
- `docs/overlay_colors.md`: confirm the existing `emergency_stop` color table still applies
  (it should — no new decision states, just a new method exposing the same colors + new reason
  strings under the existing `UNCERTAIN`/yellow bucket); add the final-`is_move` display line and
  the steering-arrow gating change.
- `docs/commands.md`: note that `main.py --modules followme` now also depends on
  `emergency_stop`'s calibration state (§5.3) for its real `is_move` output.
- `docs/architecture.md`: extend the `followme` composition description to mention `main.py` now
  also composes `emergency_stop`, independently of `followme_orchestrator` itself (isolation note
  from §0.3, restated here for whoever reads architecture.md without having read this plan file).

---

## §8 Explicit Non-Goals

- No allowlist / curated "general objects" class list — explicitly rejected in §2.1; a denylist
  only.
- No rolling/adaptive floor-color baseline — explicitly rejected in §3.1; global grayscale
  variance only, no history.
- No attempt to distinguish "camera covered" from "collision imminent" from each other — both
  intentionally resolve to the identical `UNCERTAIN` outcome (§3.1).
- No e-stop awareness inside `followme_orchestrator` itself — the AND composition lives in
  `main.py` only (§0.3, §5.1).
- No change to `--modules pretrigger` (§5.4).
- No new `EStopDecision` enum member, and no new `FollowMeCommand`/`PipelineResult` field beyond
  the `is_move` rename — the anomaly check reports through existing fields (§0.3).
