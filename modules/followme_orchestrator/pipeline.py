"""
Per-frame orchestrator sequencing (plans/08 §1): pre-trigger (face_identity ->
human_detection_roi -> gesture method -> trigger check) then, once a tracking episode is active,
post-trigger (autocar_adapter -> SteeringController; autocar_adapter's own TargetLock folds
tracking-while-present and recovery-on-loss into one state machine, so there is no separate
recovery step here). Not part of the public contract — external callers use interface.py only.

Composes across module boundaries — face_identity, human_detection_roi, the gesture methods,
autocar_adapter (a sibling file in this same package, not another module) — which is normally
forbidden by this project's own-instance isolation convention (docs/architecture.md rule #2/#3:
only main.py imports across module boundaries). This module is the ONE deliberate, documented
exception (plans/08 §0.3): it exists specifically to be the reusable, importable version of what
main.py's face_first pipeline currently does ad hoc, plus the new post-trigger tracking/steering
sequencing main.py does not do at all. It still never reaches into any composed module's PRIVATE
implementation — only public interface.py contracts, exactly as main.py already does today.

Manages a SINGLE active follow-me episode, mirroring autocar_adapter's own single-episode design
— this module can only ever follow one person at a time by construction (autocar_adapter is
itself a single-episode module-level singleton).
"""
import logging
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

import modules.face_identity.interface as face_identity_interface
import modules.human_detection_roi.interface as human_detection_roi_interface
from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
from modules.human_detection_roi.interface import evaluate as evaluate_person

from . import autocar_adapter
from .autocar_adapter import start as tracking_start, update as tracking_update
from .config import FollowMeOrchestratorConfig
from .gesture_adapter import GestureMethodAdapter
from .steering_controller import SteeringController

logger = logging.getLogger(__name__)

BboxXYWH = Tuple[int, int, int, int]


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public FollowMeCommand dataclass) so this internal module
    has no import-time dependency on interface.py — mirrors this project's established
    tuple-return convention to avoid an import cycle."""
    should_move: bool
    steering_angle_degrees: Optional[float]
    debug_state: str


class FollowMeOrchestratorPipeline:
    def __init__(self, config: FollowMeOrchestratorConfig, gesture_method: str,
                 face_registry_dir: str = "modules/face_identity/registry_data",
                 thresholds_config_path: str = "config/thresholds.yaml"):
        self.config = config
        self.registry = FaceRegistry(face_registry_dir)
        self.gesture_adapter = GestureMethodAdapter(gesture_method)
        self.steering = SteeringController(
            config.kp, config.ki, config.kd, config.max_steering_angle_degrees, config.fov_degrees,
        )

        # Eagerly load EVERY model this pipeline will need, right now, rather than lazily on
        # first use (confirmed with the user — a cold-start model load should be absorbed here,
        # at startup, not show up as a live stutter on the first real frame or — worse — at the
        # exact moment a gesture trigger fires, which is autocar_adapter.start()'s old behavior).
        # face_identity/human_detection_roi/gesture methods already build their models inside
        # configure()/their own __init__; autocar_adapter.warmup() additionally runs one
        # throwaway inference through the YOLO-pose detector and OSNet embedder, since a
        # backend's first-inference cost isn't always fully paid by construction alone.
        face_identity_interface.configure(thresholds_config_path)
        human_detection_roi_interface.configure(thresholds_config_path)
        self.gesture_adapter.warmup(thresholds_config_path)
        autocar_adapter.warmup(thresholds_config_path)

        self._tracking_active = False
        self._target_person_name: Optional[str] = None

        # Debug/visualization convenience only — the current tracked/reacquired bbox, from
        # autocar_adapter's own PUBLIC result fields (never a private reach-in from HERE; the one
        # deliberate reach into TargetLock's own internals happens inside autocar_adapter.py
        # itself, documented there). FollowMeCommand itself has no bbox field per plans/08 §0.3's
        # literal contract; visualize_followme_orchestrator.py reads this off its own private
        # pipeline instance instead, same "reach into MY OWN package's internals" pattern every
        # other module's visualize_*.py already uses.
        self.last_person_bbox: Optional[BboxXYWH] = None

        # Per-frame debug/overlay state — the raw result objects from whichever phase(s) ran
        # THIS step() call, so draw_debug() below can composite each phase's OWN draw_debug()
        # without re-running any inference. Reset at the top of whichever _step_*() branch runs;
        # not part of the tracking episode's own state (episode state lives inside
        # autocar_adapter's own engine, not here).
        self._debug_pretrigger: List[Tuple[object, object]] = []  # (FaceIdentityResult, HumanDetectionResult) per registered face
        self._debug_gesture_bbox: Optional[BboxXYWH] = None       # bbox of the last person the gesture method evaluated
        self._debug_tracking_result: Optional[object] = None       # TrackingResult, when post-trigger ran

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"followme_orchestrator: {len(missing)} steering/camera value(s) not yet "
                f"calibrated ({', '.join(missing)}) — should_move will stay False while actively "
                f"tracking, until config/thresholds.yaml's camera.fov_degrees and steering "
                f"section are filled in. Trigger detection itself is unaffected."
            )

    def step(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        if not self._tracking_active:
            return self._step_pre_trigger(frame, timestamp)
        return self._step_post_trigger(frame, timestamp)

    def _step_pre_trigger(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """Mirrors main.py's run_face_first_pipeline() exactly (plans/08 §0.5 audit item #1) —
        face_identity -> human_detection_roi -> gesture method, per registered face in frame.
        The FIRST person whose gesture reaches GREEN this frame becomes the locked target
        (only one follow-me episode can ever be active, per autocar_adapter's own design) —
        any other registered people evaluated this same frame are simply not started."""
        frame_h, frame_w = frame.shape[:2]
        face_results = [r for r in evaluate_face(frame, self.registry) if r.is_registered_match]

        self._debug_pretrigger = []
        self._debug_gesture_bbox = None
        self._debug_tracking_result = None

        # No per-frame release_track() for tracks not seen this frame (deliberately absent, not
        # just relaxed): a person briefly not matched/found (occlusion, one missed detection) is
        # not "gone for good," and every gesture method's own per-track state already self-heals
        # against real elapsed wall-clock time without any help from this loop — Method 2's
        # SequenceStateMachine resets itself via max_transition_gap_seconds the next time
        # evaluate() runs after a gap; Method 1's motion buffers and Method 3's trajectory
        # buffers both trim samples older than their own window on every call. Eagerly releasing
        # on the first missed frame wiped that state out from under those checks before they
        # ever got to run. track_id itself is bounded by the face registry's size (one entry per
        # REGISTERED person, never per stranger), so leaving state allocated indefinitely isn't a
        # real memory concern at this project's scale.
        for face in face_results:
            person = evaluate_person(frame, face.face_bbox)
            self._debug_pretrigger.append((face, person))
            if not person.person_found:
                continue
            px, py, pw, ph = person.person_bbox
            px, py = max(0, px), max(0, py)
            pw = min(pw, frame_w - px)
            ph = min(ph, frame_h - py)
            if pw <= 0 or ph <= 0:
                continue
            crop = frame[py:py + ph, px:px + pw]

            track_id = abs(hash(face.matched_person_name)) % 100000
            is_waving, _waving_state = self.gesture_adapter.evaluate(
                track_id, crop, timestamp, person_bbox_full_frame=(px, py, pw, ph),
            )
            self._debug_gesture_bbox = (px, py, pw, ph)

            if is_waving:
                tracking_start(face.matched_person_name, (px, py, pw, ph), frame, timestamp)
                self._tracking_active = True
                self._target_person_name = face.matched_person_name
                self.last_person_bbox = (px, py, pw, ph)
                self.steering.reset()
                return PipelineResult(False, None, "TRACKING_STARTED")

        return PipelineResult(False, None, "WAITING_FOR_TRIGGER")

    def _step_post_trigger(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """autocar_adapter's TargetLock folds tracking AND recovery into one state machine (see
        that module's docstring) — TRACKING while the lock holds, SEARCHING while it tries to
        reclaim a lost lock (its own ACQUIRING, running every update() call, doubling as
        recovery), LOST once recovery_timeout_seconds gives up. No separate recovery module/call
        site exists anymore."""
        self._debug_pretrigger = []
        self._debug_gesture_bbox = None

        result = tracking_update(frame, timestamp)
        self._debug_tracking_result = result

        if result.state == "TRACKING":
            if result.just_reacquired:
                # Reclaimed mid-episode after a loss — clear any stale PID windup from the gap,
                # same as target_recovery's old REACQUIRED handling.
                self.steering.reset()
            if result.person_bbox is not None:
                self.last_person_bbox = result.person_bbox
            if result.horizontal_offset is not None and self.steering.is_calibrated():
                angle = self.steering.update(result.horizontal_offset, timestamp)
                return PipelineResult(True, angle, result.state)
            # Fail-closed (confirmed with the user, matching every other module's uncalibrated
            # convention in this project): steering gains/FOV not yet set -> should_move forced
            # False even though the target is still genuinely being tracked.
            return PipelineResult(False, None, f"{result.state}_STEERING_UNCALIBRATED")

        if result.state == "SEARCHING":
            return PipelineResult(False, None, "RECOVERING")

        if result.state == "LOST":
            # Auto-resume watching for a new trigger (confirmed with the user, same convention as
            # before): flipping _tracking_active back to False here means the VERY NEXT step()
            # call falls through to _step_pre_trigger() again on its own.
            self._tracking_active = False
            self._target_person_name = None
            self.last_person_bbox = None
            return PipelineResult(False, None, "STOPPED")

        # Defensive fallback — autocar_adapter's state is always one of the three handled above.
        return PipelineResult(False, None, "UNKNOWN_TRACKING_STATE")

    def draw_debug(self, frame: np.ndarray) -> None:
        """
        Draws EVERY phase's own debug overlay onto `frame`, by calling each composed module's
        OWN draw_debug() — face_identity, human_detection_roi, the active gesture method,
        autocar_adapter's TrackingResult — rather than re-implementing any of their drawing
        logic a second time. This is the sanctioned use of this module's isolation exception
        (see interface.py's docstring): composing PUBLIC draw_debug() calls is no different from
        composing evaluate()/step() calls. Must be called AFTER step(), with the SAME frame, in
        the same iteration — it draws from state step() just populated, it does not re-run any
        inference itself.
        """
        for face, person in self._debug_pretrigger:
            face.draw_debug(frame)
            person.draw_debug(frame, face.face_bbox)

        if self._debug_gesture_bbox is not None:
            x, y, w, h = self._debug_gesture_bbox
            crop = frame[y:y + h, x:x + w]
            if crop.size > 0:
                self.gesture_adapter.draw_debug(crop, person_bbox_full_frame=self._debug_gesture_bbox)

        if self._debug_tracking_result is not None:
            self._debug_tracking_result.draw_debug(frame)
