"""
FollowMe Orchestrator + Steering Controller — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.followme_orchestrator. Everything
else in this package (config.py, gesture_adapter.py, steering_controller.py, pipeline.py) is an
internal implementation detail and may change without notice.

Purpose (plans/08_followme_orchestrator_steering.md): composes the ENTIRE face-first pipeline
plus the post-trigger tracking/recovery modules into one steppable unit, so a caller doesn't
need to hand-wire seven modules together the way main.py's script code currently does for the
pre-trigger portion only.

    step(frame, timestamp):
      NOT currently tracking -> face_identity -> human_detection_roi -> gesture method ->
        is_waving GREEN? -> modules.target_tracking.start(...)
      currently tracking -> modules.target_tracking.update(...)
        RECORDING/TRACKING -> SteeringController.update(horizontal_offset, timestamp)
        LOST -> modules.target_recovery.start/update(...)
          REACQUIRED -> modules.target_tracking.reset(...), resume steering next cycle
          TIMEOUT -> stop, auto-resume watching for a fresh trigger (confirmed with the user —
                     no separate reset call needed; the next step() call re-arms itself)
          SEARCHING -> should_move=False (robot moves forward only while actively following)

ISOLATION EXCEPTION, stated explicitly, not silently overridden (plans/08 §0.3): this module is
the ONE deliberate exception to "own-instance isolation, no module imports another's
interface.py except through main.py" (docs/architecture.md rule #2/#3). It exists specifically to
be the reusable, importable version of what main.py's script-level code currently does ad hoc for
the pre-trigger portion of the face_first pipeline — a composition root, not a rule violation.
It still never reaches into any composed module's PRIVATE implementation — only public
interface.py contracts, exactly as main.py already does today.

Manages a SINGLE active follow-me episode — mirrors target_tracking/target_recovery's own
single-episode design (both are themselves module-level singletons; this orchestrator can only
ever follow one registered person at a time by construction, the same as those two modules).

`configure(gesture_method=...)` MUST be called before the first step() — unlike every other
module's configure(), there is no sensible default gesture method to lazily initialize with
(mirrors main.py's own --gesture-method being a required flag for --modules face_first).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import FollowMeOrchestratorConfig, load_config
from .pipeline import FollowMeOrchestratorPipeline

__all__ = ["FollowMeCommand", "configure", "step", "draw_debug"]


@dataclass
class FollowMeCommand:
    should_move: bool                       # True = move forward, False = stop. No speed
                                              # parameter — speed is a downstream, separate concern.
    steering_angle_degrees: Optional[float]  # signed angle for the Ackermann servo; None when
                                              # should_move is False (no meaningful steering
                                              # target when stopped)
    debug_state: str                         # current high-level pipeline state, for logging/
                                              # visualization (e.g. "WAITING_FOR_TRIGGER",
                                              # "TRACKING_STARTED", "RECORDING", "TRACKING",
                                              # "RECORDING_STEERING_UNCALIBRATED",
                                              # "TRACKING_STEERING_UNCALIBRATED", "RECOVERING",
                                              # "REACQUIRED_RESUMING", "STOPPED") — exact state
                                              # names are this module's own design choice, not
                                              # fixed by the originating spec


_pipeline_singleton: Optional[FollowMeOrchestratorPipeline] = None


def configure(gesture_method: str, thresholds_config_path: str = "config/thresholds.yaml",
              face_registry_dir: str = "modules/face_identity/registry_data") -> None:
    """
    REQUIRED before the first step() call — (re)initializes the module-level orchestrator.
    `gesture_method`: "condition" (Method 1) | "hand_keypoint" (Method 2) |
    "trajectory_verifier" (Method 3), same choices as main.py's --gesture-method.
    """
    global _pipeline_singleton
    config: FollowMeOrchestratorConfig = load_config(thresholds_config_path)
    _pipeline_singleton = FollowMeOrchestratorPipeline(config, gesture_method, face_registry_dir)


def _get_pipeline() -> FollowMeOrchestratorPipeline:
    if _pipeline_singleton is None:
        raise RuntimeError(
            "modules.followme_orchestrator: configure(gesture_method=...) must be called before "
            "step() — there is no sensible default gesture method to lazily initialize with."
        )
    return _pipeline_singleton


def step(frame: np.ndarray, timestamp: float) -> FollowMeCommand:
    """The single method the rest of the system (or a human operator's test harness) calls, once
    per frame, to get the current robot command."""
    result = _get_pipeline().step(frame, timestamp)
    return FollowMeCommand(
        should_move=result.should_move,
        steering_angle_degrees=result.steering_angle_degrees,
        debug_state=result.debug_state,
    )


def draw_debug(frame: np.ndarray) -> None:
    """
    Draws EVERY phase's own debug overlay onto `frame` — face_identity's bbox+match,
    human_detection_roi's ROI+person bbox, the active gesture method's keypoints/skeleton/state,
    target_tracking's bbox+center-line+reverify readout, and target_recovery's search
    status+reacquired bbox — by calling each composed module's OWN draw_debug(), not
    re-implementing any of their drawing logic. This is the sanctioned use of this module's
    isolation exception: composing PUBLIC draw_debug() calls is no different from composing
    evaluate()/step() calls.

    Call this AFTER step(), with the SAME frame, in the same loop iteration — it draws from
    whichever phase(s) that step() call actually ran; it does not re-run any inference itself.
    """
    _get_pipeline().draw_debug(frame)
