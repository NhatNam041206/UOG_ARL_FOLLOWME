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
      NOT currently tracking -> face_identity -> human_detection_roi -> gesture_hand_keypoint
        (the sole TRIGGER gesture method — two others were removed, confirmed with the user) ->
        is_waving GREEN? -> autocar_adapter.start(person_name, ...)
      currently tracking -> autocar_adapter.update(...) — its TargetLock folds tracking AND
        recovery into one state machine, so there is no separate recovery step here:
        TRACKING -> SteeringController.update(horizontal_offset, timestamp); resets the PID if
                    this frame is a mid-episode reclaim (just_reacquired)
        SEARCHING -> should_move=False (robot moves forward only while actively following)
        LOST -> stop, auto-resume watching for a fresh trigger (confirmed with the user — no
                separate reset call needed; the next step() call re-arms itself)

ISOLATION EXCEPTION, stated explicitly, not silently overridden (plans/08 §0.3): this module is
the ONE deliberate exception to "own-instance isolation, no module imports another's
interface.py except through main.py" (docs/architecture.md rule #2/#3). It exists specifically to
be the reusable, importable version of what main.py's script-level code currently does ad hoc for
the pre-trigger portion of the face_first pipeline — a composition root, not a rule violation.
It still never reaches into any composed module's PRIVATE implementation — only public
interface.py contracts, exactly as main.py already does today.

Manages a SINGLE active follow-me episode — mirrors autocar_adapter's own single-episode design
(itself a module-level singleton; this orchestrator can only ever follow one registered person at
a time by construction, the same as that adapter).

`configure(...)` MUST be called before the first step() — see its own docstring below.
"""
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import FollowMeOrchestratorConfig, load_config
from .pipeline import FollowMeOrchestratorPipeline

__all__ = ["FollowMeCommand", "configure", "step", "draw_debug", "draw_steering_arrow", "debug_snapshot"]


@dataclass
class FollowMeCommand:
    should_move: bool                       # True = move forward, False = stop. No speed
                                              # parameter — speed is a downstream, separate concern.
    steering_angle_degrees: Optional[float]  # ABSOLUTE servo angle, ready to write directly to
                                              # the servo (config: steering.servo_center_degrees,
                                              # default 90 = straight ahead; no further conversion
                                              # needed downstream). None when should_move is False
                                              # (no meaningful steering target when stopped)
    debug_state: str                         # current high-level pipeline state, for logging/
                                              # visualization (e.g. "WAITING_FOR_TRIGGER",
                                              # "TRACKING_STARTED", "TRACKING", "TARGET_REACHED",
                                              # "TRACKING_STEERING_UNCALIBRATED", "RECOVERING",
                                              # "STOPPED") — exact state names are this module's
                                              # own design choice, not fixed by the originating spec


_pipeline_singleton: Optional[FollowMeOrchestratorPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml",
              face_registry_dir: str = "modules/face_identity/registry_data") -> None:
    """
    REQUIRED before the first step() call — (re)initializes the module-level orchestrator.

    Eagerly loads EVERY model this pipeline will use (face_identity, human_detection_roi,
    gesture_hand_keypoint, and autocar_adapter's YOLO-pose+OSNet) before returning — see
    FollowMeOrchestratorPipeline.__init__ — so none of them cold-start later during step(), and
    in particular never at the exact moment a gesture trigger fires. This call itself may
    therefore take several seconds; that's the intended trade — paid once, here, not live.
    """
    global _pipeline_singleton
    config: FollowMeOrchestratorConfig = load_config(thresholds_config_path)
    _pipeline_singleton = FollowMeOrchestratorPipeline(config, face_registry_dir, thresholds_config_path)


def _get_pipeline() -> FollowMeOrchestratorPipeline:
    if _pipeline_singleton is None:
        raise RuntimeError("modules.followme_orchestrator: configure(...) must be called before step().")
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


def debug_snapshot() -> dict:
    """
    Plain-dict snapshot of the CURRENT frame's decision-relevant fields (face match, gesture
    sequence progress, tracking state), for structured logging — see
    plans/10_debug_logging_observability.md. NOT part of the FollowMeCommand contract: keys may be
    added or removed without notice, unlike step()'s own typed return value. Call this AFTER
    step(), same convention as draw_debug()/draw_steering_arrow() — it reads from whichever
    phase(s) that step() call actually ran, it does not re-run any inference itself.
    """
    return _get_pipeline().debug_snapshot()


def draw_debug(frame: np.ndarray) -> None:
    """
    Draws EVERY phase's own debug overlay onto `frame` — face_identity's bbox+match,
    human_detection_roi's ROI+person bbox, the active gesture method's keypoints/skeleton/state,
    and autocar_adapter's tracked bbox+center-line+state readout — by calling each composed
    module's OWN draw_debug(), not re-implementing any of their drawing logic. This is the
    sanctioned use of this module's isolation exception: composing PUBLIC draw_debug() calls is
    no different from composing evaluate()/step() calls.

    Call this AFTER step(), with the SAME frame, in the same loop iteration — it draws from
    whichever phase(s) that step() call actually ran; it does not re-run any inference itself.
    """
    _get_pipeline().draw_debug(frame)


_ARROW_COLOR = (0, 255, 255)
_ARROW_LENGTH_PX = 100
_ARROW_ORIGIN_MARGIN_PX = 30


def draw_steering_arrow(frame: np.ndarray, command: FollowMeCommand) -> None:
    """
    Draws an arrow from bottom-center of `frame` showing the CALCULATED steering direction —
    derived from the same `steering_angle_degrees` the robot is actually being commanded with
    this frame (an ABSOLUTE servo angle, config: steering.servo_center_degrees=90=straight by
    default — see steering_controller.py's docstring), not a re-derivation of it. The arrow itself is drawn
    relative to straight-ahead: 0 degrees of visual tilt (servo angle 90) points straight up;
    positive servo angles (>90) tilt the arrow right, negative (<90) tilt it left.

    No-ops (draws nothing) when should_move is False or steering_angle_degrees is None — i.e.
    whenever the robot isn't actually being told to move, there is no "calculated direction" to
    show. Call this AFTER step(), with the SAME frame, same as draw_debug() above.
    """
    if not command.should_move or command.steering_angle_degrees is None:
        return
    frame_h, frame_w = frame.shape[:2]
    origin = (frame_w // 2, frame_h - _ARROW_ORIGIN_MARGIN_PX)
    tilt_degrees = command.steering_angle_degrees - _get_pipeline().steering.servo_center_degrees
    angle_rad = math.radians(tilt_degrees)
    tip = (
        int(origin[0] + _ARROW_LENGTH_PX * math.sin(angle_rad)),
        int(origin[1] - _ARROW_LENGTH_PX * math.cos(angle_rad)),
    )
    cv2.arrowedLine(frame, origin, tip, _ARROW_COLOR, 3, tipLength=0.3)
    cv2.putText(frame, f"servo={command.steering_angle_degrees:.1f} deg", (origin[0] + 12, origin[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _ARROW_COLOR, 2)
