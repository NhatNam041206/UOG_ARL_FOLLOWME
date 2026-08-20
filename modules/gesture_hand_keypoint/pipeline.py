"""
Per-track orchestrator: person crop -> MediaPipe hand detect -> pick the single hand (regardless
of Left/Right side) that clears the palm-height gate, highest-confidence first -> hand-shape
classification -> shared OPEN/CLOSED sequence state machine -> feed CONFIRMED pulses into the
shared RED/YELLOW/GREEN confirmation tracker -> GestureMethodResult. Not part of the public
contract — external callers use interface.py only.

Redesign (replaces the prior motion-based Method 2 entirely, no fallback to the old approach):
no wrist motion, no trajectory, no arm geometry — pure hand-shape sequence classification from
MediaPipe landmark geometry only.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

from .bbox_context import BboxContext, palm_height_gate_pass
from .config import GestureHandKeypointConfig
from .confirmation import ConfirmationTracker, GREEN
from .hand_shape import classify_hand_shape
from .hand_detector import HandLandmarkerWrapper
from .palm_orientation import palm_facing_camera_debug
from .sequence_state_machine import CONFIRMED, SequenceStateMachine, WAITING_OPEN

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """Plain-primitive result (not the public GestureMethodResult dataclass) so this internal
    module has no import-time dependency on interface.py — mirrors this project's established
    tuple-return convention to avoid an import cycle."""
    track_id: int
    is_waving: bool
    waving_state: str
    sequence_stage: str
    confidence_debug: Optional[float]
    palm_facing_camera_debug: Optional[bool]
    hands_raw: Optional[object]


@dataclass
class _TrackState:
    """Per-track_id state. A SINGLE shared SequenceStateMachine, not one per hand side: only one
    hand drives the sequence at a time, chosen each frame purely by "clears the palm-height gate
    + highest confidence" — side (Left/Right) is deliberately ignored, per the confirmed "one
    hand, regardless of side" design. Avoids two independent sequences progressing at once and
    sidesteps MediaPipe occasionally mislabeling handedness."""
    sequence_machine: SequenceStateMachine = field(default_factory=SequenceStateMachine)
    waving_tracker: ConfirmationTracker = field(default_factory=ConfirmationTracker)
    # Once any hand's sequence reaches CONFIRMED, we hold a "True" pulse into waving_tracker
    # until this timestamp — see the long comment in evaluate() below for why.
    confirmed_hold_until: Optional[float] = None


class GestureHandKeypointPipeline:
    def __init__(self, config: GestureHandKeypointConfig):
        self.config = config
        self.detector = HandLandmarkerWrapper(config.model_path)
        self._tracks: Dict[int, _TrackState] = {}

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"gesture_hand_keypoint: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — evaluate() will report is_waving=False on every call "
                f"until config/thresholds.yaml's gesture_hand_keypoint section is filled in."
            )

    def release_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def evaluate(self, track_id: int, person_crop_bgr, timestamp: Optional[float] = None,
                 person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> PipelineResult:
        if timestamp is None:
            timestamp = time.time()
        state = self._tracks.setdefault(track_id, _TrackState())

        if person_crop_bgr is None or getattr(person_crop_bgr, "size", 0) == 0:
            waving_state = state.waving_tracker.update(False, timestamp, self.config)
            return PipelineResult(track_id, False, waving_state, WAITING_OPEN, None, None, None)

        # Detection always runs — it needs no calibrated thresholds, and the mandatory
        # visualization tool needs raw keypoints to be usable even before calibration.
        hands = self.detector.detect(person_crop_bgr)

        if person_bbox_full_frame is not None:
            bbox = BboxContext.from_person_bbox(person_bbox_full_frame)
        else:
            bbox = BboxContext.whole_crop_as_bbox(person_crop_bgr.shape)

        missing = self.config.missing_keys()
        if missing:
            waving_state = state.waving_tracker.update(False, timestamp, self.config)
            return PipelineResult(track_id, False, waving_state, WAITING_OPEN, None, None, hands if hands else None)

        confirmed_this_frame = False
        best_confidence: Optional[float] = None
        best_palm_facing: Optional[bool] = None

        # Only hands clearing the confidence floor are even considered. Of those, only ones
        # ALSO clearing the palm-height gate are eligible to drive the sequence this frame; side
        # (Left/Right) plays no role in the choice.
        eligible = [h for h in hands if h.handedness_confidence >= self.config.confidence_threshold]

        if not eligible:
            # No sufficiently-confident hand at all this frame — treated like "no shape
            # reading": does not advance, does not reset (only the timeout / height-gate
            # failure do that). Mirrors the prior per-hand behavior for a hand not seen.
            stage = state.sequence_machine.stage
        else:
            gate_ok = [h for h in eligible if palm_height_gate_pass(h.landmarks_px, bbox, self.config)]
            if gate_ok:
                candidate = max(gate_ok, key=lambda h: h.handedness_confidence)
                best_confidence = candidate.handedness_confidence
                best_palm_facing = palm_facing_camera_debug(candidate.landmarks_px, candidate.handedness)
                shape = classify_hand_shape(candidate.landmarks_px, self.config)
                stage = state.sequence_machine.update(shape, True, timestamp, self.config)
            else:
                # Every confident hand is below the palm-height cutoff — immediate reset (gate
                # failure), same "no partial credit" rule as before, now evaluated once across
                # whichever hands are visible rather than per hand-side.
                candidate = max(eligible, key=lambda h: h.handedness_confidence)
                best_confidence = candidate.handedness_confidence
                best_palm_facing = palm_facing_camera_debug(candidate.landmarks_px, candidate.handedness)
                stage = state.sequence_machine.update(None, False, timestamp, self.config)

            if stage == CONFIRMED:
                confirmed_this_frame = True
                state.sequence_machine.reset()  # CONFIRMED is momentary — ready for the next gesture
                # `stage` (local var) still reports "CONFIRMED" for THIS frame's result below,
                # even though the machine object itself already reset for the next frame — the
                # visualization requirement needs to see CONFIRMED on the frame it happens.

        best_stage = stage

        # Reconciling a ONE-SHOT completion event (the 4-step sequence, momentary CONFIRMED)
        # with ConfirmationTracker (designed for a CONTINUOUS per-frame condition, spec says
        # reuse it "exactly as before"): hold a synthetic "True" pulse for
        # 2x confirmation_duration_seconds once CONFIRMED fires — the first half lets the
        # tracker complete its normal RED->YELLOW->GREEN promotion (which needs
        # confirmation_duration_seconds of continuous True), the second half keeps is_waving
        # visibly GREEN for a real dwell instead of reverting to RED the instant GREEN is
        # reached. This is a judgment call for reconciling a discrete event with a
        # continuous-condition tracker without modifying ConfirmationTracker itself.
        if confirmed_this_frame:
            state.confirmed_hold_until = timestamp + 2 * self.config.confirmation_duration_seconds
        raw_condition_pass = state.confirmed_hold_until is not None and timestamp <= state.confirmed_hold_until

        waving_state = state.waving_tracker.update(raw_condition_pass, timestamp, self.config)

        return PipelineResult(
            track_id=track_id,
            is_waving=(waving_state == GREEN),
            waving_state=waving_state,
            sequence_stage=best_stage,
            confidence_debug=best_confidence,
            palm_facing_camera_debug=best_palm_facing,
            hands_raw=hands if hands else None,
        )
