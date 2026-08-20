"""
Emergency Stop (E-Stop) module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.emergency_stop. Everything else in
this package (config.py, roi.py, detection.py, zones.py, pipeline.py) is an internal
implementation detail and may change without notice.

Safety context (see the module spec for full detail): there is currently no non-CV sensor
(sonar/ultrasonic/IR) hardware backstop. Until one exists, this CV module is the SOLE safety
layer preventing collisions. Governing principle: when uncertain, STOP. UNCERTAIN must be
treated identically to STOP by every consumer — it exists only to distinguish "confidently
clear to stop" from "not confident enough to say GO" for logging/debugging, never to be treated
as a softer state.

Frame type: no dedicated Frame dataclass exists in this codebase. `process_frame()` takes a raw
BGR frame (numpy.ndarray from cv2), the standard OpenCV convention.

Pre-implementation audit (spec §7): searched the full repository for an "Immediate Danger
Override" mechanism (DANGER_LEFT/DANGER_RIGHT/AMBIGUOUS_DANGER states, a process_video.py file,
lane-boundary-intercept geometry). None exists anywhere in this codebase — confirmed with the
user this was never built, not merely relocated. This module therefore has no naming collision
to avoid and nothing to merge/deprecate; it is a wholly new, standalone safety layer.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import EStopConfig, load_config
from .pipeline import EmergencyStopPipeline


class EStopDecision(Enum):
    GO = "GO"
    STOP = "STOP"
    UNCERTAIN = "UNCERTAIN"   # must be treated as STOP by any consumer


@dataclass
class EStopOutput:
    decision: EStopDecision
    reason: str                # human-readable cause, e.g. "near_zone_object", "low_confidence_detection"
    triggering_track_id: Optional[int] = None
    zone: Optional[str] = None    # "far" | "mid" | "near" | None
    timestamp: float = 0.0


class EmergencyStopModule:
    """
    Owns all per-run state (dwell timers, resume buffer, own detector/tracker instance). Create
    one instance and call process_frame() once per camera frame; do not share an instance across
    unrelated robot runs without re-instantiating (state like dwell timers should not carry over
    a stale session).
    """

    def __init__(self, thresholds_config_path: str = "config/thresholds.yaml"):
        config: EStopConfig = load_config(thresholds_config_path)
        self._pipeline = EmergencyStopPipeline(config)

    def process_frame(self, frame) -> EStopOutput:
        decision_str, reason, track_id, zone, timestamp = self._pipeline.process_frame(frame)
        return EStopOutput(
            decision=EStopDecision(decision_str),
            reason=reason,
            triggering_track_id=track_id,
            zone=zone,
            timestamp=timestamp,
        )

    @property
    def last_latency_ms(self) -> float:
        """Measured wall-clock time for the most recent process_frame() call, in milliseconds.
        Exposed for the frame-time benchmarking called out as an outstanding prerequisite in the
        module spec (§3.5/§6) before a latency budget can be safely set."""
        return self._pipeline.last_latency_ms
