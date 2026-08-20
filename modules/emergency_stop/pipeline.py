"""
Per-frame orchestrator for the emergency_stop module: ROI crop -> standalone detection ->
low-confidence gate -> zone logic -> resume hysteresis. Not part of the public contract —
external callers use interface.py only (which wraps this class).
"""
import logging
import time
from typing import Optional, Tuple

from .config import EStopConfig
from .detection import EStopDetector
from .roi import RunwayGeometry, build_geometry
from .zones import ZoneEvaluator

logger = logging.getLogger(__name__)


class EmergencyStopPipeline:
    def __init__(self, config: EStopConfig):
        self.config = config
        self.detector = EStopDetector(config.yolo_model_path)
        self.zone_evaluator = ZoneEvaluator()

        self._geometry_cache: Optional[RunwayGeometry] = None
        self._geometry_frame_size: Optional[Tuple[int, int]] = None

        # Resume hysteresis state (§3.6). None = "not currently in an uninterrupted clear
        # streak". Deliberately starts as None (not "already clear") — cold start requires the
        # same resume_buffer_seconds of proven-clear frames as any real recovery does, per the
        # "when uncertain, STOP" governing principle: no prior history is itself a form of
        # uncertainty, so GO is never assumed on frame 1.
        self._resume_clear_since: Optional[float] = None

        # Exposed for external latency benchmarking (spec §3.5/§6: the latency budget itself
        # can't be enforced yet, but the measurement needs to exist so it CAN be benchmarked).
        self.last_latency_ms: float = 0.0

        missing = config.missing_keys()
        if missing:
            logger.warning(
                f"emergency_stop: {len(missing)} threshold(s) not yet calibrated "
                f"({', '.join(missing)}) — module will output UNCERTAIN on every frame until "
                f"config/thresholds.yaml's emergency_stop section is fully filled in."
            )

    def process_frame(self, frame) -> Tuple[str, str, Optional[int], Optional[str], float]:
        """
        Returns (decision_str, reason, triggering_track_id, zone, timestamp). interface.py wraps
        this tuple into the public EStopOutput/EStopDecision types — kept as primitives here so
        this internal module has no import-time dependency on interface.py (avoids a cycle).
        """
        t_start = time.time()
        now = t_start
        try:
            missing = self.config.missing_keys()
            if missing:
                return self._decide("UNCERTAIN", f"uncalibrated_config:{','.join(missing)}", None, None, now)

            if frame is None or getattr(frame, "size", 0) == 0:
                return self._decide("UNCERTAIN", "invalid_frame", None, None, now)

            frame_h, frame_w = frame.shape[:2]
            geometry = self._get_geometry(frame_w, frame_h)

            rx1, ry1, rx2, ry2 = geometry.roi_rect
            roi_crop = frame[ry1:ry2, rx1:rx2]
            if roi_crop.size == 0:
                return self._decide("UNCERTAIN", "degenerate_roi", None, None, now)

            raw_dets = self.detector.track(roi_crop)
            detections = []
            for d in raw_dets:
                bx1, by1, bx2, by2 = d["bbox"]
                gx, gy = d["ground_contact"]
                detections.append({
                    **d,
                    "bbox": (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1),
                    "ground_contact": (gx + rx1, gy + ry1),
                })

            # §3.5: a detection inside the ROI below the confidence floor escalates the WHOLE
            # frame to UNCERTAIN instead of being silently dropped — a dropped low-confidence
            # detection could be a real obstacle, and dropping it would be an implicit GO.
            low_conf = [d for d in detections if d["confidence"] < self.config.min_detection_confidence]
            if low_conf:
                return self._decide("UNCERTAIN", "low_confidence_detection", low_conf[0]["track_id"], None, now)

            reason, track_id, zone = self.zone_evaluator.evaluate(
                detections, geometry,
                self.config.size_prefilter_width_px, self.config.size_prefilter_height_px,
                self.config.t_mid_seconds, now,
            )

            if reason is not None:
                self._resume_clear_since = None
                return self._decide("STOP", reason, track_id, zone, now)

            # Runway clear this frame -> resume hysteresis (§3.6). Only transitions to GO after
            # resume_buffer_seconds of UNINTERRUPTED clear frames; any interruption resets it.
            if self._resume_clear_since is None:
                self._resume_clear_since = now
            elapsed = now - self._resume_clear_since
            if elapsed >= self.config.resume_buffer_seconds:
                return self._decide("GO", "runway_clear", None, None, now)
            return self._decide("STOP", "resume_buffer_pending", None, None, now)

        except Exception as exc:
            logger.exception("emergency_stop: unhandled exception in per-frame pipeline")
            self._resume_clear_since = None
            return self._decide("UNCERTAIN", f"pipeline_exception:{type(exc).__name__}", None, None, now)
        finally:
            self.last_latency_ms = (time.time() - t_start) * 1000.0

    def _decide(self, decision, reason, track_id, zone, now):
        return decision, reason, track_id, zone, now

    def _get_geometry(self, frame_w: int, frame_h: int) -> RunwayGeometry:
        if self._geometry_cache is None or self._geometry_frame_size != (frame_w, frame_h):
            self._geometry_cache = build_geometry(
                frame_w, frame_h,
                self.config.runway_left_line, self.config.runway_right_line,
                int(self.config.roi_buffer_px),
                float(self.config.zone_far_boundary), float(self.config.zone_mid_boundary),
            )
            self._geometry_frame_size = (frame_w, frame_h)
        return self._geometry_cache
