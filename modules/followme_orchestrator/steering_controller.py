"""
PID steering controller — deliberately its OWN class, not merged into the orchestrator or any CV
module (plans/08 §3).

Rationale (preserved here, not just in the spec): PID timing correctness depends on `dt` being
REAL elapsed wall-clock time, not an assumed frame interval. CV pipeline latency varies
frame-to-frame (a slow face-match frame, a slow gesture-method frame) — embedding the PID inside
a class that also does CV inference would let that latency variance silently corrupt PID timing
(the D-term and integral accumulation both need an accurately-measured dt). Keeping this a
separate class, fed a real `timestamp` each cycle by the orchestrator, avoids that entirely. This
is a correctness requirement, not a style preference.
"""
from typing import Optional


class SteeringController:
    def __init__(self, kp: Optional[float], ki: Optional[float], kd: Optional[float],
                 max_steering_angle_degrees: Optional[float], fov_degrees: Optional[float]):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_steering_angle_degrees = max_steering_angle_degrees
        self.fov_degrees = fov_degrees
        self._integral = 0.0
        self._last_error_degrees: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def is_calibrated(self) -> bool:
        return None not in (self.kp, self.ki, self.kd, self.max_steering_angle_degrees, self.fov_degrees)

    def reset(self) -> None:
        """Clears accumulated integral/derivative state. Called whenever a new follow episode
        begins (a fresh trigger, or a recovery-driven resume) so stale error history from a
        previous episode never bleeds into a new one's first correction."""
        self._integral = 0.0
        self._last_error_degrees = None
        self._last_timestamp = None

    def update(self, horizontal_offset: float, timestamp: float) -> float:
        """
        `horizontal_offset`: -1.0..+1.0 normalized error signal from autocar_adapter (a sibling
        file in this same package — see pipeline.py).
        `timestamp`: real wall-clock time (same clock/units as everywhere else in this pipeline)
        — dt is computed internally as (timestamp - self._last_timestamp), NOT assumed.

        Converts horizontal_offset to a true angle via `fov_degrees` — this is deliberately WHERE
        that conversion happens, not inside the tracking engine itself, per the original
        target_tracking module's explicit architecture boundary (plans/06 §4.2), still honored
        here — BEFORE running PID, so kp/ki/kd are
        tuned against real degrees of error, not an abstract -1..+1 unit.

        Returns a signed angle in degrees, clamped to +/- max_steering_angle_degrees (an
        Ackermann/servo hardware limit).

        Caller must check is_calibrated() first — this raises if any gain or fov_degrees is None,
        rather than silently computing a meaningless output from missing calibration.
        """
        if not self.is_calibrated():
            raise RuntimeError(
                "SteeringController.update() called before calibration — check is_calibrated() "
                "first; the caller (pipeline.py) is responsible for the fail-closed should_move "
                "gate, not this method."
            )

        error_degrees = horizontal_offset * (self.fov_degrees / 2.0)

        dt = 0.0 if self._last_timestamp is None else max(0.0, timestamp - self._last_timestamp)

        self._integral += error_degrees * dt
        derivative = 0.0
        if self._last_error_degrees is not None and dt > 1e-6:
            derivative = (error_degrees - self._last_error_degrees) / dt

        output = self.kp * error_degrees + self.ki * self._integral + self.kd * derivative

        self._last_error_degrees = error_degrees
        self._last_timestamp = timestamp

        return max(-self.max_steering_angle_degrees, min(self.max_steering_angle_degrees, output))
