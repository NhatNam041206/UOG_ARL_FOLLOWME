"""
Gate B — motion gate (spec §4.2): independently of pose, is the wrist actually moving in a
multi-directional (not just monotonic) way over a short recent window? This is what rejects a
static held pose (e.g. a stretch/reach) that Gate A alone cannot distinguish from a real wave.

Motion is evaluated in the full 2D plane, not just horizontal — a "wave" is any back-and-forth in
displacement direction, not specifically side-to-side.

See §4.4: MOTION_MIN_DISPLACEMENT_PX is the noise floor that keeps this gate from mistaking
MoveNet's own per-frame inference jitter (on a perfectly still raised arm) for genuine small
wrist movement. It is required, not optional, and must be calibrated above the measured jitter
floor (spec §11 checklist) before this gate is trusted.
"""
from dataclasses import dataclass, field
from math import acos, degrees, sqrt
from typing import List, Tuple

from .config import WaveFacingConfig
from .constants import Keypoint

Sample = Tuple[float, float, float]  # (timestamp, x_px, y_px)
Vector = Tuple[float, float]


@dataclass
class MotionBuffer:
    """One per arm, per track_id — see pipeline.py. Lives for the track's lifetime."""
    samples: List[Sample] = field(default_factory=list)


def vector_length(v: Vector) -> float:
    return sqrt(v[0] ** 2 + v[1] ** 2)


def displacement_vectors(samples: List[Sample]) -> List[Vector]:
    return [(samples[i + 1][1] - samples[i][1], samples[i + 1][2] - samples[i][2])
            for i in range(len(samples) - 1)]


def angle_between(v1: Vector, v2: Vector) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = vector_length(v1) * vector_length(v2)
    if mag == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / mag))
    return degrees(acos(cos_angle))


def update_motion_buffer(buffer: MotionBuffer, wrist: Keypoint, timestamp: float, config: WaveFacingConfig) -> None:
    """
    Accumulates on every frame where the wrist clears its OWN confidence threshold, regardless of
    whether Gate A passed this frame — this buffer is never reset by Gate A failing (spec §4.2/§5:
    the two gates share no state). `wrist` is already in bbox pixel space (see preprocessing.py).
    """
    buffer.samples = [s for s in buffer.samples if timestamp - s[0] <= config.motion_window_seconds]
    if wrist.score >= config.motion_confidence_threshold:
        buffer.samples.append((timestamp, wrist.x, wrist.y))


def gate_b_pass(buffer: MotionBuffer, config: WaveFacingConfig) -> bool:
    if len(buffer.samples) < config.motion_min_samples:
        return False
    vectors = displacement_vectors(buffer.samples)
    # Noise floor applied BEFORE counting direction changes — sub-threshold vectors are "not
    # meaningfully moved", not a direction-change candidate (spec §4.2).
    significant_vectors = [v for v in vectors if vector_length(v) >= config.motion_min_displacement_px]
    if len(significant_vectors) < 2:
        return False
    direction_changes = 0
    for i in range(1, len(significant_vectors)):
        if angle_between(significant_vectors[i - 1], significant_vectors[i]) >= config.motion_direction_change_angle_deg:
            direction_changes += 1
    return direction_changes >= config.motion_min_direction_changes
