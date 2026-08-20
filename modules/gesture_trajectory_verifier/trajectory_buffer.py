"""
Live trajectory buffer (spec §2.1), per arm, per track_id. Independently implemented (spec
§0.3) even though the confidence-gating PATTERN mirrors Method 1's Gate B (gate closed if any of
wrist/elbow/shoulder is below threshold) — no shared code/classes.

Tracks THREE points per arm (wrist, elbow, shoulder) — spec's confirmed design correction from
an earlier wrist-only design, which was found to lose arm-shape information (bent vs. straight
arm can look the same as a wrist-only path).
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import GestureTrajectoryVerifierConfig
from .preprocessing import Keypoint

Point = Tuple[float, float]


@dataclass(frozen=True)
class TrajectorySample:
    timestamp: float
    wrist: Point
    elbow: Point
    shoulder: Point


@dataclass
class TrajectoryBuffer:
    samples: List[TrajectorySample] = field(default_factory=list)


def update_trajectory_buffer(buffer: TrajectoryBuffer, wrist: Keypoint, elbow: Keypoint,
                               shoulder: Keypoint, timestamp: float,
                               config: GestureTrajectoryVerifierConfig) -> None:
    buffer.samples = [s for s in buffer.samples if timestamp - s.timestamp <= config.trajectory_window_seconds]
    if min(wrist.score, elbow.score, shoulder.score) < config.confidence_threshold:
        return
    buffer.samples.append(TrajectorySample(
        timestamp=timestamp,
        wrist=(wrist.x, wrist.y),
        elbow=(elbow.x, elbow.y),
        shoulder=(shoulder.x, shoulder.y),
    ))
