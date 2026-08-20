"""
Trajectory normalization (spec §2.2), applied identically to live and reference trajectories:
1. Translate: subtract the first sample's position per point-track, so the trajectory
   represents relative motion from its own start, not absolute frame position.
2. Scale: divide by a stable body-scale reference — bbox height at capture time (spec's own
   proposed default, used here since wrist-to-shoulder distance itself changes during a wave and
   would distort the normalization).
"""
from typing import List

from .trajectory_buffer import TrajectorySample


def normalize_trajectory(samples: List[TrajectorySample], bbox_height_at_capture: float) -> List[TrajectorySample]:
    if not samples:
        return []
    scale = bbox_height_at_capture if bbox_height_at_capture > 1e-6 else 1.0
    origin_wrist, origin_elbow, origin_shoulder = samples[0].wrist, samples[0].elbow, samples[0].shoulder

    def _norm_point(point, origin):
        return ((point[0] - origin[0]) / scale, (point[1] - origin[1]) / scale)

    return [
        TrajectorySample(
            timestamp=s.timestamp,
            wrist=_norm_point(s.wrist, origin_wrist),
            elbow=_norm_point(s.elbow, origin_elbow),
            shoulder=_norm_point(s.shoulder, origin_shoulder),
        )
        for s in samples
    ]
