"""
Fixed-length resampling (spec §2.3) — TIME-BASED, confirmed with the user over arc-length-based
resampling (simpler; the project's own "engineered/lightweight over heavier approach unless
proven insufficient" principle; a wave is roughly periodic so non-uniform speed is a smaller
risk than for arbitrary motion — if empirical testing later shows shape fidelity suffers,
arc-length resampling is a well-scoped future upgrade, not built now).
"""
from typing import List

import numpy as np

from .trajectory_buffer import TrajectorySample


def resample_time_based(samples: List[TrajectorySample], resample_length: int) -> List[TrajectorySample]:
    """Linearly interpolates each of the three point-tracks at `resample_length` evenly spaced
    time steps across the buffer's own time span. Returns [] if there's no time span to sample
    (fewer than 2 samples, or all samples share one timestamp)."""
    if len(samples) < 2:
        return []
    t0, t1 = samples[0].timestamp, samples[-1].timestamp
    if t1 - t0 < 1e-9:
        return []

    times = np.array([s.timestamp for s in samples])
    wrist_xs = np.array([s.wrist[0] for s in samples])
    wrist_ys = np.array([s.wrist[1] for s in samples])
    elbow_xs = np.array([s.elbow[0] for s in samples])
    elbow_ys = np.array([s.elbow[1] for s in samples])
    shoulder_xs = np.array([s.shoulder[0] for s in samples])
    shoulder_ys = np.array([s.shoulder[1] for s in samples])

    query_times = np.linspace(t0, t1, resample_length)
    return [
        TrajectorySample(
            timestamp=float(qt),
            wrist=(float(np.interp(qt, times, wrist_xs)), float(np.interp(qt, times, wrist_ys))),
            elbow=(float(np.interp(qt, times, elbow_xs)), float(np.interp(qt, times, elbow_ys))),
            shoulder=(float(np.interp(qt, times, shoulder_xs)), float(np.interp(qt, times, shoulder_ys))),
        )
        for qt in query_times
    ]
