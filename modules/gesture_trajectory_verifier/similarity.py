"""
Similarity comparison (spec §2.4) — fixed-length resampling + cosine similarity, NOT DTW, per
confirmed project decision (§8 explicit non-goal: no DTW implementation unless empirically
proven insufficient later).
"""
from typing import List

import numpy as np

from .trajectory_buffer import TrajectorySample


def flatten_trajectory(samples: List[TrajectorySample]) -> np.ndarray:
    """Concatenates normalized (wrist, elbow, shoulder) x,y across all resampled points into one
    flat vector."""
    vec: List[float] = []
    for s in samples:
        vec.extend([s.wrist[0], s.wrist[1], s.elbow[0], s.elbow[1], s.shoulder[0], s.shoulder[1]])
    return np.array(vec, dtype=np.float32)


def trajectory_similarity(live_vec: np.ndarray, reference_vec: np.ndarray) -> float:
    norm_live = float(np.linalg.norm(live_vec))
    norm_ref = float(np.linalg.norm(reference_vec))
    if norm_live < 1e-9 or norm_ref < 1e-9 or live_vec.shape != reference_vec.shape:
        return 0.0
    return float(np.dot(live_vec, reference_vec) / (norm_live * norm_ref))
