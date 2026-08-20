"""
Face matching logic (spec §3). Both the live embedding and every registry embedding are already
L2-normalized (embedder.py, registry.py), so cosine similarity reduces to a plain dot product —
this is the standard distance metric for ArcFace-family embeddings (including EdgeFace, per its
own reference inference script's `compute_similarity(..., normalized=True)`), not invented here.

Known limitation, intentionally not solved here (spec §3): a single frame's match is NOT
debounced/confirmed over time by this module. If temporal confirmation across frames is needed,
that is the calling pipeline orchestration's job (e.g. an external ConfirmationTracker calling
this repeatedly) — this module always evaluates one frame in isolation.
"""
from typing import List, Optional, Tuple

import numpy as np

from .registry import RegistryEntry


def match_face(embedding: np.ndarray, registry_entries: List[RegistryEntry],
                similarity_threshold: Optional[float]) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Returns (is_match, matched_person_name, score) — score is the best similarity seen even on a
    non-match, so the caller can inspect how close the nearest miss was (spec §3), for
    calibration/debugging. If similarity_threshold is None (not yet calibrated) or the registry
    is empty, always returns (False, None, best_score_or_None) — fail closed, per the same
    "uncalibrated degrades to the safe/negative state" convention used elsewhere in this project.
    """
    if not registry_entries:
        return False, None, None

    best_name: Optional[str] = None
    best_score = -1.0
    for entry in registry_entries:
        score = float(np.dot(embedding, entry.embedding))
        if score > best_score:
            best_score = score
            best_name = entry.person_name

    if similarity_threshold is not None and best_score >= similarity_threshold:
        return True, best_name, best_score
    return False, None, best_score
