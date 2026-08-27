"""
Cosine-similarity matching for appearance_verifier. Both the candidate and every reference
embedding are already L2-normalized (embedder.py), so cosine similarity reduces to a plain dot
product — the same mathematical pattern already used by modules/face_identity's matching.py and
modules/gesture_trajectory_verifier's similarity.py, independently reimplemented here per this
project's own-instance/own-code isolation convention (not imported from either).
"""
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
