"""
Reference trajectory storage (spec §4.1) — a small SHARED, GENERIC set (not per-person, per
confirmed project decision), stored as .npz files under reference_trajectories/. Independent
storage format from modules.face_identity's registry — these hold flattened trajectory vectors,
not face embeddings, a genuinely different content shape, not a mirrored pattern.
"""
import glob
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_reference_id(raw_id: Optional[str]) -> str:
    if not raw_id or not raw_id.strip():
        raise ValueError("Reference ID cannot be empty.")
    name = re.sub(r"\s+", "_", raw_id.strip())
    name = re.sub(r"[^\w\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_-")
    if not name:
        raise ValueError(f"Reference ID '{raw_id}' contains no valid characters after sanitization.")
    return name


@dataclass(frozen=True)
class ReferenceTrajectory:
    reference_id: str
    flat_vector: np.ndarray   # (resample_length * 6,) float32 — normalized + resampled + flattened
    resample_length: int
    arm: str                   # "left" | "right" — which arm this reference was captured from
    created_at: str


class ReferenceTrajectoryStore:
    def __init__(self, reference_dir: str):
        self.reference_dir = reference_dir

    def _path(self, reference_id: str) -> str:
        return os.path.join(self.reference_dir, f"{sanitize_reference_id(reference_id)}.npz")

    def save(self, reference_id: str, flat_vector: np.ndarray, resample_length: int, arm: str) -> str:
        os.makedirs(self.reference_dir, exist_ok=True)
        path = self._path(reference_id)
        np.savez(
            path,
            flat_vector=np.asarray(flat_vector, dtype=np.float32),
            resample_length=np.array(int(resample_length)),
            arm=np.array(arm),
            created_at=np.array(datetime.now().isoformat()),
        )
        return path

    def load_all(self) -> List[ReferenceTrajectory]:
        os.makedirs(self.reference_dir, exist_ok=True)
        entries: List[ReferenceTrajectory] = []
        for path in sorted(glob.glob(os.path.join(self.reference_dir, "*.npz"))):
            reference_id = os.path.splitext(os.path.basename(path))[0]
            try:
                data = np.load(path, allow_pickle=False)
                entries.append(ReferenceTrajectory(
                    reference_id=reference_id,
                    flat_vector=data["flat_vector"],
                    resample_length=int(data["resample_length"]),
                    arm=str(data["arm"]),
                    created_at=str(data["created_at"]),
                ))
            except Exception as e:
                logger.warning(f"gesture_trajectory_verifier: skipping unreadable reference '{path}': {e}")
        return entries
