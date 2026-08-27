"""
Builds a back-of-head reference ON THE FLY during a single live run (main.py --target), instead
of relying solely on the one-time enrolled back_head_embedding - which only has
config.ENROLL_DURATION_SEC worth of samples from one enrollment session (one lighting setup, one
distance, one outfit) and doesn't generalize well to a real deployment.

identity/target_lock.py only calls capture() at a moment it's already CERTAIN the locked
track_id is still the target (their track_id is still being reported by ByteTrack - see
target_lock.py's module docstring for why that's trustworthy) and no face is visible right then
(the back-of-head case - exactly the situation where there is currently nothing to check
identity against, but identity is not in doubt either). Each accepted crop is saved as an image
file under temp_dataset/<session>/ (gitignored) so the collected data is inspectable, and its
OSNet embedding feeds a running average exposed as `.reference` -
identity/target_lock.py overwrites its own reference_back_head_embedding with this any time it
changes, so real session footage progressively replaces the (possibly much weaker) one-time
enrollment sample.

Purely in-memory + this run's temp_dataset/ folder - nothing here is reloaded on a future run.
"""
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


class SessionBackDataset:
    def __init__(self, osnet_embedder, out_dir: str, capture_interval_sec: float, max_samples: int):
        self._embedder = osnet_embedder
        self._interval = capture_interval_sec
        self._max_samples = max_samples
        self._embeddings: List[np.ndarray] = []
        self._last_capture_time: Optional[float] = None
        self._sample_count = 0

        self._out_dir = Path(out_dir) / time.strftime("session_%Y%m%d_%H%M%S")
        self._out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def reference(self) -> Optional[np.ndarray]:
        """Running-average embedding of every crop captured so far this session, or None if
        nothing's been captured yet."""
        if not self._embeddings:
            return None
        vec = np.mean(self._embeddings, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else vec

    def due(self) -> bool:
        """Whether it's worth even checking for a capture right now - lets the caller skip the
        (comparatively expensive) face-detection check entirely most frames, so this stays cheap
        enough for a live loop."""
        if len(self._embeddings) >= self._max_samples:
            return False
        return self._last_capture_time is None or time.time() - self._last_capture_time >= self._interval

    def capture(self, crop: np.ndarray) -> None:
        """Saves `crop` to this session's temp_dataset/ folder and folds its embedding into the
        running average. Caller is responsible for having already confirmed this is a genuine,
        currently-trusted back-of-head moment (see module docstring)."""
        self._last_capture_time = time.time()
        self._sample_count += 1
        cv2.imwrite(str(self._out_dir / f"back_{self._sample_count:04d}.jpg"), crop)
        self._embeddings.append(self._embedder.extract(crop))
