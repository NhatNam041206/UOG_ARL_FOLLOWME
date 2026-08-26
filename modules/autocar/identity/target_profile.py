"""Save/load ONE enrolled person's reference profile to a single .npz file - produced by
scripts/enroll_person.py, consumed by identity/target_lock.py via main.py --target. Fields:
  - face_embedding: SFace face-recognition embedding (identity/face_recognizer.py) of the front
    face - the one identity/target_lock.py actually matches against when a face is visible.
  - back_head_embedding: OSNet appearance embedding (identity/osnet_embedder.py) of the back of
    the head - matched against when no face is visible (facing away).
  - head_embedding, lower_embedding, aspect_ratio: from an older design (OSNet on a keypoint-
    guessed head-region rectangle for the front case too, plus lower-body/aspect-ratio signals) -
    no longer consumed by identity/target_lock.py, kept only so existing profile files and their
    round-trip format don't need to change shape.
face_embedding and back_head_embedding are each optional (None if this profile predates that
enrollment phase, or the person skipped it) - identity/target_lock.py simply can't score that case
(front or back) when its reference is missing, same as before either existed."""
import re
from datetime import datetime, timezone

import numpy as np


def sanitize_person_name(raw_name: str) -> str:
    """Keep letters (incl. accented/Unicode), digits, '_', '-'; spaces -> '_'; everything else
    (including '.', '/', '\\') is stripped - blocks path traversal by construction, not just
    pattern-matching '../'."""
    if raw_name is None:
        raise ValueError("Person name cannot be None.")
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError(f"Person name '{raw_name}' has no valid characters left after sanitizing.")
    return name


def save_target_profile(path: str, head_embedding: np.ndarray, lower_embedding: np.ndarray,
                         aspect_ratio: float, sample_count: int,
                         back_head_embedding: np.ndarray = None,
                         face_embedding: np.ndarray = None) -> None:
    fields = dict(
        head_embedding=head_embedding.astype(np.float32),
        lower_embedding=lower_embedding.astype(np.float32),
        aspect_ratio=np.array(aspect_ratio, dtype=np.float64),
        sample_count=np.array(sample_count, dtype=np.int64),
        created_at=np.array(datetime.now(timezone.utc).isoformat()),
        allow_pickle=False,
    )
    if back_head_embedding is not None:
        fields["back_head_embedding"] = back_head_embedding.astype(np.float32)
    if face_embedding is not None:
        fields["face_embedding"] = face_embedding.astype(np.float32)
    np.savez(path, **fields)


def load_target_profile(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "head_embedding" in data:
            head_embedding = data["head_embedding"].astype(np.float32)
            lower_embedding = data["lower_embedding"].astype(np.float32)
        else:
            # Old single-embedding profile (pre head/lower split) - use it for both regions so
            # existing enrollments keep working, just without the accuracy benefit of a real
            # split. Re-run scripts/enroll_person.py to get a proper head/lower profile.
            legacy = data["embedding"].astype(np.float32)
            head_embedding = legacy
            lower_embedding = legacy

        back_head_embedding = (data["back_head_embedding"].astype(np.float32)
                                if "back_head_embedding" in data else None)
        face_embedding = (data["face_embedding"].astype(np.float32)
                           if "face_embedding" in data else None)

        return {
            "head_embedding": head_embedding,
            "lower_embedding": lower_embedding,
            "back_head_embedding": back_head_embedding,
            "face_embedding": face_embedding,
            "aspect_ratio": float(data["aspect_ratio"]),
            "sample_count": int(data["sample_count"]),
            "created_at": str(data["created_at"]),
        }
