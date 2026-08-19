import os
import re
import glob
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REGISTRY_DIR = "logs/registry"
RAW_CAPTURES_DIR = "logs/raw_captures"


def sanitize_person_name(raw_name: Optional[str]) -> str:
    """
    Turn free-text input into a safe filename stem: keeps only word characters (letters incl.
    Unicode/diacritics, digits, underscore) and hyphens; whitespace runs become a single
    underscore. This rules out path traversal by construction — '.', '/', '\\' are not
    word/hyphen characters, so they're stripped entirely rather than merely pattern-matched.
    """
    if not raw_name or not raw_name.strip():
        raise ValueError("Person name cannot be empty.")
    name = re.sub(r"\s+", "_", raw_name.strip())
    name = re.sub(r"[^\w\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_-")
    if not name:
        raise ValueError(
            f"Person name '{raw_name}' contains no valid characters after sanitization "
            f"(letters, digits, underscore, hyphen only)."
        )
    return name


def registry_path(name: str) -> str:
    """Resolve the .npz path for a (not-yet-sanitized) person name."""
    return os.path.join(REGISTRY_DIR, f"{sanitize_person_name(name)}.npz")


def raw_capture_dir(name: str) -> str:
    """Resolve the raw captures directory path for a (not-yet-sanitized) person name."""
    return os.path.join(RAW_CAPTURES_DIR, sanitize_person_name(name))


def raw_capture_exists(name: str) -> bool:
    return os.path.exists(raw_capture_dir(name))


def person_exists(name: str) -> bool:
    return os.path.exists(registry_path(name))


def list_registry() -> List[Dict[str, Any]]:
    """
    Returns [{name, path, created_at, sample_count}, ...] for every readable .npz entry in the
    registry, sorted by name. Unreadable/corrupt files are skipped with a warning rather than
    raising, so one bad entry doesn't break the whole selector UI.
    """
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    entries = []
    for path in sorted(glob.glob(os.path.join(REGISTRY_DIR, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            data = np.load(path, allow_pickle=False)
            entries.append({
                "name": name,
                "path": path,
                "created_at": str(data["created_at"]),
                "sample_count": int(data.get("sample_count", data.get("valid_sample_count", 0))),
            })
        except Exception as e:
            logger.warning(f"Skipping unreadable registry entry '{path}': {e}")
    return entries


def save_person(
    name: str,
    embedding: np.ndarray,
    aspect_ratio: float,
    sample_count: int,
    multi_views: Optional[Dict[str, Optional[np.ndarray]]] = None,
    pose_proportions: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    """
    Save (or overwrite) a person's reference data as a single .npz file. Returns the saved path.
    """
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    path = registry_path(name)

    save_dict: Dict[str, Any] = {
        "embedding": np.asarray(embedding, dtype=np.float32),
        "aspect_ratio": np.array(aspect_ratio, dtype=np.float64),
        "created_at": np.array(datetime.now().isoformat()),
        "sample_count": np.array(int(sample_count)),
    }

    if multi_views:
        for view_name in ['front', 'right', 'back', 'left']:
            v_emb = multi_views.get(view_name)
            if v_emb is not None:
                save_dict[f"embedding_{view_name}"] = np.asarray(v_emb, dtype=np.float32)

    if pose_proportions:
        for k in ['shoulder_hip_ratio', 'leg_torso_ratio', 'shoulder_width']:
            val = pose_proportions.get(k)
            if val is not None:
                save_dict[f"pose_{k}"] = np.array(float(val), dtype=np.float64)

    np.savez(path, **save_dict)
    return path


def load_person(path: str) -> Dict[str, Any]:
    """Load a single registry entry's full data (embedding, aspect_ratio, created_at, sample_count, multi_views, pose_proportions)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Registry entry not found: '{os.path.abspath(path)}'.")
    data = np.load(path, allow_pickle=False)

    # Old pre-registry reference data (single global reference_embedding.npy / companion
    # reference_aspect_ratio.npy from before the multi-person registry existed) is a different
    # file format entirely: a bare .npy array, not an .npz archive with named fields, and it
    # predates the current camera geometry / OSNet verifier / aspect-ratio-gate pipeline this
    # registry format was built for. Detect that mismatch here with a clear, actionable message
    # instead of letting a raw KeyError/TypeError from a missing/wrong-shaped field leak out —
    # this data is not safe to guess-migrate, the person must be re-registered from scratch.
    if not hasattr(data, "files") or "aspect_ratio" not in data.files or "embedding" not in data.files:
        raise ValueError(
            f"Registry entry '{path}' is not in the expected format (missing required "
            f"'embedding'/'aspect_ratio' fields) — it is likely an old, incompatible reference "
            f"file from before the current registration pipeline. Please re-register this "
            f"person via '--mode register' instead of reusing this file."
        )

    result = {
        "aspect_ratio": float(data["aspect_ratio"]),
        "created_at": str(data["created_at"]),
        "sample_count": int(data.get("sample_count", data.get("valid_sample_count", 0))),
    }

    # Load single composite embedding if present
    if "embedding" in data:
        result["embedding"] = data["embedding"]
    
    # Load multi-view embeddings if present
    views = ['front', 'right', 'back', 'left']
    multi_views = {}
    valid_embs = []
    for v in views:
        key = f"embedding_{v}"
        if key in data:
            val = data[key]
            if val.ndim > 0 and val.size > 0:
                multi_views[v] = val
                valid_embs.append(val)
            else:
                multi_views[v] = None
    
    if multi_views:
        result["multi_views"] = multi_views
    
    # If composite 'embedding' wasn't directly in data, average available multi-view embeddings
    if "embedding" not in result:
        if valid_embs:
            mean_emb = np.mean(valid_embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-6:
                mean_emb = mean_emb / norm
            result["embedding"] = mean_emb
        else:
            raise ValueError(f"No valid embeddings found in '{path}'")

    # Load pose proportions if present
    pose_props = {}
    for k in ['shoulder_hip_ratio', 'leg_torso_ratio', 'shoulder_width']:
        for prefix in [f"pose_{k}", k]:
            if prefix in data:
                val = data[prefix]
                if val.size > 0:
                    pose_props[k] = float(val)
                break
    result["pose_proportions"] = pose_props
            
    return result


def delete_person(name: str) -> None:
    path = registry_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Registry entry not found for '{name}' at '{path}'.")
    os.remove(path)


def rename_person(old_name: str, new_name: str) -> str:
    """Rename a registry entry. Raises if the old entry is missing or the new name is taken."""
    old_path = registry_path(old_name)
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Registry entry not found for '{old_name}' at '{old_path}'.")

    new_path = registry_path(new_name)  # also validates/sanitizes new_name
    if os.path.exists(new_path):
        sanitized_new = sanitize_person_name(new_name)
        raise FileExistsError(f"A person named '{sanitized_new}' already exists in the registry.")

    os.rename(old_path, new_path)
    return new_path
