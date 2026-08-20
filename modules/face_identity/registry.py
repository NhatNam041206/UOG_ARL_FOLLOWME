"""
Face registry storage (spec §4.2). No face-registration system existed anywhere in this repo
before this module. Per user decision, this mirrors the STORAGE FORMAT of the sibling
UOG_ARL_FOLLOWME project's OSNet registry (src/registry.py: sanitized-person-name -> one .npz
file, embedding + metadata fields) as fresh, independent code — not a shared/imported module.
That project is a separate git repository with no Python import path connecting it to this one;
this file was written by reading its registry.py for the storage SHAPE only, then reimplemented
from scratch, per the "mirror the pattern, don't share state" isolation instruction (spec §0.3).

Registered by person NAME (not an abstract ID) — the sanitized name IS the registry key.

Differs from the sibling's format in one deliberate way: this stores every captured sample's
embedding (not just a composite), since spec §4.3 asks for multiple samples per person precisely
because a single reference face is fragile — the composite alone would throw that fidelity away.
"""
import glob
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_person_name(raw_name: Optional[str]) -> str:
    """
    Turn free-text input into a safe filename stem: keeps only word characters (letters incl.
    Unicode/diacritics, digits, underscore) and hyphens; whitespace runs become a single
    underscore. Rules out path traversal by construction — '.', '/', '\\' are stripped entirely.
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


@dataclass(frozen=True)
class RegistryEntry:
    person_name: str
    embedding: np.ndarray          # (512,) float32, L2-normalized composite (mean of samples)
    sample_embeddings: np.ndarray  # (N, 512) float32, every captured sample, for future reuse
    sample_count: int
    created_at: str


class FaceRegistry:
    """Owns no live model state — just reads/writes .npz files under `registry_dir`."""

    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir

    def _path(self, person_name: str) -> str:
        return os.path.join(self.registry_dir, f"{sanitize_person_name(person_name)}.npz")

    def person_exists(self, person_name: str) -> bool:
        return os.path.exists(self._path(person_name))

    def save_person(self, person_name: str, sample_embeddings: List[np.ndarray]) -> str:
        """
        Saves (or overwrites) a person's registry entry from N captured sample embeddings
        (each already L2-normalized, spec §4.3 "multiple samples per person"). The composite
        embedding is the mean of samples, re-normalized. Returns the saved path.
        """
        if not sample_embeddings:
            raise ValueError("Cannot save a registry entry with zero sample embeddings.")
        os.makedirs(self.registry_dir, exist_ok=True)

        samples = np.stack([np.asarray(e, dtype=np.float32) for e in sample_embeddings])
        composite = np.mean(samples, axis=0)
        norm = np.linalg.norm(composite)
        if norm > 1e-6:
            composite = composite / norm

        path = self._path(person_name)
        np.savez(
            path,
            embedding=composite.astype(np.float32),
            sample_embeddings=samples,
            sample_count=np.array(len(sample_embeddings)),
            created_at=np.array(datetime.now().isoformat()),
        )
        return path

    def load_person(self, person_name: str) -> RegistryEntry:
        path = self._path(person_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Registry entry not found for '{person_name}' at '{path}'.")
        return self._load_path(path, sanitize_person_name(person_name))

    def _load_path(self, path: str, person_name: str) -> RegistryEntry:
        data = np.load(path, allow_pickle=False)
        if "embedding" not in data.files:
            raise ValueError(
                f"Registry entry '{path}' is missing the required 'embedding' field — not a "
                f"valid face_identity registry file. Re-register this person."
            )
        return RegistryEntry(
            person_name=person_name,
            embedding=data["embedding"],
            sample_embeddings=data["sample_embeddings"] if "sample_embeddings" in data.files else data["embedding"][None, :],
            sample_count=int(data["sample_count"]) if "sample_count" in data.files else 1,
            created_at=str(data["created_at"]) if "created_at" in data.files else "",
        )

    def load_all(self) -> List[RegistryEntry]:
        """Every readable entry in the registry. Unreadable/corrupt files are skipped with a
        warning rather than raising, so one bad file doesn't break matching for everyone else."""
        os.makedirs(self.registry_dir, exist_ok=True)
        entries: List[RegistryEntry] = []
        for path in sorted(glob.glob(os.path.join(self.registry_dir, "*.npz"))):
            person_name = os.path.splitext(os.path.basename(path))[0]
            try:
                entries.append(self._load_path(path, person_name))
            except Exception as e:
                logger.warning(f"face_identity: skipping unreadable registry entry '{path}': {e}")
        return entries

    def delete_person(self, person_name: str) -> None:
        path = self._path(person_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Registry entry not found for '{person_name}' at '{path}'.")
        os.remove(path)

    def list_registry(self) -> List[Dict[str, Any]]:
        """Lightweight listing (no embedding arrays) for UI/CLI display."""
        return [
            {"person_name": e.person_name, "sample_count": e.sample_count, "created_at": e.created_at}
            for e in self.load_all()
        ]
