"""
Appearance Verifier — OSNet Re-ID — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.appearance_verifier. Everything else
in this package (config.py, embedder.py, matching.py, pipeline.py) is an internal implementation
detail and may change without notice.

Purpose (plans/05_appearance_verifier.md): answers "does this new person crop look like the same
person as this earlier set of reference crops?" — an appearance-based identity check, distinct
from and complementary to modules.face_identity's face-based check. Consumed by two other
modules: modules.target_tracking (a periodic re-verification sanity check during active
tracking, catching a motion tracker silently switching to a different nearby person) and
modules.target_recovery (a fallback re-acquisition path used when face-based recovery has failed
repeatedly). This module holds no per-caller state of its own — both callers may safely run their
own independent usage of these functions simultaneously.

Model: OSNet (Omni-Scale Network) via torchreid — chosen over a generic ResNet backbone because
it's purpose-trained for the same-person-across-views matching task, not repurposed generic
classification features. See embedder.py's docstring for the specific weights source
(Market1501-pretrained osnet_x1_0, auto-downloaded and cached on first use).

KNOWN LIMITATIONS — both risks below apply to EVERY caller of this module, not just one, and are
kept deliberately separate (they are two distinct, separately-worth-testing-for risks, not one
vague "may have accuracy issues" note):
  1. Similar-clothing confusion: OSNet-based appearance matching struggles to distinguish people
     wearing similar-colored/styled clothing, since appearance embeddings lean heavily on
     clothing as a feature. Test explicitly for this scenario during calibration.
  2. Cross-domain generalization drop: published OSNet benchmarks show accuracy can drop sharply
     on footage meaningfully different from its training distribution (Market-1501-family
     datasets) — this project's own campus footage/lighting/camera are an untested domain
     relative to that training data. A distinct risk from clothing confusion, not the same one.
Because of both risks, `similarity_threshold` is treated as ESPECIALLY uncalibrated — see
docs/parameters.md's note on this module specifically before trusting any starting-guess value
here more than usual.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import AppearanceVerifierConfig, load_config
from .pipeline import AppearanceVerifierPipeline, ReferenceEmbeddingSet

__all__ = [
    "AppearanceVerifierResult", "ReferenceEmbeddingSet", "build_reference_set", "verify", "configure",
]


@dataclass
class AppearanceVerifierResult:
    match_found: bool
    best_similarity_score: float   # ALWAYS a real number, never None/NaN — see
                                     # reference_frame_count for whether this was a meaningful
                                     # comparison (spec §5)
    reference_frame_count: int      # how many reference embeddings were actually compared
                                     # against; 0 means "not ready" — never confuse this with a
                                     # genuine non-match


_pipeline_singleton: Optional[AppearanceVerifierPipeline] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Optional: (re)initialize the module-level pipeline from a specific config path before the
    first build_reference_set()/verify() call. If never called, lazily initializes from the
    default "config/thresholds.yaml" on first use."""
    global _pipeline_singleton
    config: AppearanceVerifierConfig = load_config(thresholds_config_path)
    _pipeline_singleton = AppearanceVerifierPipeline(config)


def _get_pipeline() -> AppearanceVerifierPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        configure()
    return _pipeline_singleton


def build_reference_set(person_crops: List[np.ndarray]) -> ReferenceEmbeddingSet:
    """
    Takes a list of already-cropped person-bbox images (BGR) — e.g. the frames a calling
    pipeline's RECORD phase captured — and returns an embedded reference set this module can
    later compare against via verify(). Embedding happens ONCE here, not per-comparison; callers
    should build this once per episode and reuse it.
    """
    return _get_pipeline().build_reference_set(person_crops)


def verify(candidate_crop: np.ndarray, reference_set: ReferenceEmbeddingSet) -> AppearanceVerifierResult:
    """
    Embeds candidate_crop (BGR) and compares against every embedding in reference_set via cosine
    similarity. Returns the BEST score found, and match_found = best_score >=
    config.similarity_threshold. If reference_set is empty (reference_frame_count == 0), reports
    match_found=False without attempting a meaningless comparison against nothing — check
    reference_frame_count to distinguish "not ready" from a genuine non-match.
    """
    result = _get_pipeline().verify(candidate_crop, reference_set)
    return AppearanceVerifierResult(
        match_found=result.match_found,
        best_similarity_score=result.best_similarity_score,
        reference_frame_count=result.reference_frame_count,
    )
