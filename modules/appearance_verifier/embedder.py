"""
Standalone OSNet (Omni-Scale Network) Re-ID embedder wrapper. Deliberately its own model
instance — own-instance isolation convention used by every other module in this project
(docs/architecture.md rule #2).

Uses torchreid (KaiyangZhou/deep-person-reid, an installed pip dependency, MIT-licensed) — the
reference implementation OSNet's own paper and pretrained weights are published through. This is
an independent third-party package, NOT the teammate's separate UOG_ARL_FOLLOWME Re-ID pipeline
(a different git repository — see docs/technologies.md's note on the face-registry storage
format mirroring that project's format only, not its code). No code, state, or weights from that
other repository are used, imported, or reachable here.

Weights (confirmed with the user, over an ImageNet-only-backbone alternative) — IMPORTANT
correction discovered during implementation and worth recording: torchreid's own
`build_model(..., pretrained=True)` shortcut (and `FeatureExtractor` with no `model_path`) does
NOT fetch a Re-ID-trained checkpoint despite the name — it only fetches an ImageNet-CLASSIFICATION
pretrained backbone (confirmed by inspecting the actual log output: "Successfully loaded imagenet
pretrained weights..."). The real Market1501 Re-ID-trained checkpoint (94.2% rank-1, 82.6% mAP
per the official MODEL_ZOO) is a SEPARATE download, published at
https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md — this file, not
`pretrained=True`, is what actually gives this module a real person-re-identification embedding.
`_download_market1501_weights()` below fetches that specific checkpoint via `gdown` on first use
and caches it under this module's own `models/` directory (same "auto-fetch once, cache
thereafter" pattern already used for MoveNet via tensorflow_hub elsewhere in this project) —
verified working in this environment (a 10.4MB one-time download).

Preprocessing uses torchreid's OWN documented `FeatureExtractor` utility exactly as published
(image_size=(256,128), ImageNet pixel_mean/std) rather than a hand-rolled equivalent — per this
module's spec: "follow OSNet's own documented preprocessing exactly, do not invent one." The only
addition here is a BGR->RGB channel swap before handing the crop to FeatureExtractor, since this
project's universal frame convention is BGR (OpenCV, see docs/architecture.md), while
FeatureExtractor's numpy-array input path assumes RGB.
"""
import logging
import os

import cv2
import gdown
import numpy as np
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)

# The official Market1501-pretrained osnet_x1_0 checkpoint's Google Drive file id, per
# deep-person-reid's own MODEL_ZOO.md ("Same-domain ReID" section) — NOT reachable via
# build_model(pretrained=True), see module docstring above.
_MARKET1501_WEIGHTS_GDRIVE_ID = "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"
_WEIGHTS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "models", "osnet_x1_0_market1501.pth")


def _download_market1501_weights(cache_path: str) -> str:
    if os.path.exists(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info(f"appearance_verifier: downloading Market1501-pretrained OSNet weights to '{cache_path}' (one-time, ~10MB)")
    gdown.download(id=_MARKET1501_WEIGHTS_GDRIVE_ID, output=cache_path, quiet=False)
    if not os.path.exists(cache_path):
        raise RuntimeError(
            f"appearance_verifier: failed to download OSNet Market1501 weights to '{cache_path}' "
            f"— check network access to Google Drive, or supply the .pth file manually at that path."
        )
    return cache_path


class OSNetEmbedder:
    def __init__(self, model_name: str = "osnet_x1_0"):
        weights_path = _download_market1501_weights(_WEIGHTS_CACHE_PATH)
        # device="cpu": consistent with every other model in this project (see
        # docs/technologies.md — nano/lightweight models, CPU inference throughout).
        # model_path=weights_path -> FeatureExtractor loads the REAL Market1501 checkpoint on top
        # of the model (see module docstring's correction note) rather than the misleadingly-
        # named pretrained=True ImageNet-only default.
        self._extractor = FeatureExtractor(model_name=model_name, model_path=weights_path, device="cpu", verbose=False)
        logger.info(f"appearance_verifier: loaded OSNet '{model_name}' (Market1501-pretrained, '{weights_path}')")

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        `crop_bgr`: a person-bbox crop, BGR, any size. Returns an L2-normalized feature vector
        (512-D for osnet_x1_0). FeatureExtractor's own forward pass does NOT L2-normalize its
        output (see torchreid's OSNet.forward — eval mode returns the raw pooled+fc vector), so
        normalization happens here, same pattern as modules/face_identity's EdgeFace embedder.
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = self._extractor(rgb)  # [1, D] torch tensor, NOT yet normalized
        embedding = features[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        return embedding
