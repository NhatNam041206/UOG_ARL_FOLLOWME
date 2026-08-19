import os
import logging
from typing import List
import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)

# Google Drive file IDs from the official torchreid Model Zoo:
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md
#
# IMPORTANT: torchreid's `pretrained=True` default (used when no model_path is passed to
# FeatureExtractor) only downloads ImageNet-CLASSIFICATION backbone weights — the exact same
# category of problem this verifier replacement exists to fix. All checkpoints below are
# instead trained with an actual person re-identification objective (ID/triplet loss on real
# re-id datasets), which is what makes the resulting embedding identity-discriminative.
_REID_CHECKPOINT_GDRIVE_IDS = {
    # --- Width-scaled family, all trained AND evaluated on Market-1501 (single-domain,
    # "Same-domain ReID" table in the Model Zoo). Same architecture family, so these are
    # interchangeable speed/accuracy points — pick the fastest one whose Rank-1/mAP drop vs
    # osnet_x1_0 you can tolerate. All four IDs below were cross-checked directly against the
    # Model Zoo's raw markdown table (not just the linked "model" column, which points at
    # ImageNet-only backbone weights per the warning above) on 2026-08-16.
    "osnet_x1_0": "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",   # 2.2M params — Rank-1 94.2 / mAP 82.6 (baseline)
    "osnet_x0_75": "1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",  # 1.3M params — Rank-1 93.7 / mAP 81.2 (~-1.7% mAP)
    "osnet_x0_5": "1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT",   # 0.6M params — Rank-1 92.5 / mAP 79.8 (~-3.4% mAP)
    "osnet_x0_25": "1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",  # 0.2M params — Rank-1 91.2 / mAP 75.0 (~-9.2% mAP)
    # Trained on MSMT17+DukeMTMC+CUHK03 (NOT Market1501), evaluated zero-shot on Market1501 as
    # an unseen target domain. Rank-1 73.3 / mAP 45.8 ("Cross-domain ReID" table) — the Model
    # Zoo has no Market1501-only checkpoint for this variant. Picked deliberately anyway:
    # osnet_ain is architecturally built for cross-domain robustness, and a live webcam feed is
    # itself a domain none of these checkpoints were trained on — so generalizing across unseen
    # domains matters more here than the same-domain benchmark number.
    "osnet_ain_x1_0": "1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB",
}

_CHECKPOINT_DIR = "models"
# OSNet's final embedding FC projects every width variant's backbone (512/384/256/128 channels
# for x1_0/x0_75/x0_5/x0_25 respectively) up to the SAME 512-d output — feature_dim is not
# width-scaled in torchreid's OSNet implementation, confirmed by reading
# torchreid/models/osnet.py (none of the osnet_x0_* factory functions override the OSNet.
# __init__ feature_dim=512 default). So this constant holds for every variant above.
_FEATURE_DIM = 512


class OSNetVerifier:
    def __init__(self, variant: str = "osnet_x1_0"):
        """
        Person re-identification feature extractor built on OSNet (torchreid), pretrained with
        a re-id objective — NOT fine-tuned further here.

        Args:
            variant: "osnet_x1_0" (default, trained+evaluated on Market1501) or
                     "osnet_ain_x1_0" (trained on other re-id datasets, picked for cross-domain
                     robustness — see _REID_CHECKPOINT_GDRIVE_IDS comment above).
        """
        self.variant = (variant or "osnet_x1_0").strip().lower()
        if self.variant not in _REID_CHECKPOINT_GDRIVE_IDS:
            raise ValueError(
                f"Unsupported OSNet variant '{self.variant}'. Supported: "
                f"{sorted(_REID_CHECKPOINT_GDRIVE_IDS.keys())}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Limit PyTorch CPU threads to 1 per worker to prevent CPU thread contention when
        # running inside pipeline.py's ThreadPoolExecutor (kept from the MobileNetV3 verifier
        # this class replaces — confirmed necessary by prior benchmarking).
        if self.device.type == "cpu":
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        checkpoint_path = self._ensure_checkpoint(self.variant)

        # FeatureExtractor resizes to its own required input size (256x128 for OSNet) and
        # applies ImageNet-mean/std normalization internally — do NOT re-apply either step
        # ourselves in extract() below.
        self.extractor = FeatureExtractor(
            model_name=self.variant,
            model_path=checkpoint_path,
            device=self.device.type,
            verbose=False,
        )
        logger.info(
            f"OSNetVerifier initialized ({self.variant}, re-id pretrained weights: "
            f"'{checkpoint_path}') on device: {self.device}"
        )

    @staticmethod
    def _ensure_checkpoint(variant: str) -> str:
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(_CHECKPOINT_DIR, f"{variant}_reid.pth")
        if not os.path.exists(checkpoint_path):
            import gdown
            file_id = _REID_CHECKPOINT_GDRIVE_IDS[variant]
            logger.info(f"Downloading person re-id pretrained weights for '{variant}' to '{checkpoint_path}'...")
            gdown.download(id=file_id, output=checkpoint_path, quiet=False)
            if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
                raise RuntimeError(
                    f"Failed to download OSNet re-id checkpoint for variant '{variant}' "
                    f"(Google Drive id: {file_id}). Check network access, or download it "
                    f"manually from the torchreid Model Zoo and place it at '{checkpoint_path}'."
                )
        return checkpoint_path

    def extract(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized re-id feature embedding from an image crop (BGR numpy array).

        Single-crop convenience wrapper around extract_batch() — prefer extract_batch()
        directly when verifying multiple people in the same frame (a shared batched forward
        pass is dramatically cheaper per-crop than N sequential single-image calls; this was
        the single largest per-frame cost in profiling, see README perf section).

        Returns:
            1D np.ndarray (float32) feature embedding vector, L2-normalized.
        """
        return self.extract_batch([image_crop])[0]

    def extract_batch(self, image_crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract L2-normalized re-id feature embeddings for MULTIPLE crops (BGR numpy arrays)
        in a single batched forward pass through OSNet — one model launch instead of N.

        torchreid's FeatureExtractor already natively batches a list of numpy arrays into one
        stacked tensor internally (see torchreid/utils/feature_extractor.py FeatureExtractor.
        __call__), so this just needs to hand it a list instead of looping extract() one crop
        at a time. Empty/invalid crops are filtered out before the batched call (so a single
        malformed detection can't force the whole frame down the single-image path) and
        re-inserted as zero vectors at their original positions afterward.

        Returns:
            List of 1D np.ndarray (float32) embedding vectors, same length/order as
            `image_crops`, each L2-normalized (or all-zero for invalid input crops).
        """
        if not image_crops:
            return []

        valid_indices = []
        valid_rgb = []
        for i, crop in enumerate(image_crops):
            if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                logger.warning("Empty or invalid image crop passed to OSNetVerifier.extract_batch()")
                continue
            valid_indices.append(i)
            valid_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        results = [np.zeros(_FEATURE_DIM, dtype=np.float32) for _ in image_crops]
        if not valid_rgb:
            return results

        with torch.no_grad():
            features = self.extractor(valid_rgb)

        batch_vecs = features.cpu().numpy().astype(np.float32)
        for slot, orig_idx in enumerate(valid_indices):
            vec = batch_vecs[slot]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                logger.warning("Extracted embedding vector has near-zero L2 norm.")
            results[orig_idx] = vec

        return results

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two L2-normalized embedding vectors via dot product.

        Returns:
            Cosine similarity score as float [-1.0 to 1.0].
        """
        if embedding_a is None or embedding_b is None:
            return 0.0

        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0

        vec_a = embedding_a / norm_a if abs(norm_a - 1.0) > 1e-3 else embedding_a
        vec_b = embedding_b / norm_b if abs(norm_b - 1.0) > 1e-3 else embedding_b

        similarity = float(np.dot(vec_a, vec_b))
        return similarity
