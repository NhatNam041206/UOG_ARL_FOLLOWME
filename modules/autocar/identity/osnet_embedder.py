"""
Appearance embedding for target re-identification, via a local ONNX OSNet checkpoint
(models/osnet_x1_0_msmt17.onnx - see models/README.md for how it got there) run through
onnxruntime directly (CPU execution provider). No torchreid dependency: the pip-installable
`torchreid` package is an unofficial fork that fails to import (missing `tensorboard`), and the
canonical source can't be `pip install -e .`'d cleanly either - loading the already-exported
.onnx ourselves sidesteps both problems entirely, at the cost of hand-rolling the standard OSNet
preprocessing (resize to 256x128, ImageNet normalize) instead of getting it from the library.
"""
import cv2
import numpy as np
import onnxruntime as ort

_INPUT_H, _INPUT_W = 256, 128
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OSNetEmbedder:
    def __init__(self, model_path: str, device: str = "cpu"):
        providers = ["CPUExecutionProvider"]
        if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(chw, axis=0).astype(np.float32)  # -> [1, 3, 256, 128]

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Returns a 512-dim, L2-normalized embedding, or an all-zero vector for a degenerate crop."""
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
            return np.zeros(512, dtype=np.float32)

        tensor = self._preprocess(crop_bgr)
        (embedding,) = self.session.run(None, {self.input_name: tensor})
        vec = embedding[0].astype(np.float32)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else vec

    @staticmethod
    def compare(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings, in [-1, 1]."""
        if embedding_a is None or embedding_b is None:
            return 0.0
        return float(np.dot(embedding_a, embedding_b))
