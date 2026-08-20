"""
EdgeFace-XS embedding wrapper (spec §2.2, edgeface_xs_gamma_06.onnx: ~1.77M params, 99.73% LFW —
the exact variant cited in the module spec, chosen over ArcFace/InsightFace to sidestep
InsightFace's non-commercial pretrained-model licensing question entirely).

Preprocessing matches the model's own reference implementation exactly (yakhyo/edgeface-onnx's
model/edgeface.py), per spec §3's "don't invent a new metric/preprocessing, use what the model's
own documentation recommends":
  BGR->RGB swap, (pixel - 127.5) / 127.5, NCHW float32, then L2-normalize the output embedding.

Deliberately its own onnxruntime session — own model instance convention, same as every other
module in this project.
"""
import logging

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MEAN = 127.5
_SCALE = 1.0 / 127.5


class EdgeFaceEmbedder:
    def __init__(self, model_path: str):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"face_identity: loaded EdgeFace embedder from '{model_path}'")

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        `aligned_face_bgr`: a (112, 112, 3) BGR crop, already aligned (see alignment.py).
        Returns a (512,) float32 L2-normalized embedding.
        """
        blob = cv2.dnn.blobFromImage(
            aligned_face_bgr, scalefactor=_SCALE, size=(112, 112),
            mean=(_MEAN, _MEAN, _MEAN), swapRB=True, crop=False,
        )
        outputs = self._session.run(None, {self._input_name: blob})
        embedding = outputs[0][0]
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        return embedding.astype(np.float32)
