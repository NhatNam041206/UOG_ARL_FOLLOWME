"""
MoveNet Lightning singlepose wrapper (spec §2). Deliberately its own model instance — this
module owns its inference call the same way modules/emergency_stop/detection.py owns its own
YOLO instance, so the two modules never share model state.

Loaded via tensorflow_hub (confirmed with the user, in place of hunting for a pre-converted ONNX
file): `hub.load(handle)` fetches and locally caches the weights on first use — no manual model
file needs to be committed to the repo or downloaded by hand.
"""
import logging
import os
import warnings

# Quiet TensorFlow/TF-Hub startup noise (spec-irrelevant log spam from the C++ runtime and a
# transitively-imported deprecated Keras API, unrelated to this module's correctness) — must be
# set BEFORE the relevant import, since each knob only affects what's imported/logged after it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # silence TF's own C++-side INFO/WARNING logs
# TF's oneDNN-enabled notice (port.cc) ignores TF_CPP_MIN_LOG_LEVEL in this TF build and is only
# silenced by disabling oneDNN's CPU-op acceleration outright. Accepted trade-off here: MoveNet
# Lightning is a tiny model already running well inside budget (~15-20ms/frame on CPU per the
# module's own latency benchmarking), and this gate's thresholds (degrees, pixels, confidence)
# have far more tolerance than oneDNN's floating-point round-off could ever move a result by. If
# that assumption stops holding on a future, slower target device, remove this line first.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)  # before importing tensorflow_hub, which imports the
                                          # deprecated tf_keras compat shim as a side effect

import tensorflow_hub as hub

logger = logging.getLogger(__name__)


class MoveNetPoseEstimator:
    def __init__(self, tfhub_handle: str):
        self._module = hub.load(tfhub_handle)
        self._infer = self._module.signatures["serving_default"]
        logger.info(f"wave_facing_gate: loaded MoveNet from '{tfhub_handle}'")

    def estimate(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        `input_tensor`: [input_size, input_size, 3] uint8 RGB (see preprocessing.py). Returns
        [17, 3] float array of (y, x, score), normalized to [0, 1] against the input tensor —
        still in model-input space, NOT yet bbox-relative (see preprocessing.decode_keypoints).
        """
        batched = tf.cast(tf.expand_dims(input_tensor, axis=0), dtype=tf.int32)
        outputs = self._infer(batched)
        keypoints_with_scores = outputs["output_0"].numpy()  # [1, 1, 17, 3]
        return keypoints_with_scores[0, 0]
