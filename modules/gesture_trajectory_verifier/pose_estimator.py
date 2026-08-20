"""
MoveNet Lightning singlepose wrapper. Fresh, standalone instance — spec §0.3 confirms reusing
the MODEL (same TF Hub weights modules.wave_facing_gate uses) is fine, but this wrapper code is
written independently, not imported, since it's "logic operating on the model."
"""
import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # see modules/wave_facing_gate/pose_estimator.py for rationale
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

import tensorflow_hub as hub

logger = logging.getLogger(__name__)


class MoveNetPoseEstimator:
    def __init__(self, tfhub_handle: str):
        self._module = hub.load(tfhub_handle)
        self._infer = self._module.signatures["serving_default"]
        logger.info(f"gesture_trajectory_verifier: loaded MoveNet from '{tfhub_handle}'")

    def estimate(self, input_tensor: np.ndarray) -> np.ndarray:
        batched = tf.cast(tf.expand_dims(input_tensor, axis=0), dtype=tf.int32)
        outputs = self._infer(batched)
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores[0, 0]
