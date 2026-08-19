"""
MoveNet Lightning (singlepose) wrapper for demo_wave_trigger.py — Quick Demo Spec: Wave +
Facing Trigger Gate (document/implementation/followme/Project_Master_Doc.md muc 3). New setup
for this demo only, not used by the existing detect/verify pipeline (src/pipeline.py,
src/detector.py, src/verifier.py), which this module never imports or modifies.

Loaded via TF Hub ("hub.load(...)") rather than a manually-downloaded .tflite file — the model
is fetched and cached on first use, the same auto-download-on-first-use pattern Ultralytics
already uses elsewhere in this project for YOLO weights (see src/detector.py).
"""
import os
import logging
import warnings
from typing import Optional, Tuple

# Both must be set before `import tensorflow`/`import tensorflow_hub` to take effect — silences
# TF's native-backend startup chatter (oneDNN/cpu_feature_guard/"Fingerprint not found") and
# tensorflow_hub's `pkg_resources`-deprecated UserWarning (see requirements.txt's setuptools<81
# pin — that's the functional fix; this is just cosmetic console noise on top of it). Cosmetic
# only: does not affect model behavior, and actual init failures still surface as Python
# exceptions. One "WARNING: All log messages before absl::InitializeLog()..." line may still
# print once — that's absl's own pre-init preamble, printed unconditionally on first native log
# call regardless of these settings, and not suppressible from Python.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# Set BEFORE importing tensorflow: the "tensorflow" logger name is just a key in Python's
# logging registry and doesn't require the module to be imported yet — but tensorflow logs a
# tf_keras deprecation notice DURING its own import, so setting the level after import is too
# late to catch it.
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)

# Fixed COCO keypoint order MoveNet outputs — index positions must NOT change (see spec muc 3).
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

INPUT_SIZE = 192


def movenet_point_to_crop_px(y_norm: float, x_norm: float, crop_w: int, crop_h: int) -> Tuple[float, float]:
    """
    Invert `resize_with_pad`'s letterboxing: map a normalized [0,1] keypoint (relative to the
    centered, letterboxed INPUT_SIZE x INPUT_SIZE model input) back to pixel coordinates in the
    original crop. Confirmed empirically that resize_with_pad centers the scaled image
    (symmetric padding on the shorter target dimension), not top/left-aligned — so this is not a
    simple linear rescale, the padding offset must be subtracted first. Shared by
    src/wave_detector.py (posture-gate geometry) and demo_wave_trigger.py (skeleton overlay) so
    both stay consistent with how MoveNet's own preprocessing actually works.
    """
    scale = min(INPUT_SIZE / crop_h, INPUT_SIZE / crop_w)
    scaled_w, scaled_h = crop_w * scale, crop_h * scale
    pad_x, pad_y = (INPUT_SIZE - scaled_w) / 2.0, (INPUT_SIZE - scaled_h) / 2.0
    px = (x_norm * INPUT_SIZE - pad_x) / scale
    py = (y_norm * INPUT_SIZE - pad_y) / scale
    return px, py


class MoveNetPoseEstimator:
    """Runs MoveNet Lightning on a single-person crop (not the full frame)."""

    def __init__(self, model_url: str):
        logger.info(f"Loading MoveNet Lightning from '{model_url}' (downloads + caches on first use)...")
        model = hub.load(model_url)
        self._movenet = model.signatures["serving_default"]
        logger.info("MoveNet Lightning loaded.")

    def estimate(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            crop_bgr: BGR person crop (from FollowPipeline's verified bbox), any size.

        Returns:
            (17, 3) array of [y, x, confidence_score], normalized [0.0, 1.0] against the
            192x192 letterboxed (resize_with_pad) input frame — or None if the crop is empty.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        rgb = crop_bgr[:, :, ::-1]
        image = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        image = tf.image.resize_with_pad(image[tf.newaxis, ...], INPUT_SIZE, INPUT_SIZE)
        image = tf.cast(image, dtype=tf.int32)

        outputs = self._movenet(image)
        keypoints = outputs["output_0"].numpy()  # [1, 1, 17, 3]
        return keypoints[0, 0, :, :]
