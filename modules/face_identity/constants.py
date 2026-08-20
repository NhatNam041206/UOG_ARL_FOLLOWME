"""
Fixed geometric/model constants — not calibration targets, do not move to thresholds.yaml.
"""
import numpy as np

# EdgeFace's expected input size (spec §2.2's chosen model, edgeface_xs_gamma_06.onnx: 1.77M
# params, matches the plan's cited 99.73% LFW variant).
EDGEFACE_INPUT_SIZE = 112

# Canonical 5-point ArcFace-style alignment template for a 112x112 aligned face (left eye, right
# eye, nose tip, left mouth corner, right mouth corner), in that fixed order. This is the standard
# reference template EdgeFace (and ArcFace-family models generally) were trained against — sourced
# from yakhyo/uniface's face_utils.py (the same author's alignment utility used in the official
# edgeface-onnx inference demo), not invented here, per the "use what the model's own
# documentation recommends" instruction in the module spec §3.
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)
