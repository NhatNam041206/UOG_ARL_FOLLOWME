"""
MoveNet Lightning keypoint indices needed by this module — reimplemented independently here
(spec §0.3: reusing the MoveNet MODEL is fine, reusing modules.wave_facing_gate's code/classes
that operate on its output is not, even for something as small as these index constants).
"""
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10

# (wrist, elbow, shoulder) index triples per arm.
ARM_KEYPOINTS = {
    "left": (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
    "right": (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER),
}
