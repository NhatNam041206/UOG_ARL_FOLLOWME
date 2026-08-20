"""
MediaPipe Hands' fixed 21-point landmark layout — do not reorder, fixed by the model's own
output contract.
"""
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# Non-thumb finger (tip, pip) index pairs used by the open/closed check (hand_shape.py) — thumb is
# excluded since its geometry doesn't fit the same tip-farther-than-pip-from-wrist heuristic
# (the thumb's PIP/MCP joints don't line up radially from the wrist the way the other four
# fingers' do), a well-known limitation of this lightweight heuristic, not a bug.
FINGER_TIP_PIP_PAIRS = {
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}
