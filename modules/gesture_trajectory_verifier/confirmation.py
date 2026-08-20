"""
RED/YELLOW/GREEN confirmation state machine (spec §5, fixed pattern — reimplemented
independently here per spec §0.3: no shared class/instance with modules.wave_facing_gate or
modules.gesture_hand_keypoint). One instance per track_id.
"""
from typing import Optional

from .config import GestureTrajectoryVerifierConfig

RED = "RED"
YELLOW = "YELLOW"
GREEN = "GREEN"


class ConfirmationTracker:
    def __init__(self):
        self.state = RED
        self.yellow_since: Optional[float] = None

    def update(self, raw_condition_pass: bool, timestamp: float, config: GestureTrajectoryVerifierConfig) -> str:
        if not raw_condition_pass:
            self.state = RED
            self.yellow_since = None
            return self.state

        if self.state == RED:
            self.state = YELLOW
            self.yellow_since = timestamp
        elif self.state == YELLOW:
            if timestamp - self.yellow_since >= config.confirmation_duration_seconds:
                self.state = GREEN
        return self.state
