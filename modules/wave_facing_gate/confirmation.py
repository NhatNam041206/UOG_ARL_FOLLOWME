"""
RED/YELLOW/GREEN confirmation state machine (spec §6). Debounces a per-frame raw boolean into a
held signal — a single passing frame is never enough; GREEN requires CONFIRMATION_DURATION_SECONDS
of UNINTERRUPTED passing frames, with no partial credit on interruption.

One instance per (track_id, signal) — is_waving and is_facing_camera each get their own tracker
per track, fully decoupled from each other (spec §5).
"""
from typing import Optional

from .config import WaveFacingConfig

RED = "RED"
YELLOW = "YELLOW"
GREEN = "GREEN"


class ConfirmationTracker:
    def __init__(self):
        self.state = RED
        self.yellow_since: Optional[float] = None

    def update(self, raw_condition_pass: bool, timestamp: float, config: WaveFacingConfig) -> str:
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
        # GREEN stays GREEN while raw_condition_pass remains True.
        return self.state
