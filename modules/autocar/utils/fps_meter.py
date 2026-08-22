import time


class FPSMeter:
    """Exponential-moving-average FPS counter, call tick() once per processed frame."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.fps = 0.0
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.fps == 0 else self.fps * self.alpha + inst_fps * (1 - self.alpha)
        self._last = now
        return self.fps
