"""
Gesture-method dispatch adapter — mirrors main.py's own `_GestureMethodAdapter` class exactly
(the same calling-convention normalization across the three interchangeable gesture methods),
per plans/08 §0.5's audit instruction to replicate that existing, already-working sequencing
rather than reinvent it.

Physically its own independent copy living inside this module's own package — not imported from
main.py, which is a script entry point (not a library other modules import from), and this
project's own-code isolation convention treats duplicated LOGIC across modules the same way it
treats duplicated confirmation-tracker/BboxContext code elsewhere: reimplemented independently,
not shared, even when nearly identical.
"""
from typing import Optional, Tuple

from modules.wave_facing_gate.interface import WaveFacingGateModule


class GestureMethodAdapter:
    def __init__(self, method_name: str):
        self.method_name = method_name
        self._last_result = None  # stashed by evaluate(), consumed by draw_debug()
        if method_name == "condition":
            self._module = WaveFacingGateModule()
        elif method_name == "hand_keypoint":
            import modules.gesture_hand_keypoint.interface as gi
            self._module = gi
        elif method_name == "trajectory_verifier":
            import modules.gesture_trajectory_verifier.interface as gi
            self._module = gi
        else:
            raise ValueError(f"Unknown gesture method '{method_name}'")

    def warmup(self, thresholds_config_path: str = "config/thresholds.yaml") -> None:
        """Eagerly constructs the underlying model NOW rather than on the first real evaluate()
        call (confirmed with the user — startup should absorb model-load latency, not the live
        pipeline). "condition" is already fully constructed in __init__ above (WaveFacingGateModule
        loads its MoveNet instance eagerly); hand_keypoint/trajectory_verifier instead hold a
        reference to their own INTERFACE MODULE (self._module = gi above), which lazily builds its
        own pipeline singleton on first configure()/evaluate() — calling configure() here forces
        that construction immediately."""
        if self.method_name in ("hand_keypoint", "trajectory_verifier"):
            self._module.configure(thresholds_config_path)

    def evaluate(self, track_id: int, crop, timestamp: float,
                 person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str]:
        """Returns (is_waving, waving_state). `person_bbox_full_frame` is only used by
        hand_keypoint — ignored by the other two methods, exactly as in main.py's adapter. Also
        stashes the raw result object for draw_debug() below."""
        if self.method_name == "condition":
            r = self._module.process_frame(track_id=track_id, crop=crop)
            self._last_result = r
            return r.is_waving, r.waving_state

        if self.method_name == "hand_keypoint":
            r = self._module.evaluate(track_id, crop, timestamp, person_bbox_full_frame=person_bbox_full_frame)
        else:  # trajectory_verifier
            r = self._module.evaluate(track_id, crop, timestamp)
        self._last_result = r
        return r.is_waving, r.waving_state

    def draw_debug(self, crop, person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Draws the last evaluate() call's per-method debug overlay directly onto `crop` (a
        view into the caller's frame) — mirrors main.py's own _GestureMethodAdapter.draw_debug().
        No-ops if the method has no draw_debug or nothing has been evaluated yet."""
        if self._last_result is None or not hasattr(self._last_result, "draw_debug"):
            return
        if self.method_name == "hand_keypoint":
            self._last_result.draw_debug(crop, person_bbox_full_frame=person_bbox_full_frame)
        else:
            self._last_result.draw_debug(crop)

    def release_track(self, track_id: int) -> None:
        if self.method_name == "condition":
            self._module.reset_track(track_id)
        else:
            self._module.release_track(track_id)
