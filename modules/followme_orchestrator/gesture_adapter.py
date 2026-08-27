"""
Gesture-method dispatch adapter — mirrors main.py's own `_GestureMethodAdapter` class exactly
(a thin wrapper around modules.gesture_hand_keypoint, the sole TRIGGER gesture method; two others
— modules.wave_facing_gate "condition" and modules.gesture_trajectory_verifier
"trajectory_verifier" — were removed, confirmed with the user hand_keypoint is the only one kept).

Physically its own independent copy living inside this module's own package — not imported from
main.py, which is a script entry point (not a library other modules import from), and this
project's own-code isolation convention treats duplicated LOGIC across modules the same way it
treats duplicated confirmation-tracker/BboxContext code elsewhere: reimplemented independently,
not shared, even when nearly identical.
"""
from typing import Optional, Tuple

import modules.gesture_hand_keypoint.interface as gi


class GestureMethodAdapter:
    def __init__(self):
        self._module = gi
        self._last_result = None  # stashed by evaluate(), consumed by draw_debug()

    def warmup(self, thresholds_config_path: str = "config/thresholds.yaml") -> None:
        """Eagerly constructs the underlying model NOW rather than on the first real evaluate()
        call (confirmed with the user — startup should absorb model-load latency, not the live
        pipeline). self._module holds a reference to gesture_hand_keypoint's own INTERFACE
        MODULE, which lazily builds its own pipeline singleton on first configure()/evaluate() —
        calling configure() here forces that construction immediately."""
        self._module.configure(thresholds_config_path)

    def evaluate(self, track_id: int, crop, timestamp: float,
                 person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str]:
        """Returns (is_waving, waving_state). `person_bbox_full_frame` feeds the palm-height gate
        (measured against the person's full-frame bbox, not just the crop). Also stashes the raw
        result object for draw_debug() below."""
        r = self._module.evaluate(track_id, crop, timestamp, person_bbox_full_frame=person_bbox_full_frame)
        self._last_result = r
        return r.is_waving, r.waving_state

    @property
    def last_result(self):
        """The full GestureMethodResult (including sequence_stage/open_count/close_count/
        total_confirmed_count) from the most recent evaluate() call — None before the first call.
        For debug_snapshot.py's use; the (bool, str) tuple evaluate() returns is deliberately
        narrowed for the orchestrator's own trigger-check logic, this is the escape hatch for
        logging the fuller picture without changing that narrow return type."""
        return self._last_result

    def draw_debug(self, crop, person_bbox_full_frame: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Draws the last evaluate() call's debug overlay directly onto `crop` (a view into the
        caller's frame) — mirrors main.py's own _GestureMethodAdapter.draw_debug(). No-ops if
        nothing has been evaluated yet."""
        if self._last_result is None:
            return
        self._last_result.draw_debug(crop, person_bbox_full_frame=person_bbox_full_frame)

    def release_track(self, track_id: int) -> None:
        self._module.release_track(track_id)
