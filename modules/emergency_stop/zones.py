"""
Size pre-filter (spec §3.3) + three-zone runway logic (spec §3.4), including per-track mid-zone
dwell timers.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

from .roi import RunwayGeometry


class ZoneEvaluator:
    """
    Stateful (per-track mid-zone dwell timers) — one instance lives for the lifetime of the
    owning EmergencyStopModule, not per-frame.
    """

    def __init__(self):
        self._mid_zone_entry_time: Dict[int, float] = {}

    def evaluate(
        self,
        detections: List[Dict[str, Any]],
        geometry: RunwayGeometry,
        size_prefilter_width_px: float,
        size_prefilter_height_px: float,
        t_mid_seconds: float,
        now: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Returns (reason, triggering_track_id, zone) for the first STOP-worthy object found, or
        (None, None, None) if the runway is clear this frame. Near-zone hits are reported over
        mid-zone dwell hits when both occur in the same frame (near is the more urgent/immediate
        condition) — this only affects which object is named in the output, not whether the
        frame overall resolves to STOP, since both zones produce the identical STOP decision.
        """
        if now is None:
            now = time.time()

        near_hit: Optional[int] = None
        mid_hit: Optional[int] = None
        seen_mid_ids = set()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Fast-path shortcut for small/clearly-safe objects (§3.3) — NOT a replacement for
            # zone logic below; a large object can still be correctly SAFE if it maps to the far
            # zone or off-runway entirely.
            if bbox_w < size_prefilter_width_px and bbox_h < size_prefilter_height_px:
                continue

            gx, gy = det["ground_contact"]
            if not geometry.contains(gx, gy):
                continue  # ground-contact point is not on the runway at all -> safe

            zone = geometry.zone_for_y(gy)
            if zone is None or zone == "far":
                continue

            if zone == "near":
                if near_hit is None:
                    near_hit = det["track_id"]
                continue

            # zone == "mid": per-track dwell timer. Resets (does not pause/resume) if the track
            # leaves the mid zone before t_mid_seconds elapses — including if the track simply
            # isn't present in this frame's detections at all (occlusion/loss), since we can no
            # longer confirm it's still lingering in the mid zone.
            seen_mid_ids.add(det["track_id"])
            entry_time = self._mid_zone_entry_time.get(det["track_id"])
            if entry_time is None:
                entry_time = now
                self._mid_zone_entry_time[det["track_id"]] = entry_time
            if (now - entry_time) >= t_mid_seconds and mid_hit is None:
                mid_hit = det["track_id"]

        for tid in list(self._mid_zone_entry_time.keys()):
            if tid not in seen_mid_ids:
                del self._mid_zone_entry_time[tid]

        if near_hit is not None:
            return "near_zone_object", near_hit, "near"
        if mid_hit is not None:
            return "mid_zone_dwell_exceeded", mid_hit, "mid"
        return None, None, None
