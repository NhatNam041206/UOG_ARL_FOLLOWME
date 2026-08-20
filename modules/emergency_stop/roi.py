"""
Runway geometry: ROI crop rectangle + per-point zone/membership tests.

Known limitation (spec §6, intentional simplification — do not silently fix): the runway is
frame-relative only. It does NOT account for the robot's current steering angle (no
steering-angle-based skewing of the trapezoid). Accepted because turning is expected to be
slow/rare; this will under-cover the outside edge of a turn if that assumption changes.

Coordinate convention: `runway_left_line` / `runway_right_line` are each
`[[x_top_frac, y_top_frac], [x_bottom_frac, y_bottom_frac]]`, normalized 0.0-1.0 fractions of
frame width/height. `zone_far_boundary` / `zone_mid_boundary` are y-fractions in that SAME
full-frame normalized space (0 = top/far, 1 = bottom/near) — confirmed with the user so every
geometry value in thresholds.yaml lives in one coordinate system, not two.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[float, float]
Line = Tuple[Point, Point]  # (top, bottom) in pixel coordinates


@dataclass
class RunwayGeometry:
    frame_w: int
    frame_h: int
    left_line: Line
    right_line: Line
    roi_buffer_px: int
    zone_far_boundary_y: float  # pixel y
    zone_mid_boundary_y: float  # pixel y
    roi_rect: Tuple[int, int, int, int]  # x1, y1, x2, y2 — bounding crop rectangle (buffered)

    def _x_at_y(self, line: Line, y: float) -> float:
        (x_top, y_top), (x_bottom, y_bottom) = line
        if abs(y_bottom - y_top) < 1e-9:
            return x_top
        t = (y - y_top) / (y_bottom - y_top)
        return x_top + t * (x_bottom - x_top)

    def contains(self, x: float, y: float) -> bool:
        """
        Exact (unbuffered) trapezoid membership test — this is what decides whether a
        ground-contact point counts as "on the runway" for zone logic. `roi_buffer_px` only
        widens what gets cropped/fed to the detector (so bboxes near the trapezoid edge aren't
        clipped); it deliberately never widens this membership test itself, so the buffer margin
        can never make a genuinely off-runway object look like a near-zone stop trigger. This is
        also how §3.1's "near zone should use a tighter/more-trusted crop than the far zone" is
        satisfied here: every zone uses this same exact, unbuffered edge — there is no looser
        tolerance for far-zone membership than for near-zone membership.
        """
        left_x = self._x_at_y(self.left_line, y)
        right_x = self._x_at_y(self.right_line, y)
        lo, hi = min(left_x, right_x), max(left_x, right_x)
        return lo <= x <= hi

    def zone_for_y(self, y: float) -> Optional[str]:
        """Returns "far" | "mid" | "near", or None if y is above the runway's own top edge."""
        top_y = min(self.left_line[0][1], self.right_line[0][1])
        bottom_y = max(self.left_line[1][1], self.right_line[1][1])
        if y < top_y or y > bottom_y:
            return None
        if y < self.zone_far_boundary_y:
            return "far"
        if y < self.zone_mid_boundary_y:
            return "mid"
        return "near"


def build_geometry(
    frame_w: int,
    frame_h: int,
    runway_left_line: List[List[float]],
    runway_right_line: List[List[float]],
    roi_buffer_px: int,
    zone_far_boundary: float,
    zone_mid_boundary: float,
) -> RunwayGeometry:
    def to_pixels(line_frac: List[List[float]]) -> Line:
        (x1f, y1f), (x2f, y2f) = line_frac
        return (
            (x1f * frame_w, y1f * frame_h),
            (x2f * frame_w, y2f * frame_h),
        )

    left_line = to_pixels(runway_left_line)
    right_line = to_pixels(runway_right_line)

    xs = [left_line[0][0], left_line[1][0], right_line[0][0], right_line[1][0]]
    ys = [left_line[0][1], left_line[1][1], right_line[0][1], right_line[1][1]]

    x1 = max(0, int(min(xs) - roi_buffer_px))
    x2 = min(frame_w, int(max(xs) + roi_buffer_px))
    y1 = max(0, int(min(ys)))
    y2 = min(frame_h, int(max(ys)))

    return RunwayGeometry(
        frame_w=frame_w,
        frame_h=frame_h,
        left_line=left_line,
        right_line=right_line,
        roi_buffer_px=roi_buffer_px,
        zone_far_boundary_y=zone_far_boundary * frame_h,
        zone_mid_boundary_y=zone_mid_boundary * frame_h,
        roi_rect=(x1, y1, x2, y2),
    )
