"""
Rule-based wave detection + facing-camera proxy for demo_wave_trigger.py — Quick Demo Spec:
Wave + Facing Trigger Gate (document/implementation/followme/Project_Master_Doc.md muc 4-5).

All thresholds are constructor parameters sourced from config/settings.yaml's
`wave_trigger_demo` section (PLACEHOLDER values pending post-demo calibration) — never
hard-coded here, per spec muc 7.

The posture gate (muc 4.1) deliberately deviates from the spec's literal "wrist above shoulder,
gated by shoulder confidence" — replaced with body-relative geometry (wrist above the crop's own
vertical midline, wrist within a tunable horizontal band around body center) so reliability no
longer depends on the shoulder keypoint's confidence at all, and a full sideways arm extension
(not a wave) is excluded from "raised" before it ever reaches the oscillation buffer.

The facing-camera proxy (muc 5) also deviates from the spec's literal 4-keypoint-confidence-only
check — confidence alone can't tell "visible" from "oriented toward the camera" (turning the
torso while keeping the head toward the camera barely changes eye/shoulder confidence). Added a
torso-orientation ratio (shoulder width vs. shoulder-to-hip height, both measured in actual crop
pixels via movenet_point_to_crop_px — NOT MoveNet's raw normalized coordinates, since x/y axes
can have different letterbox padding and aren't directly comparable otherwise) that shrinks when
the body rotates away from the camera, on top of the original confidence floor. Requires hip
keypoints too, not read at all before this.
"""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.pose_estimator import KEYPOINT_INDEX, movenet_point_to_crop_px

logger = logging.getLogger(__name__)

_WRIST_SIDES = ("left_wrist", "right_wrist")
# "Half of body" per the posture-gate redesign — a named constant rather than a bare 0.5 in the
# comparison below, per the project's own "no scattered magic numbers" convention (spec muc 7).
_BODY_MIDLINE_FRACTION = 0.5


@dataclass
class GestureResult:
    is_waving: bool
    is_facing_camera: bool
    direction_changes: int
    amplitude_norm: float
    shoulder_torso_ratio: float


class _TrackState:
    def __init__(self, buffer_size: int):
        self.wrist_x_buffer: deque = deque(maxlen=buffer_size)
        self.bad_frame_streak: int = 0
        self.not_raised_streak: int = 0


class WaveFacingGate:
    """
    Per-track_id wave gesture + facing-camera proxy. Stateful across frames (temporal buffer for
    wave detection, muc 4.2) — call `update()` once per frame for the single verified target
    track_id; call `reset(track_id)` (or just stop calling update for that id) when the target
    is lost so a re-acquired target starts with a clean buffer.
    """

    def __init__(
        self,
        threshold_keypoint_conf_wave: float,
        threshold_keypoint_conf_facing: float,
        wave_buffer_size: int,
        wave_direction_changes_min: int,
        wave_amplitude_norm_min: float,
        max_consecutive_bad_frames: int,
        wave_horizontal_margin_percent: float,
        wave_not_raised_reset_frames: int,
        wave_min_horizontal_extent_percent: float,
        wave_max_horizontal_extent_percent: float,
        facing_shoulder_ratio_min: float,
    ):
        self.threshold_keypoint_conf_wave = threshold_keypoint_conf_wave
        self.threshold_keypoint_conf_facing = threshold_keypoint_conf_facing
        self.wave_buffer_size = wave_buffer_size
        self.wave_direction_changes_min = wave_direction_changes_min
        self.wave_amplitude_norm_min = wave_amplitude_norm_min
        self.max_consecutive_bad_frames = max_consecutive_bad_frames
        self.wave_horizontal_margin_percent = wave_horizontal_margin_percent
        self.wave_not_raised_reset_frames = wave_not_raised_reset_frames
        self.wave_min_horizontal_extent_percent = wave_min_horizontal_extent_percent
        self.wave_max_horizontal_extent_percent = wave_max_horizontal_extent_percent
        self.facing_shoulder_ratio_min = facing_shoulder_ratio_min
        self._states: Dict[int, _TrackState] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Drop buffered state for one track_id, or all of them if track_id is None."""
        if track_id is None:
            self._states.clear()
        else:
            self._states.pop(track_id, None)

    def update(self, track_id: int, keypoints: Optional[np.ndarray],
               crop_w: int = 0, crop_h: int = 0) -> GestureResult:
        if track_id not in self._states:
            self._states[track_id] = _TrackState(self.wave_buffer_size)
        state = self._states[track_id]

        if keypoints is None or crop_w <= 0 or crop_h <= 0:
            # No usable pose this frame (e.g. degenerate crop) — same fault-tolerance treatment
            # as a low-confidence frame (muc 4.3): skip, don't reset, unless the bad streak grows
            # past the limit.
            self._register_bad_frame(state)
            return self._result(state, is_facing_camera=False, shoulder_torso_ratio=0.0)

        # 4.1 posture condition, redesigned per team discussion — geometry-based instead of
        # shoulder-confidence-gated. Per wrist: wrist_conf is a coarse "MoveNet found something
        # here at all" floor (not compared to shoulder anymore), then the position itself is
        # judged plausible-and-raised by three body-relative checks:
        #   - vertical: above the crop's own vertical midline (not specifically the shoulder)
        #   - outer horizontal bound: within the crop's own width plus a tunable margin (sanity
        #     check against a wildly wrong detection, not an absolute bbox edge)
        #   - horizontal WAVE-ZONE BAND: distance from the crop's horizontal center must fall
        #     between a tunable min and max. The max excludes a full sideways arm extension (not
        #     a wave). The min excludes the opposite failure mode: touching/scratching the head
        #     or face keeps the wrist very close to body center, which used to pass here with
        #     only a max bound and no floor — confirmed false-positive, fixed by requiring the
        #     wrist to be laterally offset from center by at least this much.
        raised_side_x = None
        any_side_reliable = False
        margin_px = self.wave_horizontal_margin_percent * crop_w
        min_extent_px = self.wave_min_horizontal_extent_percent * crop_w
        max_extent_px = self.wave_max_horizontal_extent_percent * crop_w
        center_px = crop_w / 2.0
        for wrist_name in _WRIST_SIDES:
            wrist_y, wrist_x, wrist_conf = keypoints[KEYPOINT_INDEX[wrist_name]]
            if wrist_conf <= self.threshold_keypoint_conf_wave:
                continue
            any_side_reliable = True

            px, py = movenet_point_to_crop_px(wrist_y, wrist_x, crop_w, crop_h)
            within_vertical = py < (crop_h * _BODY_MIDLINE_FRACTION)
            within_outer_bound = (-margin_px) <= px <= (crop_w + margin_px)
            distance_from_center = abs(px - center_px)
            within_wave_zone = min_extent_px <= distance_from_center <= max_extent_px
            if within_vertical and within_outer_bound and within_wave_zone and raised_side_x is None:
                raised_side_x = float(wrist_x)

        if not any_side_reliable:
            # Neither wrist cleared the confidence floor this frame (occlusion/motion blur) —
            # skip the frame, don't push, don't reset, per muc 4.3.
            self._register_bad_frame(state)
        else:
            state.bad_frame_streak = 0
            if raised_side_x is not None:
                state.wrist_x_buffer.append(raised_side_x)
                state.not_raised_streak = 0
            else:
                # Arm reliably observed as NOT raised. A deque(maxlen=N) only evicts old entries
                # when something new is pushed — since nothing gets pushed while the arm stays
                # down, the buffer would otherwise freeze with stale oscillation history and
                # keep reporting is_waving=True indefinitely after the person actually stopped
                # (confirmed bug: is_waving stuck True after lowering the hand). Once the arm has
                # stayed down for wave_not_raised_reset_frames in a row, clear the buffer so
                # is_waving correctly falls back to False instead of computing on ghost data.
                state.not_raised_streak += 1
                if state.not_raised_streak > self.wave_not_raised_reset_frames:
                    state.wrist_x_buffer.clear()
                    state.not_raised_streak = 0

        is_facing_camera, shoulder_torso_ratio = self._is_facing_camera(keypoints, crop_w, crop_h)
        return self._result(state, is_facing_camera=is_facing_camera, shoulder_torso_ratio=shoulder_torso_ratio)

    def _register_bad_frame(self, state: _TrackState) -> None:
        state.bad_frame_streak += 1
        if state.bad_frame_streak > self.max_consecutive_bad_frames:
            state.wrist_x_buffer.clear()
            state.bad_frame_streak = 0

    def _is_facing_camera(self, keypoints: np.ndarray, crop_w: int, crop_h: int):
        """
        Returns (is_facing_camera, shoulder_torso_ratio). Confidence floor first (are these 6
        points even visible), then a torso-orientation check: shoulder width (horizontal spread
        between the shoulders) relative to shoulder-to-hip height. Facing the camera keeps the
        shoulders spread wide relative to torso height; rotating the torso away — even with the
        head still toward the camera — shrinks the shoulder spread via foreshortening while torso
        height barely changes, so the ratio drops. Both measured in actual crop-pixel space (not
        MoveNet's raw normalized output) since the x/y axes can carry different letterbox padding
        and aren't directly comparable otherwise (see movenet_point_to_crop_px).
        """
        names = ("left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not all(keypoints[KEYPOINT_INDEX[n]][2] > self.threshold_keypoint_conf_facing for n in names):
            return False, 0.0

        ls_y, ls_x, _ = keypoints[KEYPOINT_INDEX["left_shoulder"]]
        rs_y, rs_x, _ = keypoints[KEYPOINT_INDEX["right_shoulder"]]
        lh_y, lh_x, _ = keypoints[KEYPOINT_INDEX["left_hip"]]
        rh_y, rh_x, _ = keypoints[KEYPOINT_INDEX["right_hip"]]

        ls_px, ls_py = movenet_point_to_crop_px(ls_y, ls_x, crop_w, crop_h)
        rs_px, rs_py = movenet_point_to_crop_px(rs_y, rs_x, crop_w, crop_h)
        lh_px, lh_py = movenet_point_to_crop_px(lh_y, lh_x, crop_w, crop_h)
        rh_px, rh_py = movenet_point_to_crop_px(rh_y, rh_x, crop_w, crop_h)

        shoulder_width_px = abs(ls_px - rs_px)
        torso_height_px = abs(((ls_py + rs_py) / 2.0) - ((lh_py + rh_py) / 2.0))
        if torso_height_px <= 1e-6:
            return False, 0.0

        shoulder_torso_ratio = shoulder_width_px / torso_height_px
        return shoulder_torso_ratio >= self.facing_shoulder_ratio_min, shoulder_torso_ratio

    def _result(self, state: _TrackState, is_facing_camera: bool, shoulder_torso_ratio: float) -> GestureResult:
        buf = state.wrist_x_buffer
        if len(buf) < 2:
            return GestureResult(is_waving=False, is_facing_camera=is_facing_camera,
                                  direction_changes=0, amplitude_norm=0.0,
                                  shoulder_torso_ratio=shoulder_torso_ratio)

        values = list(buf)
        diffs = [b - a for a, b in zip(values, values[1:])]
        direction_changes = 0
        last_sign = 0
        for d in diffs:
            sign = (d > 0) - (d < 0)
            if sign == 0:
                continue
            if last_sign != 0 and sign != last_sign:
                direction_changes += 1
            last_sign = sign

        amplitude_norm = max(values) - min(values)
        is_waving = (
            direction_changes >= self.wave_direction_changes_min
            and amplitude_norm >= self.wave_amplitude_norm_min
        )
        return GestureResult(
            is_waving=is_waving,
            is_facing_camera=is_facing_camera,
            direction_changes=direction_changes,
            amplitude_norm=amplitude_norm,
            shoulder_torso_ratio=shoulder_torso_ratio,
        )
