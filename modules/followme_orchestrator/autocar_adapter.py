"""
Adapter over the vendored Autocar tracking+recovery backbone (modules/autocar/ — kept completely
unmodified; this file is the ONLY thing that imports from it, and only via their own public
classes). Not part of followme_orchestrator's own public contract — only pipeline.py in this
package calls start()/update() below.

Replaces modules.target_tracking + modules.target_recovery combined (their TargetLock already
conflates tracking-while-present and recovery-on-loss into one state machine — see the flow
notes below).

    gesture trigger GREEN -> start(person_name, initial_bbox, frame, ts)
            |
            v
       TRACKING    (force-locked via one IoU-matched detect+track pass at start(); horizontal_
            |        offset reported every frame while the lock holds)
            |  (locked track_id vanishes from the tracker's output — e.g. occlusion)
            v
       SEARCHING   (their TargetLock's own reclaim logic, checked every update() call: any
            |        brand-new track this frame gets compared against the enrolled profile)
            | (reclaimed via a re-id match)              | (recovery_timeout_seconds elapsed)
            v                                              v
       TRACKING (resumes same-frame, real offset)        LOST -> caller re-arms the trigger

Unlike modules.target_tracking's old design, there is no separate RECORDING phase: identity is
already proven by face_identity + the gesture trigger before start() is ever called, so tracking
begins TRACKING on the very next successful frame rather than first collecting reference crops.

Force-lock note: start() reaches directly into TargetLock.locked_track_id/_prev_track_ids to skip
their own ACQUIRING phase — no public method on their class does this, and their file is not
edited. This is a deliberate, narrow exception (documented here, not a general precedent for
reaching into their internals elsewhere): face_identity + the gesture trigger already proved WHO
and WHERE more precisely than a few rounds of face-only re-id sampling would, so re-deriving that
from scratch would be redundant and, with more than one person in frame, could momentarily lock
onto the wrong one.

recovery_timeout_seconds closes the one real gap their design has: ACQUIRING/reclaim retries
indefinitely with no timeout on its own. Mirrors modules.target_recovery's own exact convention
for its search_timeout_seconds (`is not None and elapsed >= timeout`) — None means never times
out, same as that module's current behavior.

A fresh BYTETracker + TargetLock is created per start() (cheap: plain Python state + a small
.npz load) rather than reused across episodes, so a long idle gap between follow-me episodes can
never leave stale track state lying around. The YOLO detector and OSNet embedder ARE reused
across episodes (expensive: real model weights) — this file's one persistent instance of each,
never shared with any other module's own detector/embedder instances (own-instance isolation,
docs/architecture.md rule #2).

KNOWN LIMITATIONS (beyond ByteTrack's usual motion-based-not-identity-verified caveat, documented
identically for modules.target_tracking/human_detection):
  - Requires a pre-enrolled profile (modules/autocar/models/enrolled_<name>.npz) for whichever
    person is being followed — unlike the old target_tracking, which captured its reference on
    the fly at trigger time. See the project's registration-tool plan for how these get built.
  - Requires modules/autocar/models/osnet_x1_0_msmt17.onnx to exist — not part of the vendored
    repo (their own .gitignore excludes it); see modules/autocar/models/README.md.
  - No PID or steering logic here — same architectural boundary target_tracking always had;
    SteeringController (steering_controller.py, this same package) still owns that.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np
import yaml

from . import autocar_bootstrap

autocar_bootstrap.ensure_on_path()

# These resolve against modules/autocar/ (their vendored tree), not this project's own top-level
# namespace — see autocar_bootstrap.py's docstring for why that's safe here.
from detector.yolov8_pose_torch import YOLOv8PoseTorch  # noqa: E402
from tracker.byte_tracker import BYTETracker  # noqa: E402
from identity.osnet_embedder import OSNetEmbedder  # noqa: E402
from identity.target_lock import TargetLock  # noqa: E402
from identity.target_profile import sanitize_person_name  # noqa: E402

__all__ = ["TrackingResult", "configure", "warmup", "start", "update"]

BboxXYWH = Tuple[int, int, int, int]
State = Literal["TRACKING", "SEARCHING", "LOST"]
_STATE_COLOR = {"TRACKING": (0, 200, 0), "SEARCHING": (0, 160, 255), "LOST": (0, 0, 255)}

_MODELS_DIR = "modules/autocar/models"


# --- config: reads the `autocar:` block of config/thresholds.yaml ---

@dataclass
class _AdapterConfig:
    detect_conf: float = 0.4
    detect_imgsz: int = 300
    pose_model_path: Optional[str] = None  # None -> vendored config.POSE_MODEL_PATH default

    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_thresh: float = 0.8
    low_match_thresh: float = 0.5
    track_buffer: int = 30

    reid_similarity_threshold: float = 0.75
    reid_acquire_rounds: int = 3
    reid_acquire_cooldown_sec: float = 0.5
    reid_model_path: Optional[str] = None  # None -> f"{_MODELS_DIR}/osnet_x1_0_msmt17.onnx"

    recovery_timeout_seconds: Optional[float] = None  # None = never times out (see module docstring)
    device: str = "cpu"


def _load_config(thresholds_path: str) -> _AdapterConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("autocar", {}) or {}

    kwargs = {}
    for field_name in (
        "detect_conf", "detect_imgsz", "pose_model_path", "track_high_thresh", "track_low_thresh",
        "new_track_thresh", "match_thresh", "low_match_thresh", "track_buffer",
        "reid_similarity_threshold", "reid_acquire_rounds", "reid_acquire_cooldown_sec",
        "reid_model_path", "recovery_timeout_seconds", "device",
    ):
        if field_name in section and section[field_name] is not None:
            kwargs[field_name] = section[field_name]
    return _AdapterConfig(**kwargs)


# --- bbox helpers (their code is xyxy throughout; this project's convention is xywh) ---

def _xywh_to_xyxy(bbox: BboxXYWH) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=float)


def _xyxy_to_xywh(bbox: np.ndarray) -> BboxXYWH:
    x1, y1, x2, y2 = bbox
    return (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- public result type ---

@dataclass
class TrackingResult:
    target_locked: bool
    horizontal_offset: Optional[float]                # normalized -1.0..+1.0, 0.0 = centered
    person_bbox: Optional[BboxXYWH]                    # (x, y, w, h), full-frame pixel space
    state: State
    just_reacquired: bool = False  # True only on the exact frame a mid-episode reclaim succeeds —
    # pipeline.py resets the steering PID on this, rather than inferring it from a state change.

    def draw_debug(self, frame: np.ndarray) -> None:
        """Draws the tracked bbox (colored by state) and a vertical frame-center reference line —
        mirrors modules.target_tracking's old draw_debug() layout so the overlay shape is familiar."""
        color = _STATE_COLOR.get(self.state, (255, 255, 255))
        frame_h, frame_w = frame.shape[:2]
        cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (120, 120, 120), 1)
        if self.person_bbox is not None:
            x, y, w, h = self.person_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        offset_str = f"{self.horizontal_offset:+.2f}" if self.horizontal_offset is not None else "None"
        cv2.putText(frame, f"tracking: state={self.state} offset={offset_str}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --- engine ---

class _AutocarEngine:
    def __init__(self, config: _AdapterConfig):
        self.config = config
        self._detector: Optional[YOLOv8PoseTorch] = None
        self._embedder: Optional[OSNetEmbedder] = None
        self._tracker: Optional[BYTETracker] = None
        self._target_lock: Optional[TargetLock] = None
        self._episode_active = False
        self._lost_since: Optional[float] = None

    def _get_detector(self) -> YOLOv8PoseTorch:
        if self._detector is None:
            self._detector = YOLOv8PoseTorch(
                model_path=self.config.pose_model_path, conf=self.config.detect_conf,
                imgsz=self.config.detect_imgsz, device=self.config.device,
            )
        return self._detector

    def _get_embedder(self) -> OSNetEmbedder:
        if self._embedder is None:
            reid_model_path = self.config.reid_model_path or f"{_MODELS_DIR}/osnet_x1_0_msmt17.onnx"
            try:
                self._embedder = OSNetEmbedder(reid_model_path, device=self.config.device)
            except Exception as e:
                raise RuntimeError(
                    f"followme_orchestrator.autocar_adapter: could not load the OSNet re-id "
                    f"weights at '{reid_model_path}'. Not part of the vendored repo (their own "
                    f".gitignore excludes it) — see modules/autocar/models/README.md."
                ) from e
        return self._embedder

    def warmup(self) -> None:
        """Eagerly constructs the detector + embedder and runs one throwaway inference through
        each (confirmed with the user — model loading AND a backend's first-inference overhead
        should both be absorbed at startup, not at the moment a gesture trigger actually fires,
        which is exactly when a stutter is most noticeable). Idempotent — _get_detector()/
        _get_embedder() only construct once; safe to call again."""
        detector = self._get_detector()
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        embedder = self._get_embedder()
        embedder.extract(np.zeros((64, 64, 3), dtype=np.uint8))

    def start(self, person_name: str, initial_bbox_xywh: BboxXYWH, frame: np.ndarray, timestamp: float) -> None:
        profile_path = f"{_MODELS_DIR}/enrolled_{sanitize_person_name(person_name)}.npz"
        try:
            self._target_lock = TargetLock(
                profile_path, embedder=self._get_embedder(),
                similarity_threshold=self.config.reid_similarity_threshold,
                acquire_rounds=self.config.reid_acquire_rounds,
                acquire_cooldown_sec=self.config.reid_acquire_cooldown_sec,
                device=self.config.device,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"followme_orchestrator.autocar_adapter: no enrolled profile at '{profile_path}' "
                f"for person '{person_name}'. Run the registration tool for this person first."
            ) from e

        self._tracker = BYTETracker(
            high_thresh=self.config.track_high_thresh, low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh, match_thresh=self.config.match_thresh,
            low_match_thresh=self.config.low_match_thresh, track_buffer=self.config.track_buffer,
        )

        detections = self._get_detector().detect(frame)
        tracks = self._tracker.update(detections)

        target_xyxy = _xywh_to_xyxy(initial_bbox_xywh)
        best_track, best_iou = None, 0.0
        for t in tracks:
            iou = _iou(t.bbox, target_xyxy)
            if iou > best_iou:
                best_track, best_iou = t, iou

        if best_track is not None:
            self._target_lock.locked_track_id = best_track.track_id
            self._target_lock._prev_track_ids = {t.track_id for t in tracks}
        # else: the trigger frame's detector missed them (rare) — target_lock stays in its own
        # ACQUIRING and will pick them up via face-only sampling within a few frames instead.

        self._episode_active = True
        self._lost_since = None

    def update(self, frame: np.ndarray, timestamp: float) -> TrackingResult:
        if not self._episode_active or self._tracker is None or self._target_lock is None:
            return TrackingResult(False, None, None, "LOST")

        detections = self._get_detector().detect(frame)
        tracks = self._tracker.update(detections)

        was_locked = self._target_lock.locked_track_id is not None
        target_id = self._target_lock.update(tracks, frame)

        if target_id is not None:
            track = next(t for t in tracks if t.track_id == target_id)
            frame_w = frame.shape[1]
            cx = (track.bbox[0] + track.bbox[2]) / 2.0
            offset = float(np.clip((cx - frame_w / 2.0) / (frame_w / 2.0), -1.0, 1.0))
            self._lost_since = None
            return TrackingResult(
                target_locked=True, horizontal_offset=offset, person_bbox=_xyxy_to_xywh(track.bbox),
                state="TRACKING", just_reacquired=not was_locked,
            )

        if self._lost_since is None:
            self._lost_since = timestamp
        elapsed = timestamp - self._lost_since
        timeout = self.config.recovery_timeout_seconds
        if timeout is not None and elapsed >= timeout:
            self._episode_active = False
            return TrackingResult(False, None, None, "LOST")

        return TrackingResult(False, None, None, "SEARCHING")


# --- module-level singleton, mirroring every other module's interface.py pattern ---

_engine_singleton: Optional[_AutocarEngine] = None


def configure(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    global _engine_singleton
    _engine_singleton = _AutocarEngine(_load_config(thresholds_config_path))


def _get_engine() -> _AutocarEngine:
    global _engine_singleton
    if _engine_singleton is None:
        configure()
    return _engine_singleton


def warmup(thresholds_config_path: str = "config/thresholds.yaml") -> None:
    """Call once, before the frame loop starts (followme_orchestrator's own pipeline __init__
    does this automatically) — eagerly loads the YOLO-pose detector and OSNet embedder so the
    FIRST real gesture trigger never pays their model-load cost live. Without this, both are
    lazily constructed inside start(), which only runs at the exact moment someone waves — the
    worst possible time for a multi-second stutter."""
    global _engine_singleton
    if _engine_singleton is None:
        configure(thresholds_config_path)
    _engine_singleton.warmup()


def start(person_name: str, initial_person_bbox: BboxXYWH, frame: np.ndarray, timestamp: float) -> None:
    _get_engine().start(person_name, initial_person_bbox, frame, timestamp)


def update(frame: np.ndarray, timestamp: float) -> TrackingResult:
    return _get_engine().update(frame, timestamp)
