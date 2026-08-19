"""
Quick Demo Spec — Wave + Facing Trigger Gate
(document/implementation/followme/Project_Master_Doc.md)

Proof-of-concept only, NOT the final implementation. Reuses the existing Follow-Me pipeline
(src/pipeline.py + src/detector.py + src/verifier.py, untouched — this script only READS their
output) to get the bbox of a track already verified as the registered person, crops that bbox
each frame, feeds the crop to MoveNet Lightning (src/pose_estimator.py, new for this demo), and
runs a rule-based wave detector + facing-camera proxy (src/wave_detector.py, new for this demo)
on the resulting keypoints.

    trigger = registered_person AND is_waving AND is_facing_camera

Does NOT include: approach/steering logic, emergency-stop, occlusion handling beyond the
short-term fault tolerance already in src/wave_detector.py, or any change to identity
matching (reused as-is from the existing pipeline).

--any-person testing mode: bypasses identity verification entirely (no registry, no OSNet by
default, no FollowPipeline) so the wave/facing/pose logic can be smoke-tested with whoever is in
frame, without registering anyone first. Uses src/detector.py's YoloDetector directly
(unmodified) plus src/any_person_tracker.py's sticky lock (see that module's docstring for the
4 selectable re-acquisition strategies). NOT a substitute for the real demo — identity is not
verified at all in this mode.

Every run writes a per-frame CSV log (path: wave_trigger_demo.log_csv_path) with trigger state
and per-module timing (detect/verify/lock/pose/gesture/overlay/total ms) for both modes, and the
overlay/console log show the same timings live — intended for profiling on constrained hardware
(Raspberry Pi 5).
"""
import os
import csv
import sys
import time
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

from src.person_selector import PersonRegistrySelector
from src.pipeline import FollowPipeline
from src.detector import YoloDetector
from src.pose_estimator import MoveNetPoseEstimator, KEYPOINT_INDEX, movenet_point_to_crop_px
from src.wave_detector import WaveFacingGate, GestureResult
from src.any_person_tracker import AnyPersonTracker, VALID_METHODS
from main import WebcamStreamThread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# COCO skeleton edges (keypoint name pairs) for the optional debug overlay.
_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
]

BBox = Tuple[int, int, int, int]

_LOG_HEADERS = [
    "timestamp", "frame_idx", "mode", "reacquisition_method", "lock_event",
    "registered_person", "track_id", "is_waving", "is_facing_camera", "trigger",
    "direction_changes", "amplitude_norm", "shoulder_torso_ratio",
    "detect_ms", "verify_ms", "lock_ms", "pose_ms", "gesture_ms", "overlay_ms", "total_ms", "fps",
]


def _init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_LOG_HEADERS)


def _log_csv_row(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(h, "") for h in _LOG_HEADERS])


def _clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)


def _frame_state_from_pipeline(pipeline: FollowPipeline, frame) -> Tuple[bool, Optional[int], Optional[BBox]]:
    """Real demo path: registered_person comes from FollowPipeline's identity verification."""
    angle_result = pipeline.process_frame(frame)
    if not angle_result.target_found:
        return False, None, None

    target_det = next(
        (d for d in pipeline.last_detections if d["track_id"] == angle_result.track_id), None
    )
    if target_det is None:
        return True, angle_result.track_id, None

    fh, fw = frame.shape[:2]
    bbox = _clamp_bbox(target_det["bbox"], fw, fh)
    return True, angle_result.track_id, bbox


def _draw_overlay(frame, bbox, keypoints, gesture: Optional[GestureResult], registered_person,
                   keypoint_display_threshold, current_fps, any_person_mode,
                   reacquisition_method, lock_event, timings: Dict[str, float],
                   gesture_gate: WaveFacingGate):
    display = frame.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop_w, crop_h = x2 - x1, y2 - y1

        if keypoints is not None and crop_w > 0 and crop_h > 0:
            pts_px = {}
            for name, idx in KEYPOINT_INDEX.items():
                ky, kx, kconf = keypoints[idx]
                if kconf > keypoint_display_threshold:
                    px, py = movenet_point_to_crop_px(ky, kx, crop_w, crop_h)
                    pts_px[name] = (x1 + int(px), y1 + int(py))

            for a, b in _SKELETON_EDGES:
                if a in pts_px and b in pts_px:
                    cv2.line(display, pts_px[a], pts_px[b], (255, 255, 0), 2)
            for pt in pts_px.values():
                cv2.circle(display, pt, 3, (0, 165, 255), -1)

    is_waving = gesture.is_waving if gesture else False
    is_facing = gesture.is_facing_camera if gesture else False
    trigger = registered_person and is_waving and is_facing

    y_offset = 30
    if any_person_mode:
        cv2.putText(display, f"ANY-PERSON MODE (bypassed) | method={reacquisition_method} | lock={lock_event}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        y_offset += 26

    lines = [
        (f"registered_person: {registered_person}", (0, 255, 0) if registered_person else (0, 0, 255)),
        (f"is_waving: {is_waving}", (0, 255, 0) if is_waving else (0, 0, 255)),
        (f"is_facing_camera: {is_facing}", (0, 255, 0) if is_facing else (0, 0, 255)),
        (f"TRIGGER: {trigger}", (0, 255, 0) if trigger else (0, 0, 255)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (20, y_offset + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    y_offset += len(lines) * 26

    if gesture is not None:
        debug_text = (f"dir_changes:{gesture.direction_changes}  amplitude:{gesture.amplitude_norm:.3f}  "
                       f"shoulder/torso:{gesture.shoulder_torso_ratio:.3f}")
        cv2.putText(display, debug_text, (20, y_offset + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 22

    # Live threshold readout — the actual active values from config, so they're visible without
    # checking the yaml file while tuning.
    g = gesture_gate
    thresholds_text_1 = (
        f"thr  conf(wave/facing):{g.threshold_keypoint_conf_wave:.2f}/{g.threshold_keypoint_conf_facing:.2f}  "
        f"facing_ratio>={g.facing_shoulder_ratio_min:.2f}  "
        f"waving(changes>={g.wave_direction_changes_min},amp>={g.wave_amplitude_norm_min:.2f})"
    )
    thresholds_text_2 = (
        f"thr  wave_zone[{g.wave_min_horizontal_extent_percent:.2f}-{g.wave_max_horizontal_extent_percent:.2f}]  "
        f"margin:{g.wave_horizontal_margin_percent:.2f}  "
        f"not_raised_reset:{g.wave_not_raised_reset_frames}  bad_frames:{g.max_consecutive_bad_frames}"
    )
    cv2.putText(display, thresholds_text_1, (20, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    y_offset += 18
    cv2.putText(display, thresholds_text_2, (20, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    y_offset += 22

    timing_text = f"ms  detect:{timings['detect_ms']:.1f}"
    if timings["verify_ms"] > 0:
        timing_text += f"  verify:{timings['verify_ms']:.1f}"
    if any_person_mode:
        timing_text += f"  lock:{timings['lock_ms']:.1f}"
    timing_text += f"  pose:{timings['pose_ms']:.1f}  gesture:{timings['gesture_ms']:.1f}  total:{timings['total_ms']:.1f}"
    cv2.putText(display, timing_text, (20, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    cv2.putText(display, f"FPS:{current_fps:.1f}", (20, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return display


def main():
    parser = argparse.ArgumentParser(description="Quick Demo: Wave + Facing Trigger Gate")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path to YAML settings file (reads the wave_trigger_demo section).")
    parser.add_argument("--no-ui", action="store_true",
                        help="Disable the display window; log trigger state to console instead.")
    parser.add_argument("--any-person", action="store_true",
                        help="Testing mode: skip identity verification entirely (no person "
                             "registration/selection needed). Uses raw YOLO person detection "
                             "with a sticky lock (src/any_person_tracker.py) so the tracked "
                             "subject doesn't jump between people in frame. NOT the real demo "
                             "— see module docstring.")
    parser.add_argument("--reacquisition-method", type=str, choices=VALID_METHODS, default=None,
                        help="Override wave_trigger_demo.any_person_tracking.reacquisition_method "
                             "from config, for quick A/B testing. --any-person mode only.")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file at '{args.config}': {e}")
        sys.exit(1)

    demo_cfg = config.get("wave_trigger_demo", {})
    if not demo_cfg:
        logger.error(f"'wave_trigger_demo' section missing from '{args.config}'.")
        sys.exit(1)

    pipeline: Optional[FollowPipeline] = None
    detector: Optional[YoloDetector] = None
    any_person_tracker: Optional[AnyPersonTracker] = None
    reacquisition_method = ""

    if args.any_person:
        any_person_cfg = demo_cfg.get("any_person_tracking", {})
        reacquisition_method = args.reacquisition_method or any_person_cfg.get("reacquisition_method", "largest_bbox")
        logger.warning(
            f"Running in --any-person mode: identity verification is BYPASSED. "
            f"registered_person is forced True for whoever the sticky lock is following "
            f"(reacquisition_method='{reacquisition_method}')."
        )
        yolo_path = config.get("yolo_model_path", "yolo11n.onnx")
        detection_imgsz = config.get("detection_imgsz")
        try:
            detector = YoloDetector(yolo_path, imgsz=detection_imgsz)
        except Exception as e:
            logger.error(f"Failed to initialize YoloDetector: {e}")
            sys.exit(1)

        osnet_verifier = None
        if reacquisition_method == "osnet":
            # Lazy: only pay OSNet's load cost when this method is actually selected, keeping
            # the other 3 methods free of it entirely (part of the point of --any-person mode).
            from src.verifier import OSNetVerifier
            try:
                osnet_verifier = OSNetVerifier(config.get("osnet_variant", "osnet_x1_0"))
            except Exception as e:
                logger.error(f"Failed to initialize OSNetVerifier for reacquisition_method='osnet': {e}")
                sys.exit(1)

        try:
            any_person_tracker = AnyPersonTracker(
                method=reacquisition_method,
                lock_grace_frames=int(any_person_cfg.get("lock_grace_frames", 5)),
                histogram_similarity_min=float(any_person_cfg.get("histogram_similarity_min", 0.5)),
                osnet_similarity_min=float(any_person_cfg.get("osnet_similarity_min", 0.5)),
                osnet_verifier=osnet_verifier,
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        # Reuse the existing person-selection + pipeline init flow exactly as main.py --mode run does.
        selector = PersonRegistrySelector(config, config_path=args.config)
        selected_path = selector.run()
        if not selected_path:
            logger.error("No person selected from the registry. Exiting.")
            sys.exit(1)

        try:
            pipeline = FollowPipeline(config_path=args.config, reference_npz_path=selected_path)
        except Exception as e:
            logger.error(f"Failed to initialize FollowPipeline: {e}")
            sys.exit(1)

    try:
        pose_estimator = MoveNetPoseEstimator(demo_cfg["pose_model_url"])
    except Exception as e:
        logger.error(f"Failed to load MoveNet: {e}")
        if pipeline:
            pipeline.close()
        sys.exit(1)

    gesture_gate = WaveFacingGate(
        threshold_keypoint_conf_wave=float(demo_cfg["threshold_keypoint_conf_wave"]),
        threshold_keypoint_conf_facing=float(demo_cfg["threshold_keypoint_conf_facing"]),
        wave_buffer_size=int(demo_cfg["wave_buffer_size"]),
        wave_direction_changes_min=int(demo_cfg["wave_direction_changes_min"]),
        wave_amplitude_norm_min=float(demo_cfg["wave_amplitude_norm_min"]),
        max_consecutive_bad_frames=int(demo_cfg["max_consecutive_bad_frames"]),
        wave_horizontal_margin_percent=float(demo_cfg["wave_horizontal_margin_percent"]),
        wave_not_raised_reset_frames=int(demo_cfg["wave_not_raised_reset_frames"]),
        wave_min_horizontal_extent_percent=float(demo_cfg["wave_min_horizontal_extent_percent"]),
        wave_max_horizontal_extent_percent=float(demo_cfg["wave_max_horizontal_extent_percent"]),
        facing_shoulder_ratio_min=float(demo_cfg["facing_shoulder_ratio_min"]),
    )
    # Debug-overlay skeleton dots: show a keypoint if it clears EITHER gate's bar (purely
    # cosmetic — don't hide a point that matters to at least one of the two checks).
    keypoint_display_threshold = min(
        gesture_gate.threshold_keypoint_conf_wave, gesture_gate.threshold_keypoint_conf_facing
    )

    log_csv_path = demo_cfg.get("log_csv_path", "logs/wave_trigger_demo_log.csv")
    _init_csv(log_csv_path)
    logger.info(f"Logging per-frame trigger state + timing to '{log_csv_path}'")

    camera_index = config.get("camera_index", 0)
    input_res = config.get("input_resolution", [640, 480])
    flip_horiz = config.get("flip_horizontal", False)

    try:
        cam_stream = WebcamStreamThread(camera_index, input_res, flip_horizontal=flip_horiz).start()
        logger.info(f"Started camera reader thread on camera index {camera_index} (flip_horizontal={flip_horiz})")
    except Exception as e:
        logger.error(f"Failed to start camera stream thread: {e}")
        if pipeline:
            pipeline.close()
        sys.exit(1)

    window_name = "Quick Demo: Wave + Facing Trigger Gate"
    if not args.no_ui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    frame_idx = 0
    fps_start_time = time.time()
    current_fps = 0.0
    last_log_time = time.time()
    # Tracks the STICKY identity (not the per-frame track_id, which is None during grace-period
    # frames) so gesture_gate state is only evicted when the locked target genuinely changes —
    # not every time a grace-period frame happens to report no bbox.
    prev_identity: Optional[int] = None

    try:
        while True:
            t_frame_start = time.time()
            ret, frame = cam_stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            frame_idx += 1

            detect_ms = verify_ms = lock_ms = 0.0
            lock_event = ""

            if args.any_person:
                t0 = time.time()
                detections = detector.track(frame)
                detect_ms = (time.time() - t0) * 1000.0
                registered_person, track_id, bbox = any_person_tracker.update(frame, detections)
                lock_ms = any_person_tracker.last_lock_ms
                lock_event = any_person_tracker.last_event
                current_identity = any_person_tracker.locked_track_id
            else:
                registered_person, track_id, bbox = _frame_state_from_pipeline(pipeline, frame)
                detect_ms = pipeline.last_timing_ms["detect_ms"]
                verify_ms = pipeline.last_timing_ms["verify_ms"]
                current_identity = pipeline.active_target_id

            if current_identity != prev_identity:
                if prev_identity is not None:
                    gesture_gate.reset(track_id=prev_identity)
                prev_identity = current_identity

            keypoints = None
            gesture = None
            crop_w = crop_h = 0

            t0 = time.time()
            if registered_person and track_id is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                crop_w, crop_h = x2 - x1, y2 - y1
                crop = frame[y1:y2, x1:x2]
                keypoints = pose_estimator.estimate(crop)
            pose_ms = (time.time() - t0) * 1000.0

            t0 = time.time()
            if registered_person and track_id is not None:
                gesture = gesture_gate.update(track_id, keypoints, crop_w, crop_h)
            gesture_ms = (time.time() - t0) * 1000.0

            trigger = registered_person and bool(gesture and gesture.is_waving) and bool(gesture and gesture.is_facing_camera)

            overlay_ms = 0.0
            timings = {
                "detect_ms": detect_ms, "verify_ms": verify_ms, "lock_ms": lock_ms,
                "pose_ms": pose_ms, "gesture_ms": gesture_ms, "total_ms": 0.0,
            }
            if not args.no_ui:
                t0 = time.time()
                timings["total_ms"] = (time.time() - t_frame_start) * 1000.0
                display = _draw_overlay(
                    frame, bbox, keypoints, gesture, registered_person,
                    keypoint_display_threshold, current_fps, args.any_person,
                    reacquisition_method, lock_event, timings, gesture_gate,
                )
                overlay_ms = (time.time() - t0) * 1000.0
                cv2.imshow(window_name, display)

            total_ms = (time.time() - t_frame_start) * 1000.0

            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            _log_csv_row(log_csv_path, {
                "timestamp": time.time(), "frame_idx": frame_idx,
                "mode": "any_person" if args.any_person else "real",
                "reacquisition_method": reacquisition_method, "lock_event": lock_event,
                "registered_person": registered_person, "track_id": track_id if track_id is not None else "",
                "is_waving": gesture.is_waving if gesture else False,
                "is_facing_camera": gesture.is_facing_camera if gesture else False,
                "trigger": trigger,
                "direction_changes": gesture.direction_changes if gesture else 0,
                "amplitude_norm": round(gesture.amplitude_norm, 4) if gesture else 0.0,
                "shoulder_torso_ratio": round(gesture.shoulder_torso_ratio, 4) if gesture else 0.0,
                "detect_ms": round(detect_ms, 2), "verify_ms": round(verify_ms, 2), "lock_ms": round(lock_ms, 2),
                "pose_ms": round(pose_ms, 2), "gesture_ms": round(gesture_ms, 2),
                "overlay_ms": round(overlay_ms, 2), "total_ms": round(total_ms, 2),
                "fps": round(current_fps, 2),
            })

            if time.time() - last_log_time >= 1.0:
                is_waving = gesture.is_waving if gesture else False
                is_facing = gesture.is_facing_camera if gesture else False
                logger.info(
                    f"FPS={current_fps:.1f} registered_person={registered_person} "
                    f"is_waving={is_waving} is_facing_camera={is_facing} TRIGGER={trigger} | "
                    f"Timing(ms) detect={detect_ms:.1f} verify={verify_ms:.1f} lock={lock_ms:.1f} "
                    f"pose={pose_ms:.1f} gesture={gesture_ms:.1f} overlay={overlay_ms:.1f} total={total_ms:.1f}"
                )
                last_log_time = time.time()

            if not args.no_ui:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exiting demo on user command ('q' pressed).")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Shutting down...")

    finally:
        cam_stream.stop()
        if pipeline:
            pipeline.close()
        if not args.no_ui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
