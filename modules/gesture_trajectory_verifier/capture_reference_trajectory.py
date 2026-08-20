"""
Standalone reference-trajectory capture script (spec §4.2). Records a person performing ONE
clean wave (webcam or pre-recorded video), extracts the wrist/elbow/shoulder trajectory using
this module's own MoveNet-based extraction (same as live inference), normalizes+resamples it
per §2.2-2.3, and saves it as a new reference entry in the SHARED, GENERIC reference set (spec
§4: not per-person).

Whichever arm accumulates more total motion (path length) during the recording is treated as
"the waving arm" for this capture and is what gets saved — deliberately simple (spec §4.2: "don't
over-engineer this into a large dataset-collection tool").

No upstream human-detection/ROI-scoping is wired into this standalone tool — the WHOLE frame is
used, same documented simplification as this project's other standalone test/capture scripts
(e.g. modules/wave_facing_gate/test_wave_facing.py). Frame height stands in for "bbox height at
capture time" (spec §2.2's scale reference).

Usage:
    python -m modules.gesture_trajectory_verifier.capture_reference_trajectory <reference_id> --mode camera [--camera-index 0]
    python -m modules.gesture_trajectory_verifier.capture_reference_trajectory <reference_id> --mode video --video path.mp4
"""
import argparse
import sys
import time
from math import dist

import cv2

from modules.gesture_trajectory_verifier.config import load_config
from modules.gesture_trajectory_verifier.constants import ARM_KEYPOINTS
from modules.gesture_trajectory_verifier.normalization import normalize_trajectory
from modules.gesture_trajectory_verifier.pose_estimator import MoveNetPoseEstimator
from modules.gesture_trajectory_verifier.preprocessing import decode_keypoints, preprocess_crop
from modules.gesture_trajectory_verifier.reference_store import ReferenceTrajectoryStore, sanitize_reference_id
from modules.gesture_trajectory_verifier.resampling import resample_time_based
from modules.gesture_trajectory_verifier.similarity import flatten_trajectory
from modules.gesture_trajectory_verifier.trajectory_buffer import TrajectoryBuffer, update_trajectory_buffer


def _path_length(buffer: TrajectoryBuffer) -> float:
    points = [s.wrist for s in buffer.samples]
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1)) if len(points) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a reference wave trajectory for the shared reference set.")
    parser.add_argument("reference_id", help="Unique ID for this reference trajectory.")
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    try:
        reference_id = sanitize_reference_id(args.reference_id)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    if config.resample_length is None or config.min_samples_for_comparison is None or config.confidence_threshold is None:
        print("ERROR: resample_length, min_samples_for_comparison, and confidence_threshold must "
              "be calibrated in config/thresholds.yaml's gesture_trajectory_verifier section before capturing.", file=sys.stderr)
        return 1

    pose_estimator = MoveNetPoseEstimator(config.movenet_tfhub_handle)

    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: could not open capture source", file=sys.stderr)
        return 1

    left_buffer, right_buffer = TrajectoryBuffer(), TrajectoryBuffer()
    print("Recording... perform ONE clean wave, then press 'q' to finish and save.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pre = preprocess_crop(frame, config.movenet_input_size)
            if pre is not None:
                raw_kp = pose_estimator.estimate(pre.tensor)
                keypoints = decode_keypoints(raw_kp, pre)
                now = time.time()
                for arm, (wrist_idx, elbow_idx, shoulder_idx) in ARM_KEYPOINTS.items():
                    buffer = left_buffer if arm == "left" else right_buffer
                    update_trajectory_buffer(buffer, keypoints[wrist_idx], keypoints[elbow_idx],
                                               keypoints[shoulder_idx], now, config)

            cv2.putText(frame, f"left samples: {len(left_buffer.samples)}  right samples: {len(right_buffer.samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("capture_reference_trajectory", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    left_len, right_len = _path_length(left_buffer), _path_length(right_buffer)
    chosen_arm, chosen_buffer = ("left", left_buffer) if left_len >= right_len else ("right", right_buffer)
    print(f"Chosen arm: {chosen_arm} (path length {max(left_len, right_len):.1f}px, {len(chosen_buffer.samples)} samples)")

    if len(chosen_buffer.samples) < config.min_samples_for_comparison:
        print(f"ERROR: only {len(chosen_buffer.samples)} samples captured for the chosen arm, "
              f"need >= {config.min_samples_for_comparison}. Try again with a clearer wave.", file=sys.stderr)
        return 1

    frame_h = frame.shape[0] if frame is not None else 1.0
    normalized = normalize_trajectory(chosen_buffer.samples, bbox_height_at_capture=frame_h)
    resampled = resample_time_based(normalized, config.resample_length)
    if not resampled:
        print("ERROR: could not resample the captured trajectory (insufficient time span).", file=sys.stderr)
        return 1

    flat = flatten_trajectory(resampled)
    store = ReferenceTrajectoryStore(config.reference_dir)
    path = store.save(reference_id, flat, config.resample_length, chosen_arm)
    print(f"Saved reference trajectory '{reference_id}' (arm={chosen_arm}) to '{path}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
