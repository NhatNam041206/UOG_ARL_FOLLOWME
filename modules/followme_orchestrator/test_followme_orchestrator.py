"""
Standalone test entry point for the followme_orchestrator module. Imports ONLY from
modules.followme_orchestrator.interface (mirrors modules/face_identity/test_face_identity.py) —
runs the full composed pipeline against a video, printing FollowMeCommand per frame.

Usage:
    python -m modules.followme_orchestrator.test_followme_orchestrator <video_path> --gesture-method hand_keypoint
"""
import argparse
import sys
import time

import cv2

from modules.followme_orchestrator.interface import configure, step


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the followme_orchestrator pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--gesture-method", choices=["condition", "hand_keypoint", "trajectory_verifier"], required=True)
    parser.add_argument("--face-registry-dir", default="modules/face_identity/registry_data")
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    configure(args.gesture_method, thresholds_config_path=args.config, face_registry_dir=args.face_registry_dir)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            command = step(frame, timestamp)
            print(
                f"frame={frame_idx:06d} debug_state={command.debug_state:28s} "
                f"should_move={command.should_move} steering_angle_degrees={command.steering_angle_degrees}"
            )

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
