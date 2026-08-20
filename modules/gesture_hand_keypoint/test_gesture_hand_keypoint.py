"""
Standalone test entry point for the gesture_hand_keypoint module. Imports ONLY from
modules.gesture_hand_keypoint.interface — no other project module is imported or required, so
this can run in isolation (mirrors modules/emergency_stop/test_estop.py). Treats the whole frame
as a single track_id=1 person crop, same documented simplification as
modules/wave_facing_gate/test_wave_facing.py — point the camera/video at one person.

Usage:
    python -m modules.gesture_hand_keypoint.test_gesture_hand_keypoint <path-to-video-file>
"""
import sys
import time
import argparse

import cv2

from modules.gesture_hand_keypoint.interface import evaluate, release_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gesture_hand_keypoint pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    args = parser.parse_args()

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

            result = evaluate(track_id=1, person_crop_bgr=frame, timestamp=time.time())
            print(
                f"frame={frame_idx:06d} is_waving={result.is_waving} waving_state={result.waving_state} "
                f"sequence_stage={result.sequence_stage} confidence_debug={result.confidence_debug} "
                f"palm_facing_camera_debug={result.palm_facing_camera_debug}"
            )

            frame_idx += 1
    finally:
        cap.release()
        release_track(1)

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
