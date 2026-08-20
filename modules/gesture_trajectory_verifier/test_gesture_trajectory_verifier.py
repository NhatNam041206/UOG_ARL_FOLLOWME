"""
Standalone test entry point for the gesture_trajectory_verifier module. Imports ONLY from
modules.gesture_trajectory_verifier.interface — no other project module is imported or required.
Treats the whole frame as a single track_id=1 person crop, same documented simplification as
modules/wave_facing_gate/test_wave_facing.py — point the camera/video at one person.

Usage:
    python -m modules.gesture_trajectory_verifier.test_gesture_trajectory_verifier <path-to-video-file>
"""
import argparse
import sys
import time

import cv2

from modules.gesture_trajectory_verifier.interface import evaluate, release_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gesture_trajectory_verifier pipeline standalone against a recorded video.")
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
                f"confidence_debug={result.confidence_debug} matched_reference_id={result.matched_reference_id} "
                f"arm={result.arm} reference_count={result.reference_count}"
            )

            frame_idx += 1
    finally:
        cap.release()
        release_track(1)

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
