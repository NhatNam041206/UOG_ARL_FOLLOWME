"""
Standalone test entry point for the target_tracking module. Imports ONLY from
modules.target_tracking.interface (mirrors modules/face_identity/test_face_identity.py) — no
gesture-trigger pipeline is required; the initial bbox is drawn on the FIRST frame automatically
(a small centered box) purely so this module's own mechanics (locking, RECORDING, TRACKING,
periodic re-verify, LOST) can be exercised without any manual interaction. For a realistic,
interactive initial-bbox handoff, see visualize_target_tracking.py instead.

Usage:
    python -m modules.target_tracking.test_target_tracking <video_path>
"""
import argparse
import sys
import time

import cv2

from modules.target_tracking.interface import start, update


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the target_tracking pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    frame_idx = 0
    started = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            if not started:
                h, w = frame.shape[:2]
                # A small centered placeholder bbox stands in for a real gesture-trigger handoff
                # — this script exercises the module's own state machine in isolation, not a
                # realistic initial-bbox source.
                initial_bbox = (w // 4, h // 4, w // 2, h // 2)
                start(initial_bbox, frame, timestamp)
                started = True

            result = update(frame, timestamp)
            print(
                f"frame={frame_idx:06d} state={result.state} target_locked={result.target_locked} "
                f"horizontal_offset={result.horizontal_offset} person_bbox={result.person_bbox} "
                f"reference_set_size={len(result.reference_set.embeddings) if result.reference_set else None}"
            )

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
