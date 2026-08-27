"""
Standalone test entry point for the human_detection_roi module. Imports ONLY from
modules.human_detection_roi.interface — no other project module (including modules.face_identity)
is imported or required, so this can run in isolation. A face bbox must be supplied manually via
--face-bbox since this module has no face detector of its own — in real use that bbox comes from
modules.face_identity, but this test validates ROI-scoping/person-detection in isolation from it.

Usage:
    python -m project_tests.human_detection_roi.test_human_detection_roi <path-to-video-file> --face-bbox X Y W H
"""
import argparse
import sys

import cv2

from modules.human_detection_roi.interface import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the human_detection_roi pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--face-bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"), required=True,
                         help="Fixed face bbox (full-frame pixel space) to scope detection around, every frame.")
    args = parser.parse_args()
    face_bbox = tuple(args.face_bbox)

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

            result = evaluate(frame, face_bbox)
            print(
                f"frame={frame_idx:06d} person_found={result.person_found} "
                f"person_bbox={result.person_bbox} detection_confidence={result.detection_confidence}"
            )

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
