"""
Standalone test entry point for the face_identity module. Imports ONLY from
modules.face_identity.interface — no other project module is imported or required, so this can
run in isolation to validate the module on its own (mirrors modules/emergency_stop/test_estop.py).

Usage:
    python -m modules.face_identity.test_face_identity <path-to-video-file>
"""
import argparse
import sys

import cv2

from modules.face_identity.interface import FaceRegistry, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the face_identity pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--registry-dir", default="modules/face_identity/registry_data")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    registry = FaceRegistry(args.registry_dir)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = evaluate(frame, registry)
            print(f"frame={frame_idx:06d} num_faces={len(results):2d}")
            for r in results:
                print(
                    f"  face_bbox={r.face_bbox} face_detection_confidence={r.face_detection_confidence} "
                    f"is_registered_match={r.is_registered_match} matched_person_name={r.matched_person_name} "
                    f"match_confidence={r.match_confidence}"
                )

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
