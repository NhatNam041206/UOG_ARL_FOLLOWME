"""
Standalone test entry point for the human_detection module. Imports ONLY from
modules.human_detection.interface — no other project module is imported or required, so this
can run in isolation to validate the module on its own (mirrors
project_tests/emergency_stop/test_estop.py).

Usage:
    python -m project_tests.human_detection.test_human_detection <path-to-video-file> [--show]
"""
import argparse
import sys

import cv2

from modules.human_detection.interface import HumanDetectionModule

_BOX_COLOR = (0, 200, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the human_detection pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--show", action="store_true", help="Display frames with bbox overlay while processing.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    detector = HumanDetectionModule()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            print(f"frame={frame_idx:06d} num_people={len(detections):2d} latency_ms={detector.last_latency_ms:6.1f}")
            for det in detections:
                print(f"  track_id={det.track_id} bbox={det.bbox} confidence={det.confidence:.2f}")
                if args.show:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
                    cv2.putText(frame, f"id={det.track_id} {det.confidence:.2f}", (x1, max(15, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 2)

            if args.show:
                cv2.imshow("human_detection test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
