"""
Standalone test entry point for the emergency_stop module (spec §5). Imports ONLY from
modules.emergency_stop.interface — no other project module is imported or required, so this can
run in isolation to validate the module on its own.

Usage:
    python -m modules.emergency_stop.test_estop <path-to-video-file> [--show]

`<path-to-video-file>` must be supplied by the caller — no default video is assumed to exist.
"""
import argparse
import sys

import cv2

from modules.emergency_stop.interface import EmergencyStopModule, EStopDecision


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the emergency_stop pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--show", action="store_true", help="Display the raw frames in a window while processing.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    estop = EmergencyStopModule()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = estop.process_frame(frame)
            print(
                f"frame={frame_idx:06d} decision={output.decision.value:9s} "
                f"reason={output.reason:28s} track_id={str(output.triggering_track_id):>6s} "
                f"zone={str(output.zone):5s} latency_ms={estop.last_latency_ms:6.1f}"
            )

            if output.decision in (EStopDecision.STOP, EStopDecision.UNCERTAIN):
                pass  # placeholder for wiring into an actual actuator stop signal downstream

            if args.show:
                cv2.imshow("emergency_stop test", frame)
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
