"""
Standalone test entry point for the wave_facing_gate module. Imports ONLY from
modules.wave_facing_gate.interface — no other project module is imported or required, so this
can run in isolation to validate the module on its own (mirrors
modules/emergency_stop/test_estop.py).

This module's real input is a per-track bbox crop from the teammate's detection/tracking/Re-ID
pipeline (spec §1). That pipeline doesn't exist in this standalone test, so — same simplification
as any single-module isolation test — the WHOLE frame is fed in as a single track_id=1 crop.
Point the camera/video at one person for a meaningful result.

Usage:
    python -m modules.wave_facing_gate.test_wave_facing <path-to-video-file> [--show]
"""
import argparse
import sys

import cv2

from modules.wave_facing_gate.interface import WaveFacingGateModule

# Visualization only (spec §6 UI/debug note leaves combined-vs-separate unspecified for Stage 2;
# this standalone test picks "both GREEN" for a single combined box color, purely for eyeballing
# results while calibrating — not a Stage-2 decision).
_STATE_COLOR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the wave_facing_gate pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--show", action="store_true", help="Display frames with overlay while processing.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    gate = WaveFacingGateModule()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = gate.process_frame(track_id=1, crop=frame)
            print(
                f"frame={frame_idx:06d} is_waving={str(result.is_waving):5s} "
                f"is_facing={str(result.is_facing_camera):5s} "
                f"waving_state={result.waving_state:6s} facing_state={result.facing_state:6s} "
                f"wave_arm={str(result.wave_arm):5s} "
                f"facing_conf_min={result.facing_confidence_min} "
                f"latency_ms={gate.last_latency_ms:6.1f}"
            )

            if args.show:
                color = _STATE_COLOR["GREEN"] if (result.is_waving and result.is_facing_camera) else (
                    _STATE_COLOR["YELLOW"] if "YELLOW" in (result.waving_state, result.facing_state) else _STATE_COLOR["RED"]
                )
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
                cv2.putText(frame, f"wave={result.waving_state} facing={result.facing_state}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("wave_facing_gate test", frame)
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
