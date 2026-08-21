"""
Mandatory standalone visualization entry point for followme_orchestrator (spec §5) — the only
visualization tool in this whole feature set that exercises EVERYTHING together (face -> ROI ->
gesture -> tracking -> recovery -> steering), since every other module's visualizer is
deliberately scoped to just that one module.

Displays debug_state, should_move, steering_angle_degrees, a frame-center reference line, and
the current tracked/reacquired bbox (read off this script's own private pipeline instance's
last_person_bbox — a debug-only convenience field, not part of FollowMeCommand's public
contract; see pipeline.py). Logs FollowMeCommand fields to console per frame.

Usage:
    python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode camera [--camera-index 0]
    python -m modules.followme_orchestrator.visualize_followme_orchestrator --gesture-method hand_keypoint --mode video --video path.mp4
"""
import argparse
import sys
import time

import cv2

from modules.followme_orchestrator.config import load_config
from modules.followme_orchestrator.pipeline import FollowMeOrchestratorPipeline

_STATE_COLOR = {
    "WAITING_FOR_TRIGGER": (180, 180, 180),
    "TRACKING_STARTED": (0, 220, 255),
    "RECORDING": (0, 220, 255),
    "TRACKING": (0, 200, 0),
    "RECORDING_STEERING_UNCALIBRATED": (0, 160, 255),
    "TRACKING_STEERING_UNCALIBRATED": (0, 160, 255),
    "RECOVERING": (0, 140, 255),
    "REACQUIRED_RESUMING": (0, 200, 0),
    "STOPPED": (0, 0, 255),
}


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone followme_orchestrator end-to-end visualization/debugging tool.")
    parser.add_argument("--gesture-method", choices=["condition", "hand_keypoint", "trajectory_verifier"], required=True)
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--face-registry-dir", default="modules/face_identity/registry_data")
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    # Own pipeline instance (not the module-level singleton behind interface.step()) so this
    # debug tool can read last_person_bbox for drawing — a reach-in that's fine here since this
    # file lives inside the module's own package (same pattern every other visualize_*.py uses).
    pipeline = FollowMeOrchestratorPipeline(config, args.gesture_method, args.face_registry_dir)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            result = pipeline.step(frame, timestamp)
            color = _STATE_COLOR.get(result.debug_state, (255, 255, 255))

            frame_h, frame_w = frame.shape[:2]
            cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (120, 120, 120), 1)

            if pipeline.last_person_bbox is not None:
                x, y, w, h = pipeline.last_person_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            angle_str = f"{result.steering_angle_degrees:+.1f}deg" if result.steering_angle_degrees is not None else "None"
            label1 = f"state={result.debug_state}"
            label2 = f"should_move={result.should_move}  steering={angle_str}"
            cv2.putText(frame, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(
                f"frame={frame_idx:06d} debug_state={result.debug_state:28s} "
                f"should_move={result.should_move} steering_angle_degrees={result.steering_angle_degrees}"
            )

            cv2.imshow("visualize_followme_orchestrator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
