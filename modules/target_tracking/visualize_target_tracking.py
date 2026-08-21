"""
Mandatory standalone visualization entry point for target_tracking (spec §7). Runs the full
RECORDING -> TRACKING -> LOST cycle against a webcam feed or video file.

Initial bbox input method: CLICK-AND-DRAG on the first frame (a design choice, since the spec
left this an open "confirm with the user" question that wasn't part of this module's blocking
design decisions — click-and-drag was chosen as the more direct, visually-verifiable way to
simulate the gesture-trigger handoff than typing raw pixel coordinates). Press 'r' at any time to
re-arm bbox selection and start a fresh episode (simulates a new gesture trigger).

Draws the tracked bbox, a vertical line at frame-center, and a live readout of state,
horizontal_offset, and the periodic re-verify's last score/pass-fail (shown only on the frames
where it actually ran, per spec §7 — not every frame). Logs TrackingResult fields to console per
frame.

Usage:
    python -m modules.target_tracking.visualize_target_tracking --mode camera [--camera-index 0]
    python -m modules.target_tracking.visualize_target_tracking --mode video --video path.mp4
"""
import argparse
import sys
import time

import cv2

from modules.target_tracking.pipeline import TargetTrackingPipeline
from modules.target_tracking.config import load_config

_drag_state = {"dragging": False, "start": None, "end": None, "armed": True}


def _on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_state["dragging"] = True
        _drag_state["start"] = (x, y)
        _drag_state["end"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drag_state["dragging"]:
        _drag_state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _drag_state["dragging"]:
        _drag_state["dragging"] = False
        _drag_state["end"] = (x, y)


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone target_tracking visualization/debugging tool.")
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    # Own pipeline instance (not the module-level singleton behind interface.start()/update()) so
    # this debug tool can read internal episode state (last_reverify_score/last_reverify_pass)
    # for the "only visible on frames where re-verify actually ran" requirement — a reach-in
    # that's fine here since this file lives inside the module's own package.
    pipeline = TargetTrackingPipeline(config)

    window_name = "visualize_target_tracking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    print("Click-and-drag a box around a person to start tracking. Press 'r' to re-arm, 'q' to quit.")

    frame_idx = 0
    episode_active = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            if _drag_state["start"] and _drag_state["end"]:
                x1, y1 = _drag_state["start"]
                x2, y2 = _drag_state["end"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

            if not _drag_state["dragging"] and _drag_state["armed"] and _drag_state["start"] and _drag_state["end"]:
                x1, y1 = _drag_state["start"]
                x2, y2 = _drag_state["end"]
                bx, by = min(x1, x2), min(y1, y2)
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                if bw > 5 and bh > 5:
                    pipeline.start((bx, by, bw, bh), frame, timestamp)
                    episode_active = True
                    _drag_state["armed"] = False
                    print(f"frame={frame_idx:06d} STARTED episode with initial bbox=({bx},{by},{bw},{bh})")

            if episode_active:
                prev_reverify_time = pipeline._episode.last_reverify_time
                result = pipeline.update(frame, timestamp)

                # Draws bbox (colored by state) + frame-center line + state/offset/reverify
                # readout — the same reusable overlay any external caller gets (main.py,
                # modules.followme_orchestrator) via TrackingResult.draw_debug(), not
                # hand-rolled here a second time.
                result.draw_debug(frame)

                # Console-only: print the reverify readout ONLY on the frame it actually ran
                # (last_reverify_time changed this call) — per spec §7, not every frame. This is
                # a print-cadence choice specific to this console log, separate from the overlay
                # above (which persists the last-known value on screen, more useful visually).
                if pipeline._episode.last_reverify_time != prev_reverify_time:
                    print(f"frame={frame_idx:06d} REVERIFY score={result.last_reverify_score} pass={result.last_reverify_pass}")

                print(
                    f"frame={frame_idx:06d} state={result.state} target_locked={result.target_locked} "
                    f"horizontal_offset={result.horizontal_offset} person_bbox={result.person_bbox}"
                )

                if result.state == "LOST":
                    episode_active = False
                    _drag_state["start"] = None
                    _drag_state["end"] = None
                    _drag_state["armed"] = True
                    print(f"frame={frame_idx:06d} LOST — press-and-drag again to start a new episode.")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                episode_active = False
                _drag_state["start"] = None
                _drag_state["end"] = None
                _drag_state["armed"] = True

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
