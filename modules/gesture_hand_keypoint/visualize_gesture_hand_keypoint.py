"""
Mandatory standalone visualization entry point for gesture_hand_keypoint (spec §6). Chained LIVE
with modules.face_identity + modules.human_detection_roi (both already built and working in this
repo, so live-chaining the full upstream pipeline is the more useful default) — consumes only
their public evaluate()/dataclass outputs, per each module's own isolation rule. Passes
person.person_bbox through as person_bbox_full_frame, required for the palm-height gate.

Draws, per frame, per person crop:
  - All 21 MediaPipe hand landmarks + skeleton on each detected hand — the skeleton/keypoints
    are colored YELLOW when that hand currently classifies as OPEN, GREEN when CLOSED, gray if
    NEITHER (ambiguous) — the whole-hand read the sequence state machine consumes
  - Each finger's key edge ALSO colored green (extended) or red (curled), thumb included — a
    separate, more granular per-finger diagnostic, made visible (GestureMethodResult.draw_debug())
  - A red DOTTED horizontal line at the palm_height_fraction cutoff, for visually calibrating
    the palm-height gate against the person's bbox
  - The current OPEN->CLOSED->OPEN->CLOSED sequence stage (WAITING_OPEN / WAITING_CLOSE_1 /
    WAITING_OPEN_2 / WAITING_CLOSE_2 / CONFIRMED) — needed to debug the sequence itself, not
    just the debounced red/yellow/green result
  - A bbox-colored (red/yellow/green) indicator reflecting the current confirmation state
Logs GestureMethodResult fields to console per frame.

Usage:
    python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode camera [--camera-index 0]
    python -m modules.gesture_hand_keypoint.visualize_gesture_hand_keypoint --mode video --video path/to/file.mp4
"""
import argparse
import sys
import time

import cv2

from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
from modules.gesture_hand_keypoint.interface import evaluate as evaluate_gesture
from modules.human_detection_roi.interface import evaluate as evaluate_person

_STATE_COLOR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone gesture_hand_keypoint visualization/debugging tool.")
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--face-registry-dir", default="modules/face_identity/registry_data")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    face_registry = FaceRegistry(args.face_registry_dir)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            face_results = [r for r in evaluate_face(frame, face_registry) if r.is_registered_match]
            for face in face_results:
                person = evaluate_person(frame, face.face_bbox)
                if not person.person_found:
                    continue

                px, py, pw, ph = person.person_bbox
                px, py = max(0, px), max(0, py)
                crop = frame[py:py + ph, px:px + pw]
                if crop.size == 0:
                    continue

                track_id = abs(hash(face.matched_person_name)) % 100000  # stand-in track_id for this debug tool
                gesture = evaluate_gesture(track_id, crop, timestamp, person_bbox_full_frame=person.person_bbox)
                color = _STATE_COLOR.get(gesture.waving_state, (255, 255, 255))

                # Draws skeleton + keypoints (yellow=OPEN/green=CLOSED/gray=NEITHER) + per-finger
                # coloring + the red dotted palm-height calibration line directly onto `crop` (a
                # view into `frame`) using keypoints_raw — no extra inference. Same person_bbox
                # passed to evaluate_gesture() above, so the threshold line lines up correctly.
                gesture.draw_debug(crop, person_bbox_full_frame=person.person_bbox)

                cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
                label1 = f"{face.matched_person_name}: state={gesture.waving_state} stage={gesture.sequence_stage}"
                label2 = f"conf={gesture.confidence_debug} palm_facing={gesture.palm_facing_camera_debug}"
                cv2.putText(frame, label1, (px, max(15, py - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label2, (px, max(15, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(
                    f"frame={frame_idx:06d} track_id={track_id} is_waving={gesture.is_waving} "
                    f"waving_state={gesture.waving_state} sequence_stage={gesture.sequence_stage} "
                    f"confidence_debug={gesture.confidence_debug} "
                    f"palm_facing_camera_debug={gesture.palm_facing_camera_debug}"
                )

            cv2.imshow("visualize_gesture_hand_keypoint", frame)
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
