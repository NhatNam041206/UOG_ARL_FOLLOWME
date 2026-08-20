"""
Mandatory standalone visualization entry point for face_identity (spec §5). Runs the module
fully independently of any other pipeline stage (human detection, gesture methods) — imports
ONLY from modules.face_identity.interface, matching this project's isolation convention.

Draws, per frame:
  - A bounding box around every detected face (regardless of match status)
  - A distinct color for faces that match a registered person (green) vs. don't (red)
  - The matched person's ID + similarity score as on-screen text, for any match
Logs FaceIdentityResult fields to console for every frame a face is detected.

Usage:
    python -m modules.face_identity.visualize_face_identity --mode camera [--camera-index 0]
    python -m modules.face_identity.visualize_face_identity --mode video --video path/to/file.mp4
"""
import argparse
import sys

import cv2

from modules.face_identity.interface import FaceRegistry, evaluate

_MATCH_COLOR = (0, 200, 0)
_NO_MATCH_COLOR = (0, 0, 255)


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone face_identity visualization/debugging tool.")
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--registry-dir", default="modules/face_identity/registry_data")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    registry = FaceRegistry(args.registry_dir)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = evaluate(frame, registry)
            for r in results:
                print(
                    f"frame={frame_idx:06d} face_bbox={r.face_bbox} "
                    f"face_detection_confidence={r.face_detection_confidence} "
                    f"is_registered_match={r.is_registered_match} "
                    f"matched_person_name={r.matched_person_name} match_confidence={r.match_confidence}"
                )
                x, y, w, h = r.face_bbox
                color = _MATCH_COLOR if r.is_registered_match else _NO_MATCH_COLOR
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = (f"{r.matched_person_name} ({r.match_confidence:.2f})"
                         if r.is_registered_match else f"no match ({r.match_confidence})")
                cv2.putText(frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.putText(frame, f"faces: {len(results)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("visualize_face_identity", frame)
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
