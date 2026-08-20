"""
Mandatory standalone visualization entry point for human_detection_roi (spec §5). Chained LIVE
with modules.face_identity (this repo already has that module built and working, so live
chaining is the more useful default over accepting pre-recorded FaceIdentityResult data) —
consumes only modules.face_identity's public evaluate()/FaceIdentityResult output, per the
isolation rule (spec §0.3: no reaching into face_identity's internals).

Draws, per frame, per matched face:
  - The face bbox (input to this module) in BLUE
  - The computed ROI search region (spec §2) in YELLOW
  - The final detected person bbox (output) in GREEN
Logs HumanDetectionResult fields to console for inspection without watching video.

Usage:
    python -m modules.human_detection_roi.visualize_human_detection_roi --mode camera [--camera-index 0]
    python -m modules.human_detection_roi.visualize_human_detection_roi --mode video --video path/to/file.mp4
"""
import argparse
import sys

import cv2

from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
from modules.human_detection_roi.config import load_config
from modules.human_detection_roi.interface import evaluate as evaluate_person
from modules.human_detection_roi.roi import compute_roi

_FACE_COLOR = (255, 100, 0)     # blue-ish
_ROI_COLOR = (0, 220, 255)      # yellow
_PERSON_COLOR = (0, 200, 0)     # green


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone human_detection_roi visualization/debugging tool.")
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

    face_registry = FaceRegistry(args.face_registry_dir)
    roi_config = load_config(args.config)  # only used to redraw the ROI region for visualization

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_results = [r for r in evaluate_face(frame, face_registry) if r.is_registered_match]
            for face in face_results:
                fx, fy, fw, fh = face.face_bbox
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), _FACE_COLOR, 2)

                if roi_config.roi_expansion_factor is not None:
                    rx1, ry1, rx2, ry2 = compute_roi(
                        face.face_bbox, frame.shape, roi_config.roi_expansion_factor,
                        upward_fraction=roi_config.roi_upward_fraction, width_fraction=roi_config.roi_width_fraction,
                    )
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), _ROI_COLOR, 2)

                person = evaluate_person(frame, face.face_bbox)
                print(
                    f"frame={frame_idx:06d} matched_person_name={face.matched_person_name} "
                    f"person_found={person.person_found} person_bbox={person.person_bbox} "
                    f"detection_confidence={person.detection_confidence}"
                )
                if person.person_found:
                    px, py, pw, ph = person.person_bbox
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), _PERSON_COLOR, 2)
                    cv2.putText(frame, f"{face.matched_person_name}", (px, max(15, py - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _PERSON_COLOR, 2)

            cv2.imshow("visualize_human_detection_roi", frame)
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
