"""
Mandatory standalone visualization entry point for target_recovery (spec §6). Simulates a search
episode against a webcam feed or video file.

Reference-set supply method: a FOLDER OF REFERENCE IMAGES, embedded once via
appearance_verifier.build_reference_set() at startup — same approach as
modules/appearance_verifier/visualize_appearance_verifier.py, chosen so this tool can run without
first driving a live modules.target_tracking RECORDING phase to completion (a design choice; the
spec left the exact supply method an open "confirm with the user" question that wasn't one of
this module's blocking design decisions).

Displays which path is currently active/being attempted each frame, the running
face_search_fail_count, and a visible countdown of elapsed_search_seconds against
search_timeout_seconds. On REACQUIRED, clearly shows which path succeeded (reacquired_via) and
draws the resulting bbox. Logs RecoveryResult fields to console per frame.

Usage:
    python -m modules.target_recovery.visualize_target_recovery --target-person-name Nam --reference-dir <folder> --mode camera [--camera-index 0]
    python -m modules.target_recovery.visualize_target_recovery --target-person-name Nam --reference-dir <folder> --mode video --video path.mp4
"""
import argparse
import glob
import os
import sys
import time

import cv2

from modules.appearance_verifier.interface import build_reference_set
from modules.face_identity.interface import FaceRegistry
from modules.target_recovery.config import load_config
from modules.target_recovery.interface import start, update

_STATUS_COLOR = {"SEARCHING": (0, 220, 255), "REACQUIRED": (0, 200, 0), "TIMEOUT": (0, 0, 255)}


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone target_recovery visualization/debugging tool.")
    parser.add_argument("--target-person-name", required=True, help="Registered person name Path A should filter for.")
    parser.add_argument("--reference-dir", required=True, help="Folder of reference person-crop images for the appearance reference set.")
    parser.add_argument("--face-registry-dir", default="modules/face_identity/registry_data")
    parser.add_argument("--mode", choices=["camera", "video"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")

    config = load_config(args.config)

    ref_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(args.reference_dir, ext))
    )
    ref_crops = [c for c in (cv2.imread(p) for p in ref_paths) if c is not None]
    if not ref_crops:
        print(f"ERROR: no reference images found in '{args.reference_dir}'", file=sys.stderr)
        return 1
    reference_set = build_reference_set(ref_crops)
    print(f"Loaded {len(ref_crops)} reference image(s) from '{args.reference_dir}'.")

    registry = FaceRegistry(args.face_registry_dir)

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    start(reference_set, args.target_person_name, time.time())
    print(f"Searching for '{args.target_person_name}'...")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            result = update(frame, registry, timestamp)
            color = _STATUS_COLOR.get(result.status, (255, 255, 255))

            grace = config.face_search_grace_attempts
            path_b_active = grace is not None and result.face_search_fail_count >= grace
            active_path = "Path B (appearance fallback) active" if path_b_active else "Path A (face match) primary"

            timeout_str = f"/{config.search_timeout_seconds:.0f}s" if config.search_timeout_seconds is not None else ""
            label1 = f"status={result.status}  {active_path}"
            label2 = f"face_search_fail_count={result.face_search_fail_count}  elapsed={result.elapsed_search_seconds:.1f}s{timeout_str}"
            cv2.putText(frame, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if result.status == "REACQUIRED" and result.reacquired_person_bbox is not None:
                x, y, w, h = result.reacquired_person_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, f"REACQUIRED via {result.reacquired_via}", (x, max(15, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(
                f"frame={frame_idx:06d} status={result.status} reacquired_via={result.reacquired_via} "
                f"reacquired_person_bbox={result.reacquired_person_bbox} "
                f"face_search_fail_count={result.face_search_fail_count} "
                f"elapsed_search_seconds={result.elapsed_search_seconds:.1f}"
            )

            cv2.imshow("visualize_target_recovery", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if result.status in ("REACQUIRED", "TIMEOUT"):
                print(f"Search episode ended: {result.status}. Press 'q' to quit, or any other key to keep viewing the final frame.")
                cv2.waitKey(0)
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
