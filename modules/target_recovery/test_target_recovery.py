"""
Standalone test entry point for the target_recovery module. Imports ONLY from
modules.target_recovery.interface plus modules.face_identity and modules.appearance_verifier
(needed to supply a registry and a reference_set — mirrors modules/face_identity/test_face_identity.py's
"no other project module required beyond direct inputs" style).

Usage:
    python -m modules.target_recovery.test_target_recovery <video_path> --target-person-name Nam --reference-dir <folder-of-reference-images> [--face-registry-dir modules/face_identity/registry_data]
"""
import argparse
import glob
import os
import sys
import time

import cv2

from modules.appearance_verifier.interface import build_reference_set
from modules.face_identity.interface import FaceRegistry
from modules.target_recovery.interface import start, update


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the target_recovery pipeline standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4).")
    parser.add_argument("--target-person-name", required=True, help="Registered person name Path A should filter for.")
    parser.add_argument("--reference-dir", required=True, help="Folder of reference person-crop images for the appearance reference set.")
    parser.add_argument("--face-registry-dir", default="modules/face_identity/registry_data")
    args = parser.parse_args()

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

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    start(reference_set, args.target_person_name, time.time())

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            result = update(frame, registry, timestamp)
            print(
                f"frame={frame_idx:06d} status={result.status} reacquired_via={result.reacquired_via} "
                f"reacquired_person_bbox={result.reacquired_person_bbox} "
                f"face_search_fail_count={result.face_search_fail_count} "
                f"elapsed_search_seconds={result.elapsed_search_seconds:.1f}"
            )
            if result.status in ("REACQUIRED", "TIMEOUT"):
                print(f"Search episode ended: {result.status}")
                break

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
