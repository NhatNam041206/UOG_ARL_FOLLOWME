"""
Standalone test entry point for the appearance_verifier module. Imports ONLY from
modules.appearance_verifier.interface — no other project module is imported or required
(mirrors modules/face_identity/test_face_identity.py).

Compares every frame of a video against a small set of reference images to sanity-check
embedding + matching end to end, without needing a live person detector.

Usage:
    python -m modules.appearance_verifier.test_appearance_verifier <video_path> --reference-dir <folder-of-reference-images>
"""
import argparse
import glob
import os
import sys

import cv2

from modules.appearance_verifier.interface import build_reference_set, verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Run appearance_verifier standalone against a recorded video.")
    parser.add_argument("video_path", help="Path to a recorded video file (e.g. an .mp4). Each frame is treated as a whole-frame candidate crop.")
    parser.add_argument("--reference-dir", required=True, help="Folder of reference person-crop images (.jpg/.png).")
    args = parser.parse_args()

    ref_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(args.reference_dir, ext))
    )
    if not ref_paths:
        print(f"ERROR: no reference images found in '{args.reference_dir}'", file=sys.stderr)
        return 1

    ref_crops = [cv2.imread(p) for p in ref_paths]
    ref_crops = [c for c in ref_crops if c is not None]
    print(f"Loaded {len(ref_crops)} reference image(s) from '{args.reference_dir}'.")
    reference_set = build_reference_set(ref_crops)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file '{args.video_path}'", file=sys.stderr)
        return 1

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = verify(frame, reference_set)
            print(
                f"frame={frame_idx:06d} match_found={result.match_found} "
                f"best_similarity_score={result.best_similarity_score:.4f} "
                f"reference_frame_count={result.reference_frame_count}"
            )

            frame_idx += 1
    finally:
        cap.release()

    print(f"Processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
