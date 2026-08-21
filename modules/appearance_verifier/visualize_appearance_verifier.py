"""
Mandatory standalone visualization entry point for appearance_verifier (spec §6). Runs in
complete isolation from any tracking/recovery logic — accepts a folder of "reference" images and
either a single "candidate" image or a live feed ("candidate" mode).

Displays the best similarity score prominently, plus a clear match/no-match indicator against
the configured threshold. Logs AppearanceVerifierResult fields to console per comparison.

Suggested test procedure for this module's two named accuracy risks (spec §2, §6):
  1. Similar-clothing confusion: build a reference set from person A, then run a candidate feed
     of a DIFFERENT person B wearing similar-colored/styled clothing to A. This SHOULD score low
     (no match), but may not — that is exactly the risk being tested for, not a bug in this tool.
  2. Cross-domain generalization: build a reference set from person A in one lighting/distance
     condition, then run a candidate feed of the SAME person A under noticeably different
     lighting or distance. This should score high (match) if OSNet generalizes acceptably to
     this project's own footage; a low score here is evidence of the cross-domain risk, worth
     recording during calibration, not silently dismissing.

Usage:
    python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir <folder> --mode camera [--camera-index 0]
    python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir <folder> --mode video --video path.mp4
    python -m modules.appearance_verifier.visualize_appearance_verifier --reference-dir <folder> --mode image --image path.jpg
"""
import argparse
import glob
import os
import sys

import cv2

from modules.appearance_verifier.config import load_config
from modules.appearance_verifier.interface import build_reference_set, verify

_MATCH_COLOR = (0, 200, 0)
_NO_MATCH_COLOR = (0, 0, 255)


def load_reference_set(reference_dir: str):
    ref_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(reference_dir, ext))
    )
    crops = [c for c in (cv2.imread(p) for p in ref_paths) if c is not None]
    return crops


def draw_result(frame, result, threshold):
    color = _MATCH_COLOR if result.match_found else _NO_MATCH_COLOR
    if result.reference_frame_count == 0:
        label1 = "NOT READY (reference_frame_count=0)"
        color = (200, 200, 200)
    else:
        label1 = f"{'MATCH' if result.match_found else 'NO MATCH'}  score={result.best_similarity_score:.4f}  threshold={threshold}"
    label2 = f"reference_frame_count={result.reference_frame_count}"
    cv2.putText(frame, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, label2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone appearance_verifier visualization/debugging tool.")
    parser.add_argument("--reference-dir", required=True, help="Folder of reference person-crop images (.jpg/.png).")
    parser.add_argument("--mode", choices=["camera", "video", "image"], required=True)
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")
    parser.add_argument("--image", help="Path to a single candidate image. Required when --mode image.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="config/thresholds.yaml")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")
    if args.mode == "image" and not args.image:
        parser.error("--image is required when --mode image")

    config = load_config(args.config)

    ref_crops = load_reference_set(args.reference_dir)
    if not ref_crops:
        print(f"ERROR: no reference images found in '{args.reference_dir}'", file=sys.stderr)
        return 1
    print(f"Loaded {len(ref_crops)} reference image(s) from '{args.reference_dir}'.")
    reference_set = build_reference_set(ref_crops)

    if args.mode == "image":
        candidate = cv2.imread(args.image)
        if candidate is None:
            print(f"ERROR: could not read image '{args.image}'", file=sys.stderr)
            return 1
        result = verify(candidate, reference_set)
        print(
            f"match_found={result.match_found} best_similarity_score={result.best_similarity_score:.4f} "
            f"reference_frame_count={result.reference_frame_count} threshold={config.similarity_threshold}"
        )
        draw_result(candidate, result, config.similarity_threshold)
        cv2.imshow("visualize_appearance_verifier", candidate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
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
            draw_result(frame, result, config.similarity_threshold)

            cv2.imshow("visualize_appearance_verifier", frame)
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
