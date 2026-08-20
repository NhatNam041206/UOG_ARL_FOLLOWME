"""
Face registration — Phase 1: raw image capture.

Registration is split into two phases (per user request, mirroring — in FORMAT only, as fresh
standalone code, per spec §0.3 isolation — the sibling UOG_ARL_FOLLOWME project's
RawDataCapturer/TargetRegistrar two-phase pattern in its src/registration.py):

    Phase 1 (this script): capture raw face images from a webcam, save them as JPEG files.
    Phase 2 (build_face_registry.py): turn a folder of images into a registry entry.

The point of the split: after this script runs, the images sitting in
raw_captures/{person_name}/ are just ordinary JPEG files — actual face photos (a padded crop
around the detected face, not the whole scene), so the folder is directly useful to skim through.
You can open that folder, delete a bad shot, add a better photo from somewhere else, or skip this
script entirely and drop in your own photos by hand — Phase 2 doesn't care where the images came
from, it just re-detects the face in each one fresh, exactly like live inference does.

Usage:
    python -m modules.face_identity.capture_face_images <person_name> [--camera-index 0] [--samples 5]
"""
import argparse
import glob
import os
import sys
import time

import cv2

from modules.face_identity.config import load_config
from modules.face_identity.face_detector import YuNetFaceDetector
from modules.face_identity.registry import sanitize_person_name

_CAPTURE_INTERVAL_SECONDS = 1.0  # gap between accepted samples, so they're not near-duplicate frames

# Fraction of the detected bbox's width/height added as padding on each side before cropping and
# saving. NOT tight-cropped to the bbox — a tight crop can clip landmarks right at the edge and
# make Phase 2's re-detection on the saved image fail/degrade; this margin keeps enough context
# around the face for YuNet to reliably re-detect it from the saved file alone.
_CROP_PADDING_FRACTION = 0.6


def _padded_face_crop(frame, bbox):
    x, y, w, h = bbox
    pad_x, pad_y = int(w * _CROP_PADDING_FRACTION), int(h * _CROP_PADDING_FRACTION)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    return frame[y1:y2, x1:x2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: capture raw face images for one person from a live webcam.")
    parser.add_argument("person_name", help="This person's name (sanitized to a safe directory name).")
    parser.add_argument("--camera-index", type=int, default=0, help="OS camera device index (default 0).")
    parser.add_argument("--samples", type=int, default=5, help="Number of images to capture this run (default 5).")
    parser.add_argument("--config", default="config/thresholds.yaml", help="Path to thresholds.yaml.")
    args = parser.parse_args()

    try:
        person_name = sanitize_person_name(args.person_name)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    detector = YuNetFaceDetector(config.yunet_model_path)

    person_dir = os.path.join(config.raw_captures_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    existing_count = len(glob.glob(os.path.join(person_dir, "*.jpg")))
    if existing_count:
        print(f"NOTE: '{person_dir}' already has {existing_count} image(s) — new captures are added alongside them.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera_index}", file=sys.stderr)
        return 1

    print(f"Capturing images for '{person_name}': need {args.samples} more. Look at the camera.")
    print("Press 'q' to stop early (images already saved are kept).")

    saved = 0
    last_capture_time = 0.0
    try:
        while saved < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            raw_faces = detector.detect(frame)
            display = frame.copy()
            best_face = max(raw_faces, key=lambda f: f.score) if raw_faces else None
            if best_face is not None:
                x, y, w, h = best_face.bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)

            now = time.time()
            if best_face is not None and (now - last_capture_time) >= _CAPTURE_INTERVAL_SECONDS:
                out_path = os.path.join(person_dir, f"{existing_count + saved + 1:03d}.jpg")
                face_crop = _padded_face_crop(frame, best_face.bbox)
                cv2.imwrite(out_path, face_crop)
                saved += 1
                last_capture_time = now
                print(f"  saved '{out_path}' ({saved}/{args.samples})")

            cv2.putText(display, f"images: {saved}/{args.samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("capture_face_images", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done: {saved} new image(s) saved to '{person_dir}' ({existing_count + saved} total).")
    print(f"Next: python -m modules.face_identity.build_face_registry {person_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
