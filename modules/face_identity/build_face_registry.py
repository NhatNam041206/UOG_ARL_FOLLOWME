"""
Face registration — Phase 2: build a registry entry from a folder of images.

Reads every image in a person's folder (default: raw_captures/{person_name}/, produced by
capture_face_images.py — but ANY folder works, which is the whole point: point --images-dir at
your own photos to register someone without a webcam at all), runs the SAME face detection +
alignment + embedding pipeline used at live-inference time on each image, and saves the
resulting embeddings as that person's registry entry (modules.face_identity.registry).

Images where no face is found, or where the face can't be aligned, are skipped individually
(logged, not fatal) — one bad photo in the folder doesn't block the rest.

Usage:
    python -m modules.face_identity.build_face_registry <person_name> [--images-dir path]
"""
import argparse
import glob
import os
import sys

import cv2

from modules.face_identity.alignment import align_face
from modules.face_identity.config import load_config
from modules.face_identity.embedder import EdgeFaceEmbedder
from modules.face_identity.face_detector import YuNetFaceDetector
from modules.face_identity.registry import FaceRegistry, sanitize_person_name

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: build a face registry entry from a folder of images.")
    parser.add_argument("person_name", help="This person's name (used as the registry key).")
    parser.add_argument(
        "--images-dir", default=None,
        help="Folder of images to build from. Defaults to raw_captures/{person_name}/ (from "
             "capture_face_images.py) — point this anywhere to use your own photos instead.",
    )
    parser.add_argument("--config", default="config/thresholds.yaml", help="Path to thresholds.yaml.")
    args = parser.parse_args()

    try:
        person_name = sanitize_person_name(args.person_name)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    images_dir = args.images_dir or os.path.join(config.raw_captures_dir, person_name)
    if not os.path.isdir(images_dir):
        print(
            f"ERROR: images directory '{images_dir}' does not exist. Run "
            f"'python -m modules.face_identity.capture_face_images {person_name}' first, "
            f"or pass --images-dir pointing at a folder of your own photos.", file=sys.stderr,
        )
        return 1

    image_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"ERROR: no images found in '{images_dir}' (looked for {_IMAGE_EXTENSIONS}).", file=sys.stderr)
        return 1

    detector = YuNetFaceDetector(config.yunet_model_path)
    embedder = EdgeFaceEmbedder(config.edgeface_model_path)
    registry = FaceRegistry(config.registry_dir)

    embeddings = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  SKIP '{path}': could not read image file.")
            continue

        faces = detector.detect(image)
        if not faces:
            print(f"  SKIP '{path}': no face detected.")
            continue
        if len(faces) > 1:
            print(f"  NOTE '{path}': {len(faces)} faces detected, using the highest-confidence one.")
        best_face = max(faces, key=lambda f: f.score)

        aligned = align_face(image, best_face.landmarks)
        if aligned is None:
            print(f"  SKIP '{path}': could not align face (degenerate landmarks).")
            continue

        embeddings.append(embedder.embed(aligned))
        print(f"  OK '{path}' (detection score={best_face.score:.2f})")

    if not embeddings:
        print(f"ERROR: no usable face embeddings extracted from '{images_dir}'.", file=sys.stderr)
        return 1

    saved_path = registry.save_person(person_name, embeddings)
    print(f"Saved '{person_name}' to '{saved_path}' ({len(embeddings)}/{len(image_paths)} images used).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
