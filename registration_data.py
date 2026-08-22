"""
Registration data layer — Layer 1 of register_person's three-layer design (data / overlay /
interaction; see register_person.py's own docstring for the full split). Everything here is pure
data plumbing: filesystem state, building the two registry files, CRUD on a person's
registration. No cv2.imshow/waitKey and no camera reads anywhere in this file — register_person.py
(Layer 3) owns all of that and calls into this layer instead.

Capture is split into two persisted, inspectable phases (confirmed with the user), not one:

    1. RAW      — save_raw_capture(): the exact camera frame, uncropped, one per accepted sample.
    2. CROPPED  — build_cropped_roi(): reads every RAW image back and crops it to the configured
                  ROI, saving the result as its own separate file. A real, on-disk artifact you
                  can open and look at — this exists specifically so the ROI crop can be visually
                  checked BEFORE anything downstream (detection, embedding) ever runs on it.

Both npz-building functions read from the CROPPED folder, never the raw one directly — detection
still only ever happens in the data-building phase (build_face_registry / build_target_profile),
on the cropped images, same "detection happens after capture, never during" design as before.

CRUD mapping:
    Create -> save_raw_capture() during a live session, then build_cropped_roi() once it ends
    Read   -> list_people() / get_status()
    Update -> re-running a capture session for a name that already exists — reset_captures()
              always wipes RAW+CROPPED first (confirmed with the user: never mix old and new
              photos), then the same Create flow runs, followed by rebuild_registries()
    Delete -> delete_person()
"""
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

import registration_overlay as overlay
from modules.followme_orchestrator import autocar_bootstrap

CAPTURES_DIR = "registration_captures"
FACE_REGISTRY_DIR = "modules/face_identity/registry_data"
AUTOCAR_MODELS_DIR = "modules/autocar/models"


@dataclass
class PersonStatus:
    name: str
    raw_front_count: int
    raw_back_count: int
    cropped_front_count: int
    cropped_back_count: int
    has_face_registry: bool
    has_target_profile: bool

    @property
    def ready_for_followme(self) -> bool:
        return self.has_face_registry and self.has_target_profile


def _raw_dir(name: str, phase: str) -> str:
    return os.path.join(CAPTURES_DIR, name, "raw", phase)


def _cropped_dir(name: str, phase: str) -> str:
    return os.path.join(CAPTURES_DIR, name, "cropped", phase)


def _count_jpgs(directory: str) -> int:
    return len(glob.glob(os.path.join(directory, "*.jpg"))) if os.path.isdir(directory) else 0


def list_people() -> List[PersonStatus]:
    """READ (all) — every name with a raw-capture folder OR an already-built registry file,
    whichever exists; someone mid-registration (captures but no built registry yet) still shows
    up, so the menu can offer to finish/rebuild for them."""
    names = set()
    if os.path.isdir(CAPTURES_DIR):
        names.update(os.listdir(CAPTURES_DIR))
    if os.path.isdir(FACE_REGISTRY_DIR):
        names.update(os.path.splitext(f)[0] for f in os.listdir(FACE_REGISTRY_DIR) if f.endswith(".npz"))
    if os.path.isdir(AUTOCAR_MODELS_DIR):
        names.update(
            f[len("enrolled_"):-len(".npz")] for f in os.listdir(AUTOCAR_MODELS_DIR)
            if f.startswith("enrolled_") and f.endswith(".npz")
        )
    return sorted((get_status(name) for name in names), key=lambda p: p.name)


def get_status(name: str) -> PersonStatus:
    """READ (one)."""
    return PersonStatus(
        name=name,
        raw_front_count=_count_jpgs(_raw_dir(name, "front")),
        raw_back_count=_count_jpgs(_raw_dir(name, "back")),
        cropped_front_count=_count_jpgs(_cropped_dir(name, "front")),
        cropped_back_count=_count_jpgs(_cropped_dir(name, "back")),
        has_face_registry=os.path.exists(os.path.join(FACE_REGISTRY_DIR, f"{name}.npz")),
        has_target_profile=os.path.exists(os.path.join(AUTOCAR_MODELS_DIR, f"enrolled_{name}.npz")),
    )


def ensure_capture_dirs(name: str) -> None:
    for phase in ("front", "back"):
        os.makedirs(_raw_dir(name, phase), exist_ok=True)
        os.makedirs(_cropped_dir(name, phase), exist_ok=True)


def reset_captures(name: str) -> None:
    """Removes any previously captured RAW and CROPPED frames for `name` before a fresh capture
    session starts (confirmed with the user) — every registration run, whether it's a brand-new
    person or a re-capture of an existing one, always begins from a clean slate rather than mixing
    old and new photos together. Does NOT touch the already-BUILT registry files — those are only
    overwritten once rebuild_registries() actually succeeds, so a session that fails partway never
    destroys the last known-good profile."""
    shutil.rmtree(os.path.join(CAPTURES_DIR, name), ignore_errors=True)
    ensure_capture_dirs(name)


def save_raw_capture(name: str, phase: str, frame: np.ndarray) -> str:
    """CREATE, phase 1 of 2 — persists one RAW camera frame, exactly as captured, no cropping
    and no detection. `phase` is "front" or "back"."""
    directory = _raw_dir(name, phase)
    index = _count_jpgs(directory) + 1
    path = os.path.join(directory, f"{index:03d}.jpg")
    cv2.imwrite(path, frame)
    return path


def build_cropped_roi(name: str, phase: str, roi_percent) -> int:
    """CREATE, phase 2 of 2 — reads every RAW image for this phase back off disk, crops each to
    `roi_percent` (registration_overlay.crop_to_roi, the same pure crop math the live preview
    draws a box for), and saves the result as its own file under the CROPPED folder — a real,
    inspectable artifact, not a value computed and discarded. Returns how many were written.
    Overwrites any previous cropped output for this phase (RAW is the source of truth)."""
    raw_dir, cropped_dir = _raw_dir(name, phase), _cropped_dir(name, phase)
    shutil.rmtree(cropped_dir, ignore_errors=True)
    os.makedirs(cropped_dir, exist_ok=True)

    written = 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jpg"))):
        image = cv2.imread(path)
        if image is None:
            continue
        cropped = overlay.crop_to_roi(image, roi_percent)
        if cropped.size == 0:
            continue
        cv2.imwrite(os.path.join(cropped_dir, os.path.basename(path)), cropped)
        written += 1
    return written


def delete_person(name: str) -> None:
    """DELETE — removes RAW + CROPPED captures + both built registry files. Irreversible;
    register_person.py must confirm with the operator before calling this."""
    shutil.rmtree(os.path.join(CAPTURES_DIR, name), ignore_errors=True)
    face_path = os.path.join(FACE_REGISTRY_DIR, f"{name}.npz")
    if os.path.exists(face_path):
        os.remove(face_path)
    profile_path = os.path.join(AUTOCAR_MODELS_DIR, f"enrolled_{name}.npz")
    if os.path.exists(profile_path):
        os.remove(profile_path)


def build_face_registry(name: str, config_path: str = "config/thresholds.yaml") -> bool:
    """Shells out to modules/face_identity/build_face_registry.py, reused completely unchanged,
    pointed at this person's CROPPED front frames (not raw — see module docstring)."""
    print("\nBuilding face_identity registry entry...")
    result = subprocess.run([
        sys.executable, "-m", "modules.face_identity.build_face_registry",
        name, "--images-dir", _cropped_dir(name, "front"), "--config", config_path,
    ])
    return result.returncode == 0


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def build_target_profile(name: str) -> bool:
    """Builds modules/autocar/models/enrolled_<name>.npz from this person's CROPPED front/back
    images (not raw — see module docstring) — mirrors modules/autocar/scripts/enroll_person.py's
    own composite-and-save logic exactly, just image-driven instead of a live loop. Reaches into
    modules/autocar's vendored identity/ code via autocar_bootstrap, the same bridge
    autocar_adapter.py already uses — modules/autocar/ itself is never touched by this.

    This is where ALL detection for the target profile happens — capture and cropping both run no
    detection at all (see register_person.py / build_cropped_roi above). Every detection in an
    image is a candidate; the LARGEST bbox wins (confirmed with the user) — since each image is
    already cropped to the operator-configured ROI the subject was asked to stand inside, the
    subject is the dominant figure in frame, and any smaller detection is background the ROI crop
    didn't fully exclude, not a second person to reject the whole image over."""
    print("\nBuilding autocar re-id profile...")
    autocar_bootstrap.ensure_on_path()
    from detector.yolov8_pose_torch import YOLOv8PoseTorch  # noqa: E402
    from identity import face_region, pose_gate  # noqa: E402
    from identity.osnet_embedder import OSNetEmbedder  # noqa: E402
    from identity.target_profile import save_target_profile  # noqa: E402

    detector = YOLOv8PoseTorch()
    embedder = OSNetEmbedder(f"{AUTOCAR_MODELS_DIR}/osnet_x1_0_msmt17.onnx")

    def _collect(directory: str, want_face_visible: bool):
        head_embeddings, lower_embeddings, aspect_ratios = [], [], []
        for path in sorted(glob.glob(os.path.join(directory, "*.jpg"))):
            image = cv2.imread(path)
            if image is None:
                continue
            detections = detector.detect(image)
            if not detections:
                continue
            det = max(detections, key=lambda d: _bbox_area(d.bbox))  # largest bbox = the subject
            if face_region.is_face_visible(det.keypoints) != want_face_visible:
                continue
            frame_h, frame_w = image.shape[:2]
            head_crop, lower_crop = face_region.crop_head_lower(image, det.bbox, det.keypoints, frame_w, frame_h)
            if head_crop.size == 0:
                continue
            head_embeddings.append(embedder.extract(head_crop))
            if lower_crop.size > 0:
                lower_embeddings.append(embedder.extract(lower_crop))
            aspect_ratios.append(pose_gate.aspect_ratio_from_bbox(det.bbox))
        return head_embeddings, lower_embeddings, aspect_ratios

    front_head, front_lower, front_ratios = _collect(_cropped_dir(name, "front"), want_face_visible=True)
    back_head, _, _ = _collect(_cropped_dir(name, "back"), want_face_visible=False)

    if not front_head or not back_head:
        print(f"FAILED: usable samples front={len(front_head)} back={len(back_head)} (need >= 1 each).")
        return False

    def _composite(vectors):
        vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else vec

    head_composite = _composite(front_head)
    back_head_composite = _composite(back_head)
    lower_composite = _composite(front_lower) if front_lower else np.zeros_like(head_composite)
    median_ratio = float(np.median(front_ratios)) if front_ratios else 1.0

    os.makedirs(AUTOCAR_MODELS_DIR, exist_ok=True)
    out_path = f"{AUTOCAR_MODELS_DIR}/enrolled_{name}.npz"
    save_target_profile(out_path, head_composite, lower_composite, median_ratio,
                         len(front_head), back_head_embedding=back_head_composite)
    print(f"Saved {len(front_head)} front + {len(back_head)} back samples -> '{out_path}'")
    return True


def rebuild_registries(name: str, config_path: str = "config/thresholds.yaml") -> bool:
    """UPDATE (and the tail end of Create) — regenerates BOTH registry files from whatever CROPPED
    images currently exist for `name`."""
    face_ok = build_face_registry(name, config_path)
    target_ok = build_target_profile(name)
    return face_ok and target_ok
