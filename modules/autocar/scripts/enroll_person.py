"""
Enroll ONE person's appearance for main.py --target re-identification.

Two phases, back to back:
  1. FRONT - stand inside the on-screen ROI box, press SPACE to start a countdown, then hold
     still, facing the camera (the face must be visible - see identity/face_region.py), while it
     collects samples over config.ENROLL_DURATION_SEC seconds.
  2. BACK - turn around, back to the camera, press SPACE to start a second countdown, then hold
     still while it collects back-of-head samples over the same duration. This lets main.py
     --target still recognize the person while they're walking away with their back to the
     camera, instead of only from the front.

Saves averaged front-head, back-of-head, and lower-body OSNet embeddings + aspect ratio to
models/enrolled_<name>.npz.

    python scripts/enroll_person.py alice --source 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import config
from detector.yolov8_pose_torch import YOLOv8PoseTorch
from identity import face_region, pose_gate
from identity.osnet_embedder import OSNetEmbedder
from identity.target_profile import save_target_profile, sanitize_person_name
from utils.video_source import VideoSource

WINDOW_NAME = "Enroll Target"


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll one person's appearance for --target re-identification")
    parser.add_argument("name", help="Person's name - used as the output filename")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 (Jetson Nano target is cpu)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def _roi_box(frame_w: int, frame_h: int):
    x1p, y1p, x2p, y2p = config.ENROLL_ROI_PERCENT
    return int(x1p * frame_w), int(y1p * frame_h), int(x2p * frame_w), int(y2p * frame_h)


def _bbox_center_in_roi(bbox, rx1, ry1, rx2, ry2) -> bool:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def main():
    args = parse_args()
    name = sanitize_person_name(args.name)
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Loading detector + OSNet embedder (device={args.device})...")
    detector = YOLOv8PoseTorch(device=args.device)
    embedder = OSNetEmbedder(config.REID_MODEL_PATH, device=args.device)
    video = VideoSource(source, width=args.width, height=args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(f"Enrolling '{name}'. Phase 1/2: FRONT - stand inside the yellow box facing the "
          f"camera, press SPACE to start ('q' to cancel).")

    state = "waiting_front"  # waiting_front -> countdown_front -> collecting_front
    # -> waiting_back -> countdown_back -> collecting_back
    countdown_start = 0.0
    collect_start = 0.0
    frame_idx = 0
    last_sample_frame_idx = -config.ENROLL_SAMPLE_INTERVAL_FRAMES

    head_embeddings, lower_embeddings, aspect_ratios = [], [], []
    back_head_embeddings = []
    cancelled = False

    try:
        for frame in video:
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rx1, ry1, rx2, ry2 = _roi_box(frame_w, frame_h)

            display = frame.copy()
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

            if state == "waiting_front":
                cv2.putText(display, "FRONT: face the camera, stand in box, press SPACE",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif state == "waiting_back":
                cv2.putText(display, "BACK: turn around (back to camera), press SPACE",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif state in ("countdown_front", "countdown_back"):
                remaining = config.ENROLL_COUNTDOWN_SEC - (time.time() - countdown_start)
                if remaining <= 0:
                    state = "collecting_front" if state == "countdown_front" else "collecting_back"
                    collect_start = time.time()
                else:
                    cv2.putText(display, f"Starting in {remaining:.1f}s", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            elif state == "collecting_front":
                elapsed = time.time() - collect_start
                cv2.putText(display, f"Collecting FRONT... {elapsed:.1f}/{config.ENROLL_DURATION_SEC}s "
                                      f"({len(head_embeddings)} samples)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if frame_idx - last_sample_frame_idx >= config.ENROLL_SAMPLE_INTERVAL_FRAMES:
                    last_sample_frame_idx = frame_idx
                    detections = detector.detect(frame)
                    roi_dets = [d for d in detections if _bbox_center_in_roi(d.bbox, rx1, ry1, rx2, ry2)]

                    if len(roi_dets) == 1:
                        det = roi_dets[0]
                        if not face_region.is_face_visible(det.keypoints):
                            print(f"  skip frame {frame_idx}: face not visible - look at the camera")
                        else:
                            head_crop, lower_crop = face_region.crop_head_lower(
                                frame, det.bbox, det.keypoints, frame_w, frame_h
                            )
                            if head_crop.size > 0 and lower_crop.size > 0:
                                head_embeddings.append(embedder.extract(head_crop))
                                lower_embeddings.append(embedder.extract(lower_crop))
                                aspect_ratios.append(pose_gate.aspect_ratio_from_bbox(det.bbox))
                    else:
                        print(f"  skip frame {frame_idx}: {len(roi_dets)} people in ROI (need exactly 1)")

                if elapsed >= config.ENROLL_DURATION_SEC:
                    state = "waiting_back"
                    print("Phase 1/2 done. Phase 2/2: BACK - turn around, back to the camera, "
                          "press SPACE ('q' to cancel).")

            elif state == "collecting_back":
                elapsed = time.time() - collect_start
                cv2.putText(display, f"Collecting BACK... {elapsed:.1f}/{config.ENROLL_DURATION_SEC}s "
                                      f"({len(back_head_embeddings)} samples)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if frame_idx - last_sample_frame_idx >= config.ENROLL_SAMPLE_INTERVAL_FRAMES:
                    last_sample_frame_idx = frame_idx
                    detections = detector.detect(frame)
                    roi_dets = [d for d in detections if _bbox_center_in_roi(d.bbox, rx1, ry1, rx2, ry2)]

                    if len(roi_dets) == 1:
                        det = roi_dets[0]
                        if face_region.is_face_visible(det.keypoints):
                            print(f"  skip frame {frame_idx}: face visible - turn all the way around")
                        else:
                            head_crop, _ = face_region.crop_head_lower(
                                frame, det.bbox, det.keypoints, frame_w, frame_h
                            )
                            if head_crop.size > 0:
                                back_head_embeddings.append(embedder.extract(head_crop))
                    else:
                        print(f"  skip frame {frame_idx}: {len(roi_dets)} people in ROI (need exactly 1)")

                if elapsed >= config.ENROLL_DURATION_SEC:
                    break

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cancelled = True
                break
            if key == ord(" ") and state == "waiting_front":
                state = "countdown_front"
                countdown_start = time.time()
            elif key == ord(" ") and state == "waiting_back":
                state = "countdown_back"
                countdown_start = time.time()

            frame_idx += 1
    finally:
        video.release()
        cv2.destroyAllWindows()

    if cancelled:
        print("Cancelled.")
        sys.exit(1)

    if len(head_embeddings) < config.ENROLL_MIN_SAMPLES:
        print(f"FAILED: only {len(head_embeddings)} valid FRONT samples collected "
              f"(need >= {config.ENROLL_MIN_SAMPLES}). Try again - better lighting, look at the "
              f"camera, or make sure only 1 person is inside the ROI box while collecting.")
        sys.exit(1)

    if len(back_head_embeddings) < config.ENROLL_MIN_SAMPLES:
        print(f"FAILED: only {len(back_head_embeddings)} valid BACK samples collected "
              f"(need >= {config.ENROLL_MIN_SAMPLES}). Try again - make sure you're fully turned "
              f"around (face not visible) and only 1 person is inside the ROI box while collecting.")
        sys.exit(1)

    def _composite(embeddings):
        vec = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else vec

    head_composite = _composite(head_embeddings)
    back_head_composite = _composite(back_head_embeddings)
    lower_composite = _composite(lower_embeddings)
    median_aspect_ratio = float(np.median(aspect_ratios))

    out_path = f"models/enrolled_{name}.npz"
    save_target_profile(out_path, head_composite, lower_composite, median_aspect_ratio,
                         len(head_embeddings), back_head_embedding=back_head_composite)
    print(f"Saved {len(head_embeddings)} front + {len(back_head_embeddings)} back samples -> "
          f"'{out_path}' (aspect_ratio={median_aspect_ratio:.3f})")
    print(f"Run: python main.py --source 0 --target {out_path}")


if __name__ == "__main__":
    main()
