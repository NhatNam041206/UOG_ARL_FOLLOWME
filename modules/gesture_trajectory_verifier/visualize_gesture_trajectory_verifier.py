"""
Mandatory standalone visualization entry point for gesture_trajectory_verifier (spec §7).
Chained LIVE with modules.face_identity + modules.human_detection_roi (both already built in
this repo) — consumes only their public evaluate()/dataclass outputs.

Draws, per frame, per person:
  - The live wrist trajectory (both arms) directly on the person crop, as it accumulates
  - A separate small inset panel plotting the NORMALIZED live wrist path against the
    best-matching reference trajectory's wrist path, side by side in a shared relative-scale
    space (spec: "the most valuable debug view for this method, since 'does the shape look
    similar' is the whole point") — only the wrist sub-path is drawn in the inset for legibility,
    even though matching itself compares all three points (wrist+elbow+shoulder)
  - The current best similarity score and which reference/arm produced it
Logs GestureMethodResult fields to console per frame.

Usage:
    python -m modules.gesture_trajectory_verifier.visualize_gesture_trajectory_verifier --mode camera [--camera-index 0]
    python -m modules.gesture_trajectory_verifier.visualize_gesture_trajectory_verifier --mode video --video path.mp4
"""
import argparse
import sys
import time

import cv2
import numpy as np

from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
from modules.gesture_trajectory_verifier.config import load_config
from modules.gesture_trajectory_verifier.normalization import normalize_trajectory
from modules.gesture_trajectory_verifier.pipeline import GestureTrajectoryVerifierPipeline
from modules.gesture_trajectory_verifier.reference_store import ReferenceTrajectoryStore
from modules.human_detection_roi.interface import evaluate as evaluate_person

_STATE_COLOR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}
_LIVE_COLOR = (255, 150, 0)
_REF_COLOR = (0, 200, 255)
_INSET_SIZE = 220


def draw_inset(live_wrist_points, ref_wrist_points):
    """Plots two normalized (translated-to-own-start, scale-divided) wrist paths in a shared
    fixed-size panel, auto-scaled to fit both. Returns a BGR image."""
    panel = np.full((_INSET_SIZE, _INSET_SIZE, 3), 30, dtype=np.uint8)
    cv2.line(panel, (_INSET_SIZE // 2, 0), (_INSET_SIZE // 2, _INSET_SIZE), (70, 70, 70), 1)
    cv2.line(panel, (0, _INSET_SIZE // 2), (_INSET_SIZE, _INSET_SIZE // 2), (70, 70, 70), 1)

    all_points = list(live_wrist_points) + list(ref_wrist_points)
    if not all_points:
        return panel
    max_extent = max(max(abs(x), abs(y)) for x, y in all_points) or 1.0
    scale = (_INSET_SIZE * 0.4) / max_extent

    def _to_px(pt):
        x, y = pt
        return (int(_INSET_SIZE // 2 + x * scale), int(_INSET_SIZE // 2 + y * scale))

    for color, points in ((_LIVE_COLOR, live_wrist_points), (_REF_COLOR, ref_wrist_points)):
        px_points = [_to_px(p) for p in points]
        for i in range(len(px_points) - 1):
            cv2.line(panel, px_points[i], px_points[i + 1], color, 2)
        for p in px_points:
            cv2.circle(panel, p, 2, color, -1)

    cv2.putText(panel, "live", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LIVE_COLOR, 1)
    cv2.putText(panel, "reference", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _REF_COLOR, 1)
    return panel


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone gesture_trajectory_verifier visualization/debugging tool.")
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
    gesture_config = load_config(args.config)
    reference_store = ReferenceTrajectoryStore(gesture_config.reference_dir)
    # Own pipeline instance (not the module-level singleton behind interface.evaluate()) so this
    # debug tool can read internal per-arm buffer state for drawing the live trajectory — a
    # reach-in that's fine here since this file lives inside the module's own package.
    gesture_pipeline = GestureTrajectoryVerifierPipeline(gesture_config)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            face_results = [r for r in evaluate_face(frame, face_registry) if r.is_registered_match]
            for face in face_results:
                person = evaluate_person(frame, face.face_bbox)
                if not person.person_found:
                    continue
                px, py, pw, ph = person.person_bbox
                px, py = max(0, px), max(0, py)
                crop = frame[py:py + ph, px:px + pw]
                if crop.size == 0:
                    continue

                track_id = abs(hash(face.matched_person_name)) % 100000  # stand-in track_id for this debug tool
                raw = gesture_pipeline.evaluate(track_id, crop, timestamp)
                gesture_state = raw.waving_state
                color = _STATE_COLOR.get(gesture_state, (255, 255, 255))

                cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
                label1 = f"{face.matched_person_name}: state={raw.waving_state} arm={raw.arm}"
                label2 = f"score={raw.confidence_debug} ref={raw.matched_reference_id} refs_available={raw.reference_count}"
                cv2.putText(frame, label1, (px, max(15, py - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label2, (px, max(15, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(
                    f"frame={frame_idx:06d} track_id={track_id} is_waving={raw.is_waving} "
                    f"waving_state={raw.waving_state} confidence_debug={raw.confidence_debug} "
                    f"matched_reference_id={raw.matched_reference_id} arm={raw.arm} "
                    f"reference_count={raw.reference_count}"
                )

                # Live wrist path: read the internal per-arm buffer for whichever arm the
                # comparison used (or 'left' by default if none matched yet), normalized the
                # same way the comparison itself normalizes it, for a fair visual comparison.
                track_state = gesture_pipeline._tracks.get(track_id)
                live_points = []
                if track_state is not None:
                    arm_key = raw.arm or "left"
                    buffer = track_state.motion_buffers.get(arm_key)
                    if buffer and len(buffer.samples) >= 2:
                        normalized = normalize_trajectory(buffer.samples, float(ph))
                        live_points = [s.wrist for s in normalized]
                        for i in range(len(live_points) - 1):
                            p1 = (int(px + buffer.samples[i].wrist[0]), int(py + buffer.samples[i].wrist[1]))
                            p2 = (int(px + buffer.samples[i + 1].wrist[0]), int(py + buffer.samples[i + 1].wrist[1]))
                            cv2.line(frame, p1, p2, _LIVE_COLOR, 2)

                # Inset: normalized live wrist path vs the best-matching reference's wrist
                # sub-path, if one was found — the spec's own "most valuable debug view".
                ref_points = []
                if raw.matched_reference_id is not None:
                    ref_entries = {r.reference_id: r for r in reference_store.load_all()}
                    ref = ref_entries.get(raw.matched_reference_id)
                    if ref is not None:
                        ref_points = [tuple(p) for p in ref.flat_vector.reshape(-1, 6)[:, 0:2]]
                if live_points or ref_points:
                    inset = draw_inset(live_points, ref_points)
                    ih, iw = inset.shape[:2]
                    if frame.shape[0] >= ih and frame.shape[1] >= iw:
                        frame[0:ih, frame.shape[1] - iw:frame.shape[1]] = inset

            cv2.imshow("visualize_gesture_trajectory_verifier", frame)
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
