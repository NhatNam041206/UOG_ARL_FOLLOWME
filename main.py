"""
Entry point for running the project's modules against either a live camera or a recorded video.
Imports ONLY from each module's public interface.py — all camera/video input handling and
module selection/wiring lives here in the root entry point, not inside any module itself.

--modules selects which pipeline runs per frame:
    estop        Emergency Stop only (full-frame runway/collision logic).
    wave_facing  Human detection -> per-person Wave Gesture + Facing-Camera Gate (the ORIGINAL
                 demo pipeline, whole-frame human detection, no identity verification).
    both         Both of the above, independently, on the SAME frame each iteration.
    face_first   The face-first exploratory pipeline from plans/01-04: face detect+match
                 (modules.face_identity) -> ROI-scoped human detection (modules.human_detection_roi)
                 -> ONE of three interchangeable gesture methods (--gesture-method), per matched
                 person. TRIGGER = registered_person (implied True, a face already matched) AND
                 is_waving from the chosen gesture method. This is the pipeline plans/01-04 describe.

Bbox color for wave_facing/both (only drawn with --show): GREEN once both is_waving and
is_facing_camera are GREEN (confirmed), YELLOW while either signal is still building, RED
otherwise. For face_first: GREEN means TRIGGER=True, YELLOW means confirmation is building, RED
otherwise.

Standalone single-module test/visualization scripts (all support --show, some chain live with
their own upstream modules where the module's own spec calls for it — see each script's
docstring): modules/emergency_stop/test_estop.py, modules/human_detection/test_human_detection.py,
modules/wave_facing_gate/test_wave_facing.py, modules/face_identity/{test_face_identity,
visualize_face_identity}.py, modules/human_detection_roi/{test_human_detection_roi,
visualize_human_detection_roi}.py, modules/gesture_hand_keypoint/{test_gesture_hand_keypoint,
visualize_gesture_hand_keypoint}.py, modules/gesture_trajectory_verifier/
{test_gesture_trajectory_verifier,visualize_gesture_trajectory_verifier}.py. This file (main.py)
is the general runner that combines modules for actual multi-module/multi-person operation.

Usage:
    python main.py --mode camera --modules estop
    python main.py --mode camera --modules wave_facing --show --debug
    python main.py --mode camera --modules face_first --gesture-method hand_keypoint --show
    python main.py --mode video --video path.mp4 --modules face_first --gesture-method trajectory_verifier --show
"""
import argparse
import os
import sys
import time

import cv2
import yaml

from modules.emergency_stop.interface import EmergencyStopModule
from modules.human_detection.interface import HumanDetectionModule
from modules.wave_facing_gate.interface import WaveFacingGateModule

_ESTOP_COLOR = {"GO": (0, 200, 0), "STOP": (0, 0, 255), "UNCERTAIN": (0, 220, 255)}
_WAVE_STATE_COLOR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}


def wave_bbox_color(wave_result):
    """GREEN once both signals are confirmed, YELLOW while either is still building, else RED."""
    if wave_result.is_waving and wave_result.is_facing_camera:
        return _WAVE_STATE_COLOR["GREEN"]
    if "YELLOW" in (wave_result.waving_state, wave_result.facing_state):
        return _WAVE_STATE_COLOR["YELLOW"]
    return _WAVE_STATE_COLOR["RED"]


def load_camera_config(config_path: str = "config/thresholds.yaml") -> int:
    """Load camera_index from config/thresholds.yaml's camera section, defaulting to 0."""
    if not os.path.exists(config_path):
        return 0
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        camera_config = config.get("camera", {})
        return camera_config.get("camera_index", 0)
    except Exception:
        return 0


def draw_lines(frame, lines, start_y: int, color, line_height: int = 26) -> int:
    """Draws each string in `lines` on its own row starting at `start_y`; returns the y position
    just below the last line, so callers can stack multiple modules' overlays without overlap."""
    y = start_y
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += line_height
    return y


def open_capture(args: argparse.Namespace):
    if args.mode == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"camera index {args.camera_index}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = f"video '{args.video}'"
    return cap, source_desc


class _GestureMethodAdapter:
    """
    Normalizes the calling-convention difference between modules.wave_facing_gate ("condition" —
    Method 1, predates the plans' shared GestureMethodResult contract, has its own two-signal
    is_waving/is_facing_camera output) and modules.gesture_hand_keypoint / .gesture_trajectory_verifier
    ("hand_keypoint" / "trajectory_verifier" — Methods 2/3, share the plans' GestureMethodResult
    contract exactly). Lets run_face_first_pipeline() call any of the three the same way.
    """

    def __init__(self, method_name: str):
        self.method_name = method_name
        self._last_result = None  # stashed by evaluate(), consumed by draw_debug()
        if method_name == "condition":
            self._module = WaveFacingGateModule()
        elif method_name == "hand_keypoint":
            import modules.gesture_hand_keypoint.interface as gi
            self._module = gi
        elif method_name == "trajectory_verifier":
            import modules.gesture_trajectory_verifier.interface as gi
            self._module = gi
        else:
            raise ValueError(f"Unknown gesture method '{method_name}'")

    def evaluate(self, track_id: int, crop, timestamp: float, person_bbox_full_frame=None):
        """Returns (is_waving, waving_state, extra_debug_label). `person_bbox_full_frame` is
        only used by hand_keypoint (its palm-height gate needs the person's full-frame bbox,
        not just the crop) — ignored by the other two methods. Also stashes the raw result
        object for draw_debug() below, since the tuple return here is print/state-only."""
        if self.method_name == "condition":
            r = self._module.process_frame(track_id=track_id, crop=crop)
            extra = f"facing={r.facing_state} wave_arm={r.wave_arm}"
            self._last_result = r
            return r.is_waving, r.waving_state, extra

        if self.method_name == "hand_keypoint":
            r = self._module.evaluate(track_id, crop, timestamp, person_bbox_full_frame=person_bbox_full_frame)
            extra = f"conf={r.confidence_debug} stage={r.sequence_stage} palm_facing={r.palm_facing_camera_debug}"
        else:  # trajectory_verifier
            r = self._module.evaluate(track_id, crop, timestamp)
            extra = f"conf={r.confidence_debug} ref={r.matched_reference_id} arm={r.arm} refs={r.reference_count}"
        self._last_result = r
        return r.is_waving, r.waving_state, extra

    def draw_debug(self, crop, person_bbox_full_frame=None) -> None:
        """Draws the last evaluate() call's per-method debug overlay directly onto `crop` (a view
        into the caller's frame) — same overlay each method's own standalone visualize_*.py
        script draws. No-ops if the method has no draw_debug (gesture_trajectory_verifier
        doesn't define one yet) or nothing has been evaluated this track yet."""
        if self._last_result is None or not hasattr(self._last_result, "draw_debug"):
            return
        if self.method_name == "hand_keypoint":
            self._last_result.draw_debug(crop, person_bbox_full_frame=person_bbox_full_frame)
        else:
            self._last_result.draw_debug(crop)

    def release_track(self, track_id: int) -> None:
        if self.method_name == "condition":
            self._module.reset_track(track_id)
        else:
            self._module.release_track(track_id)


def run_legacy_pipeline(cap, args: argparse.Namespace, source_desc: str) -> int:
    """--modules estop | wave_facing | both — the original whole-frame-detection demo pipeline,
    no identity verification."""
    run_estop = args.modules in ("estop", "both")
    run_wave_facing = args.modules in ("wave_facing", "both")

    estop = EmergencyStopModule() if run_estop else None
    human_detector = HumanDetectionModule() if run_wave_facing else None
    wave_facing = WaveFacingGateModule() if run_wave_facing else None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            line = f"frame={frame_idx:06d}"
            overlay_y = 30

            if estop is not None:
                estop_output = estop.process_frame(frame)
                line += (
                    f" estop.decision={estop_output.decision.value:9s} "
                    f"estop.reason={estop_output.reason:28s} "
                    f"estop.track_id={str(estop_output.triggering_track_id):>6s} "
                    f"estop.zone={str(estop_output.zone):5s} "
                    f"estop.latency_ms={estop.last_latency_ms:6.1f}"
                )
                if args.show:
                    color = _ESTOP_COLOR[estop_output.decision.value]
                    overlay_y = draw_lines(frame, [
                        f"ESTOP: {estop_output.decision.value} ({estop_output.reason})",
                        f"  track_id={estop_output.triggering_track_id} zone={estop_output.zone}",
                    ], overlay_y, color)

            if human_detector is not None and wave_facing is not None:
                detections = human_detector.detect(frame)
                line += f" people={len(detections):2d} human_detection.latency_ms={human_detector.last_latency_ms:6.1f}"
                if args.show:
                    overlay_y = draw_lines(frame, [f"PEOPLE: {len(detections)} detected"], overlay_y, (255, 255, 255))

                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]  # a VIEW into frame — drawing on it updates frame in place

                    wave_result = wave_facing.process_frame(track_id=det.track_id, crop=crop)
                    line += (
                        f" | person(track_id={det.track_id} bbox=({x1},{y1},{x2},{y2}) conf={det.confidence:.2f} "
                        f"is_waving={wave_result.is_waving} is_facing={wave_result.is_facing_camera} "
                        f"waving_state={wave_result.waving_state} facing_state={wave_result.facing_state} "
                        f"wave_arm={wave_result.wave_arm} facing_conf_min={wave_result.facing_confidence_min} "
                        f"wave.latency_ms={wave_facing.last_latency_ms:.1f})"
                    )

                    if args.show:
                        color = wave_bbox_color(wave_result)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"id={det.track_id} wave={wave_result.waving_state} facing={wave_result.facing_state}"
                        cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if args.debug:
                            wave_result.draw_debug(crop)

            print(line)

            if args.show:
                cv2.imshow("main", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


def run_face_first_pipeline(cap, args: argparse.Namespace, source_desc: str) -> int:
    """
    --modules face_first — the exploratory pipeline from plans/01-04:
        full frame -> face detect+match -> ROI-scoped human detection -> chosen gesture method
    TRIGGER = registered_person (a face already matched a registry entry) AND is_waving from the
    chosen gesture method. Multiple registered people in the same frame are all evaluated
    independently (face_identity.evaluate() already returns a list, spec §1: it does not pick
    "the" person).
    """
    from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
    from modules.human_detection_roi.interface import evaluate as evaluate_person

    face_registry = FaceRegistry(args.face_registry_dir)
    gesture = _GestureMethodAdapter(args.gesture_method)
    active_track_ids = set()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()
            frame_h, frame_w = frame.shape[:2]

            face_results = [r for r in evaluate_face(frame, face_registry) if r.is_registered_match]
            line = f"frame={frame_idx:06d} faces_matched={len(face_results)}"
            seen_track_ids = set()

            for face in face_results:
                person = evaluate_person(frame, face.face_bbox)
                if not person.person_found:
                    line += f" | {face.matched_person_name}: person_not_found"
                    continue

                px, py, pw, ph = person.person_bbox
                px, py = max(0, px), max(0, py)
                pw = min(pw, frame_w - px)
                ph = min(ph, frame_h - py)
                if pw <= 0 or ph <= 0:
                    continue
                crop = frame[py:py + ph, px:px + pw]  # a VIEW into frame — drawing on it updates frame in place

                # human_detection_roi is ROI-scoped, stateless per call — no persistent
                # track_id (a per-frame-shifting ROI crop doesn't give ByteTrack a stable
                # coordinate frame to track against). Key gesture-method state off the
                # registered person's name instead — stable across frames on its own, since the
                # same name only ever maps to one physical person.
                track_id = abs(hash(face.matched_person_name)) % 100000
                seen_track_ids.add(track_id)
                is_waving, waving_state, extra = gesture.evaluate(
                    track_id, crop, timestamp, person_bbox_full_frame=(px, py, pw, ph),
                )
                trigger = is_waving  # registered_person already implied True (face matched above)

                line += (
                    f" | {face.matched_person_name}: is_waving={is_waving} state={waving_state} "
                    f"TRIGGER={trigger} bbox=({px},{py},{pw},{ph}) ({extra})"
                )

                if args.show:
                    if args.debug:
                        gesture.draw_debug(crop, person_bbox_full_frame=(px, py, pw, ph))
                    color = _WAVE_STATE_COLOR["GREEN"] if trigger else (
                        _WAVE_STATE_COLOR["YELLOW"] if waving_state == "YELLOW" else _WAVE_STATE_COLOR["RED"]
                    )
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
                    label = f"{face.matched_person_name}: {waving_state}" + ("  TRIGGER!" if trigger else "")
                    cv2.putText(frame, label, (px, max(15, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Release gesture-method state for tracks that dropped out this frame (face no
            # longer matched/found) — bounds memory, mirrors each method's own reset_track/
            # release_track hygiene hook.
            for stale_id in active_track_ids - seen_track_ids:
                gesture.release_track(stale_id)
            active_track_ids = seen_track_ids

            print(line)

            if args.show:
                cv2.imshow("main (face-first pipeline)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run project modules against a live camera or a recorded video."
    )
    parser.add_argument(
        "--mode", choices=["camera", "video"], required=True,
        help="Input source: 'camera' for a live webcam, 'video' for a recorded file.",
    )
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")

    config_camera_index = load_camera_config("config/thresholds.yaml")
    parser.add_argument(
        "--camera-index", type=int, default=config_camera_index,
        help=f"OS camera device index (default {config_camera_index} from config, or 0 if not set). Used when --mode camera.",
    )
    parser.add_argument(
        "--modules", choices=["estop", "wave_facing", "both", "face_first"], default="estop",
        help="Which pipeline to run: 'estop'/'wave_facing'/'both' are the original whole-frame "
             "demo pipeline; 'face_first' is the face-first exploratory pipeline from "
             "plans/01-04 (requires --gesture-method). Default: estop.",
    )
    parser.add_argument(
        "--gesture-method", choices=["condition", "hand_keypoint", "trajectory_verifier"],
        help="Which gesture method the face_first pipeline uses: 'condition' = Method 1 "
             "(modules.wave_facing_gate), 'hand_keypoint' = Method 2, 'trajectory_verifier' = "
             "Method 3. Required when --modules face_first.",
    )
    parser.add_argument(
        "--face-registry-dir", default="modules/face_identity/registry_data",
        help="Path to the face_identity registry directory. Used when --modules face_first.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable per-person debug overlay (keypoints/skeleton/gate state), drawn by whichever "
             "module is active: wave_facing's own pose debug for --modules wave_facing/both, or the "
             "chosen --gesture-method's draw_debug() for --modules face_first (no-op for "
             "trajectory_verifier, which doesn't define one).",
    )
    parser.add_argument("--show", action="store_true", help="Display frames in a window while processing.")
    args = parser.parse_args()

    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")
    if args.modules == "face_first" and not args.gesture_method:
        parser.error("--gesture-method is required when --modules face_first")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    if args.modules == "face_first":
        return run_face_first_pipeline(cap, args, source_desc)
    return run_legacy_pipeline(cap, args, source_desc)


if __name__ == "__main__":
    sys.exit(main())
