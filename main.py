"""
Entry point for running the project's modules against either a live camera or a recorded video.
Imports ONLY from each module's public interface.py — all camera/video input handling and
module selection/wiring lives here in the root entry point, not inside any module itself.

--modules selects which pipeline runs per frame:
    pretrigger   The face-first exploratory pipeline from plans/01-04, STOPPING at the trigger:
                 face detect+match (modules.face_identity) -> ROI-scoped human detection
                 (modules.human_detection_roi) -> ONE of three interchangeable gesture methods
                 (--gesture-method), per matched person. TRIGGER = registered_person (implied
                 True, a face already matched) AND is_waving from the chosen gesture method.
                 Renamed from the original "face_first" (plans/01-04 still call it that) once
                 "followme" below started continuing PAST this same trigger point — this mode
                 exists for calibrating/testing just the pre-trigger stages in isolation, same
                 as it always did.
    followme     The FULL pipeline (plans/01-08): everything "pretrigger" does, but continuing
                 past the trigger into modules.followme_orchestrator — target_tracking (record +
                 follow), target_recovery (re-acquire on loss), and PID steering. Composes
                 modules.followme_orchestrator.interface (configure()/step()) rather than
                 re-implementing that sequencing here — see docs/architecture.md's isolation
                 exception note for why that module, not this file, owns the composition.
    register     Not a per-frame pipeline — hands off to register_person instead (see that file).
                 With --person-name: headless, registers just that one person (front-facing,
                 then turned around) and exits — builds BOTH the face_identity registry entry
                 and the autocar re-id enrollment profile. Without --person-name: opens
                 register_person.RegistrationApp, a Tkinter CRUD UI listing everyone
                 registered/in-progress, to register a new person OR pick/re-capture/delete an
                 existing one interactively. Ignores --mode/--show/--debug entirely
                 (--gesture-method only required to actually start following someone afterward).
                 Add --then-followme (headless path) to chain straight into camera followme mode
                 once registration succeeds; from the UI, select a fully-registered person and
                 click "Follow Me" to do the same thing interactively. Either way, one main.py
                 invocation covers register -> followme end to end, instead of two separate runs.

Bbox color for pretrigger (only drawn with --show): GREEN means TRIGGER=True, YELLOW means
confirmation is building, RED otherwise. For followme: GREEN/YELLOW/RED reflect
FollowMeCommand.debug_state (see modules/followme_orchestrator/visualize_followme_orchestrator.py's
own color table).

Standalone single-module test/visualization scripts (all support --show, some chain live with
their own upstream modules where the module's own spec calls for it — see each script's
docstring): modules/emergency_stop/test_estop.py, modules/human_detection/test_human_detection.py,
modules/wave_facing_gate/test_wave_facing.py, modules/face_identity/{test_face_identity,
visualize_face_identity}.py, modules/human_detection_roi/{test_human_detection_roi,
visualize_human_detection_roi}.py, modules/gesture_hand_keypoint/{test_gesture_hand_keypoint,
visualize_gesture_hand_keypoint}.py, modules/gesture_trajectory_verifier/
{test_gesture_trajectory_verifier,visualize_gesture_trajectory_verifier}.py,
modules/appearance_verifier/{test_appearance_verifier,visualize_appearance_verifier}.py,
modules/target_tracking/{test_target_tracking,visualize_target_tracking}.py,
modules/target_recovery/{test_target_recovery,visualize_target_recovery}.py,
modules/followme_orchestrator/{test_followme_orchestrator,visualize_followme_orchestrator}.py.
This file (main.py) is the general runner that combines modules for actual multi-module/
multi-person operation.

Usage:
    python main.py
        # ^ the simplest form — --modules defaults to "register", opening the Tkinter UI to
        # register a new person or pick/re-capture/delete an existing one. Pass --gesture-method
        # too if you want to use the UI's "Follow Me" button afterward (see below).
    python main.py --mode camera --modules pretrigger --gesture-method hand_keypoint --show
    python main.py --mode camera --modules followme --gesture-method hand_keypoint --show
    python main.py --mode video --video path.mp4 --modules followme --gesture-method trajectory_verifier --show
    python main.py --modules register --person-name Nam --camera-index 0
    python main.py --modules register --person-name Nam --then-followme --gesture-method hand_keypoint --show
    python main.py --modules register --gesture-method hand_keypoint --show
"""
import argparse
import os
import sys
import time

import cv2
import yaml

from modules.wave_facing_gate.interface import WaveFacingGateModule

_WAVE_STATE_COLOR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}


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
    contract exactly). Lets run_pretrigger_pipeline() call any of the three the same way.
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
        script draws. All three methods now define draw_debug(); this no-ops only if nothing has
        been evaluated for this track yet."""
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


def run_pretrigger_pipeline(cap, args: argparse.Namespace, source_desc: str) -> int:
    """
    --modules pretrigger — the exploratory pipeline from plans/01-04, stopping at the trigger:
        full frame -> face detect+match -> ROI-scoped human detection -> chosen gesture method
    TRIGGER = registered_person (a face already matched a registry entry) AND is_waving from the
    chosen gesture method. Multiple registered people in the same frame are all evaluated
    independently (face_identity.evaluate() already returns a list, spec §1: it does not pick
    "the" person). For the FULL pipeline that continues past this trigger, see
    run_followme_pipeline() below.
    """
    from modules.face_identity.interface import FaceRegistry, evaluate as evaluate_face
    from modules.human_detection_roi.interface import evaluate as evaluate_person

    face_registry = FaceRegistry(args.face_registry_dir)
    gesture = _GestureMethodAdapter(args.gesture_method)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()
            frame_h, frame_w = frame.shape[:2]

            face_results = [r for r in evaluate_face(frame, face_registry) if r.is_registered_match]
            line = f"mode=PRETRIGGER frame={frame_idx:06d} faces_matched={len(face_results)}"

            for face in face_results:
                person = evaluate_person(frame, face.face_bbox)

                if args.show and args.debug:
                    # Phase 1/2 overlay: face bbox + ROI region + person bbox, drawn even when
                    # person_found is False (the ROI itself is still useful to see) — the exact
                    # overlay modules/face_identity's and modules/human_detection_roi's own
                    # visualize_*.py scripts draw, reused here rather than re-implemented.
                    face.draw_debug(frame)
                    person.draw_debug(frame, face.face_bbox)

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
                        # Phase 3 overlay: gesture method's own keypoints/skeleton/state, drawn
                        # on top of phase 1/2's overlay above.
                        gesture.draw_debug(crop, person_bbox_full_frame=(px, py, pw, ph))
                    color = _WAVE_STATE_COLOR["GREEN"] if trigger else (
                        _WAVE_STATE_COLOR["YELLOW"] if waving_state == "YELLOW" else _WAVE_STATE_COLOR["RED"]
                    )
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
                    label = f"{face.matched_person_name}: {waving_state}" + ("  TRIGGER!" if trigger else "")
                    cv2.putText(frame, label, (px, max(15, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # No per-frame release_track() here (deliberately removed, not just relaxed): a
            # person briefly not matched/found (occlusion, one missed detection) is NOT the
            # same as "gone for good," and every gesture method's own per-track state is already
            # self-healing against real elapsed wall-clock time without any help from this loop
            # — Method 2's SequenceStateMachine resets itself via max_transition_gap_seconds the
            # next time evaluate() runs after a gap, Method 1's motion buffers and Method 3's
            # trajectory buffers both trim samples older than their own window on every call.
            # Eagerly releasing on the first missed frame was wiping that state out from under
            # those checks before they ever got to run — a bug, not the "hygiene" it looked like.
            # track_id itself is bounded by the face registry's size (one entry per REGISTERED
            # person, never per stranger), so leaving a track's state allocated indefinitely
            # isn't a real memory concern at this project's scale.
            print(line)

            if args.show:
                cv2.imshow("main (pretrigger pipeline)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames from {source_desc}.")
    return 0


_FOLLOWME_STATE_COLOR = {
    "WAITING_FOR_TRIGGER": (180, 180, 180),
    "TRACKING_STARTED": (0, 220, 255),
    "TRACKING": (0, 200, 0),
    "TRACKING_STEERING_UNCALIBRATED": (0, 160, 255),
    "RECOVERING": (0, 140, 255),
    "STOPPED": (0, 0, 255),
}


def run_followme_pipeline(cap, args: argparse.Namespace, source_desc: str) -> int:
    """
    --modules followme — the FULL pipeline (plans/01-08): everything run_pretrigger_pipeline()
    above does, continuing PAST the trigger into modules.followme_orchestrator — target_tracking
    (record + follow), target_recovery (re-acquire on loss), and PID steering.

    Composes modules.followme_orchestrator.interface (configure() once, then step() per frame)
    rather than re-implementing that sequencing here — that module is the one deliberate,
    documented exception to "only main.py composes across module boundaries"
    (docs/architecture.md design rule #2), built specifically to be this reusable, importable
    version of the pipeline. This function is a thin CLI wrapper around it, mirroring exactly
    what modules/followme_orchestrator/visualize_followme_orchestrator.py already does
    standalone, just reusing main.py's own --mode/--camera-index/--video plumbing instead of
    duplicating that argument handling a second time.

    No special TIMEOUT handling needed here — modules.followme_orchestrator already auto-resumes
    watching for a fresh trigger once a search episode times out (confirmed with the user): the
    very next step() call in this same per-frame loop does that on its own.

    --debug draws EVERY phase's overlay (face, ROI, gesture, tracking, recovery) via
    modules.followme_orchestrator.interface.draw_debug() — that function composes each composed
    module's OWN draw_debug(), so this file draws none of it itself. The steering-direction arrow
    (draw_steering_arrow()) is drawn separately, whenever --show is on, independent of --debug —
    it's the actual calculated robot command, not a per-phase debug readout.
    """
    from modules.followme_orchestrator.interface import configure, draw_debug, draw_steering_arrow, step

    configure(args.gesture_method, thresholds_config_path=args.config, face_registry_dir=args.face_registry_dir)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()

            command = step(frame, timestamp)
            color = _FOLLOWME_STATE_COLOR.get(command.debug_state, (255, 255, 255))
            angle_str = f"{command.steering_angle_degrees:+.1f}" if command.steering_angle_degrees is not None else "None"
            line = (
                f"mode=FOLLOWME frame={frame_idx:06d} debug_state={command.debug_state:28s} "
                f"should_move={command.should_move} steering_angle_degrees={angle_str}"
            )
            print(line)

            if args.show:
                if args.debug:
                    # Phase overlays (face/ROI/gesture/tracking/recovery), drawn first so the
                    # summary text below isn't occluded by it — same layering convention as
                    # run_pretrigger_pipeline.
                    draw_debug(frame)
                draw_lines(frame, [
                    f"state={command.debug_state}",
                    f"should_move={command.should_move}  steering={angle_str}deg",
                ], 30, color)
                # The calculated direction to the person — drawn regardless of --debug (it's the
                # actual robot command, not a per-phase debug readout); no-ops on its own when
                # should_move is False, so it simply doesn't appear while stopped.
                draw_steering_arrow(frame, command)
                cv2.imshow("main (followme pipeline)", frame)
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
        "--mode", choices=["camera", "video"],
        help="Input source: 'camera' for a live webcam, 'video' for a recorded file. Required "
             "for --modules pretrigger/followme; ignored (and not required) for --modules register.",
    )
    parser.add_argument("--video", help="Path to a recorded video file. Required when --mode video.")

    config_camera_index = load_camera_config("config/thresholds.yaml")
    parser.add_argument(
        "--camera-index", type=int, default=config_camera_index,
        help=f"OS camera device index (default {config_camera_index} from config, or 0 if not set). Used when --mode camera.",
    )
    parser.add_argument(
        "--modules", choices=["pretrigger", "followme", "register"], default="register",
        help="Which pipeline to run: 'pretrigger' is the face-first exploratory pipeline from "
             "plans/01-04, stopping at TRIGGER (requires --gesture-method); 'followme' is the "
             "FULL pipeline (plans/01-08) continuing past the trigger into tracking/recovery/"
             "steering via modules.followme_orchestrator (also requires --gesture-method); "
             "'register' (the default — the natural starting point, register-or-choose then "
             "follow) hands off to register_person instead of a per-frame pipeline (see that "
             "file, and --person-name above).",
    )
    parser.add_argument(
        "--gesture-method", choices=["condition", "hand_keypoint", "trajectory_verifier"],
        help="Which gesture method 'pretrigger'/'followme' use: 'condition' = Method 1 "
             "(modules.wave_facing_gate), 'hand_keypoint' = Method 2, 'trajectory_verifier' = "
             "Method 3. Required when --modules pretrigger or followme.",
    )
    parser.add_argument(
        "--face-registry-dir", default="modules/face_identity/registry_data",
        help="Path to the face_identity registry directory. Used when --modules pretrigger or followme.",
    )
    parser.add_argument(
        "--config", default="config/thresholds.yaml",
        help="Path to thresholds.yaml. Used when --modules followme (passed through to "
             "modules.followme_orchestrator.configure()) or --modules register.",
    )
    parser.add_argument(
        "--person-name",
        help="--modules register only: register just this one person headlessly, no UI. Omit "
             "entirely to open the Tkinter registration UI instead (list/register/re-capture/"
             "delete people interactively).",
    )
    parser.add_argument(
        "--front-samples", type=int, default=15,
        help="--modules register only: how many face-forward samples to capture.",
    )
    parser.add_argument(
        "--back-samples", type=int, default=15,
        help="--modules register only: how many turned-around samples to capture.",
    )
    parser.add_argument(
        "--then-followme", action="store_true",
        help="--modules register --person-name (headless path) only: on successful registration, "
             "immediately continue into camera followme mode (as if --mode camera --modules "
             "followme had been run right after). Requires --gesture-method. Skipped entirely if "
             "registration itself fails. The UI path (no --person-name) does this interactively "
             "instead, via its own \"Follow Me\" button.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable the full per-phase debug overlay: face bbox + ROI region + the chosen "
             "--gesture-method's keypoints/skeleton/state for --modules pretrigger; all of that "
             "PLUS target_tracking's bbox/center-line/reverify readout and target_recovery's "
             "search status for --modules followme (via "
             "modules.followme_orchestrator.draw_debug(), which composes each module's own "
             "draw_debug() — see docs/architecture.md).",
    )
    parser.add_argument("--show", action="store_true", help="Display frames in a window while processing.")
    args = parser.parse_args()

    if args.modules == "register":
        import register_person

        if args.person_name:
            # Headless path — register exactly this one person, no GUI.
            if args.then_followme and not args.gesture_method:
                parser.error("--gesture-method is required when --then-followme is set")

            result = register_person.run(
                args.person_name, args.camera_index, args.front_samples, args.back_samples, args.config,
            )
            if result != 0 or not args.then_followme:
                return result
            chosen_name = args.person_name
            print(f"\nRegistration succeeded — continuing into followme mode for '{chosen_name}'.")

        else:
            # No --person-name — open the Tkinter CRUD app instead: register a new person or
            # manage/re-capture/delete existing ones, then optionally click "Follow Me" on a
            # fully-registered person to hand off into the SAME followme loop below. Blocks here
            # until the app window closes; app.chosen_name is None unless "Follow Me" was used.
            app = register_person.RegistrationApp(
                args.camera_index, args.front_samples, args.back_samples, args.config,
                can_follow_me=bool(args.gesture_method),
            )
            app.mainloop()
            if not app.chosen_name:
                return 0  # operator managed data and closed the window — nothing more to do
            chosen_name = app.chosen_name
            print(f"\n'{chosen_name}' selected — continuing into followme mode.")

        # Fall through into the SAME followme camera loop below (args.mode/args.modules
        # rewritten so the existing dispatch code just below handles it identically to a fresh
        # `--mode camera --modules followme` run; the camera was already released before
        # returning here, either by register_person.run() or by closing the app, so reopening it
        # is safe). followme itself takes no person_name — whoever waves and matches the face
        # registry triggers it, same as always; chosen_name only got us to this point.
        args.mode = "camera"
        args.modules = "followme"

    if not args.mode:
        parser.error("--mode is required when --modules pretrigger or followme")
    if args.mode == "video" and not args.video:
        parser.error("--video is required when --mode video")
    if not args.gesture_method:
        parser.error(f"--gesture-method is required when --modules {args.modules}")

    cap, source_desc = open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: could not open {source_desc}", file=sys.stderr)
        return 1

    # "Initial" stage: one-time banner identifying which mode is about to run and its key
    # settings, before any per-frame logging starts — so a terminal log is never ambiguous about
    # which pipeline produced the lines that follow it.
    print(
        f"mode={args.modules.upper()} source={source_desc} "
        f"gesture_method={args.gesture_method} show={args.show} debug={args.debug}"
    )

    if args.modules == "pretrigger":
        return run_pretrigger_pipeline(cap, args, source_desc)
    return run_followme_pipeline(cap, args, source_desc)


if __name__ == "__main__":
    sys.exit(main())
