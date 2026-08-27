"""
Person registration UI — Layer 3 (display + interact) of a three-layer design:

    registration_data.py     Layer 1 — filesystem state, CRUD, building both registry files.
                              ALL face/person detection lives here too — see that file's
                              build_target_profile() docstring.
    registration_overlay.py  Layer 2 — pure frame-in/image-out drawing + the ROI crop, no I/O
                              whatsoever, no detection.
    register_person.py       Layer 3 (this file) — the ONLY place that reads the camera, opens a
                              window, or reads input. A Tkinter app owning the CRUD menu +
                              capture flow, calling into the other two layers for everything else.

CRUD, from the Tkinter app (run with no arguments):
    New          Create — capture a NEW person (front + back), then build both registry files.
    (the list)   Read   — every registered/in-progress person, with sample counts + build status.
    Re-capture   Update — re-captures the SELECTED person from scratch (see "remove the old
                 ones" below), then rebuilds both registry files.
    Delete       Delete — removes the selected person's raw captures AND both built registry
                 files. Requires confirming a dialog.

Also runnable non-interactively for one person, skipping the GUI entirely:
    python register_person.py <person_name> [--camera-index 0] [--front-samples 15] [--back-samples 15]
    python main.py --modules register --person-name <person_name> [--camera-index 0]
Both call run() below directly — same flow either way, still built on Layer 1/2, just a plain
cv2 window instead of the Tkinter app (used by main.py's --then-followme chain, where a second
GUI event loop competing with the followme camera loop right after would be unwelcome).

"Remove the old ones" (confirmed with the user): every capture session — New AND Re-capture
alike — starts by deleting that person's existing RAW and CROPPED photos first
(registration_data.reset_captures()), so a session never mixes old and new photos together. The
already-BUILT registry files are left alone until a rebuild actually SUCCEEDS at the end, so a
session that fails partway never destroys the last known-good profile.

Capture itself runs NO cropping and no IDENTITY detection at all — every RAW frame is saved
exactly as the camera produced it (registration_data.save_raw_capture()); the ROI box is only
drawn on screen so the operator can see where to stand, never applied to the saved file. It DOES
run one live, lightweight check while capturing: registration_data.LiveSubjectDetector counts how
many people have a bbox center inside the ROI this tick, and a frame is only accepted (saved) when
that count is exactly 1 — the ROI box is drawn green when accepted, yellow when 0 or 2+ people are
in view (confirmed with the user: closes the gap where a second/extra person in frame could
silently corrupt a sample). This is a person-COUNT check only, not identity — no face detection,
no embeddings computed here.

Once a phase's raw capture finishes, registration_data.build_cropped_roi() reads those RAW files
back and crops each one to the configured ROI (config/thresholds.yaml's register_person: section),
saving the result as its own separate, inspectable file — a real on-disk artifact under
registration_captures/<name>/cropped/ you can open and check BEFORE anything downstream ever runs
on it (confirmed with the user; the Tkinter flow pauses here for exactly that reason — see
CaptureWindow's "cropping" state below).
ALL IDENTITY detection — face detection for the face registry, pose/person detection for the re-id
profile (picking the LARGEST bbox in frame, then splitting head/lower) — happens afterward,
entirely inside the "data building" phase (registration_data.build_face_registry /
build_target_profile), reading the CROPPED images, never the raw ones. Three phases in order: RAW
capture (gated by a live person-count check) -> CROPPED ROI -> data building; identity detection
only ever runs in the third.

FRONT feeds both consumers (face registry + the re-id profile's front-head/lower-body
embeddings); BACK feeds only the re-id profile's back-of-head embedding. Once both registry files
build successfully, follow-me can begin for that person immediately — see main.py's
--then-followme flag, which chains straight from a successful registration into camera followme
mode without a second command.
"""
import argparse
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Optional, Tuple

import cv2
import yaml
from PIL import Image, ImageTk

import registration_data as data
import registration_overlay as overlay
from debug_stream import DebugStreamServer
from modules.face_identity.registry import sanitize_person_name
from run_logging import RunLogger

_CAPTURE_INTERVAL_SECONDS = 1.0  # gap between accepted samples, so they're not near-duplicate frames
_COUNTDOWN_SECONDS = 3.0
_PERSON_CHECK_INTERVAL_SECONDS = 0.2  # live person-count check cadence — throttled below the tick
# rate so the pose detector doesn't stutter the live preview (mirrors modules/autocar's own
# ENROLL_SAMPLE_INTERVAL_FRAMES-based throttling of its live per-frame checks)
_WINDOW_NAME = "register_person"  # cv2 window title, used only by the headless run() path below


def _load_roi_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        section = (yaml.safe_load(f) or {}).get("register_person", {}) or {}
    front_roi = section.get("front_roi_percent", [0.20, 0.05, 0.80, 0.95])
    back_roi = section.get("back_roi_percent", [0.15, 0.0, 0.85, 1.0])
    return front_roi, back_roi


# --------------------------------------------------------------------------------------------
# Headless CLI path — used by main.py (--modules register) and this file's own `<name>` argv form.
# --------------------------------------------------------------------------------------------

def _capture_phase_cli(cap, detector: data.LiveSubjectDetector, roi_percent, instruction: str,
                        name: str, phase: str, samples_needed: int, show: bool,
                        stream: Optional[DebugStreamServer], frame_idx: int,
                        logger: Optional[RunLogger]) -> "tuple[int, int]":
    """Returns (saved, frame_idx) — frame_idx is the running per-run() counter threaded through
    both phase calls (front then back), so decisions.jsonl frame numbers are unique/monotonic
    across the whole registration, not reset per phase (mirrors main.py's own frame_idx
    convention in run_pretrigger_pipeline/run_followme_pipeline)."""
    want_overlay = show or stream is not None  # draw even without a display if streaming
    start = time.time()
    while time.time() - start < _COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            return 0, frame_idx
        remaining = _COUNTDOWN_SECONDS - (time.time() - start)
        if want_overlay:
            frame = overlay.draw_countdown(frame, roi_percent, instruction, remaining)
        if stream is not None:
            stream.update_frame(frame)
        if logger is not None:
            logger.log_frame(frame=frame_idx, ts=time.time(), mode="register", stage="countdown",
                              person_name=name, phase=phase, saved=0, samples_needed=samples_needed,
                              person_count=None)
        frame_idx += 1
        if show:
            cv2.imshow(_WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early.")
                return 0, frame_idx

    saved, last_save_time, last_check_time, person_count = 0, 0.0, 0.0, 0
    while saved < samples_needed:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        if (now - last_check_time) >= _PERSON_CHECK_INTERVAL_SECONDS:
            person_count = detector.count_in_roi(frame, roi_percent)
            last_check_time = now
        if person_count == 1 and (now - last_save_time) >= _CAPTURE_INTERVAL_SECONDS:
            path = data.save_raw_capture(name, phase, frame)  # RAW — see module docstring
            saved += 1
            last_save_time = now
            print(f"  saved '{path}' ({saved}/{samples_needed})")
        if want_overlay:
            frame = overlay.draw_capture(frame, roi_percent, instruction, saved, samples_needed, person_count)
        if stream is not None:
            stream.update_frame(frame)
        if logger is not None:
            logger.log_frame(frame=frame_idx, ts=now, mode="register", stage="capture",
                              person_name=name, phase=phase, saved=saved, samples_needed=samples_needed,
                              person_count=person_count)
        frame_idx += 1
        if show:
            cv2.imshow(_WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early.")
                break
    return saved, frame_idx


def run(person_name: str, camera_index: int = 0, front_samples: int = 15,
        back_samples: int = 15, config_path: str = "config/thresholds.yaml",
        show: bool = False, stream: Optional[DebugStreamServer] = None,
        logger: Optional[RunLogger] = None, close_logger_on_exit: bool = True) -> int:
    """The headless entry point for registering (or re-registering) ONE person via a plain cv2
    window — main.py, this file's own standalone `<name>` CLI form, AND run_interactive() (the
    REPL, chunk 8) all call this directly, all passing `show`/`stream`/`logger` explicitly (their
    own --show/--stream/--log-dir flags, or the REPL's own shared instances).

    `show` defaults to False (NOT this function's old always-on-window behavior — that was a real
    gap: a true SSH session with no X11 forwarding would hang on cv2.imshow with no way to opt
    out). Pass `show=True` to open a local display window. `stream`, if given, receives the same
    ROI/person-count overlay frame `show` would have displayed, published live over HTTP (see
    debug_stream.DebugStreamServer) — independent of `show`, so you can stream without a local
    window, show a local window without streaming, both, or neither.

    `logger`, if given, must already have had `.start(...)` called by the caller (mirrors
    main.py's run_pretrigger_pipeline/run_followme_pipeline convention: caller starts, the
    function that owns the frame loop closes) — this function calls `.close()` on it in its own
    `finally`, once, covering both phases (plans/11_registration_interactive_console.md chunk 7)
    — UNLESS `close_logger_on_exit=False`, needed by run_interactive() (chunk 8): the REPL shares
    ONE logger/decisions.jsonl across MANY `register <name>` commands in one session, so no
    single `run()` call may close it — the REPL itself owns closing it once, on exit. Headless
    single-shot callers (main.py, this file's own CLI) leave this at its default `True`.
    """
    exit_reason = "completed"
    frame_idx = 0
    try:
        try:
            person_name = sanitize_person_name(person_name)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            exit_reason = "error"
            return 1

        data.reset_captures(person_name)  # "remove the old ones" — see module docstring
        front_roi, back_roi = _load_roi_config(config_path)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: could not open camera index {camera_index}", file=sys.stderr)
            exit_reason = "error"
            return 1

        print("Loading person detector (live single-person check)...")
        detector = data.LiveSubjectDetector()

        try:
            print(f"FRONT: face the camera, stand inside the box. Need {front_samples} samples — 'q' to stop early.")
            saved, frame_idx = _capture_phase_cli(cap, detector, front_roi, "FACE THE CAMERA", person_name,
                                                   "front", front_samples, show, stream, frame_idx, logger)
            if saved == 0:
                print("ERROR: no FRONT samples captured.", file=sys.stderr)
                exit_reason = "stopped_early"
                return 1
            print("Phase 1/2 done.")

            print(f"BACK: turn around, stand inside the box. Need {back_samples} samples — 'q' to stop early.")
            saved, frame_idx = _capture_phase_cli(cap, detector, back_roi, "TURN AROUND - BACK TO CAMERA",
                                                   person_name, "back", back_samples, show, stream, frame_idx, logger)
            if saved == 0:
                print("ERROR: no BACK samples captured.", file=sys.stderr)
                exit_reason = "stopped_early"
                return 1
            print("Phase 2/2 done.")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()

        front_cropped = data.build_cropped_roi(person_name, "front", front_roi)
        back_cropped = data.build_cropped_roi(person_name, "back", back_roi)
        print(
            f"\nCropped {front_cropped} front + {back_cropped} back image(s) — see "
            f"'registration_captures/{person_name}/cropped/' to check them before the build below uses them."
        )

        if data.rebuild_registries(person_name, config_path):
            print(f"\nDone: '{person_name}' is registered for both pipelines.")
            return 0
        print("\nDone with errors — see build output above.")
        exit_reason = "error"
        return 1
    except KeyboardInterrupt:
        # Caught HERE, not just at the top-level caller, deliberately: this makes Ctrl+C mean
        # "cancel THIS registration" in both callers, not "always kill the whole process" — the
        # one-shot CLI path (main.py, this file's own <person_name> form) has nothing left to do
        # after a cancelled registration anyway, so returning 1 ends it just the same either way;
        # but run_interactive() (chunk 8) calls run() from inside its own command loop, where
        # catching it here means Ctrl+C during an active `register <name>` cancels only that
        # command and returns control to the REPL's own >>> prompt — Ctrl+C while just sitting at
        # the prompt (no registration in progress) is a SEPARATE catch, in run_interactive()
        # itself, and means "exit the whole console" instead. Same keystroke, different meaning,
        # by design — whichever operation is actually running gets cancelled, nothing more.
        print("\nRegistration cancelled (Ctrl+C).")
        exit_reason = "stopped_early"
        return 1
    except Exception:
        exit_reason = "error"
        raise
    finally:
        if logger is not None and close_logger_on_exit:
            logger.close(frame_count=frame_idx, exit_reason=exit_reason)


# --------------------------------------------------------------------------------------------
# Interactive console path — operates registration (list/register/delete) over a plain SSH
# session, no display needed at all. See plans/11_registration_interactive_console.md.
# --------------------------------------------------------------------------------------------

def run_interactive(camera_index: int = 0, front_samples: int = 15, back_samples: int = 15,
                     config_path: str = "config/thresholds.yaml",
                     stream: Optional[DebugStreamServer] = None,
                     logger: Optional[RunLogger] = None) -> Tuple[int, Optional[str]]:
    """
    Interactive registration console (plans/11_registration_interactive_console.md chunk 8) — a
    REPL for operating registration over a plain SSH session, no display needed at all:
    `list` (Read), `register <name>` (Create/Update — calls run() completely unchanged), `delete
    <name>` (Delete), `follow <name>` (select a fully-registered person and exit — the console's
    equivalent of the Tkinter UI's "Follow Me" button, see below), `quit`/`exit`/Ctrl+D to stop
    without selecting anyone.

    Returns `(exit_code, chosen_name)` — `chosen_name` is `None` unless `follow <name>` was used,
    mirroring `RegistrationApp.chosen_name` (an instance attribute there, since Tkinter's own
    `mainloop()` returns nothing useful; a return value here, since this is a plain function) —
    see that class's own `_on_follow_me()` for the exact parallel. The caller (`main.py`'s
    `--interactive` dispatch) reads `chosen_name` the same way it already reads
    `RegistrationApp.chosen_name`, falling through into the SAME followme camera loop the other
    two register paths share.

    Architecture (plans/11 §3): this is the ONLY writer among the three channels that plan
    describes. `stream`, if given, is passed straight through to run() for each `register`
    command — the exact same live overlay the headless --person-name path already publishes,
    completely unmodified (invariant §3.3.1/§3.3.4 — never repurposed as a control channel here).
    `logger`, if given, must already be started by the caller (same convention as run()) — this
    function owns it for the WHOLE session and closes it once, on exit (not per-command); each
    `register <name>` command therefore passes close_logger_on_exit=False into run() to keep the
    shared decisions.jsonl open across multiple commands (see run()'s own docstring).

    MVP scope, deliberately deferred not forgotten (plans/11 §3.3.7): no per-command sample-count
    override (uses the session-wide front_samples/back_samples); no `stop` command — a REPL
    command already blocks synchronously, so Ctrl+C is the natural cancel, caught below and
    exits cleanly rather than as a raw traceback; no camera auto-reconnect. Each `register <name>`
    command's own logged frame numbers restart at 0 per command (not threaded globally across an
    entire REPL session) — an accepted simplification: every record still carries a real `ts`
    timestamp plus `person_name`/`phase`/`command_index` for disambiguation, so log ordering is
    never actually ambiguous, just not globally-sequential in the `frame` field specifically.
    """
    print("Registration console. Commands: list, register <name>, delete <name>, follow <name>, quit")
    command_index = 0
    chosen_name: Optional[str] = None
    exit_reason = "user_quit"  # every normal way this loop ends (quit/exit/follow/EOF/Ctrl+C) IS a user quit
    try:
        while True:
            try:
                line = input(">>> ").strip()
            except EOFError:
                print()
                break

            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit"):
                break

            elif cmd == "list":
                people = data.list_people()
                if not people:
                    print("  (no one registered yet)")
                for p in people:
                    print(f"  {p.name:20s} front={p.cropped_front_count:3d} back={p.cropped_back_count:3d} "
                          f"ready={p.ready_for_followme}")
                if logger is not None:
                    logger.log_frame(ts=time.time(), mode="register", stage="list",
                                      command_index=command_index, people_count=len(people))
                command_index += 1

            elif cmd == "register" and len(parts) >= 2:
                name = parts[1]
                result = run(name, camera_index, front_samples, back_samples, config_path,
                             show=False, stream=stream, logger=logger, close_logger_on_exit=False)
                print("OK" if result == 0 else "FAILED")
                command_index += 1

            elif cmd == "delete" and len(parts) >= 2:
                name = parts[1]
                confirmed = input(f"Delete '{name}'? [y/N] ").strip().lower() == "y"
                if confirmed:
                    data.delete_person(name)
                    print("deleted")
                else:
                    print("cancelled")
                if logger is not None:
                    logger.log_frame(ts=time.time(), mode="register", stage="delete",
                                      command_index=command_index, person_name=name, confirmed=confirmed)
                command_index += 1

            elif cmd == "follow" and len(parts) >= 2:
                name = parts[1]
                status = data.get_status(name)
                if not status.ready_for_followme:
                    print(f"'{name}' isn't fully registered yet — needs both a face registry "
                          f"entry and a target profile. Use 'register {name}' first.")
                else:
                    chosen_name = name
                    print(f"Selected '{name}' — exiting console to start followme mode.")
                if logger is not None:
                    logger.log_frame(ts=time.time(), mode="register", stage="follow",
                                      command_index=command_index, person_name=name,
                                      accepted=chosen_name == name)
                command_index += 1
                if chosen_name is not None:
                    break

            else:
                print("Unknown command. Commands: list, register <name>, delete <name>, follow <name>, quit")
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
    except Exception:
        exit_reason = "error"
        raise
    finally:
        if logger is not None:
            logger.close(frame_count=command_index, exit_reason=exit_reason)
    return 0, chosen_name


# --------------------------------------------------------------------------------------------
# Tkinter GUI path — the interactive CRUD app, launched when register_person.py is run with no args.
# --------------------------------------------------------------------------------------------

class CaptureWindow(tk.Toplevel):
    """One capture session for one person: countdown -> RAW front -> countdown -> RAW back ->
    CROP both phases -> pause for the operator to check the crops -> build. No IDENTITY detection
    during capture (see module docstring) — each accepted tick just saves the live frame AS-IS via
    registration_data.save_raw_capture(); cropping (registration_data.build_cropped_roi()) and all
    identity detection (registration_data.rebuild_registries()) both happen only after capture
    ends. Capture ticks DO run a live person-COUNT check (self.detector) that gates which frames
    get accepted — see _tick()'s capture_front/capture_back branch."""

    def __init__(self, parent, name: str, camera_index: int, front_samples: int, back_samples: int,
                 front_roi, back_roi, config_path: str, on_done):
        super().__init__(parent)
        self.title(f"Registering '{name}'")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.name = name
        self.front_samples, self.back_samples = front_samples, back_samples
        self.front_roi, self.back_roi = front_roi, back_roi
        self.config_path = config_path
        self.on_done = on_done
        self._cancelled = False
        self._photo = None  # keeps a live reference — Tkinter/PIL drops the image otherwise

        self.video_label = tk.Label(self)
        self.video_label.pack()
        self.status_label = tk.Label(self, text="Starting...", font=("Segoe UI", 13))
        self.status_label.pack(pady=8)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera index {camera_index}", parent=self)
            self.destroy()
            return

        data.reset_captures(name)  # "remove the old ones" — see register_person.py's module docstring

        self.status_label.configure(text="Loading person detector...")
        self.update_idletasks()
        self.detector = data.LiveSubjectDetector()

        self._state = "countdown_front"
        self._state_start = time.time()
        self._saved = 0
        self._last_save_time = 0.0
        self._person_count = 0
        self._last_check_time = 0.0
        self.after(0, self._tick)

    def _cancel(self):
        self._cancelled = True
        try:
            self.cap.release()
        except Exception:
            pass
        self.destroy()

    def _show(self, bgr_image) -> None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)

    def _tick(self):
        if self._cancelled:
            return

        # These two states run after the camera is already released (see
        # _do_crop_and_pause) — check them BEFORE touching self.cap.read() at all.
        if self._state == "cropping":
            self._do_crop_and_pause()
            return
        if self._state == "building":
            self._do_build()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self._tick)
            return

        if self._state in ("countdown_front", "countdown_back"):
            roi = self.front_roi if self._state == "countdown_front" else self.back_roi
            instruction = "FACE THE CAMERA" if self._state == "countdown_front" else "TURN AROUND - BACK TO CAMERA"
            remaining = _COUNTDOWN_SECONDS - (time.time() - self._state_start)
            if remaining <= 0:
                self._state = "capture_front" if self._state == "countdown_front" else "capture_back"
                self._saved, self._last_save_time = 0, 0.0
                self._person_count, self._last_check_time = 0, 0.0
            else:
                self._show(overlay.draw_countdown(frame, roi, instruction, remaining))
                self.status_label.configure(text=f"{instruction} — starting in {remaining:.1f}s")

        elif self._state in ("capture_front", "capture_back"):
            phase = "front" if self._state == "capture_front" else "back"
            roi = self.front_roi if phase == "front" else self.back_roi
            instruction = "FACE THE CAMERA" if phase == "front" else "TURN AROUND - BACK TO CAMERA"
            needed = self.front_samples if phase == "front" else self.back_samples

            now = time.time()
            if (now - self._last_check_time) >= _PERSON_CHECK_INTERVAL_SECONDS:
                self._person_count = self.detector.count_in_roi(frame, roi)
                self._last_check_time = now

            if self._person_count == 1 and (now - self._last_save_time) >= _CAPTURE_INTERVAL_SECONDS:
                data.save_raw_capture(self.name, phase, frame)  # RAW — see class docstring
                self._saved += 1
                self._last_save_time = now

            self._show(overlay.draw_capture(frame, roi, instruction, self._saved, needed, self._person_count))
            status_suffix = "ACCEPTED" if self._person_count == 1 else f"REJECTED ({self._person_count} in frame)"
            self.status_label.configure(text=f"{instruction} — {self._saved}/{needed} — {status_suffix}")

            if self._saved >= needed:
                if phase == "front":
                    self._state, self._state_start = "countdown_back", time.time()
                else:
                    self._state = "cropping"  # handled at the top of _tick() on the next call

        self.after(30, self._tick)

    def _do_crop_and_pause(self):
        """RAW capture is done for both phases — crop each into its own inspectable folder, then
        STOP and let the operator actually look at them before anything gets built (confirmed
        with the user: this is the whole point of splitting raw/cropped into two phases)."""
        try:
            self.cap.release()
        except Exception:
            pass
        self.status_label.configure(text="Cropping to ROI...")
        self.update_idletasks()

        front_n = data.build_cropped_roi(self.name, "front", self.front_roi)
        back_n = data.build_cropped_roi(self.name, "back", self.back_roi)
        cropped_path = f"registration_captures/{self.name}/cropped/"

        proceed = messagebox.askokcancel(
            "Check the crops",
            f"Cropped {front_n} front + {back_n} back image(s) to:\n{cropped_path}\n\n"
            f"Open that folder and check the crops look right, then click OK to build the "
            f"registry files from them — or Cancel to stop here without building (the raw and "
            f"cropped photos stay on disk either way).",
            parent=self,
        )
        if not proceed:
            self.on_done()
            self.destroy()
            return
        self._state = "building"
        self.after(0, self._tick)

    def _do_build(self):
        self.status_label.configure(text="Building registry files...")
        self.update_idletasks()

        ok = data.rebuild_registries(self.name, self.config_path)
        self.on_done()
        if ok:
            messagebox.showinfo("Done", f"'{self.name}' is registered for both pipelines.", parent=self)
        else:
            messagebox.showwarning("Done with errors", "See console output for details.", parent=self)
        self.destroy()


class RegistrationApp(tk.Tk):
    """The CRUD menu — a list of everyone registered/in-progress, plus New/Re-capture/Delete."""

    _COLUMNS = ("name", "raw_front", "raw_back", "crop_front", "crop_back", "face", "target", "status")
    _HEADINGS = ("Name", "Raw F", "Raw B", "Crop F", "Crop B", "Face", "Target", "Status")

    def __init__(self, camera_index: int, front_samples: int, back_samples: int, config_path: str,
                 can_follow_me: bool = False):
        super().__init__()
        self.title("Person Registration")
        self.camera_index = camera_index
        self.front_samples, self.back_samples = front_samples, back_samples
        self.config_path = config_path
        self.front_roi, self.back_roi = _load_roi_config(config_path)

        # can_follow_me: whether the caller (main.py) can actually chain into followme mode after
        # this app closes — "Follow Me" only ever hands a name back via self.chosen_name; it never
        # starts the camera loop itself (that stays main.py's job, same seam --then-followme
        # already uses). False only when this file is run standalone (see main() below).
        self.can_follow_me = can_follow_me
        self.chosen_name: Optional[str] = None

        self._build_widgets()
        self._refresh()

    def _build_widgets(self):
        self.tree = ttk.Treeview(self, columns=self._COLUMNS, show="headings", height=12)
        for col, heading in zip(self._COLUMNS, self._HEADINGS):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=100, anchor="center")
        self.tree.column("name", width=160, anchor="w")
        self.tree.pack(padx=10, pady=10, fill="both", expand=True)

        buttons = tk.Frame(self)
        buttons.pack(pady=(0, 10))
        tk.Button(buttons, text="New", width=12, command=self._on_new).pack(side="left", padx=4)
        tk.Button(buttons, text="Re-capture", width=12, command=self._on_recapture).pack(side="left", padx=4)
        tk.Button(buttons, text="Delete", width=12, command=self._on_delete).pack(side="left", padx=4)
        tk.Button(buttons, text="Refresh", width=12, command=self._refresh).pack(side="left", padx=4)
        tk.Button(buttons, text="Follow Me", width=12, command=self._on_follow_me).pack(side="left", padx=4)

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for person in data.list_people():
            self.tree.insert("", "end", iid=person.name, values=(
                person.name,
                person.raw_front_count, person.raw_back_count,
                person.cropped_front_count, person.cropped_back_count,
                "Y" if person.has_face_registry else "N",
                "Y" if person.has_target_profile else "N",
                "READY" if person.ready_for_followme else "incomplete",
            ))

    def _selected_name(self) -> Optional[str]:
        selection = self.tree.selection()
        return selection[0] if selection else None

    def _on_new(self):
        name = simpledialog.askstring("New person", "Person's name:", parent=self)
        if name:
            self._start_capture(name)

    def _on_recapture(self):
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Re-capture", "Select a person first.", parent=self)
            return
        self._start_capture(name)

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Delete", "Select a person first.", parent=self)
            return
        if messagebox.askyesno("Delete", f"Delete '{name}' — captures + both registry files?", parent=self):
            data.delete_person(name)
            self._refresh()

    def _on_follow_me(self):
        """Hands the selected, fully-registered person's name back to the caller (main.py) via
        self.chosen_name and closes the app — main.py reads it after mainloop() returns and
        falls through into the SAME followme camera loop --then-followme uses. This method never
        starts a camera loop itself; that responsibility stays entirely with main.py."""
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Follow Me", "Select a person first.", parent=self)
            return
        status = data.get_status(name)
        if not status.ready_for_followme:
            messagebox.showwarning(
                "Follow Me", f"'{name}' isn't fully registered yet — needs both a face "
                f"registry entry and a target profile.", parent=self,
            )
            return
        if not self.can_follow_me:
            messagebox.showwarning(
                "Follow Me", "This app was launched standalone (register_person.py directly), "
                "which has no camera loop to chain into. Relaunch with "
                "`python main.py --modules register` to use this.",
                parent=self,
            )
            return
        self.chosen_name = name
        self.destroy()

    def _start_capture(self, raw_name: str):
        try:
            name = sanitize_person_name(raw_name)
        except ValueError as e:
            messagebox.showerror("Invalid name", str(e), parent=self)
            return
        CaptureWindow(
            self, name, self.camera_index, self.front_samples, self.back_samples,
            self.front_roi, self.back_roi, self.config_path, on_done=self._refresh,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register people for both the face-match and re-id tracking pipelines."
    )
    parser.add_argument(
        "person_name", nargs="?",
        help="Register just this one person headlessly and exit, skipping the Tkinter CRUD app.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--front-samples", type=int, default=15)
    parser.add_argument("--back-samples", type=int, default=15)
    parser.add_argument("--config", default="config/thresholds.yaml")
    parser.add_argument(
        "--show", action="store_true",
        help="Open a local display window for the capture overlay (ROI box + person-count gate). "
             "Off by default — same convention as main.py's own --show — since a real SSH "
             "session with no X11 forwarding would otherwise hang here. Independent of --stream: "
             "use either, both, or neither.",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Also publish the live capture overlay (ROI box + person-count gate) over HTTP at "
             "127.0.0.1:8080, for watching a headless (no local display) capture session over an "
             "SSH port-forward. Independent of --show.",
    )
    parser.add_argument(
        "--log-dir", default="runs",
        help="Directory to write per-run structured logs into (runs/<timestamp>_register/"
             "meta.json + decisions.jsonl — same convention as main.py's pretrigger/followme "
             "runs, see plans/10_debug_logging_observability.md and "
             "plans/11_registration_interactive_console.md chunks 7-8). Always on for the "
             "headless <person_name>/--interactive paths — this is what makes a headless SSH "
             "session reviewable afterward without --show.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Open an interactive registration console instead (list/register <name>/delete "
             "<name>/quit) — operate registration over a plain SSH session, no display needed at "
             "all. See plans/11_registration_interactive_console.md. Mutually exclusive with "
             "passing a person_name positionally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interactive and args.person_name:
        print("ERROR: --interactive and a positional person_name are mutually exclusive.", file=sys.stderr)
        return 1

    if args.interactive:
        stream = DebugStreamServer() if args.stream else None
        if stream is not None:
            print(f"streaming to {stream.start()}")
        logger = RunLogger(log_root=args.log_dir)
        run_dir = logger.start("register", sys.argv[1:], f"camera:{args.camera_index}", args.config)
        print(f"logging to {run_dir}")
        try:
            result, chosen_name = run_interactive(args.camera_index, args.front_samples, args.back_samples,
                                                    args.config, stream=stream, logger=logger)
        finally:
            if stream is not None:
                stream.stop()
        if chosen_name:
            # Same limitation as the --person-name path just below: this standalone script has
            # no camera-loop machinery of its own to hand a chosen person off to. Unlike that
            # path, run_interactive() already offered 'follow <name>' inside the console itself —
            # tell the operator to relaunch through main.py to actually act on that choice.
            print(f"\n'{chosen_name}' selected — relaunch with `python main.py --modules register "
                  f"--interactive` to continue into followme mode with this console.")
        return result

    if args.person_name:
        stream = DebugStreamServer() if args.stream else None
        if stream is not None:
            print(f"streaming to {stream.start()}")
        logger = RunLogger(log_root=args.log_dir)
        run_dir = logger.start("register", sys.argv[1:], f"camera:{args.camera_index}", args.config)
        print(f"logging to {run_dir}")
        try:
            return run(args.person_name, args.camera_index, args.front_samples, args.back_samples,
                       args.config, show=args.show, stream=stream, logger=logger)
        finally:
            if stream is not None:
                stream.stop()
    # can_follow_me=False (default) when launched standalone — this file has no camera-loop
    # machinery of its own to hand a chosen person off to; "Follow Me" tells the operator to use
    # `python main.py --modules register` instead, which does.
    RegistrationApp(args.camera_index, args.front_samples, args.back_samples, args.config).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
