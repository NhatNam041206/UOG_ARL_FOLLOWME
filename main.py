import sys
import time
import yaml
import logging
import argparse
import threading
import cv2
from typing import Optional

from src.registration import TargetRegistrar
from src.pipeline import FollowPipeline
from src.camera_utils import configure_capture

# Configure standard python logging (DO NOT use print())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class WebcamStreamThread:
    """
    Non-blocking multi-threaded camera frame reader.
    Continuously grabs frames in a background thread to eliminate camera I/O wait latency.
    """
    def __init__(self, camera_index: int, input_resolution: Optional[list] = None, flip_horizontal: bool = False):
        self.camera_index = camera_index
        self.input_resolution = input_resolution
        self.flip_horizontal = flip_horizontal
        self.cap = cv2.VideoCapture(camera_index)
        self.running = False
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera index {self.camera_index} failed to open. Check webcam connection.")
        if self.input_resolution and len(self.input_resolution) == 2:
            configure_capture(self.cap, int(self.input_resolution[0]), int(self.input_resolution[1]))
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                if self.input_resolution and len(self.input_resolution) == 2:
                    rw, rh = int(self.input_resolution[0]), int(self.input_resolution[1])
                    frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="CV Follow-Me Module (Stage 1 & Stage 2)")
    parser.add_argument("--mode", type=str, choices=["capture", "register", "run"], default="run",
                        help="Operating mode: 'capture' (Phase 1 raw data), 'register' (Phase 1+2), or 'run' (Stage 2 tracking)")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path to YAML settings file")
    parser.add_argument("--ui", action="store_true",
                        help="Show the debug overlay window (bboxes, ROI region, telemetry). "
                             "Default is headless/background (no window) — the intended mode "
                             "for unattended/production runs; pass --ui only for local debugging.")
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file at '{args.config}': {e}")
        sys.exit(1)

    if args.mode == "capture":
        from src.registration import RawDataCapturer
        logger.info("Executing Phase 1: Raw Data Capture...")
        try:
            capturer = RawDataCapturer(config)
            sanitized_name = capturer.run()
            logger.info(f"Phase 1 Raw Data Capture completed successfully for '{sanitized_name}'!")
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            sys.exit(1)

    elif args.mode == "register":
        from src.registration import TargetRegistrar
        logger.info("Executing Stage 1: Target Registration (Phase 1 Capture + Phase 2 Build)...")
        try:
            registrar = TargetRegistrar(config)
            saved_path = registrar.run()
            logger.info(f"Stage 1 Registration completed successfully! Saved to '{saved_path}'")
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            sys.exit(1)

    elif args.mode == "run":
        # Person selection happens BEFORE anything pipeline/camera-related is touched. The
        # selector itself auto-redirects to a fresh registration if the registry is empty — see
        # PersonRegistrySelector.run(). If the user closes it without picking anyone, exit
        # cleanly here rather than falling through to run() with no/stale reference data.
        from src.person_selector import PersonRegistrySelector
        selector = PersonRegistrySelector(config, config_path=args.config)
        selected_path = selector.run()
        if not selected_path:
            logger.error("No person selected from the registry. Exiting.")
            sys.exit(1)

        logger.info(f"Executing Stage 2: Multi-Threaded Follow-Me Pipeline (target: '{selected_path}')...")
        try:
            pipeline = FollowPipeline(config_path=args.config, reference_npz_path=selected_path)
        except Exception as e:
            logger.error(f"Failed to initialize FollowPipeline: {e}")
            sys.exit(1)

        camera_index = config.get("camera_index", 0)
        input_res = config.get("input_resolution", [640, 480])
        flip_horiz = config.get("flip_horizontal", False)

        try:
            cam_stream = WebcamStreamThread(camera_index, input_res, flip_horizontal=flip_horiz).start()
            logger.info(f"Started multi-threaded camera reader thread on camera index {camera_index} (flip_horizontal={flip_horiz})")
        except Exception as e:
            logger.error(f"Failed to start camera stream thread: {e}")
            pipeline.close()
            sys.exit(1)

        # Debug overlay window is entirely opt-in via --ui. Import lazily so the default
        # (headless/background) run path never touches cv2's GUI-drawing code at all.
        window_name = "Stage 2: CV Follow-Me Pipeline (Debug UI)"
        if args.ui:
            from src.debug_overlay import render as render_debug_overlay
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info("Debug UI enabled (--ui): showing bboxes, ROI region, and telemetry overlay.")
        else:
            logger.info("Running headless (no --ui): no window shown, state visible via periodic log lines.")

        # FPS calculation metrics
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0

        try:
            while True:
                ret, frame = cam_stream.read()

                # Frame decode error handling: skip frame, log warning, do not crash loop
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                # Process frame via multi-threaded pipeline
                angle_result = pipeline.process_frame(frame)

                if args.ui:
                    display = render_debug_overlay(frame, pipeline, angle_result, current_fps)
                    cv2.imshow(window_name, display)

                # Measuring real FPS & logging once per second — this is the "what is the
                # pipeline doing" signal in headless mode, so it includes ROI/mode state too.
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    roi_state = "ROI" if pipeline.last_used_roi else "FULL-FRAME"
                    timing = pipeline.last_timing_ms
                    logger.info(
                        f"Performance: FPS = {current_fps:.2f} | Target Found = {angle_result.target_found} | "
                        f"Detect = {roi_state} | ROIFail = {pipeline._roi_failure_count}/{pipeline.roi_failure_max_frames} | "
                        f"Timing(ms) detect={timing['detect_ms']:.1f} verify={timing['verify_ms']:.1f} "
                        f"total={timing['total_ms']:.1f}"
                    )
                    frame_count = 0
                    fps_start_time = time.time()

                if args.ui:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Exiting pipeline on user command ('q' pressed).")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Shutting down...")

        finally:
            cam_stream.stop()
            pipeline.close()
            if args.ui:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
