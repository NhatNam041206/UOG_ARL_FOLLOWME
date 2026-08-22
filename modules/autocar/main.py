"""Live entry point: webcam/video -> YOLOv8n-pose -> ByteTrack -> display."""
import argparse

import cv2

import config
from detector.yolov8_pose_torch import YOLOv8PoseTorch
from tracker.byte_tracker import BYTETracker
from utils.draw import draw_tracks
from utils.fps_meter import FPSMeter
from utils.video_source import VideoSource

WINDOW_NAME = "Human Tracking"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n-pose + ByteTrack human tracking")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument("--model", default=config.POSE_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=config.DETECT_CONF)
    parser.add_argument("--imgsz", type=int, default=config.DETECT_IMGSZ)
    parser.add_argument("--device", default="cpu",
                         help="cpu or cuda:0. Defaults to cpu to match the Jetson Nano deployment target "
                              "(which has no CUDA-capable desktop GPU) - pass --device cuda:0 to use a local GPU.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--no-mirror", action="store_true",
                         help="Disable horizontal flip (mirror is on by default, meant for a front-facing webcam)")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no cv2 window)")
    parser.add_argument("--target", default=None,
                         help="Path to an enrolled person profile (see scripts/enroll_person.py) - when given, "
                              "that person's track is picked out and highlighted (green, 'TARGET' label) among "
                              "everyone else detected. Off by default: generic multi-person tracking, unchanged.")
    return parser.parse_args()


class HumanTrackingApp:
    """Owns the live pipeline (video source -> detector -> tracker -> optional target lock ->
    display) for one run. Built once from parsed CLI args; run() drives the frame loop until 'q'
    is pressed or the video source ends."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        source = int(args.source) if args.source.isdigit() else args.source

        print(f"Using device: {args.device}")
        self.detector = YOLOv8PoseTorch(model_path=args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
        self.tracker = BYTETracker()
        self.fps_meter = FPSMeter()

        self.target_lock = None
        if args.target:
            from identity.target_lock import TargetLock
            self.target_lock = TargetLock(args.target, device=args.device)
            print(f"Loaded target profile '{args.target}' - will highlight that person's track once seen.")

        self.video = VideoSource(source, width=args.width, height=args.height)
        if not args.no_display:
            # WINDOW_NORMAL (resizable) instead of the imshow default (fixed-size,
            # pinned to the frame's native pixel size) - otherwise dragging/
            # maximizing the window just leaves the extra space blank instead of
            # scaling the video into it.
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def run(self) -> None:
        try:
            for frame in self.video:
                frame = self._process_frame(frame)
                if not self.args.no_display:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.video.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Runs detection/tracking/(optional) target-lock/drawing on one frame and returns the
        annotated frame ready for display."""
        if not self.args.no_mirror:
            frame = cv2.flip(frame, 1)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        target_id = self.target_lock.update(tracks, frame) if self.target_lock else None

        fps = self.fps_meter.tick()
        target_confidence = self.target_lock.last_verify_score if self.target_lock else None
        candidate_scores = self.target_lock.candidate_scores if self.target_lock else None
        draw_tracks(frame, tracks, target_id=target_id, target_mode=self.target_lock is not None,
                    target_confidence=target_confidence, candidate_scores=candidate_scores)

        status = f"FPS: {fps:.1f} | People: {len(tracks)}"
        if self.target_lock is not None:
            status += " | Target: LOCKED" if target_id is not None else " | Target: searching..."
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame


def main():
    args = parse_args()
    app = HumanTrackingApp(args)
    app.run()


if __name__ == "__main__":
    main()
