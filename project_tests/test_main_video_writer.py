"""
Unit tests for main.open_debug_video_writer — real cv2.VideoWriter, no CV models/mediapipe/
onnxruntime needed (unlike main.py's actual per-frame pipelines, which this dev environment can't
import — see plans/10_debug_logging_observability.md chunk 5's verification note).

Run with:
    python -m pytest project_tests/test_main_video_writer.py -v
"""
import os

import cv2
import numpy as np
import pytest

from main import open_debug_video_writer


class _FakeCap:
    """Just enough of cv2.VideoCapture's interface for open_debug_video_writer's .get() calls."""

    def __init__(self, fps: float, width: int, height: int):
        self._props = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }

    def get(self, prop_id):
        return self._props.get(prop_id, 0)


def test_uses_reported_fps_and_resolution(tmp_path):
    cap = _FakeCap(fps=24.0, width=320, height=240)
    writer, video_path, fps, resolution = open_debug_video_writer(cap, str(tmp_path))

    try:
        assert writer.isOpened()
        assert video_path == os.path.join(str(tmp_path), "debug.avi")
        assert fps == 24.0
        assert resolution == (320, 240)
    finally:
        writer.release()


def test_falls_back_when_fps_unreported(tmp_path):
    """Many webcams report 0 for CAP_PROP_FPS — must not produce a 0-fps (unplayable) video."""
    cap = _FakeCap(fps=0.0, width=640, height=480)
    writer, video_path, fps, resolution = open_debug_video_writer(cap, str(tmp_path))

    try:
        assert fps == 20.0
        assert resolution == (640, 480)
    finally:
        writer.release()


def test_falls_back_when_resolution_unreported(tmp_path):
    cap = _FakeCap(fps=15.0, width=0, height=0)
    writer, video_path, fps, resolution = open_debug_video_writer(cap, str(tmp_path))

    try:
        assert resolution == (640, 480)
    finally:
        writer.release()


def test_writer_actually_writes_frames(tmp_path):
    cap = _FakeCap(fps=10.0, width=64, height=48)
    writer, video_path, fps, resolution = open_debug_video_writer(cap, str(tmp_path))

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()

    assert os.path.exists(video_path)
    assert os.path.getsize(video_path) > 0
