"""
Unit tests for scripts.debug_stream.DebugStreamServer — real HTTP requests against a real server
bound to an OS-assigned port (port=0), no camera/model dependency (cv2 itself has none). Run with:

    python -m pytest project_tests/test_debug_stream.py -v
"""
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from scripts.debug_stream import DebugStreamServer


@pytest.fixture
def server():
    s = DebugStreamServer()
    yield s
    s.stop()


def test_start_binds_localhost_only_and_returns_url(server):
    url = server.start(port=0)
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/")


def test_index_page_served(server):
    url = server.start(port=0)
    resp = urllib.request.urlopen(url, timeout=2)
    try:
        assert resp.status == 200
        assert b"stream.mjpg" in resp.read()
    finally:
        resp.close()


def test_unknown_path_is_404(server):
    url = server.start(port=0)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url + "nonexistent", timeout=2)
    assert exc_info.value.code == 404


def test_stream_serves_pushed_frame_as_mjpeg(server):
    url = server.start(port=0, throttle_every_n_frames=1, min_interval_seconds=0.01)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    server.update_frame(frame)
    time.sleep(0.1)

    resp = urllib.request.urlopen(url + "stream.mjpg", timeout=2)
    try:
        content_type = resp.headers.get("Content-Type", "")
        assert content_type.startswith("multipart/x-mixed-replace")
        # read1(), not read(): read() blocks trying to fill the full buffer from this
        # keep-alive multipart stream, which never sends more than one small JPEG without
        # another update_frame() call; read1() returns after each individual chunk the server
        # wrote instead, so accumulate a few of them (each wfile.write() call is its own chunk).
        received = b""
        for _ in range(10):
            received += resp.read1(4096)
            if b"\xff\xd8" in received:
                break
        assert b"Content-Type: image/jpeg" in received
        assert b"\xff\xd8" in received  # JPEG SOI marker — confirms a real JPEG was embedded
    finally:
        resp.close()


def test_throttling_skips_non_multiple_pushes(server):
    server.start(port=0, throttle_every_n_frames=5)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in range(4):
        server.update_frame(frame)
    assert server._buffer.get_jpeg() is None  # none of the first 4 calls were the 5th

    server.update_frame(frame)
    assert server._buffer.get_jpeg() is not None  # the 5th call publishes


def test_update_frame_before_start_does_not_raise():
    server = DebugStreamServer()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    server.update_frame(frame)  # no server started — should be a silent no-op-ish write to buffer only
    server.stop()  # also must not raise when never started


def test_stop_is_idempotent(server):
    server.start(port=0)
    server.stop()
    server.stop()  # second call must not raise
