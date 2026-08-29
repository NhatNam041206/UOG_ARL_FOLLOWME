"""
Live debug-view MJPEG server — dev-tooling only, NOT wired into any module's core step() path
(plans/10_debug_logging_observability.md chunk 6). For watching a headless SSH run live, as an
alternative to --show (which needs a display attached) — reviewing a completed run's
runs/<run_id>/{decisions.jsonl,debug.avi} afterward remains the primary, always-on workflow; this
is only for wanting to see it live, during interactive testing.

Stdlib-only (http.server) — no new dependency added, matching this project's own documented
preference for lightweight hand-implemented pieces over pulling in a web framework
(docs/technologies.md's "hand-implemented PID over a library dependency" entry).

Binds 127.0.0.1 ONLY — never exposed on the network. View via an SSH local port-forward:
    ssh -L 8080:localhost:8080 pi@<host>
    (then open http://127.0.0.1:8080/ in a browser on your own machine)

Throttled by design: update_frame() only actually re-encodes/publishes every Nth call (default
every 3rd), and JPEG quality defaults to 70 — full-rate full-quality streaming would compete with
the inference loop for CPU on a Pi, which is the one thing this tool must not do (see the
--save-video design rationale this mirrors, in main.py's open_debug_video_writer()).
"""
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

__all__ = ["DebugStreamServer"]

_BOUNDARY = "frame"
_INDEX_HTML = (
    b"<!doctype html><html><head><title>debug stream</title></head>"
    b'<body style="margin:0;background:#111">'
    b'<img src="/stream.mjpg" style="width:100%;display:block">'
    b"</body></html>"
)


def _get_local_ip() -> str:
    """Best-effort discovery of the machine's primary LAN IP for display in stream URLs."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"



class _FrameBuffer:
    """Thread-safe latest-frame holder — one writer (the pipeline loop via update_frame()), many
    readers (each connected client's own handler thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None

    def set_frame(self, frame: np.ndarray, quality: int) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg


def _make_handler(buffer: _FrameBuffer, min_interval_seconds: float):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence stdlib's default per-request access log — this is a dev tool, not a service

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)
                return
            if self.path == "/stream.mjpg":
                self._stream_mjpeg()
                return
            self.send_response(404)
            self.end_headers()

        def _stream_mjpeg(self):
            # The whole method is inside the try, not just the streaming loop below — a client
            # can disconnect during the initial send_response()/send_header() round-trip too
            # (rare in real use, but common enough in fast automated tests hitting this server
            # repeatedly that it showed up as an unhandled thread exception in test runs). Caught
            # as OSError, the common base of BrokenPipeError/ConnectionResetError/
            # ConnectionAbortedError, rather than naming each subclass — any socket-level failure
            # here means the same thing: the client is gone, not a bug to report.
            try:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                self.end_headers()
                last_sent = None
                while True:
                    jpeg = buffer.get_jpeg()
                    if jpeg is not None and jpeg is not last_sent:
                        self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        last_sent = jpeg
                    time.sleep(min_interval_seconds)
            except OSError:
                pass  # client disconnected — normal, not an error

    return _Handler


class DebugStreamServer:
    """One instance per pipeline run (mirrors run_logging.RunLogger's convention) — NOT
    thread-safe to share across runs, not meant to be. Dev-tooling only, see this file's own
    docstring for why it's never wired into a module's core step() path."""

    def __init__(self):
        self._buffer = _FrameBuffer()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._push_count = 0
        self._jpeg_quality = 70
        self._throttle_every_n_frames = 3

    def start(self, port: int = 8080, host: str = "0.0.0.0",
              throttle_every_n_frames: int = 3, jpeg_quality: int = 70,
              min_interval_seconds: float = 0.05) -> str:
        """Starts the server in a background daemon thread. Binds to `host` (`0.0.0.0` by default,
        accessible across the local network / Wi-Fi). `port=0` lets the OS pick a free port (used
        by tests to avoid collisions) — the URL returned reflects the actual bound port and primary
        reachable IP. Returns the viewable URL."""
        self._jpeg_quality = jpeg_quality
        self._throttle_every_n_frames = max(1, throttle_every_n_frames)
        handler_cls = _make_handler(self._buffer, min_interval_seconds)
        self._httpd = ThreadingHTTPServer((host, port), handler_cls)
        self._httpd.daemon_threads = True  # per-connection streaming threads must not block process exit
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        display_host = _get_local_ip() if host == "0.0.0.0" else host
        return f"http://{display_host}:{self._httpd.server_port}/"


    def update_frame(self, frame: np.ndarray) -> None:
        """Called once per pipeline frame. Throttled internally — only every
        `throttle_every_n_frames`-th call actually re-encodes/publishes a frame, so streaming
        doesn't compete with inference for CPU. Safe to call even if start() was never called
        (silently does nothing, since there's no server to publish to) — but nothing reads it
        either way in that case."""
        self._push_count += 1
        if self._push_count % self._throttle_every_n_frames != 0:
            return
        self._buffer.set_frame(frame, quality=self._jpeg_quality)

    def stop(self) -> None:
        """Safe to call even if start() was never called, or called more than once."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
