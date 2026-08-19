import logging
import cv2

logger = logging.getLogger(__name__)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, target_fps: int = 30) -> None:
    """
    Request MJPG (compressed) capture, a target resolution/FPS, and a 1-frame internal buffer.

    Without this, cv2.VideoCapture() defaults vary wildly by camera/driver — many fall back to
    an uncompressed format (YUY2/raw) that can't sustain the requested resolution over USB
    bandwidth, so the driver silently drops to a lower native resolution (frames then get
    upscaled blurrily by our own cv2.resize) and/or a low framerate; an unset buffer also lets
    frames queue up, so read() returns increasingly stale frames once processing can't keep up.
    MJPG asks the camera to do the compression on-device, freeing USB bandwidth for higher
    resolution/framerate. Not all backends/cameras honor every property — failures here are
    silently ignored by OpenCV, which is fine; the actual resulting properties are logged so
    it's visible whether the request was actually honored.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_int else "?"
    logger.info(
        f"Camera capture configured: requested {width}x{height}@{target_fps}fps MJPG -> "
        f"actual {actual_w}x{actual_h}@{actual_fps:.1f}fps {actual_fourcc} "
        f"(camera/driver may not honor every request — mismatch here explains resize blur or "
        f"a lower achievable FPS than requested)"
    )
