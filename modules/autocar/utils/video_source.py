import cv2


class VideoSource:
    """Thin iterator over a webcam index or a video file path."""

    def __init__(self, source, width=None, height=None):
        self.cap = cv2.VideoCapture(source)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()
