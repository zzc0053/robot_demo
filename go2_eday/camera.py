import cv2
class MacCameraCapture:
    def __init__(self, device_index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device_index={device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)

    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        return frame

    def stop(self):
        try:
            self.cap.release()
        except Exception:
            pass
