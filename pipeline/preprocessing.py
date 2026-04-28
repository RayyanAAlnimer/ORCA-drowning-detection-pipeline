from __future__ import annotations

import cv2


class FramePreprocessor:
    """Lightweight frame cleanup before pose detection/tracking."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "center_zoom_enhance",
        zoom: float = 1.35,
        y_bias: float = 0.0,
    ):
        self.enabled = bool(enabled)
        self.mode = mode
        self.zoom = float(zoom)
        self.y_bias = float(y_bias)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply(self, frame):
        if not self.enabled:
            return frame
        if self.mode == "enhance":
            return self._enhance(frame)
        if self.mode == "center_zoom_enhance":
            return self._enhance(self._crop_zoom(frame, self.zoom, 0.0))
        if self.mode == "lower_center_zoom_enhance":
            return self._enhance(self._crop_zoom(frame, self.zoom, self.y_bias or 0.12))
        raise ValueError(f"Unknown preprocessing mode: {self.mode}")

    def _enhance(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = self._clahe.apply(l_chan)
        enhanced = cv2.merge((l_chan, a_chan, b_chan))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)

    @staticmethod
    def _crop_zoom(frame, zoom: float, y_bias: float):
        if zoom <= 1.0:
            return frame
        h, w = frame.shape[:2]
        crop_w = max(1, int(round(w / zoom)))
        crop_h = max(1, int(round(h / zoom)))
        cx = w // 2
        cy = int(round((h / 2) + (float(y_bias) * h)))
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
