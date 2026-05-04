from ultralytics import YOLO


class PoseTracker:
    """
    Stage 2: YOLO Pose + tracking IDs via Ultralytics built-in tracker.

    Supports:
      - track_video(video_path)  → generator of Results per frame
      - track_frame(frame)       → single Results object (for webcam/live use)

    Ultralytics persist=True keeps track IDs stable across frames.
    """

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5):
        # This is the first learned model in the runtime path: it converts pixels
        # into boxes, keypoints, confidences, and tracker IDs.
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    # ---------------------------------------------------------
    # File-based tracking (existing behavior, unchanged)
    # ---------------------------------------------------------
    def track_video(self, video_path: str):
        """
        Input: video file path
        Output: generator yielding Results objects frame-by-frame
        """
        return self.model.track(
            source=video_path,
            stream=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            # persist=True is the handoff from detection to tracking: YOLO keeps
            # identities alive across frames so Stage 3 can build per-person history.
            persist=True
        )

    # ---------------------------------------------------------
    # Frame-based tracking (NEW: for webcam/live streaming)
    # ---------------------------------------------------------
    def track_frame(self, frame):
        """
        Input: single frame (numpy array)
        Output: single Results object

        Uses persist=True to maintain tracking state across calls.
        """
        results = self.model.track(
            source=frame,
            stream=False,   # single inference
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            # The camera loop calls this once per frame; persistence keeps IDs
            # stable between calls instead of treating each frame independently.
            persist=True
        )

        # Ultralytics returns a list even for single frame
        return results[0]
