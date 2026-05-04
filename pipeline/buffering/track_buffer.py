from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Any


@dataclass
class FrameFeature:
    """
    A single time-step feature for one tracked person.
    Keep this lightweight and consistent across frames.
    """
    frame_idx: int
    bbox_xyxy: List[float]                 # [x1, y1, x2, y2]
    conf: float
    kpts_xyc: Optional[Any] = None         # e.g., 17x3 list/np array; keep consistent
    extra: Optional[Dict[str, Any]] = None # anything else you want (speed, center, etc.)


class TrackBufferManager:
    """
    Maintains a rolling window of FrameFeature per track_id.
    - window_size_frames: number of frames per sequence
    - age_out_frames: how long to keep a track buffer after it disappears
    """
    def __init__(self, window_size_frames: int, age_out_frames: int):
        if window_size_frames <= 0:
            raise ValueError("window_size_frames must be > 0")
        if age_out_frames < 0:
            raise ValueError("age_out_frames must be >= 0")

        self.window = int(window_size_frames)
        self.age_out = int(age_out_frames)

        self.buffers: Dict[int, Deque[FrameFeature]] = {}
        self.last_seen: Dict[int, int] = {}

    def update(self, observations: List[dict], frame_idx: int) -> Dict[int, List[FrameFeature]]:
        """
        observations: list of dicts, each must include:
          - track_id
          - frame_idx
          - bbox_xyxy
          - conf
          - optional kpts_xyc
          - optional extra (dict)

        Returns:
          ready_sequences: dict[track_id] = list[FrameFeature] (length == window)
          Only for tracks whose buffer has filled to the window length.
        """
        # This is where frame-wise detections become temporal data: every track ID
        # owns its own rolling deque, so swimmer histories do not get mixed.
        for obs in observations:
            tid = int(obs["track_id"])

            feat = FrameFeature(
                frame_idx=int(obs.get("frame_idx", frame_idx)),
                bbox_xyxy=[float(x) for x in obs["bbox_xyxy"]],
                conf=float(obs.get("conf", 0.0)),
                kpts_xyc=obs.get("kpts_xyc", None),
                extra=obs.get("extra", None),
            )

            if tid not in self.buffers:
                self.buffers[tid] = deque(maxlen=self.window)

            self.buffers[tid].append(feat)
            self.last_seen[tid] = int(frame_idx)

        # If a swimmer disappears briefly, keep the buffer for age_out frames.
        # If they stay missing longer, drop stale state to prevent old motion from
        # contaminating a future detection with the same ID.
        if self.age_out >= 0:
            to_delete = []
            for tid, last in self.last_seen.items():
                if frame_idx - last > self.age_out:
                    to_delete.append(tid)

            for tid in to_delete:
                self.buffers.pop(tid, None)
                self.last_seen.pop(tid, None)

        # Only full windows are emitted. This protects Stage 4/5 from making a
        # distress prediction before enough movement history exists.
        ready: Dict[int, List[FrameFeature]] = {}
        for tid, buf in self.buffers.items():
            if len(buf) >= self.window:
                ready[tid] = list(buf)

        return ready

    def get_active_track_ids(self) -> List[int]:
        return list(self.buffers.keys())

    def get_buffer_len(self, track_id: int) -> int:
        buf = self.buffers.get(int(track_id))
        return len(buf) if buf is not None else 0
