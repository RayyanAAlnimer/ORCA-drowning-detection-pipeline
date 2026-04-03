from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import defaultdict
from pathlib import Path

from pipeline.tracker import PoseTracker
from pipeline.buffering import TrackBufferManager
from pipeline.featurization import PoseFeatureExtractor
from pipeline.training.dataset_builder import build_dataset_from_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed sequence dataset from tracked videos.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--label", type=int, default=0, help="Binary label assigned to exported sequences")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for .pt outputs")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--age-out-seconds", type=float, default=1.0)
    parser.add_argument("--augment-horizontal-flip", action="store_true", help="Export an additional horizontally mirrored sample for each saved sequence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    tracker = PoseTracker(args.model)
    bufman = TrackBufferManager(
        window_size_frames=max(1, int(round(args.window_seconds * fps))),
        age_out_frames=max(0, int(round(args.age_out_seconds * fps))),
    )
    extractor = PoseFeatureExtractor(frame_size=(w, h))

    saved_sequences = {}
    labels_by_track = defaultdict(lambda: args.label)
    frame_idx = 0

    for r0 in tracker.track_video(args.video):
        observations = []
        if r0.boxes is not None and r0.boxes.id is not None and len(r0.boxes) > 0:
            ids_t = r0.boxes.id.int()
            xyxy_t = r0.boxes.xyxy
            conf_t = r0.boxes.conf if hasattr(r0.boxes, "conf") else None
            kpts_t = None
            if getattr(r0, "keypoints", None) is not None and getattr(r0.keypoints, "data", None) is not None:
                kpts_t = r0.keypoints.data

            for i in range(len(r0.boxes)):
                observations.append({
                    "track_id": int(ids_t[i].item()),
                    "frame_idx": frame_idx,
                    "bbox_xyxy": xyxy_t[i].tolist(),
                    "conf": float(conf_t[i].item()) if conf_t is not None else 1.0,
                    "kpts_xyc": kpts_t[i].tolist() if kpts_t is not None and i < kpts_t.shape[0] else None,
                })

        ready = bufman.update(observations, frame_idx)
        for track_id, seq in ready.items():
            saved_sequences[track_id] = seq
        frame_idx += 1

    saved = build_dataset_from_tracks(
        tracks=saved_sequences,
        output_dir=args.output_dir,
        labels_by_track=labels_by_track,
        extractor=extractor,
        prefix=Path(args.video).stem,
        augment_horizontal_flip=args.augment_horizontal_flip,
    )
    print(f"Saved {len(saved)} sequences to {args.output_dir}")


if __name__ == "__main__":
    main()
