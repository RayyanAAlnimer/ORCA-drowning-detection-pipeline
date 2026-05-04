from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.buffering import TrackBufferManager
from pipeline.decision import DecisionLevel, DecisionManager
from pipeline.featurization import PoseFeatureExtractor
from pipeline.models import TCNInferencer
from pipeline.preprocessing import FramePreprocessor
from pipeline.tracker import PoseTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an MVP-v1 alarm demo video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint", default="models/checkpoints/mvp-v1/tcn_best.pt")
    parser.add_argument("--yolo-model", default="models/pose/yolov8n-pose.pt")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--age-out-seconds", type=float, default=1.0)
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--possible-th", type=float, default=0.60)
    parser.add_argument("--confirm-th", type=float, default=0.80)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--preprocess-mode", default="center_zoom_enhance")
    parser.add_argument("--preprocess-zoom", type=float, default=1.35)
    return parser.parse_args()


def draw_caption(frame, text: str, color=(0, 0, 255)) -> None:
    h, w = frame.shape[:2]
    margin = max(16, int(0.02 * w))
    box_h = max(70, int(0.10 * h))
    y1 = h - box_h - margin
    y2 = h - margin
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, y1), (w - margin, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    font_scale = max(0.8, min(1.6, w / 1050))
    thickness = max(2, int(round(font_scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(margin + 16, (w - tw) // 2)
    y = y1 + (box_h + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracker = PoseTracker(args.yolo_model)
    preprocessor = FramePreprocessor(enabled=True, mode=args.preprocess_mode, zoom=args.preprocess_zoom)
    buffer = TrackBufferManager(
        window_size_frames=max(1, int(round(args.window_seconds * fps))),
        age_out_frames=max(0, int(round(args.age_out_seconds * fps))),
    )
    extractor = PoseFeatureExtractor(frame_size=(w, h))
    tcn = TCNInferencer(feature_dim=50, weights_path=str(ROOT / args.checkpoint), device="cpu")
    decision = DecisionManager(
        ema_alpha=0.30,
        possible_th=args.possible_th,
        confirm_th=args.confirm_th,
        hold_seconds=args.hold_seconds,
        cooldown_seconds=args.cooldown_seconds,
        stale_track_seconds=3.0,
    )

    frame_idx = 0
    latest_raw = 0.0
    confirmed_at = None
    confirmed_active = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        processed = preprocessor.apply(frame)
        result = tracker.track_frame(processed)

        observations = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy_t = result.boxes.xyxy
            areas = (xyxy_t[:, 2] - xyxy_t[:, 0]) * (xyxy_t[:, 3] - xyxy_t[:, 1])
            i = int(areas.argmax().item())
            conf_t = result.boxes.conf if hasattr(result.boxes, "conf") else None
            kpts_t = None
            if getattr(result, "keypoints", None) is not None and getattr(result.keypoints, "data", None) is not None:
                kpts_t = result.keypoints.data
            observations.append(
                {
                    "track_id": 1,
                    "frame_idx": frame_idx,
                    "bbox_xyxy": xyxy_t[i].tolist(),
                    "conf": float(conf_t[i].item()) if conf_t is not None else 1.0,
                    "kpts_xyc": kpts_t[i].tolist() if kpts_t is not None and i < kpts_t.shape[0] else None,
                }
            )

        ready = buffer.update(observations, frame_idx)
        video_ts = frame_idx / float(fps)
        if ready and frame_idx % max(1, args.infer_every) == 0 and 1 in ready:
            batch = extractor.extract(ready[1])
            latest_raw = tcn.predict_proba(batch.X)
            events = decision.update_batch({1: latest_raw}, ts=video_ts)
            for event in events:
                if event.level == DecisionLevel.CONFIRMED:
                    confirmed_active = True
                    if confirmed_at is None:
                        confirmed_at = video_ts

        annotated = result.plot()
        snapshot = decision.snapshot()
        level, smooth_prob, _, _ = snapshot.get(1, (DecisionLevel.NORMAL, 0.0, 0.0, 0.0))
        if level == DecisionLevel.CONFIRMED:
            confirmed_active = True

        status = f"MVP-v1 distress p={smooth_prob:.2f} | status={level.value.upper()}"
        cv2.putText(annotated, status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 1, cv2.LINE_AA)

        if confirmed_active:
            draw_caption(annotated, "ALARM: DISTRESS DETECTED", (0, 0, 255))
        elif level == DecisionLevel.POSSIBLE:
            draw_caption(annotated, "Possible distress - monitoring", (0, 165, 255))

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"output={output_path}")
    print(f"frames={frame_idx} fps={fps:.2f}")
    print(f"confirmed_at_seconds={confirmed_at}")


if __name__ == "__main__":
    main()
