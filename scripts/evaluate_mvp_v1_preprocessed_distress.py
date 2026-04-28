from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.models import TCNBinaryClassifier
from pipeline.tracker import PoseTracker
from scripts.run_randomized_baseline import build_sequences_for_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mvp-v1 on preprocessed real distress clips.")
    parser.add_argument("--input-dir", default="data/raw/distress/thrashing")
    parser.add_argument("--pattern", default="IMG*.mp4")
    parser.add_argument("--output-dir", default="outputs/logs/mvp_v1_preprocessed_distress_eval")
    parser.add_argument("--checkpoint", default="models/checkpoints/mvp-v1/tcn_best.pt")
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    enhanced = cv2.merge((l_chan, a_chan, b_chan))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)


def crop_zoom(frame, zoom: float, y_bias: float = 0.0):
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(round(w / zoom)))
    crop_h = max(1, int(round(h / zoom)))
    cx = w // 2
    cy = int(round((h / 2) + (y_bias * h)))
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_video(input_path: Path, output_path: Path, variant: str) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if variant == "enhance":
            frame = enhance_frame(frame)
        elif variant == "center_zoom_enhance":
            frame = enhance_frame(crop_zoom(frame, zoom=1.5, y_bias=0.0))
        elif variant == "lower_center_zoom_enhance":
            frame = enhance_frame(crop_zoom(frame, zoom=1.45, y_bias=0.12))
        else:
            raise ValueError(f"Unknown variant: {variant}")
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    return {"fps": fps, "width": w, "height": h, "frame_count": frame_count}


def load_mvp_v1(checkpoint: Path) -> TCNBinaryClassifier:
    model = TCNBinaryClassifier(in_channels=50, channels=[64, 64, 128], kernel_size=3, dropout=0.2)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def score_sequences(model: TCNBinaryClassifier, sequence_dir: Path, threshold: float) -> dict:
    probs: list[float] = []
    with torch.no_grad():
        for sample_path in sorted(sequence_dir.glob("*.pt")):
            sample = torch.load(sample_path, map_location="cpu")
            x = sample["X"].float().transpose(0, 1).unsqueeze(0)
            probs.append(float(torch.sigmoid(model(x)).item()))
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "sequence_count": len(probs),
        "probabilities": probs,
        "predictions": preds,
        "max_probability": max(probs) if probs else None,
        "mean_probability": (sum(probs) / len(probs)) if probs else None,
        "distress_windows": sum(preds),
    }


def clear_pt_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for sample_path in path.glob("*.pt"):
        sample_path.unlink()


def main() -> None:
    args = parse_args()
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    video_dir = output_dir / "videos"
    sequence_root = output_dir / "sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    sequence_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob(args.pattern))
    tracker = PoseTracker(args.yolo_model)
    model = load_mvp_v1(ROOT / args.checkpoint)
    variants = ["original", "enhance", "center_zoom_enhance", "lower_center_zoom_enhance"]
    rows = []

    print(f"[run] videos={len(videos)} variants={len(variants)}", flush=True)
    for video in videos:
        for variant in variants:
            if variant == "original":
                eval_video = video
                video_meta = None
            else:
                variant_dir = video_dir / variant
                variant_dir.mkdir(parents=True, exist_ok=True)
                eval_video = variant_dir / video.name
                video_meta = preprocess_video(video, eval_video, variant)

            sequence_dir = sequence_root / variant / video.stem
            clear_pt_files(sequence_dir)
            saved_count = build_sequences_for_video(
                video_path=eval_video,
                label=1,
                output_dir=sequence_dir,
                tracker=tracker,
                window_seconds=2.0,
                stride_frames=30,
                age_out_seconds=1.0,
            )
            score = score_sequences(model, sequence_dir, args.threshold)
            row = {
                "video_name": video.name,
                "source_video": str(video),
                "eval_video": str(eval_video),
                "variant": variant,
                "preprocess_meta": video_meta,
                "saved_count": saved_count,
                **score,
            }
            rows.append(row)
            print(
                f"[{video.name}][{variant}] seq={score['sequence_count']} "
                f"max={score['max_probability']} distress_windows={score['distress_windows']}",
                flush=True,
            )

    summary = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "variants": variants,
        "video_count": len(videos),
        "rows": rows,
    }
    out_path = output_dir / "mvp_v1_preprocessed_distress_eval.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
