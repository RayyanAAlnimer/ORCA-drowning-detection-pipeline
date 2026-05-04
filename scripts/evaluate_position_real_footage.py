from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.buffering import TrackBufferManager
from pipeline.featurization import PoseFeatureExtractor
from pipeline.models import TCNInferencer
from pipeline.preprocessing import FramePreprocessor
from pipeline.tracker import PoseTracker


VIDEO_RE = re.compile(r"^(distress|nondistress)_pos([1-4])_\d+\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MVP-v1 and pose detection quality on real position clips."
    )
    parser.add_argument("--video-dir", default="assets/videos")
    parser.add_argument("--output-dir", default="outputs/logs/real_position_mvp_v1_eval")
    parser.add_argument("--checkpoint", default="models/checkpoints/mvp-v1/tcn_best.pt")
    parser.add_argument("--yolo-model", default="models/pose/yolov8n-pose.pt")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--age-out-seconds", type=float, default=1.0)
    parser.add_argument("--infer-every-n-frames", type=int, default=5)
    parser.add_argument("--min-visible-keypoints", type=int, default=5)
    parser.add_argument("--preprocess-mode", default="center_zoom_enhance")
    parser.add_argument("--preprocess-zoom", type=float, default=1.35)
    parser.add_argument("--preprocess-y-bias", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def discover_videos(video_dir: Path) -> list[dict]:
    videos = []
    for path in sorted(video_dir.glob("*.mp4")):
        match = VIDEO_RE.match(path.name)
        if not match:
            continue
        action, pos = match.groups()
        videos.append(
            {
                "path": path,
                "video_name": path.name,
                "action": action,
                "position": f"pos{pos}",
                "position_index": int(pos),
                "expected_label": 1 if action == "distress" else 0,
            }
        )
    return videos


def selected_pose_success(result, min_visible_keypoints: int) -> tuple[bool, float | None, int, list[float] | None, list | None]:
    if result.boxes is None or len(result.boxes) == 0:
        return False, None, 0, None, None

    xyxy_t = result.boxes.xyxy
    areas = (xyxy_t[:, 2] - xyxy_t[:, 0]) * (xyxy_t[:, 3] - xyxy_t[:, 1])
    idx = int(areas.argmax().item())

    conf_t = result.boxes.conf if hasattr(result.boxes, "conf") else None
    conf = float(conf_t[idx].item()) if conf_t is not None else 1.0
    bbox_xyxy = xyxy_t[idx].tolist()

    kpts_xyc = None
    visible = 0
    if (
        getattr(result, "keypoints", None) is not None
        and getattr(result.keypoints, "data", None) is not None
        and idx < result.keypoints.data.shape[0]
    ):
        kpts_xyc = result.keypoints.data[idx].tolist()
        arr = np.asarray(kpts_xyc, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            visible = int(np.sum(arr[:, 2] > 0.2))

    success = kpts_xyc is not None and visible >= min_visible_keypoints
    return success, conf, visible, bbox_xyxy, kpts_xyc


def evaluate_video(
    item: dict,
    tracker: PoseTracker,
    preprocessor: FramePreprocessor,
    tcn: TCNInferencer,
    args: argparse.Namespace,
) -> dict:
    video_path = item["path"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    bufman = TrackBufferManager(
        window_size_frames=max(1, int(round(args.window_seconds * fps))),
        age_out_frames=max(0, int(round(args.age_out_seconds * fps))),
    )
    extractor = PoseFeatureExtractor(frame_size=(width, height))

    total_frames = 0
    frames_with_box = 0
    successful_pose_frames = 0
    detection_confidences = []
    visible_keypoint_counts = []
    probabilities = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = preprocessor.apply(frame)
        result = tracker.track_frame(frame)

        success, conf, visible, bbox_xyxy, kpts_xyc = selected_pose_success(
            result, args.min_visible_keypoints
        )

        observations = []
        if bbox_xyxy is not None:
            frames_with_box += 1
            if success:
                successful_pose_frames += 1
                if conf is not None:
                    detection_confidences.append(conf)
                visible_keypoint_counts.append(visible)

            observations.append(
                {
                    "track_id": 1,
                    "frame_idx": total_frames,
                    "bbox_xyxy": bbox_xyxy,
                    "conf": conf if conf is not None else 1.0,
                    "kpts_xyc": kpts_xyc,
                }
            )

        ready = bufman.update(observations, total_frames)
        if ready and total_frames % max(1, args.infer_every_n_frames) == 0:
            seq = ready.get(1)
            if seq is not None:
                batch = extractor.extract(seq)
                if batch.X.size:
                    probabilities.append(tcn.predict_proba(batch.X))

        total_frames += 1

    cap.release()

    expected_label = item["expected_label"]
    predictions = [1 if p >= args.threshold else 0 for p in probabilities]
    correct_windows = sum(1 for pred in predictions if pred == expected_label)
    distress_window_count = sum(predictions)

    mean_distress_probability = float(np.mean(probabilities)) if probabilities else None
    correct_class_confidence = None
    if probabilities:
        if expected_label == 1:
            correct_class_confidence = mean_distress_probability
        else:
            correct_class_confidence = float(np.mean([1.0 - p for p in probabilities]))

    return {
        **{k: v for k, v in item.items() if k != "path"},
        "source_video": str(video_path),
        "fps": float(fps),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames_with_box": frames_with_box,
        "successful_pose_frames": successful_pose_frames,
        "pose_detection_success_rate": successful_pose_frames / total_frames if total_frames else 0.0,
        "person_box_detection_rate": frames_with_box / total_frames if total_frames else 0.0,
        "average_successful_detection_confidence": float(np.mean(detection_confidences))
        if detection_confidences
        else None,
        "average_visible_keypoints_on_success": float(np.mean(visible_keypoint_counts))
        if visible_keypoint_counts
        else None,
        "window_count": len(probabilities),
        "mean_distress_probability": mean_distress_probability,
        "correct_class_confidence": correct_class_confidence,
        "distress_window_count": distress_window_count,
        "correct_window_count": correct_windows,
        "classification_window_accuracy": correct_windows / len(probabilities) if probabilities else None,
        "probabilities": probabilities,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["action"], row["position"], row["position_index"])].append(row)

    out = []
    for (action, position, position_index), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2])):
        out.append(
            {
                "action": action,
                "position": position,
                "position_index": position_index,
                "video_count": len(items),
                "average_pose_detection_success_rate": mean_or_none(
                    [r["pose_detection_success_rate"] for r in items]
                ),
                "average_person_box_detection_rate": mean_or_none(
                    [r["person_box_detection_rate"] for r in items]
                ),
                "average_successful_detection_confidence": mean_or_none(
                    [r["average_successful_detection_confidence"] for r in items]
                ),
                "average_visible_keypoints_on_success": mean_or_none(
                    [r["average_visible_keypoints_on_success"] for r in items]
                ),
                "average_distress_probability": mean_or_none(
                    [r["mean_distress_probability"] for r in items]
                ),
                "average_correct_class_confidence": mean_or_none(
                    [r["correct_class_confidence"] for r in items]
                ),
                "average_classification_window_accuracy": mean_or_none(
                    [r["classification_window_accuracy"] for r in items]
                ),
                "total_windows": sum(r["window_count"] for r in items),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary_rows: list[dict], action: str, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    rows = [r for r in summary_rows if r["action"] == action]
    rows.sort(key=lambda r: r["position_index"])
    x = [r["position_index"] for r in rows]
    y = [(r[metric] or 0.0) * 100.0 for r in rows]
    labels = [r["position"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2.4)
    plt.xticks(x, labels)
    plt.ylim(0, 105)
    plt.xlabel("Swimmer position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 2, f"{yi:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(ROOT / args.video_dir)
    tracker = PoseTracker(args.yolo_model)
    preprocessor = FramePreprocessor(
        enabled=True,
        mode=args.preprocess_mode,
        zoom=args.preprocess_zoom,
        y_bias=args.preprocess_y_bias,
    )
    tcn = TCNInferencer(feature_dim=50, weights_path=str(ROOT / args.checkpoint), device="cpu")

    print(f"[run] videos={len(videos)} output={output_dir}", flush=True)
    rows = []
    start = time.time()
    for idx, item in enumerate(videos, start=1):
        row = evaluate_video(item, tracker, preprocessor, tcn, args)
        rows.append(row)
        print(
            f"[{idx:02d}/{len(videos)}] {row['video_name']} "
            f"pose={row['pose_detection_success_rate'] * 100:.1f}% "
            f"windows={row['window_count']} mean_p={row['mean_distress_probability']}",
            flush=True,
        )

    summary_rows = aggregate(rows)

    result = {
        "checkpoint": args.checkpoint,
        "yolo_model": args.yolo_model,
        "threshold": args.threshold,
        "preprocessing": {
            "enabled": True,
            "mode": args.preprocess_mode,
            "zoom": args.preprocess_zoom,
            "y_bias": args.preprocess_y_bias,
        },
        "metric_definitions": {
            "pose_detection_success_rate": "successful pose frames / total frames; success requires selected swimmer pose with at least min_visible_keypoints",
            "average_successful_detection_confidence": "mean YOLO person confidence over frames where pose detection succeeded",
            "mean_distress_probability": "mean MVP-v1 distress probability over evaluated temporal windows",
            "correct_class_confidence": "distress probability for distress clips, 1 - distress probability for nondistress clips",
        },
        "min_visible_keypoints": args.min_visible_keypoints,
        "video_count": len(rows),
        "elapsed_seconds": time.time() - start,
        "videos": rows,
        "position_summary": summary_rows,
    }

    json_path = output_dir / "real_position_mvp_v1_eval.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    video_csv = output_dir / "real_position_mvp_v1_per_video.csv"
    summary_csv = output_dir / "real_position_mvp_v1_position_summary.csv"
    write_csv(
        video_csv,
        rows,
        [
            "video_name",
            "action",
            "position",
            "position_index",
            "total_frames",
            "successful_pose_frames",
            "pose_detection_success_rate",
            "average_successful_detection_confidence",
            "average_visible_keypoints_on_success",
            "window_count",
            "mean_distress_probability",
            "correct_class_confidence",
            "classification_window_accuracy",
        ],
    )
    write_csv(
        summary_csv,
        summary_rows,
        [
            "action",
            "position",
            "position_index",
            "video_count",
            "average_pose_detection_success_rate",
            "average_person_box_detection_rate",
            "average_successful_detection_confidence",
            "average_visible_keypoints_on_success",
            "average_distress_probability",
            "average_correct_class_confidence",
            "average_classification_window_accuracy",
            "total_windows",
        ],
    )

    plot_metric(
        summary_rows,
        "distress",
        "average_pose_detection_success_rate",
        "Average successful pose detection (%)",
        "Distress Clips: Pose Detection vs Swimmer Position",
        output_dir / "distress_pose_detection_vs_position.png",
    )
    plot_metric(
        summary_rows,
        "nondistress",
        "average_pose_detection_success_rate",
        "Average successful pose detection (%)",
        "Non-Distress Clips: Pose Detection vs Swimmer Position",
        output_dir / "nondistress_pose_detection_vs_position.png",
    )
    plot_metric(
        summary_rows,
        "distress",
        "average_correct_class_confidence",
        "Average correct-class confidence (%)",
        "Distress Clips: MVP-v1 Confidence vs Swimmer Position",
        output_dir / "distress_mvp_v1_confidence_vs_position.png",
    )
    plot_metric(
        summary_rows,
        "nondistress",
        "average_correct_class_confidence",
        "Average correct-class confidence (%)",
        "Non-Distress Clips: MVP-v1 Confidence vs Swimmer Position",
        output_dir / "nondistress_mvp_v1_confidence_vs_position.png",
    )

    print(f"[done] {json_path}", flush=True)
    print(f"[done] {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
