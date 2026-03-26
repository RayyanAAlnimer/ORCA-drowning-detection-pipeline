from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pose_estimator import PoseEstimator
from pipeline.visualizer import draw_results

MODEL_NAME = "yolov8n-pose.pt"
DEFAULT_INPUT_PATH = "data/raw/input.mp4"
DEFAULT_OUTPUT_PATH = "outputs/yolo_only_out.mp4"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose on a single image or video and save annotated output."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to an input image or video")
    parser.add_argument("--output", default=None, help="Path to annotated output media")
    parser.add_argument("--model", default=MODEL_NAME, help="Path to YOLO pose model")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return Path("outputs") / f"{input_path.stem}_annotated.jpg"
    return Path("outputs") / f"{input_path.stem}_annotated.mp4"


def run_image(pose: PoseEstimator, input_path: Path, output_path: Path) -> None:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"Could not open image: {input_path}")

    result = pose.infer(frame)
    annotated = draw_results(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), annotated)
    if not ok:
        raise RuntimeError(f"Could not write annotated image: {output_path}")

    num_people = 0 if result.boxes is None else len(result.boxes)
    print(f"[YOLO-Only] image={input_path} people_detected={num_people}")
    print(f"[YOLO-Only] wrote annotated image -> {output_path}")


def run_video(pose: PoseEstimator, input_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pose.infer(frame)
        annotated = draw_results(result)
        out.write(annotated)

        if frame_idx % 30 == 0:
            num_people = 0 if result.boxes is None else len(result.boxes)
            print(f"[YOLO-Only] frame={frame_idx:05d} people_detected={num_people}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"[YOLO-Only] wrote annotated video -> {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    pose = PoseEstimator(args.model)

    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        run_image(pose, input_path, output_path)
    else:
        run_video(pose, input_path, output_path)


if __name__ == "__main__":
    main()
