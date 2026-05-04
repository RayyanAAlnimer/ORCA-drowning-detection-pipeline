from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from statistics import mean

from pipeline.buffering import TrackBufferManager
from pipeline.tracker import PoseTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO detection continuity on the new raw batch.")
    parser.add_argument('--distress-dir', default='data/raw/distress/thrashing')
    parser.add_argument('--nondistress-dir', default='data/raw/nondistress/treading')
    parser.add_argument('--model', default='models/pose/yolov8n-pose.pt')
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--age-out-seconds', type=float, default=1.0)
    parser.add_argument('--stride-frames', type=int, default=30)
    parser.add_argument('--output', default='outputs/logs/yolo_test_20260411/new_batch_yolo_summary.json')
    return parser.parse_args()


def longest_run(flags: list[int]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_exportable_sequences(tracker: PoseTracker, video_path: Path, window_frames: int, age_out_frames: int, stride_frames: int) -> int:
    bufman = TrackBufferManager(window_size_frames=window_frames, age_out_frames=age_out_frames)
    frame_idx = 0
    last_saved_frame_by_track: dict[int, int] = {}
    saved_count = 0
    for result in tracker.track_video(str(video_path)):
        observations = []
        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            ids_t = result.boxes.id.int()
            xyxy_t = result.boxes.xyxy
            conf_t = result.boxes.conf if hasattr(result.boxes, 'conf') else None
            kpts_t = None
            if getattr(result, 'keypoints', None) is not None and getattr(result.keypoints, 'data', None) is not None:
                kpts_t = result.keypoints.data
            for i in range(len(result.boxes)):
                observations.append({
                    'track_id': int(ids_t[i].item()),
                    'frame_idx': frame_idx,
                    'bbox_xyxy': xyxy_t[i].tolist(),
                    'conf': float(conf_t[i].item()) if conf_t is not None else 1.0,
                    'kpts_xyc': kpts_t[i].tolist() if kpts_t is not None and i < kpts_t.shape[0] else None,
                })
        ready = bufman.update(observations, frame_idx)
        for track_id in ready:
            last_saved = last_saved_frame_by_track.get(track_id)
            if last_saved is None or (frame_idx - last_saved) >= max(1, stride_frames):
                saved_count += 1
                last_saved_frame_by_track[track_id] = frame_idx
        frame_idx += 1
    return saved_count


def analyze_video(tracker: PoseTracker, video_path: Path, label: str, window_frames: int, age_out_frames: int, stride_frames: int) -> dict:
    detect_flags = []
    confs = []
    track_counts = []

    for result in tracker.track_video(str(video_path)):
        detected = int(result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0)
        detect_flags.append(detected)
        if detected:
            conf_t = result.boxes.conf if hasattr(result.boxes, 'conf') else None
            if conf_t is not None:
                confs.extend([float(v.item()) for v in conf_t])
            ids_t = result.boxes.id.int()
            track_counts.append(len(set(int(v.item()) for v in ids_t)))
        else:
            track_counts.append(0)

    total_frames = len(detect_flags)
    frames_with_detection = int(sum(detect_flags))
    saved_sequences = count_exportable_sequences(tracker, video_path, window_frames, age_out_frames, stride_frames)

    if frames_with_detection == 0:
        failure_mode = 'no_detection'
    elif saved_sequences == 0:
        failure_mode = 'fragmented_detection'
    else:
        failure_mode = 'exportable'

    return {
        'video_name': video_path.name,
        'video_path': str(video_path),
        'class': label,
        'total_frames': total_frames,
        'frames_with_detection': frames_with_detection,
        'detection_ratio': (frames_with_detection / total_frames) if total_frames else 0.0,
        'longest_detection_run': longest_run(detect_flags),
        'mean_confidence': mean(confs) if confs else 0.0,
        'max_tracks_in_frame': max(track_counts) if track_counts else 0,
        'saved_sequences': saved_sequences,
        'failure_mode': failure_mode,
    }


def main() -> None:
    args = parse_args()
    distress_dir = ROOT / args.distress_dir
    nondistress_dir = ROOT / args.nondistress_dir
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = PoseTracker(args.model)
    window_frames = max(1, int(round(args.window_seconds * args.fps)))
    age_out_frames = max(0, int(round(args.age_out_seconds * args.fps)))

    distress_files = sorted([p for p in distress_dir.glob('thrashing*.mp4')])
    nondistress_files = sorted([p for p in nondistress_dir.glob('treading*.mp4')])

    rows = []
    for p in distress_files:
        row = analyze_video(tracker, p, 'distress', window_frames, age_out_frames, args.stride_frames)
        rows.append(row)
        print(f"[distress] {p.name} -> failure_mode={row['failure_mode']} saved_sequences={row['saved_sequences']} longest_run={row['longest_detection_run']}", flush=True)
    for p in nondistress_files:
        row = analyze_video(tracker, p, 'nondistress', window_frames, age_out_frames, args.stride_frames)
        rows.append(row)
        print(f"[nondistress] {p.name} -> failure_mode={row['failure_mode']} saved_sequences={row['saved_sequences']} longest_run={row['longest_detection_run']}", flush=True)

    summary = {
        'window_frames': window_frames,
        'stride_frames': args.stride_frames,
        'age_out_frames': age_out_frames,
        'counts': {
            'distress_total': len(distress_files),
            'nondistress_total': len(nondistress_files),
            'exportable': sum(1 for r in rows if r['failure_mode'] == 'exportable'),
            'fragmented_detection': sum(1 for r in rows if r['failure_mode'] == 'fragmented_detection'),
            'no_detection': sum(1 for r in rows if r['failure_mode'] == 'no_detection'),
        },
        'rows': rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[done] saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
