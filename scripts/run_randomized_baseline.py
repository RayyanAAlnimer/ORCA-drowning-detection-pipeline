from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.buffering import TrackBufferManager
from pipeline.evaluation.evaluator import evaluate_model
from pipeline.featurization import PoseFeatureExtractor
from pipeline.models import TCNBinaryClassifier
from pipeline.tracker import PoseTracker
from pipeline.training.dataloader import TemporalSequenceDataset
from pipeline.training.dataset_builder import build_dataset_from_tracks
from pipeline.training.trainer import train_model
from pipeline.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run randomized raw-video baseline training.')
    parser.add_argument('--distress-dir', default='data/raw/distress/thrashing')
    parser.add_argument('--nondistress-dir', default='data/raw/nondistress/treading')
    parser.add_argument('--model', default='tcn', choices=['tcn'])
    parser.add_argument('--yolo-model', default='models/pose/yolov8n-pose.pt')
    parser.add_argument('--model-config', default='configs/model/tcn.yaml')
    parser.add_argument('--training-config', default='configs/training/default.yaml')
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--stride-frames', type=int, default=30)
    parser.add_argument('--age-out-seconds', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', default='baseline_train_20260410_randomized_70_15_15')
    parser.add_argument('--skip-build', action='store_true')
    return parser.parse_args()


def split_counts(n: int) -> tuple[int, int, int]:
    train = round(0.70 * n)
    val = round(0.15 * n)
    if train + val > n:
        val = max(0, n - train)
    test = n - train - val
    return train, val, test


def stratified_video_split(distress_files: list[Path], nondistress_files: list[Path], seed: int) -> dict[str, list[tuple[Path, int, str]]]:
    rng = random.Random(seed)
    distress = distress_files[:]
    nondistress = nondistress_files[:]
    rng.shuffle(distress)
    rng.shuffle(nondistress)

    d_train, d_val, d_test = split_counts(len(distress))
    n_train, n_val, n_test = split_counts(len(nondistress))

    split_map = {
        'train': [(p, 1, 'distress') for p in distress[:d_train]] + [(p, 0, 'nondistress') for p in nondistress[:n_train]],
        'val': [(p, 1, 'distress') for p in distress[d_train:d_train + d_val]] + [(p, 0, 'nondistress') for p in nondistress[n_train:n_train + n_val]],
        'test': [(p, 1, 'distress') for p in distress[d_train + d_val:]] + [(p, 0, 'nondistress') for p in nondistress[n_train + n_val:]],
    }
    for key in split_map:
        rng.shuffle(split_map[key])
    return split_map


def build_sequences_for_video(video_path: Path, label: int, output_dir: Path, tracker: PoseTracker, window_seconds: float, stride_frames: int, age_out_seconds: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    bufman = TrackBufferManager(
        window_size_frames=max(1, int(round(window_seconds * fps))),
        age_out_frames=max(0, int(round(age_out_seconds * fps))),
    )
    extractor = PoseFeatureExtractor(frame_size=(w, h))

    saved_sequences: dict[int, list] = {}
    labels_by_track: dict[int, int] = {}
    last_saved_frame_by_track: dict[int, int] = {}
    sample_track_id = 0
    frame_idx = 0

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
        for track_id, seq in ready.items():
            last_saved = last_saved_frame_by_track.get(track_id)
            if last_saved is None or (frame_idx - last_saved) >= max(1, int(stride_frames)):
                sample_track_id += 1
                saved_sequences[sample_track_id] = list(seq)
                labels_by_track[sample_track_id] = label
                last_saved_frame_by_track[track_id] = frame_idx
        frame_idx += 1

    saved = build_dataset_from_tracks(
        tracks=saved_sequences,
        output_dir=str(output_dir),
        labels_by_track=labels_by_track,
        extractor=extractor,
        prefix=video_path.stem,
    )
    return len(saved)




def temporal_collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    padded = []
    for x in xs:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            x = F.pad(x, (0, 0, pad_len, 0))
        padded.append(x)
    return torch.stack(padded, dim=0), torch.stack(list(ys), dim=0)

def plot_accuracy(history: list[dict], out_svg: Path, out_png: Path) -> None:
    epochs = [row['epoch'] for row in history]
    train_acc = [row['train_accuracy'] for row in history]
    val_acc = [row['val_accuracy'] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker='o', label='Train Accuracy')
    plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], out_svg: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Pred Distress', 'Pred Non-Distress'])
    ax.set_yticks([0, 1], labels=['Actual Distress', 'Actual Non-Distress'])
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_name = args.run_name
    processed_root = ROOT / 'data' / 'processed' / run_name
    log_dir = ROOT / 'outputs' / 'logs' / run_name
    checkpoint_dir = ROOT / 'models' / 'checkpoints' / run_name

    for path in [processed_root / 'train', processed_root / 'val', processed_root / 'test', log_dir, checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)

    distress_files = sorted(Path(args.distress_dir).glob('*.mp4'))
    nondistress_files = sorted(Path(args.nondistress_dir).glob('*.mp4'))
    splits = stratified_video_split(distress_files, nondistress_files, args.seed)

    split_summary = {
        'run_name': run_name,
        'seed': args.seed,
        'window_seconds': args.window_seconds,
        'stride_frames': args.stride_frames,
        'age_out_seconds': args.age_out_seconds,
        'raw_counts': {
            'distress': len(distress_files),
            'nondistress': len(nondistress_files),
        },
        'raw_split_counts': {},
        'processed_split_counts': {},
        'videos': {},
    }

    print(f'[run] {run_name}', flush=True)
    print(f'[raw] distress={len(distress_files)} nondistress={len(nondistress_files)}', flush=True)

    for split_name, items in splits.items():
        split_summary['raw_split_counts'][split_name] = {
            'total': len(items),
            'distress': sum(1 for _, label, _ in items if label == 1),
            'nondistress': sum(1 for _, label, _ in items if label == 0),
        }
        split_summary['videos'][split_name] = [str(path) for path, _, _ in items]

    if not args.skip_build:
        tracker = PoseTracker(args.yolo_model)
        per_video_rows: list[dict] = []
        for split_name, items in splits.items():
            split_output_dir = processed_root / split_name
            split_counter = Counter()
            for idx, (video_path, label, class_name) in enumerate(items, start=1):
                saved_count = build_sequences_for_video(
                    video_path=video_path,
                    label=label,
                    output_dir=split_output_dir,
                    tracker=tracker,
                    window_seconds=args.window_seconds,
                    stride_frames=args.stride_frames,
                    age_out_seconds=args.age_out_seconds,
                )
                split_counter['videos'] += 1
                split_counter['sequences'] += saved_count
                split_counter[f'{class_name}_videos'] += 1
                split_counter[f'{class_name}_sequences'] += saved_count
                row = {
                    'split': split_name,
                    'class': class_name,
                    'label': label,
                    'video': str(video_path),
                    'video_name': video_path.name,
                    'sequence_count': saved_count,
                }
                per_video_rows.append(row)
                print(f'[build][{split_name}][{idx}/{len(items)}][{class_name}] {video_path.name} -> {saved_count} sequences', flush=True)

            split_summary['processed_split_counts'][split_name] = dict(split_counter)
            print(f'[build][{split_name}] total_sequences={split_counter["sequences"]}', flush=True)

        (log_dir / 'split_summary.json').write_text(json.dumps(split_summary, indent=2), encoding='utf-8')
        (log_dir / 'per_video_sequence_counts.json').write_text(json.dumps(per_video_rows, indent=2), encoding='utf-8')
    else:
        print('[build] skipped; reusing existing processed split', flush=True)
        split_summary_path = log_dir / 'split_summary.json'
        if split_summary_path.exists():
            split_summary = json.loads(split_summary_path.read_text(encoding='utf-8'))

    training_cfg = load_config(args.training_config)
    model_cfg = load_config(args.model_config)

    train_dataset = TemporalSequenceDataset(str(processed_root / 'train'))
    val_dataset = TemporalSequenceDataset(str(processed_root / 'val'))
    test_dataset = TemporalSequenceDataset(str(processed_root / 'test'))

    batch_size = int(training_cfg.get('batch_size', 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate)

    model = TCNBinaryClassifier(
        in_channels=int(model_cfg.get('input_dim', 50)),
        channels=list(model_cfg.get('num_channels', [64, 64, 128])),
        kernel_size=int(model_cfg.get('kernel_size', 3)),
        dropout=float(model_cfg.get('dropout', 0.2)),
    )

    print('[train] starting TCN training', flush=True)
    metrics_payload = train_model(
        args.model,
        model,
        train_loader,
        val_loader,
        training_cfg,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )
    print('[train] finished', flush=True)

    ckpt = torch.load(checkpoint_dir / f'{args.model}_best.pt', map_location='cpu')
    best_model = TCNBinaryClassifier(
        in_channels=int(model_cfg.get('input_dim', 50)),
        channels=list(model_cfg.get('num_channels', [64, 64, 128])),
        kernel_size=int(model_cfg.get('kernel_size', 3)),
        dropout=float(model_cfg.get('dropout', 0.2)),
    )
    best_model.load_state_dict(ckpt['state_dict'])

    threshold = float(training_cfg.get('threshold', 0.5))
    val_eval = evaluate_model(args.model, best_model, val_loader, device=training_cfg.get('device', 'cpu'), threshold=threshold)
    test_eval = evaluate_model(args.model, best_model, test_loader, device=training_cfg.get('device', 'cpu'), threshold=threshold)

    (log_dir / 'tcn_val_evaluation.json').write_text(json.dumps(val_eval, indent=2), encoding='utf-8')
    (log_dir / 'tcn_test_evaluation.json').write_text(json.dumps(test_eval, indent=2), encoding='utf-8')

    history = metrics_payload['history']
    best_epoch = max(history, key=lambda row: row.get('val_f1', float('-inf')))

    preds = test_eval['predictions']
    targets = test_eval['targets']
    tp = sum(1 for p, t in zip(preds, targets) if p >= threshold and t == 1.0)
    tn = sum(1 for p, t in zip(preds, targets) if p < threshold and t == 0.0)
    fp = sum(1 for p, t in zip(preds, targets) if p >= threshold and t == 0.0)
    fn = sum(1 for p, t in zip(preds, targets) if p < threshold and t == 1.0)
    confusion = [[tp, fn], [fp, tn]]

    summary = {
        'run_name': run_name,
        'best_validation': {
            'epoch': best_epoch['epoch'],
            'accuracy': best_epoch['val_accuracy'],
            'loss': best_epoch['val_loss'],
            'precision': best_epoch['val_precision'],
            'recall': best_epoch['val_recall'],
            'f1': best_epoch['val_f1'],
            'map': best_epoch['val_map'],
        },
        'test_metrics': {
            'accuracy': test_eval['accuracy'],
            'precision': test_eval['precision'],
            'recall': test_eval['recall'],
            'f1': test_eval['f1'],
            'map': test_eval['map'],
        },
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        },
    }
    (log_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    plot_accuracy(history, log_dir / 'tcn_accuracy_plot_matplotlib.svg', log_dir / 'tcn_accuracy_plot_matplotlib.png')
    plot_confusion_matrix(confusion, log_dir / 'tcn_confusion_matrix_matplotlib.svg', log_dir / 'tcn_confusion_matrix_matplotlib.png')

    print(f'[done] summary saved to {log_dir / "summary.json"}', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
