from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from pipeline.featurization import PoseFeatureExtractor


def build_dataset_from_tracks(
    tracks: Mapping[int, Sequence],
    output_dir: str,
    labels_by_track: Mapping[int, int] | None = None,
    extractor: PoseFeatureExtractor | None = None,
    prefix: str = "sample",
) -> list[Path]:
    extractor = extractor or PoseFeatureExtractor(frame_size=None)
    labels_by_track = labels_by_track or {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx, (track_id, seq) in enumerate(tracks.items()):
        batch = extractor.extract(list(seq))
        payload = {
            "X": torch.tensor(batch.X, dtype=torch.float32),
            "y": torch.tensor(float(labels_by_track.get(track_id, 0)), dtype=torch.float32),
            "track_id": int(track_id),
            "meta": batch.meta,
        }
        out_path = out_dir / f"{prefix}_{idx:05d}.pt"
        torch.save(payload, out_path)
        saved_paths.append(out_path)

    return saved_paths
