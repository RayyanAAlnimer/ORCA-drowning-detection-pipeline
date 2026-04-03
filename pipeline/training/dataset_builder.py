from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from pipeline.featurization import PoseFeatureExtractor


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _horizontal_flip_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    """Mirror a Stage-4 feature tensor to simulate the opposite camera side."""
    flipped = x.clone()

    # Pose layout is [x0, y0, x1, y1, ...]; mirror x only.
    if flipped.shape[1] >= 34:
        flipped[:, 0:34:2] = -flipped[:, 0:34:2]

    # BBox layout is [cx, cy, w, h] and cx is normalized to frame width.
    if flipped.shape[1] >= 38:
        flipped[:, 34] = 1.0 - flipped[:, 34]

    # Velocity layout is [delta_cx, delta_cy]; horizontal motion changes sign.
    if flipped.shape[1] >= 42:
        flipped[:, 40] = -flipped[:, 40]

    # Body angle is measured from torso direction; horizontal mirroring reflects it.
    if flipped.shape[1] >= 49:
        flipped[:, 48] = _wrap_to_pi(torch.pi - flipped[:, 48])

    return flipped


def build_dataset_from_tracks(
    tracks: Mapping[int, Sequence],
    output_dir: str,
    labels_by_track: Mapping[int, int] | None = None,
    extractor: PoseFeatureExtractor | None = None,
    prefix: str = "sample",
    augment_horizontal_flip: bool = False,
) -> list[Path]:
    extractor = extractor or PoseFeatureExtractor(frame_size=None)
    if labels_by_track is None:
        labels_by_track = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx, (track_id, seq) in enumerate(tracks.items()):
        batch = extractor.extract(list(seq))
        try:
            label = labels_by_track[track_id]
        except KeyError:
            label = 0

        x_tensor = torch.tensor(batch.X, dtype=torch.float32)
        base_meta = dict(batch.meta)
        payload = {
            "X": x_tensor,
            "y": torch.tensor(float(label), dtype=torch.float32),
            "track_id": int(track_id),
            "meta": {**base_meta, "augmentation": "none"},
        }
        out_path = out_dir / f"{prefix}_{idx:05d}.pt"
        torch.save(payload, out_path)
        saved_paths.append(out_path)

        if augment_horizontal_flip:
            aug_payload = {
                "X": _horizontal_flip_feature_tensor(x_tensor),
                "y": torch.tensor(float(label), dtype=torch.float32),
                "track_id": int(track_id),
                "meta": {**base_meta, "augmentation": "horizontal_flip"},
            }
            aug_path = out_dir / f"{prefix}_{idx:05d}_hflip.pt"
            torch.save(aug_payload, aug_path)
            saved_paths.append(aug_path)

    return saved_paths
