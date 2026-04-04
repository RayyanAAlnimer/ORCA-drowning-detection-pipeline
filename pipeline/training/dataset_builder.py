from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import random

import torch

from pipeline.featurization import PoseFeatureExtractor


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _horizontal_flip_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    """Mirror a Stage-4 feature tensor to simulate the opposite camera side."""
    flipped = x.clone()

    if flipped.shape[1] >= 34:
        flipped[:, 0:34:2] = -flipped[:, 0:34:2]
    if flipped.shape[1] >= 38:
        flipped[:, 34] = 1.0 - flipped[:, 34]
    if flipped.shape[1] >= 42:
        flipped[:, 40] = -flipped[:, 40]
    if flipped.shape[1] >= 49:
        flipped[:, 48] = _wrap_to_pi(torch.pi - flipped[:, 48])

    return flipped


def _lighting_jitter_feature_tensor(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Proxy for glare / lighting change by degrading confidence and adding mild noise."""
    aug = x.clone()
    conf_scale = rng.uniform(0.6, 0.9)
    noise_scale = rng.uniform(0.01, 0.03)

    if aug.shape[1] >= 34:
        aug[:, :34] = aug[:, :34] + torch.randn_like(aug[:, :34]) * noise_scale
    if aug.shape[1] >= 40:
        aug[:, 34:40] = aug[:, 34:40] + torch.randn_like(aug[:, 34:40]) * (noise_scale * 0.5)
        aug[:, 34:38] = aug[:, 34:38].clamp(0.0, 1.0)
        aug[:, 38:40] = (aug[:, 38:40] * conf_scale).clamp(0.0, 1.0)
    if aug.shape[1] >= 50:
        aug[:, 47] = (aug[:, 47] * conf_scale).clamp(0.0, 1.0)

    return aug


def _occlusion_feature_tensor(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Proxy for partial occlusion by hiding a small subset of keypoints and reducing visibility."""
    aug = x.clone()

    if aug.shape[1] >= 34:
        num_points = rng.randint(2, 5)
        kpts = list(range(17))
        rng.shuffle(kpts)
        for kp in kpts[:num_points]:
            aug[:, kp * 2: kp * 2 + 2] = 0.0

    if aug.shape[1] >= 40:
        aug[:, 38:40] = (aug[:, 38:40] * rng.uniform(0.4, 0.8)).clamp(0.0, 1.0)
    if aug.shape[1] >= 50:
        aug[:, 47] = (aug[:, 47] * rng.uniform(0.4, 0.8)).clamp(0.0, 1.0)

    return aug


def _build_augmented_tensor(
    x: torch.Tensor,
    rng: random.Random,
    allow_flip: bool,
    allow_lighting: bool,
    allow_occlusion: bool,
) -> tuple[torch.Tensor, str]:
    choices = []
    if allow_flip:
        choices.append(("horizontal_flip", _horizontal_flip_feature_tensor))
    if allow_lighting:
        choices.append(("lighting_jitter", lambda t: _lighting_jitter_feature_tensor(t, rng)))
    if allow_occlusion:
        choices.append(("occlusion", lambda t: _occlusion_feature_tensor(t, rng)))

    if not choices:
        return x.clone(), "none"

    name, fn = rng.choice(choices)
    aug = fn(x)

    # Occasionally combine a geometric change with one visibility-related perturbation.
    if name == "horizontal_flip" and rng.random() < 0.35:
        combo_choices = []
        if allow_lighting:
            combo_choices.append(("lighting_jitter", lambda t: _lighting_jitter_feature_tensor(t, rng)))
        if allow_occlusion:
            combo_choices.append(("occlusion", lambda t: _occlusion_feature_tensor(t, rng)))
        if combo_choices:
            combo_name, combo_fn = rng.choice(combo_choices)
            aug = combo_fn(aug)
            name = f"horizontal_flip+{combo_name}"

    return aug, name


def build_dataset_from_tracks(
    tracks: Mapping[int, Sequence],
    output_dir: str,
    labels_by_track: Mapping[int, int] | None = None,
    extractor: PoseFeatureExtractor | None = None,
    prefix: str = "sample",
    augment_horizontal_flip: bool = False,
    augment_lighting_jitter: bool = False,
    augment_occlusion: bool = False,
    augmentation_probability: float = 0.35,
    augmentation_seed: int = 42,
) -> list[Path]:
    extractor = extractor or PoseFeatureExtractor(frame_size=None)
    if labels_by_track is None:
        labels_by_track = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(augmentation_seed))

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

        use_augmentation = any([augment_horizontal_flip, augment_lighting_jitter, augment_occlusion])
        if use_augmentation and rng.random() < float(augmentation_probability):
            aug_tensor, aug_name = _build_augmented_tensor(
                x_tensor,
                rng,
                allow_flip=augment_horizontal_flip,
                allow_lighting=augment_lighting_jitter,
                allow_occlusion=augment_occlusion,
            )
            aug_payload = {
                "X": aug_tensor,
                "y": torch.tensor(float(label), dtype=torch.float32),
                "track_id": int(track_id),
                "meta": {**base_meta, "augmentation": aug_name},
            }
            aug_path = out_dir / f"{prefix}_{idx:05d}_aug.pt"
            torch.save(aug_payload, aug_path)
            saved_paths.append(aug_path)

    return saved_paths
