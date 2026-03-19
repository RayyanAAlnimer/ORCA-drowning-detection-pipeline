from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from pipeline.models import MODEL_REGISTRY, TCNBinaryClassifier
from pipeline.training.dataloader import TemporalSequenceDataset
from pipeline.training.trainer import train_model
from pipeline.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal model.")
    parser.add_argument("--model", choices=["tcn", "cnn", "tsm"], required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--training-config", default="configs/training/default.yaml")
    return parser.parse_args()


def build_model(model_name: str, model_cfg: dict):
    if model_name == "tcn":
        return TCNBinaryClassifier(
            in_channels=int(model_cfg.get("input_dim", 42)),
            channels=list(model_cfg.get("num_channels", [64, 64, 128])),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(input_dim=int(model_cfg.get("input_dim", 42)))


def main() -> None:
    args = parse_args()
    model_cfg_path = args.model_config or f"configs/model/{args.model}.yaml"
    model_cfg = load_config(model_cfg_path)
    training_cfg = load_config(args.training_config)

    dataset = TemporalSequenceDataset(args.data_dir)
    val_size = max(1, int(0.2 * len(dataset))) if len(dataset) > 1 else 1
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = int(training_cfg.get("batch_size", 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(args.model, model_cfg)
    metrics = train_model(args.model, model, train_loader, val_loader, training_cfg)
    print(metrics)


if __name__ == "__main__":
    main()
