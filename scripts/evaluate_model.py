from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline.evaluation.evaluator import evaluate_model
from pipeline.models import MODEL_REGISTRY, TCNBinaryClassifier
from pipeline.training.dataloader import TemporalSequenceDataset
from pipeline.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained temporal model.")
    parser.add_argument("--model", choices=["tcn", "tsm", "lrcn"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_model(model_name: str, model_cfg: dict):
    if model_name == "tcn":
        return TCNBinaryClassifier(
            in_channels=int(model_cfg.get("input_dim", 50)),
            channels=list(model_cfg.get("num_channels", [64, 64, 128])),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    return MODEL_REGISTRY[model_name](input_dim=int(model_cfg.get("input_dim", 50)))


def main() -> None:
    args = parse_args()
    model_cfg_path = args.model_config or f"configs/model/{args.model}.yaml"
    model_cfg = load_config(model_cfg_path)
    checkpoint_path = args.checkpoint or f"models/checkpoints/{args.model}_best.pt"

    model = build_model(args.model, model_cfg)
    state = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(state["state_dict"])

    dataset = TemporalSequenceDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    metrics = evaluate_model(args.model, model, loader, device=args.device)

    out_path = Path("outputs/logs") / f"{args.model}_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
