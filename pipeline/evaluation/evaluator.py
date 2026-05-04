from __future__ import annotations

from pathlib import Path
import json

import torch

from pipeline.training.metrics import accuracy, precision, recall, f1, mean_average_precision


def evaluate_model(model_name: str, model: torch.nn.Module, loader, device: str = "cpu", threshold: float = 0.5) -> dict:
    model.eval()
    device_obj = torch.device(device)
    model = model.to(device_obj)
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device_obj)
            y = y.to(device_obj)
            if model_name == "tcn":
                # Dataloader returns (batch, time, features); TCN expects
                # (batch, features, time), matching the runtime inferencer.
                logits = model(x.transpose(1, 2))
            else:
                logits = model(x)
            logits = logits.squeeze(-1) if logits.ndim > 1 else logits
            preds = torch.sigmoid(logits)
            preds_all.append(preds.cpu())
            targets_all.append(y.cpu())

    preds = torch.cat(preds_all) if preds_all else torch.empty(0)
    targets = torch.cat(targets_all) if targets_all else torch.empty(0)
    # Keep raw probabilities and targets in the JSON so confusion matrices,
    # threshold experiments, and report plots can be reproduced later.
    return {
        "accuracy": accuracy(preds, targets, threshold),
        "precision": precision(preds, targets, threshold),
        "recall": recall(preds, targets, threshold),
        "f1": f1(preds, targets, threshold),
        "map": mean_average_precision(preds, targets),
        "predictions": preds.tolist(),
        "targets": targets.tolist(),
    }
