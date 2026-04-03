from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch

from pipeline.models import TCNBinaryClassifier
from .losses import build_loss
from .metrics import accuracy, precision, recall, f1


def _forward(model_name: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model_name == "tcn":
        return model(x.transpose(1, 2))
    logits = model(x)
    return logits.squeeze(-1) if logits.ndim > 1 else logits


def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def _evaluate(model_name: str, model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_targets = []
    loss_fn = build_loss()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = _forward(model_name, model, x)
            loss = loss_fn(logits, y)
            probs = _probs_from_logits(logits)
            batch_size = y.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_probs) if all_probs else torch.empty(0)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0)
    return {
        "loss": (total_loss / total_items) if total_items else 0.0,
        "accuracy": accuracy(preds, targets),
        "precision": precision(preds, targets),
        "recall": recall(preds, targets),
        "f1": f1(preds, targets),
    }


def train_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict[str, Any],
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "outputs/logs",
) -> dict[str, Any]:
    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    loss_fn = build_loss()

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    epochs = int(config.get("epochs", 20))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        item_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = _forward(model_name, model, x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            running_loss += float(loss.item()) * batch_size
            item_count += batch_size

        train_loss = (running_loss / item_count) if item_count else 0.0
        train_metrics = _evaluate(model_name, model, train_loader, device)
        val_metrics = _evaluate(model_name, model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_path / f"{model_name}_best.pt",
            )

    metrics_payload = {
        "model_name": model_name,
        "best_f1": best_f1,
        "history": history,
    }
    with (log_path / f"{model_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    return metrics_payload
