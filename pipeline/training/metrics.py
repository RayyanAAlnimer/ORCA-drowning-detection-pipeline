from __future__ import annotations

import torch


def _binarize(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (preds >= threshold).int()


def accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds_b = _binarize(preds, threshold)
    targets_b = targets.int()
    return float((preds_b == targets_b).float().mean().item())


def precision(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds_b = _binarize(preds, threshold)
    targets_b = targets.int()
    tp = ((preds_b == 1) & (targets_b == 1)).sum().item()
    fp = ((preds_b == 1) & (targets_b == 0)).sum().item()
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds_b = _binarize(preds, threshold)
    targets_b = targets.int()
    tp = ((preds_b == 1) & (targets_b == 1)).sum().item()
    fn = ((preds_b == 0) & (targets_b == 1)).sum().item()
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    p = precision(preds, targets, threshold)
    r = recall(preds, targets, threshold)
    return float((2 * p * r) / (p + r)) if (p + r) else 0.0
