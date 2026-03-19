from __future__ import annotations

from typing import Iterable


def _compute_f1(predictions, targets, threshold: float) -> float:
    preds = [1 if p >= threshold else 0 for p in predictions]
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def sweep_thresholds(predictions: list[float], targets: list[float]) -> dict:
    best = None
    for i in range(5, 95, 5):
        threshold = i / 100.0
        score = _compute_f1(predictions, targets, threshold)
        if best is None or score > best["f1"]:
            best = {"threshold": threshold, "f1": score}

    threshold = best["threshold"] if best is not None else 0.5
    return {
        "possible_threshold": max(0.0, threshold - 0.1),
        "confirmed_threshold": threshold,
        "ema_alpha": 0.3,
        "hold_time": 1.5,
        "cooldown": 3.0,
        "best_f1": best["f1"] if best is not None else 0.0,
    }
