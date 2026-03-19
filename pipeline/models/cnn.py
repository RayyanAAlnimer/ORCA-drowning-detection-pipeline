from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("cnn")
class CNNTemporalClassifier(nn.Module):
    def __init__(self, input_dim: int = 42, hidden_channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        return torch.sigmoid(self.classifier(x)).squeeze(1)
