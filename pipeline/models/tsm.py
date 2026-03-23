from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("tsm")
class TSMTemporalClassifier(nn.Module):
    def __init__(self, input_dim: int = 50, hidden_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden_dim, 1)

    def _temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        fold = max(1, c // 4)
        out = x.clone()
        if t > 1:
            out[:, 1:, :fold] = x[:, :-1, :fold]
            out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        x = self.input_proj(x)
        x = self._temporal_shift(x)
        x = x.transpose(1, 2)
        x = torch.relu(self.temporal_conv(x))
        x = x.mean(dim=-1)
        return torch.sigmoid(self.head(x)).squeeze(1)
