from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("lrcn")
class LRCNTemporalClassifier(nn.Module):
    """Lightweight recurrent baseline for sequence classification."""

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")

        self.pooling = pooling
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")

        # Input arrives as (B, T, 50): batch, timesteps, engineered features.
        x = self.input_proj(x)

        # The LSTM models temporal flow across the feature sequence.
        x, _ = self.temporal_model(x)

        # Last-step pooling keeps the latest state; mean pooling smooths the clip.
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        # Return a single logit; sigmoid is applied outside the model.
        return self.head(x).squeeze(1)
