from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("lrcn")
class LRCNTemporalClassifier(nn.Module):
    """Lightweight placeholder LRCN for future experimentation.

    The class preserves the same training-side model-registry pattern used by
    the other experimental models without affecting the runtime inference path.
    """

    def __init__(self, input_dim: int = 42, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        x = self.frame_encoder(x)
        x, _ = self.temporal_model(x)
        x = x[:, -1, :]
        return torch.sigmoid(self.head(x)).squeeze(1)
