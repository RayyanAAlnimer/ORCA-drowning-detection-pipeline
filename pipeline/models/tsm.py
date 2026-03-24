from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


def temporal_shift(x: torch.Tensor, fold_div: int = 8) -> torch.Tensor:
    """Shift a small portion of channels forward/backward in time.

    x has shape (B, T, C). Only part of the channels are shifted so we keep
    most features aligned while still injecting cheap temporal context.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected input shape (B, T, C), got {tuple(x.shape)}")

    b, t, c = x.shape
    if t <= 1 or c == 0:
        return x

    fold = max(1, c // fold_div)
    out = x.clone()

    # Shift one group forward in time.
    out[:, 1:, :fold] = x[:, :-1, :fold]

    # Shift one group backward in time.
    out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]

    # The remaining channels stay in place as an anchor.
    return out


class TSMBlock(nn.Module):
    def __init__(self, hidden_dim: int, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div
        self.temporal_mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Shift only part of the channels to exchange neighboring-timestep context.
        x = temporal_shift(x, fold_div=self.fold_div)

        # Conv1d mixes the shifted sequence across time at low cost.
        x = x.transpose(1, 2)
        x = self.temporal_mix(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        return x + residual


@register_model("tsm")
class TSMTemporalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        fold_div: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [TSMBlock(hidden_dim=hidden_dim, fold_div=fold_div) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")

        # Input is a feature sequence shaped (B, T, 50).
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Mean pooling keeps the classifier lightweight and stable across clip length.
        x = x.mean(dim=1)

        # Return a single logit; sigmoid is applied outside the model.
        return self.head(x).squeeze(1)
