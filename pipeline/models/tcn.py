from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Cuts off padding on the right so the conv is causal (no future leakage)."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding on the left via padding + chomp

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        # Init
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBinaryClassifier(nn.Module):
    """
    Input:  x of shape (B, C, T) where C = feature_dim (e.g. 42), T = window length (e.g. 60)
    Output: logits of shape (B,)  (apply sigmoid for probability)
    """
    def __init__(
        self,
        in_channels: int,
        channels: list[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(c_in, c_out, kernel_size, dilation, dropout))
            c_in = c_out

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        y = self.tcn(x)              # (B, C', T)
        y_last = y[:, :, -1]         # take last timestep (causal)
        logit = self.head(y_last)    # (B, 1)
        return logit.squeeze(1)      # (B,)
