from __future__ import annotations

import torch.nn as nn


def build_loss() -> nn.Module:
    return nn.BCELoss()
