from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from .tcn import TCNBinaryClassifier


class TCNInferencer:
    """
    Thin wrapper to run TCN on Stage-4 output.
    Expects X as numpy array of shape (T, D).
    """
    def __init__(self, feature_dim: int, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TCNBinaryClassifier(in_channels=feature_dim).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            # support either raw state_dict or {"state_dict": ...}
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> float:
        # X: (T, D) -> (1, D, T)
        if X.ndim != 2:
            raise ValueError(f"Expected X shape (T, D), got {X.shape}")

        x = torch.from_numpy(X.astype(np.float32)).to(self.device)  # (T, D)
        x = x.transpose(0, 1).unsqueeze(0)                           # (1, D, T)

        logits = self.model(x)                                       # (1,)
        p = torch.sigmoid(logits).item()
        return float(p)
