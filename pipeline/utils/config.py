from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must contain a mapping at the top level")
    return data
