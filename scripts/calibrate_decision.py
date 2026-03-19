from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import json
import yaml

from pipeline.evaluation.calibration import sweep_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds from evaluation outputs.")
    parser.add_argument("--eval-json", default="outputs/logs/tcn_evaluation.json")
    parser.add_argument("--output", default="configs/decision/thresholds.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    thresholds = sweep_thresholds(payload["predictions"], payload["targets"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "possible_threshold": float(thresholds["possible_threshold"]),
                "confirmed_threshold": float(thresholds["confirmed_threshold"]),
                "ema_alpha": float(thresholds["ema_alpha"]),
                "hold_time": float(thresholds["hold_time"]),
                "cooldown": float(thresholds["cooldown"]),
            },
            f,
            sort_keys=False,
        )
    print(thresholds)


if __name__ == "__main__":
    main()
