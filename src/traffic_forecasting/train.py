"""Fit the traffic benchmark models with a chronological validation block."""

from __future__ import annotations

import json
from pathlib import Path

from traffic_forecasting.data import load_prepared
from traffic_forecasting.models import train_models

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    summary = train_models(
        load_prepared(),
        str(ROOT / "models" / "traffic_models.joblib"),
        str(ROOT / "reports" / "validation_metrics.json"),
    )
    print(
        json.dumps(
            {
                "selected_alpha": summary["selected_alpha"],
                "selected_single_lag_slot": summary["selected_lag_slot"],
                "block_lengths": summary["block_lengths"],
                "status": "models fit; locked test labels were not accessed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
