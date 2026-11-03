"""One-time evaluation on the official chronologically later UCI test sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from traffic_forecasting.data import load_prepared
from traffic_forecasting.features import make_permanent_outage_mask
from traffic_forecasting.models import MODEL_LABELS, predict_ridge

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "traffic_models.joblib"
OUTPUT_PATH = ROOT / "reports" / "evaluation.json"
SCENARIO_PATH = ROOT / "reports" / "scenario_metrics.csv"
SENSOR_PATH = ROOT / "reports" / "sensor_metrics.csv"
OUTAGE_RATES = (0.0, 0.10, 0.20)
MODEL_KEYS = ("best_single_lag", "own_ridge", "graph_ridge", "graph_robust_ridge")


def _calibration_radius(
    actual: np.ndarray,
    predicted: np.ndarray,
    coverage_level: float = 0.90,
) -> np.ndarray:
    """Compute per-sensor finite-sample absolute-residual calibration radii."""
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("calibration arrays must share shape (time, sensors)")
    if not 0.0 < coverage_level < 1.0:
        raise ValueError("coverage_level must be in (0, 1)")
    residuals = np.sort(np.abs(actual - predicted), axis=0)
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage_level))
    return residuals[rank - 1]


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    residual = predicted - actual
    sensor_mae = np.mean(np.abs(residual), axis=0)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_sensor_mae": float(np.mean(sensor_mae)),
        "worst_sensor_mae": float(np.max(sensor_mae)),
        "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def _prediction(
    model_key: str,
    package: dict[str, Any],
    x: np.ndarray,
    adjacency: np.ndarray,
    outage_mask: np.ndarray,
) -> np.ndarray:
    if model_key == "best_single_lag":
        prediction = x[:, :, package["best_lag_slot"]].copy()
        prediction[outage_mask] = 0.0
        return np.clip(prediction, 0.0, 1.0)
    return predict_ridge(
        package["models"][model_key],
        x,
        adjacency,
        outage_mask,
        include_neighbors=model_key.startswith("graph_"),
    )


def evaluate(force: bool = False) -> dict[str, Any]:
    """Evaluate fixed models and outage scenarios once against the locked test sequence."""
    if OUTPUT_PATH.exists() and not force:
        raise FileExistsError(
            "Locked test report already exists. Review it instead of tuning on test labels; "
            "pass --force only after a documented code correction."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Fitted model file is missing; run `make train` first.")

    data = load_prepared()
    package = joblib.load(MODEL_PATH)
    split = package["split"]
    x_cal = data["x_train"][split["validation_end"] :]
    y_cal = data["y_train"][split["validation_end"] :]
    x_test, y_test = data["x_test"], data["y_test"]
    adjacency = data["adjacency"]
    scenario_rows = []
    sensor_rows = []

    for rate in OUTAGE_RATES:
        cal_mask = make_permanent_outage_mask(len(x_cal), x_cal.shape[1], rate, seed=2468)
        test_mask = make_permanent_outage_mask(len(x_test), x_test.shape[1], rate, seed=2468)
        failed_ids = np.flatnonzero(test_mask[0]).tolist()
        for model_key in MODEL_KEYS:
            calibration_prediction = _prediction(model_key, package, x_cal, adjacency, cal_mask)
            calibration_radius = _calibration_radius(y_cal, calibration_prediction)
            prediction = _prediction(model_key, package, x_test, adjacency, test_mask)
            lower = np.clip(prediction - calibration_radius[None, :], 0.0, 1.0)
            upper = np.clip(prediction + calibration_radius[None, :], 0.0, 1.0)
            values = _metrics(y_test, prediction, lower, upper)
            n_failed = int(test_mask[0].sum())
            scenario_rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "outage_rate": rate,
                    "failed_sensors": n_failed,
                    "failed_sensor_ids": ",".join(map(str, failed_ids)),
                    **values,
                }
            )
            for sensor in range(y_test.shape[1]):
                error = prediction[:, sensor] - y_test[:, sensor]
                sensor_rows.append(
                    {
                        "model_key": model_key,
                        "outage_rate": rate,
                        "sensor_id": sensor,
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "interval_coverage": float(
                            np.mean(
                                (y_test[:, sensor] >= lower[:, sensor])
                                & (y_test[:, sensor] <= upper[:, sensor])
                            )
                        ),
                    }
                )

    metric_fields = list(scenario_rows[0])
    sensor_fields = list(sensor_rows[0])
    SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCENARIO_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scenario_rows)
    with SENSOR_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sensor_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sensor_rows)

    report = {
        "evaluation_protocol": (
            "Models and hyperparameters were selected on source training/validation blocks. "
            "Per-sensor 90% absolute-residual intervals use the separate chronological "
            "calibration block. The official later sequence was evaluated once under the "
            "predeclared clean, 10%, and 20% permanent-sensor-outage scenarios."
        ),
        "test_rows": int(len(y_test)),
        "sensor_count": int(y_test.shape[1]),
        "outage_rate_note": (
            "Rates are nominal scenario levels rounded with round(rate * sensor_count): "
            "10% maps to 4/36 sensors and 20% maps to 7/36. The same deterministic failure "
            "sets are used for calibration and test."
        ),
        "target_units": "normalized traffic-flow units (the source provides no inverse scale)",
        "interval_interpretation": (
            "Empirical per-sensor calibration intervals. Temporal dependence means the "
            "exchangeability condition for formal split-conformal coverage is not established; "
            "test coverage is descriptive."
        ),
        "source_calendar_caveat": (
            "The arrays provide temporal train/test order, but the paper's 2017-01-02 to "
            "2017-02-01 date range, bundled DOCX labels, and 2,101 quarter-hour sample count "
            "(2,100 intervals = 21 days 21 hours) do not reconcile. No exact calendar boundary "
            "is asserted."
        ),
        "selected_alpha": package["selected_alpha"],
        "validation_selected_lag_slot": package["best_lag_slot"],
        "scenarios": scenario_rows,
        "per_sensor_report": "reports/sensor_metrics.csv",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute the locked test report after a documented code correction",
    )
    args = parser.parse_args()
    result = evaluate(force=args.force)
    for row in result["scenarios"]:
        print(
            f"{row['model_key']:20s} outage={row['outage_rate']:.0%} "
            f"MAE={row['mae']:.5f} RMSE={row['rmse']:.5f} "
            f"coverage={row['interval_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()
