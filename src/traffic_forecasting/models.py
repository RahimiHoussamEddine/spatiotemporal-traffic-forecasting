"""Validation-only model fitting and prediction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from traffic_forecasting.features import (
    build_design,
    flatten_target,
    make_permanent_outage_mask,
    make_random_outage_masks,
    reshape_predictions,
)

FloatArray = NDArray[np.float64]
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
MODEL_LABELS = {
    "best_single_lag": "Validation-selected single-lag baseline",
    "own_ridge": "Pooled own-sensor/context Ridge",
    "graph_ridge": "Pooled adjacency-aware Ridge",
    "graph_robust_ridge": "Outage-augmented pooled graph Ridge",
}


def make_ridge(alpha: float) -> Pipeline:
    """Create the same deterministic standardized ridge pipeline for every candidate."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def predict_ridge(
    model: Pipeline,
    x: FloatArray,
    adjacency: FloatArray,
    outage_mask: NDArray[np.bool_] | None,
    include_neighbors: bool,
) -> FloatArray:
    """Predict normalized flow for every time and sensor row."""
    n_times, n_nodes, _ = x.shape
    design = build_design(x, adjacency, outage_mask, include_neighbors)
    return reshape_predictions(model.predict(design), n_times, n_nodes)


def best_lag_on_validation(
    x_val: FloatArray, y_val: FloatArray
) -> tuple[int, list[dict[str, Any]]]:
    """Choose among the ten source lag slots using the validation block only."""
    results = []
    for lag in range(x_val.shape[2] if x_val.ndim == 3 else 0):
        if lag >= 10:
            break
        error = x_val[:, :, lag] - y_val
        results.append(
            {
                "lag_slot": lag,
                "validation_mae": float(np.mean(np.abs(error))),
                "validation_rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    selected = min(results, key=lambda row: row["validation_rmse"])["lag_slot"]
    return int(selected), results


def _fit_candidate(
    alpha: float,
    x_train: FloatArray,
    y_train: FloatArray,
    adjacency: FloatArray,
    include_neighbors: bool,
    train_masks: NDArray[np.bool_] | None,
) -> Pipeline:
    design = build_design(x_train, adjacency, train_masks, include_neighbors)
    model = make_ridge(alpha)
    model.fit(design, flatten_target(y_train))
    return model


def _validation_rmse(
    model: Pipeline,
    x_val: FloatArray,
    y_val: FloatArray,
    adjacency: FloatArray,
    include_neighbors: bool,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    prediction = predict_ridge(model, x_val, adjacency, mask, include_neighbors)
    return float(np.sqrt(np.mean((prediction - y_val) ** 2)))


def train_models(data: dict[str, FloatArray], model_path: str, report_path: str) -> dict[str, Any]:
    """Fit fixed candidate families and select parameters without touching test labels."""
    from pathlib import Path

    import joblib

    x_source = data["x_train"]
    y_source = data["y_train"]
    adjacency = data["adjacency"]
    train_end = int(0.70 * len(x_source))
    validation_end = int(0.85 * len(x_source))
    x_train, y_train = x_source[:train_end], y_source[:train_end]
    x_val, y_val = x_source[train_end:validation_end], y_source[train_end:validation_end]
    x_cal = x_source[validation_end:]
    if min(len(x_train), len(x_val), len(x_cal)) < 1:
        raise ValueError("Source training sequence is too short for the fixed temporal split.")

    best_lag, lag_scores = best_lag_on_validation(x_val, y_val)
    fixed_validation_mask = make_permanent_outage_mask(len(x_val), x_val.shape[1], 0.20, seed=2468)
    augmentation_masks = make_random_outage_masks(len(x_train), x_train.shape[1], 0.20, seed=9051)

    model_specs = {
        "own_ridge": (False, None),
        "graph_ridge": (True, None),
    }
    models: dict[str, Pipeline] = {}
    selected_alpha: dict[str, float] = {}
    candidate_scores: dict[str, list[dict[str, Any]]] = {}

    for model_name, (include_neighbors, training_masks) in model_specs.items():
        rows = []
        for alpha in ALPHAS:
            model = _fit_candidate(
                alpha,
                x_train,
                y_train,
                adjacency,
                include_neighbors,
                training_masks,
            )
            rmse = _validation_rmse(model, x_val, y_val, adjacency, include_neighbors)
            rows.append({"alpha": alpha, "validation_clean_rmse": rmse})
        selected = min(rows, key=lambda row: row["validation_clean_rmse"])
        alpha = float(selected["alpha"])
        models[model_name] = _fit_candidate(
            alpha,
            x_train,
            y_train,
            adjacency,
            include_neighbors,
            training_masks,
        )
        selected_alpha[model_name] = alpha
        candidate_scores[model_name] = rows

    robust_rows = []
    x_train_augmented = np.concatenate([x_train, x_train], axis=0)
    y_train_augmented = np.concatenate([y_train, y_train], axis=0)
    robust_masks = np.concatenate([np.zeros_like(augmentation_masks), augmentation_masks], axis=0)
    for alpha in ALPHAS:
        clean_model = _fit_candidate(
            alpha,
            x_train_augmented,
            y_train_augmented,
            adjacency,
            True,
            robust_masks,
        )
        clean_rmse = _validation_rmse(clean_model, x_val, y_val, adjacency, True)
        outage_rmse = _validation_rmse(
            clean_model,
            x_val,
            y_val,
            adjacency,
            True,
            fixed_validation_mask,
        )
        robust_rows.append(
            {
                "alpha": alpha,
                "validation_clean_rmse": clean_rmse,
                "validation_20pct_outage_rmse": outage_rmse,
                "selection_objective_mean_rmse": 0.5 * (clean_rmse + outage_rmse),
            }
        )
    selected_robust = min(robust_rows, key=lambda row: row["selection_objective_mean_rmse"])
    robust_alpha = float(selected_robust["alpha"])
    models["graph_robust_ridge"] = _fit_candidate(
        robust_alpha,
        x_train_augmented,
        y_train_augmented,
        adjacency,
        True,
        robust_masks,
    )
    selected_alpha["graph_robust_ridge"] = robust_alpha
    candidate_scores["graph_robust_ridge"] = robust_rows

    val_predictions = {
        name: predict_ridge(
            model,
            x_val,
            adjacency,
            None,
            name != "own_ridge",
        )
        for name, model in models.items()
    }
    validation_summary = {
        "selected_alpha": selected_alpha,
        "lag_candidates": lag_scores,
        "selected_lag_slot": best_lag,
        "candidates": candidate_scores,
        "block_lengths": {
            "model_fit": len(x_train),
            "parameter_selection": len(x_val),
            "interval_calibration": len(x_cal),
            "locked_test": len(data["x_test"]),
        },
        "robust_selection_rule": (
            "For graph_robust_ridge, select alpha by the mean of clean and fixed "
            "20% sensor-outage validation RMSE. Other Ridge alphas use clean validation RMSE."
        ),
        "validation_predictions": val_predictions,
    }
    package = {
        "models": models,
        "selected_alpha": selected_alpha,
        "best_lag_slot": best_lag,
        "candidate_scores": candidate_scores,
        "split": {
            "train_end": train_end,
            "validation_end": validation_end,
            "training_rows": len(x_train),
            "validation_rows": len(x_val),
            "calibration_rows": len(x_cal),
        },
    }
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_file, compress=3)
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    import json

    report_file.write_text(
        json.dumps(
            {
                key: value
                for key, value in validation_summary.items()
                if key != "validation_predictions"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return validation_summary
