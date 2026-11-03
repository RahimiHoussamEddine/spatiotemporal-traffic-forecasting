"""Interactive sensor-network traffic forecasting and outage review app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_forecasting.data import load_prepared
from traffic_forecasting.evaluate import (
    MODEL_KEYS,
    _calibration_radius,
    _prediction,
)
from traffic_forecasting.features import make_permanent_outage_mask
from traffic_forecasting.models import MODEL_LABELS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "traffic_models.joblib"
EVALUATION_PATH = ROOT / "reports" / "evaluation.json"


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, np.ndarray]:
    return load_prepared()


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Building the selected forecast and calibration band…")
def forecast_for_view(model_key: str, outage_rate: float) -> tuple[np.ndarray, ...]:
    data = load_data()
    package = load_models()
    split = package["split"]
    x_cal = data["x_train"][split["validation_end"] :]
    y_cal = data["y_train"][split["validation_end"] :]
    x_test, adjacency = data["x_test"], data["adjacency"]
    cal_mask = make_permanent_outage_mask(len(x_cal), x_cal.shape[1], outage_rate)
    test_mask = make_permanent_outage_mask(len(x_test), x_test.shape[1], outage_rate)
    cal_prediction = _prediction(model_key, package, x_cal, adjacency, cal_mask)
    radius = _calibration_radius(y_cal, cal_prediction)
    prediction = _prediction(model_key, package, x_test, adjacency, test_mask)
    lower = np.clip(prediction - radius[None, :], 0.0, 1.0)
    upper = np.clip(prediction + radius[None, :], 0.0, 1.0)
    return prediction, lower, upper, test_mask


def network_figure(
    adjacency: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    failed: np.ndarray,
) -> go.Figure:
    n_nodes = len(actual)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for left, right in zip(*np.where(np.triu(adjacency, k=1)), strict=True):
        edge_x.extend([positions[left, 0], positions[right, 0], None])
        edge_y.extend([positions[left, 1], positions[right, 1], None])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#9aa6b2", "width": 1},
            hoverinfo="skip",
            name="Published adjacency",
        )
    )
    symbols = ["x" if is_failed else "circle" for is_failed in failed]
    figure.add_trace(
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers+text",
            text=[str(index) for index in range(n_nodes)],
            textposition="top center",
            marker={
                "size": 17,
                "color": prediction,
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "Forecast<br>(normalized)"},
                "symbol": symbols,
                "line": {"width": 1, "color": "#17212b"},
            },
            customdata=np.column_stack([actual, prediction, prediction - actual, failed]),
            hovertemplate=(
                "Sensor %{text}<br>Observed %{customdata[0]:.3f}"
                "<br>Forecast %{customdata[1]:.3f}"
                "<br>Error %{customdata[2]:+.3f}"
                "<br>Sensor unavailable %{customdata[3]}<extra></extra>"
            ),
            name="Sensors",
        )
    )
    figure.update_layout(
        height=560,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        title="Sensor network schematic (layout is not geographic)",
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Traffic Forecasting Workbench",
        page_icon="🚦",
        layout="wide",
    )
    st.title("Spatiotemporal Traffic Forecasting")
    st.caption(
        "One-step 15-minute forecasts across 36 sensors, with graph context, "
        "calibrated intervals and permanent sensor-outage scenarios."
    )

    if not MODEL_PATH.exists() or not EVALUATION_PATH.exists():
        st.error(
            "Project artifacts are missing. Run `make prepare`, `make train`, then `make evaluate`."
        )
        st.stop()

    data = load_data()
    evaluation = pd.read_csv(ROOT / "reports" / "scenario_metrics.csv")
    model_key = st.selectbox(
        "Forecast model",
        options=list(MODEL_KEYS),
        format_func=lambda key: MODEL_LABELS[key],
        key="model_select",
    )
    outage_label = st.selectbox(
        "Requested permanent sensor outage",
        options=["0%", "10%", "20%"],
        help="The fixed scenario rounds the requested rate to an integer number of 36 sensors.",
        key="outage_select",
    )
    outage_rate = int(outage_label[:-1]) / 100.0
    index = st.slider(
        "Locked test-sequence sample (15-minute step)",
        min_value=0,
        max_value=len(data["y_test"]) - 1,
        value=0,
        key="test_offset",
    )
    sensor = st.slider(
        "Sensor for the local time-series view",
        min_value=0,
        max_value=data["y_test"].shape[1] - 1,
        value=0,
        key="sensor_select",
    )

    chosen = evaluation.loc[
        (evaluation["model_key"] == model_key) & np.isclose(evaluation["outage_rate"], outage_rate)
    ].iloc[0]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Test MAE", f"{chosen['mae']:.4f}")
    metric_cols[1].metric("Test RMSE", f"{chosen['rmse']:.4f}")
    metric_cols[2].metric("Mean sensor MAE", f"{chosen['mean_sensor_mae']:.4f}")
    metric_cols[3].metric("90% interval coverage", f"{chosen['interval_coverage']:.1%}")
    metric_cols[4].metric("Sensors offline", int(chosen["failed_sensors"]))

    prediction, lower, upper, masks = forecast_for_view(model_key, outage_rate)
    failed = masks[index]
    tab_network, tab_sensor, tab_compare = st.tabs(
        ["Network forecast", "Sensor detail", "Model comparison"]
    )
    with tab_network:
        st.plotly_chart(
            network_figure(
                data["adjacency"],
                data["y_test"][index],
                prediction[index],
                failed,
            ),
            width="stretch",
        )
        st.caption(
            "Colors show normalized forecasts. Crosses mark sensors whose historical inputs "
            "were removed in this scenario. The graph comes from the source adjacency matrix; "
            "the circular placement is only a schematic."
        )
    with tab_sensor:
        first = max(0, index - 24)
        last = min(len(data["y_test"]), index + 25)
        time_index = np.arange(first, last)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=time_index,
                y=data["y_test"][first:last, sensor],
                name="Observed",
                mode="lines",
                line={"color": "#17212b"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=time_index,
                y=prediction[first:last, sensor],
                name="Forecast",
                mode="lines",
                line={"color": "#167d9a"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([time_index, time_index[::-1]]),
                y=np.concatenate([upper[first:last, sensor], lower[first:last, sensor][::-1]]),
                name="Empirical 90% calibration band",
                fill="toself",
                fillcolor="rgba(22,125,154,0.18)",
                line={"color": "rgba(22,125,154,0)"},
                hoverinfo="skip",
            )
        )
        figure.update_layout(
            height=420,
            xaxis_title="Official test-sequence sample (exact calendar dates unavailable)",
            yaxis_title="Normalized traffic flow",
            legend={"orientation": "h"},
        )
        st.plotly_chart(figure, width="stretch")
        st.write(
            {
                "sensor_id": sensor,
                "sample_offset": index,
                "observed": float(data["y_test"][index, sensor]),
                "forecast": float(prediction[index, sensor]),
                "calibration_interval": [
                    float(lower[index, sensor]),
                    float(upper[index, sensor]),
                ],
                "sensor_failed": bool(failed[sensor]),
            }
        )
    with tab_compare:
        table = evaluation.loc[np.isclose(evaluation["outage_rate"], outage_rate)].copy()
        table["model"] = table["model_key"].map(MODEL_LABELS)
        st.dataframe(
            table[
                [
                    "model",
                    "failed_sensors",
                    "mae",
                    "rmse",
                    "worst_sensor_mae",
                    "interval_coverage",
                    "mean_interval_width",
                ]
            ].sort_values("rmse"),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Intervals use a separate chronological calibration block. Coverage is descriptive "
            "because temporal dependence weakens formal split-conformal guarantees."
        )


if __name__ == "__main__":
    main()
