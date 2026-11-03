import numpy as np

from traffic_forecasting.evaluate import _calibration_radius, _metrics


def test_sensorwise_calibration_radius_uses_finite_sample_upper_order_statistic():
    actual = np.arange(20, dtype=float).reshape(10, 2)
    predicted = np.zeros_like(actual)
    radius = _calibration_radius(actual, predicted, coverage_level=0.8)
    assert radius.shape == (2,)
    assert radius.tolist() == [16.0, 17.0]


def test_metrics_report_normalized_error_and_interval_coverage():
    actual = np.array([[0.2, 0.6], [0.4, 0.8]])
    predicted = np.array([[0.3, 0.5], [0.4, 0.9]])
    lower = predicted - 0.1
    upper = predicted + 0.1
    scores = _metrics(actual, predicted, lower, upper)
    assert np.isclose(scores["mae"], 0.075)
    assert np.isclose(scores["rmse"], np.sqrt(0.0075))
    assert scores["interval_coverage"] == 1.0
