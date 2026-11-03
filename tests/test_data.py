import numpy as np

from traffic_forecasting.data import load_prepared, validate_arrays


def test_prepared_uci_dataset_has_expected_split_shapes():
    data = load_prepared()
    summary = validate_arrays(data)
    assert data["x_train"].shape == (1261, 36, 48)
    assert data["y_train"].shape == (1261, 36)
    assert data["x_test"].shape == (840, 36, 48)
    assert data["y_test"].shape == (840, 36)
    assert summary["adjacency_edges_undirected"] > 0
    assert np.isfinite(data["x_train"]).all()


def test_road_adjacency_is_symmetric_binary_without_self_edges():
    adjacency = load_prepared()["adjacency"]
    assert np.array_equal(adjacency, adjacency.T)
    assert set(np.unique(adjacency)).issubset({0.0, 1.0})
    assert np.all(np.diag(adjacency) == 0)
