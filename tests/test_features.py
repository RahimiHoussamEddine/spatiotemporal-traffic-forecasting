import numpy as np
import pytest

from traffic_forecasting.features import (
    build_design,
    make_permanent_outage_mask,
    make_random_outage_masks,
)


def toy_arrays():
    x = np.zeros((2, 3, 48), dtype=float)
    for time in range(2):
        for node in range(3):
            x[time, node, :10] = node + 10 * time
            x[time, node, 10:] = 0.25
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    return x, adjacency


def test_design_preserves_own_history_context_and_graph_mean():
    x, adjacency = toy_arrays()
    clean = build_design(x, adjacency, include_neighbors=True)
    assert clean.shape == (6, 10 + 10 + 38 + 3 + 1 + 1)
    assert np.all(clean[0, :10] == 0)
    assert np.all(clean[0, 10:20] == 1)
    assert np.all(clean[0, 20:58] == 0.25)
    assert clean[0, -2] == 1
    assert clean[0, -1] == 1


def test_missing_sensor_is_zeroed_and_excluded_from_neighbor_average():
    x, adjacency = toy_arrays()
    mask = np.array([[True, False, False], [False, False, False]])
    design = build_design(x, adjacency, mask, include_neighbors=True)
    node_zero_time_zero = design[0]
    assert np.all(node_zero_time_zero[:10] == 0)
    assert np.all(node_zero_time_zero[10:20] == 1)
    assert node_zero_time_zero[-2] == 0
    node_one_time_zero = design[1]
    assert np.all(node_one_time_zero[10:20] == 2)
    assert node_one_time_zero[-1] == 0.5


def test_masks_are_deterministic_and_reject_invalid_rates():
    first = make_permanent_outage_mask(5, 36, 0.2, seed=12)
    second = make_permanent_outage_mask(5, 36, 0.2, seed=12)
    assert np.array_equal(first, second)
    assert np.all(first == first[0])
    assert first.sum(axis=1)[0] == 7
    assert make_random_outage_masks(3, 36, 0.2, seed=9).sum(axis=1).tolist() == [7, 7, 7]
    with pytest.raises(ValueError):
        make_permanent_outage_mask(2, 3, 1.0)


def test_feature_builder_checks_shapes():
    with pytest.raises(ValueError):
        build_design(np.zeros((2, 3, 47)), np.zeros((3, 3)))
