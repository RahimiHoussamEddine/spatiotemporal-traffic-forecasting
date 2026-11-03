"""Feature construction and controlled sensor-outage scenarios."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _normalise_mask(mask: NDArray[np.bool_] | None, n_times: int, n_nodes: int) -> BoolArray:
    if mask is None:
        return np.zeros((n_times, n_nodes), dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.shape == (n_nodes,):
        array = np.broadcast_to(array, (n_times, n_nodes)).copy()
    if array.shape != (n_times, n_nodes):
        raise ValueError(f"outage mask must have shape {(n_times, n_nodes)} or {(n_nodes,)}")
    return array


def make_permanent_outage_mask(
    n_times: int,
    n_nodes: int,
    rate: float,
    seed: int = 2468,
) -> BoolArray:
    """Mask a deterministic set of sensors for every timestamp in a scenario."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("outage rate must be in [0, 1)")
    n_failed = int(round(n_nodes * rate))
    failed = np.zeros(n_nodes, dtype=bool)
    if n_failed:
        selected = np.random.default_rng(seed).permutation(n_nodes)[:n_failed]
        failed[selected] = True
    return np.broadcast_to(failed, (n_times, n_nodes)).copy()


def make_random_outage_masks(
    n_times: int,
    n_nodes: int,
    rate: float,
    seed: int = 9051,
) -> BoolArray:
    """Create per-timestamp random missing-sensor masks for augmentation."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("outage rate must be in [0, 1)")
    n_failed = int(round(n_nodes * rate))
    masks = np.zeros((n_times, n_nodes), dtype=bool)
    rng = np.random.default_rng(seed)
    for row in masks:
        if n_failed:
            row[rng.choice(n_nodes, size=n_failed, replace=False)] = True
    return masks


def build_design(
    x: FloatArray,
    adjacency: FloatArray,
    outage_mask: NDArray[np.bool_] | None = None,
    include_neighbors: bool = False,
) -> FloatArray:
    """Build pooled sensor rows from ten traffic lags, source context and graph history.

    The UCI file has no names for the 38 non-lag columns. They are preserved in
    source order, without assigning unsupported feature semantics.
    """
    x = np.asarray(x, dtype=np.float64)
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 48:
        raise ValueError("x must have shape (time, nodes, 48 source features)")
    n_times, n_nodes, _ = x.shape
    if adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("adjacency dimensions must match the node count")
    mask = _normalise_mask(outage_mask, n_times, n_nodes)

    own_history = x[:, :, :10].copy()
    own_history[mask] = 0.0
    context = x[:, :, 10:]
    node_identity = np.broadcast_to(np.eye(n_nodes), (n_times, n_nodes, n_nodes))
    own_available = (~mask).astype(np.float64)[..., None]
    blocks = [own_history, context, node_identity, own_available]

    if include_neighbors:
        degrees = adjacency.sum(axis=1)
        available_neighbors = (~mask).astype(np.float64)
        neighbor_weights = adjacency[None, :, :] * available_neighbors[:, None, :]
        counts = neighbor_weights.sum(axis=2)
        neighbor_sum = np.einsum("tij,tjf->tif", neighbor_weights, own_history, optimize=True)
        neighbor_history = np.divide(
            neighbor_sum,
            counts[..., None],
            out=np.zeros_like(neighbor_sum),
            where=counts[..., None] > 0,
        )
        neighbor_coverage = np.divide(
            counts,
            degrees[None, :],
            out=np.zeros_like(counts),
            where=degrees[None, :] > 0,
        )[..., None]
        blocks[1:1] = [neighbor_history]
        blocks.append(neighbor_coverage)

    return np.concatenate(blocks, axis=2).reshape(n_times * n_nodes, -1)


def flatten_target(y: FloatArray) -> FloatArray:
    """Return time-major pooled scalar targets."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("y must have shape (time, nodes)")
    return values.reshape(-1)


def reshape_predictions(prediction: FloatArray, n_times: int, n_nodes: int) -> FloatArray:
    """Restore pooled prediction rows to time x sensor and bound normalized flow."""
    values = np.asarray(prediction, dtype=np.float64).reshape(n_times, n_nodes)
    return np.clip(values, 0.0, 1.0)
