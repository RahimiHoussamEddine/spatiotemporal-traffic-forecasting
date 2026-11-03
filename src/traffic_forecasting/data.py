"""Acquire, validate, and prepare the licensed UCI traffic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = ROOT / "data" / "raw" / "traffic_flow_forecasting.zip"
PREPARED_PATH = ROOT / "data" / "processed" / "traffic_prepared.npz"
MANIFEST_PATH = ROOT / "reports" / "data_manifest.json"
QUALITY_PATH = ROOT / "reports" / "data_quality.json"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/734/traffic%2Bflow%2Bforecasting-1.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/734/traffic%2Bflow%2Bforecasting-1"
EXPECTED_ARCHIVE_SHA256 = "b09ebe53ec44228bfd008d35fb34d12174b5523f7a735d159d9eea326163bc45"
N_TRAIN = 1261
N_TEST = 840
N_NODES = 36
N_FEATURES = 48


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def download_archive(force: bool = False) -> Path:
    """Download the fixed UCI archive and require the reviewed checksum."""
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists() and not force:
        payload = ARCHIVE_PATH.read_bytes()
        if sha256_bytes(payload) == EXPECTED_ARCHIVE_SHA256:
            return ARCHIVE_PATH
        ARCHIVE_PATH.unlink()

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "spatiotemporal-traffic-forecasting/0.1.0"},
    )
    temporary = ARCHIVE_PATH.with_suffix(".zip.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(
                "UCI archive checksum changed. Review the source before updating the pin."
            )
        temporary.replace(ARCHIVE_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return ARCHIVE_PATH


def _stack_input(value: np.ndarray) -> np.ndarray:
    """Convert the MATLAB 1 x time object array to time x node x feature."""
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"Expected MATLAB input shape (1, time), got {value.shape}.")
    frames = []
    for frame in value[0]:
        dense = frame.toarray() if sparse.issparse(frame) else np.asarray(frame)
        if dense.shape != (N_NODES, N_FEATURES):
            raise ValueError(f"Unexpected input frame shape {dense.shape}.")
        frames.append(np.asarray(dense, dtype=np.float64))
    return np.stack(frames, axis=0)


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    """Read the pinned MATLAB arrays without extracting unrelated archive files."""
    payload = path.read_bytes()
    if sha256_bytes(payload) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Traffic archive does not match the reviewed SHA-256 pin.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Corrupt member in UCI archive: {damaged}")
        mat_bytes = archive.read("traffic_dataset.mat")
    source = loadmat(io.BytesIO(mat_bytes), squeeze_me=False)
    x_train = _stack_input(source["tra_X_tr"])
    x_test = _stack_input(source["tra_X_te"])
    y_train = np.asarray(source["tra_Y_tr"], dtype=np.float64).T
    y_test = np.asarray(source["tra_Y_te"], dtype=np.float64).T
    adjacency = np.asarray(source["tra_adj_mat"], dtype=np.float64)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "adjacency": adjacency,
    }


def validate_arrays(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Assert source dimensions, finite values, and graph integrity."""
    expected = {
        "x_train": (N_TRAIN, N_NODES, N_FEATURES),
        "y_train": (N_TRAIN, N_NODES),
        "x_test": (N_TEST, N_NODES, N_FEATURES),
        "y_test": (N_TEST, N_NODES),
        "adjacency": (N_NODES, N_NODES),
    }
    for key, shape in expected.items():
        if key not in data or data[key].shape != shape:
            raise ValueError(
                f"{key} must have shape {shape}; got {data.get(key, np.array([])).shape}."
            )
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains NaN or infinite values.")

    adjacency = data["adjacency"]
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("The published road adjacency matrix is not symmetric.")
    if not np.all(np.diag(adjacency) == 0):
        raise ValueError("The published road adjacency matrix has self-edges.")
    if not np.isin(adjacency, [0.0, 1.0]).all():
        raise ValueError("The published road adjacency matrix is not binary.")
    if data["y_train"].min() < 0 or data["y_train"].max() > 1.000001:
        raise ValueError("Training targets fall outside the documented normalized range.")
    if data["y_test"].min() < 0 or data["y_test"].max() > 1.000001:
        raise ValueError("Test targets fall outside the documented normalized range.")

    degrees = adjacency.sum(axis=1).astype(int)
    return {
        "dimensions": {key: list(value.shape) for key, value in data.items()},
        "training_samples": N_TRAIN,
        "locked_test_samples": N_TEST,
        "nodes": N_NODES,
        "features_per_node": N_FEATURES,
        "historical_traffic_features": 10,
        "remaining_encoded_features": N_FEATURES - 10,
        "adjacency_edges_undirected": int(np.triu(adjacency, k=1).sum()),
        "nodes_with_no_published_neighbors": int((degrees == 0).sum()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "normalized_target_range_train": [
            float(data["y_train"].min()),
            float(data["y_train"].max()),
        ],
        "normalized_target_range_test": [
            float(data["y_test"].min()),
            float(data["y_test"].max()),
        ],
    }


def prepare(force_download: bool = False) -> dict[str, Any]:
    """Download, validate, and write ignored local arrays plus aggregate manifests."""
    path = download_archive(force=force_download)
    data = _read_archive(path)
    details = validate_arrays(data)
    PREPARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREPARED_PATH.open("wb") as file:
        np.savez_compressed(file, **data)
    archive_bytes = path.read_bytes()
    manifest = {
        "source": "UCI Machine Learning Repository, Traffic Flow Forecasting (dataset 734)",
        "source_url": UCI_PAGE,
        "download_url": SOURCE_URL,
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "creator": "Liang Zhao",
        "source_archive_sha256": sha256_bytes(archive_bytes),
        "source_archive_bytes": len(archive_bytes),
        "prepared_arrays_sha256": hashlib.sha256(PREPARED_PATH.read_bytes()).hexdigest(),
        "time_order_note": (
            "The source provides separate contiguous train and test sequences. The source paper "
            "describes the earlier span as training and the later span as testing. The paper's "
            "2017-01-02 to 2017-02-01 date range, the bundled DOCX train/test date labels, and "
            "the 2,101 quarter-hour sample count (2,100 intervals would span 21 days 21 hours) "
            "do not reconcile. The arrays have no timestamps, so this project "
            "preserves their published order and does not claim exact calendar boundaries."
        ),
        "feature_count_note": (
            "UCI describes 47 feature groups, while the downloaded MATLAB matrices have 48 "
            "columns; the page describes 36 sensors, while the paper's dataset table lists 37 "
            "road segments and the matrices have 36 nodes. The first ten columns are historical "
            "traffic inputs; all remaining columns are preserved without unsupported labels."
        ),
        **details,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    QUALITY_PATH.write_text(
        json.dumps(
            {
                "archive_integrity": "ZIP CRC and pinned SHA-256 passed",
                "schema_validation": "passed",
                "finite_value_validation": "passed",
                "adjacency_validation": "passed",
                "missing_value_count": 0,
                "source_metadata_discrepancy": (
                    "UCI describes 47 features and 36 sensors; MATLAB arrays contain 48 columns "
                    "and 36 nodes, while the paper lists 37 segments. The paper date range and "
                    "bundled DOCX labels do not reconcile with 2,101 quarter-hour rows."
                ),
                "raw_and_prepared_arrays_committed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_prepared() -> dict[str, np.ndarray]:
    """Load local prepared arrays or explain how to create them."""
    if not PREPARED_PATH.exists():
        raise FileNotFoundError("Prepared data are missing; run `make prepare` first.")
    with np.load(PREPARED_PATH, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    validate_arrays(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate"])
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(force_download=args.force_download)
        print(json.dumps(report, indent=2))
    else:
        data = load_prepared()
        print(json.dumps(validate_arrays(data), indent=2))


if __name__ == "__main__":
    main()
