# Spatiotemporal Traffic Forecasting

A local forecasting workbench for 36 connected highway sensors. It compares
short-lag and pooled Ridge baselines with a simple adjacency-aware model, then
tests how fixed sensor outages affect 15-minute-ahead forecasts. A Streamlit
app explores forecasts, the published sensor graph, per-sensor errors, and
empirical calibration bands.

This is a compact benchmark study, not a live traffic service or a geographic
map. The source provides a graph but not sensor coordinates or a raw target
scale.

## Data source and license

The project uses Liang Zhao's [UCI Traffic Flow Forecasting dataset
734](https://archive.ics.uci.edu/dataset/734/traffic%2Bflow%2Bforecasting-1),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source
contains 1,261 training and 840 later test samples at 15-minute intervals for
36 sensors. Each sensor row has 48 inputs: ten historical traffic values and
38 encoded context values. The full feature names are not supplied for every
column; this project preserves the input order and does not invent labels. The
target is normalized to approximately `[0, 1]`; the source does not provide the
inverse scaling constant needed to report vehicle counts.

The archive and extracted arrays are not committed. `make prepare` downloads
the official UCI archive, checks its pinned SHA-256 and ZIP CRC, validates the
schema and adjacency matrix, then writes local ignored arrays and aggregate
provenance reports. The current archive hash is
`b09ebe53ec44228bfd008d35fb34d12174b5523f7a735d159d9eea326163bc45`.

**Attribution:** Zhao, Liang. “Traffic Flow Forecasting.” UCI Machine Learning
Repository, DOI [10.24432/C57897](https://doi.org/10.24432/C57897). The
associated methodology paper is Zhao, Gkountouna, and Pfoser (2019),
[“Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial
Topological Constraints”](https://doi.org/10.1145/3339823).

The paper describes the first contiguous time span as training and the later
span as testing. However, its stated 2017-01-02 to 2017-02-01 date range, the
bundled UCI document's train/test date labels, and the 2,101 quarter-hour
sample count do not reconcile: 2,100 intervals span 21 days and 21 hours if
uniformly spaced. The arrays have no timestamps, so this project preserves
their published order and does not claim exact calendar boundaries. UCI's page
describes 47 feature groups and 36 sensors, while the downloaded MATLAB arrays
contain 48 columns and 36 nodes; the paper's dataset table lists 37 road
segments. Only the first ten traffic-history columns have an unambiguous order.

## Method

The 36-by-36 binary adjacency matrix is symmetric, has 43 undirected edges, and
has no isolated sensors. The first 70% of the supplied training sequence (882
steps) fits models, the next 15% (189 steps) selects model settings, and the
last 15% (190 steps) calibrates per-sensor absolute-residual intervals. The
separate 840-step source test sequence is evaluated only after those choices
are fixed.

Compared approaches:

1. A single historical-lag baseline. The best of ten supplied lag slots is
   selected on validation only.
2. A pooled Ridge model using the target sensor's history, source context, and
   a sensor identity indicator.
3. A pooled adjacency-aware Ridge model that adds the observed-neighbor mean
   for each historical traffic slot.
4. An outage-augmented adjacency Ridge model trained on clean rows and a
   second copy with a deterministic 20% random sensor outage per time step.

Ridge regularization is selected from a fixed logarithmic grid on validation.
The outage-augmented model uses the mean of clean and fixed 20%-outage
validation RMSE for selection. Test scenarios use fixed nested sensor sets at
0%, 10%, and 20% outage; outage rates are not tuned against test performance.

Empirical 90% per-sensor intervals use absolute errors from the separate
calibration block. Since the data are time-dependent, exchangeability is not
established; test interval coverage is reported as a diagnostic rather than a
formal coverage guarantee.

## Locked test results

MAE and RMSE are on the source's normalized target scale. The outage-augmented
model is designed for sensor failures; it gives up some clean-data accuracy.

| Model | Offline sensors | MAE | RMSE | Worst sensor MAE | 90% interval coverage |
|---|---:|---:|---:|---:|---:|
| Selected single lag | 0 | 0.03723 | 0.05455 | 0.05335 | 0.890 |
| Own-sensor/context Ridge | 0 | 0.03155 | 0.04636 | 0.04657 | 0.895 |
| Adjacency-aware Ridge | 0 | 0.03148 | 0.04599 | 0.04591 | 0.895 |
| Outage-augmented graph Ridge | 0 | 0.03695 | 0.05080 | 0.05310 | 0.880 |
| Adjacency-aware Ridge | 4 (10%) | 0.06126 | 0.12173 | 0.33266 | 0.892 |
| Outage-augmented graph Ridge | 4 (10%) | 0.04590 | 0.06569 | 0.13558 | 0.881 |
| Adjacency-aware Ridge | 7 (20%) | 0.08806 | 0.17004 | 0.41012 | 0.891 |
| Outage-augmented graph Ridge | 7 (20%) | 0.05729 | 0.08511 | 0.20055 | 0.881 |

Outage percentages are requested scenario levels rounded to whole sensors:
10% selects 4/36 sensors (11.1%) and 20% selects 7/36 (19.4%). The exact
failed sensor IDs are recorded in `reports/scenario_metrics.csv`.

At 20% outage, outage augmentation lowers test MAE by 34.9% against the clean-
trained adjacency Ridge, while clean MAE is 17.4% higher than that model. This
is a scenario result on one small network and one locked sequence, not evidence
of transfer to other cities. Complete metrics, sensor results, selected
parameters and data checks are versioned in `reports/`.

## Run locally

Python 3.12 is required. From this directory:

```bash
python3.12 -m venv .venv
make install-lock
make prepare
make train
make evaluate
make test
make quality
make smoke-app
```

`make prepare` downloads and validates the source. `make train` fits models
using only the chronological fit block and validation block. `make evaluate`
uses the locked test sequence and writes aggregate results; it stops if a test
report already exists. A deliberate rerun after a documented code correction
requires `make evaluate` with `--force` passed to the module:

```bash
.venv/bin/python -m traffic_forecasting.evaluate --force
```

Launch the graphical workbench:

```bash
make app
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Choose a model, sensor
outage rate, test-sequence step, and sensor. The network view colors sensors by
forecast and marks unavailable sensors; the detail view compares observed and
forecast values with the calibration band. The circular node placement is a
schematic, not a geographic map.

The current local validation has 8 passing tests, clean Ruff checks, a
successful locked dependency installation, successful Streamlit AppTest
interactions, and an HTTP 200 response from the launched localhost app. CI is
configured but has not run remotely.

## Project files

- `src/traffic_forecasting/data.py`: pinned download, MATLAB parsing, schema
  checks, and source manifest.
- `src/traffic_forecasting/features.py`: pooled node features, graph
  aggregation, and deterministic outage masks.
- `src/traffic_forecasting/models.py`: validation-only baseline and Ridge
  selection.
- `src/traffic_forecasting/evaluate.py`: one locked test evaluation and
  per-sensor calibration diagnostics.
- `app.py`: interactive network and forecast explorer.
- `tests/`: source, graph features, outage masks, and interval-metric tests.
- `scripts/smoke_app.py`: Streamlit interaction smoke check.
- `.github/workflows/ci.yml`: configured clean-checkout validation; no remote
  CI run has occurred.

There is no API or Docker image because neither adds useful functionality to
this small local forecasting workbench. GitHub publication is pending; this
repository has no remote.

## Limitations

- There are only 2,101 ordered 15-minute samples for one 36-sensor network.
- The bundled document's calendar labels conflict. The official array order is
  retained, but exact start and end timestamps are not asserted.
- The input column names and target inverse scale are not supplied. Metrics are
  therefore reported only in normalized units.
- The graph model's clean-data gain over pooled own-sensor Ridge is small.
- Outage robustness comes from one predefined random masking protocol, not
  from real hardware failure data. Some sensor failures may be harder.
- Empirical interval coverage is slightly below 90% on the test sequence and
  has no formal time-series guarantee.
- The evaluation is one locked time span. It does not demonstrate robustness
  across networks, months, weather, incidents, or demand regimes.
- Remote CI is configured but has not run. Models and source data remain local
  and ignored; the code, source manifest, aggregate reports, and docs are the
  intended publication contents.
