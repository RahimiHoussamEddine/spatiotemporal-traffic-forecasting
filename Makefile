PYTHON ?= .venv/bin/python
PIP_CACHE_DIR := $(CURDIR)/.cache/pip
PIP_TMP_DIR := $(CURDIR)/.cache/tmp
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1

.PHONY: install-lock prepare validate-data train evaluate test quality smoke-app app

install-lock:
	mkdir -p $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -r requirements.lock.txt
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install --no-deps --no-build-isolation -e .

prepare:
	$(PYTHON) -m traffic_forecasting.data prepare

validate-data:
	$(PYTHON) -m traffic_forecasting.data validate

train: prepare
	$(PYTHON) -m traffic_forecasting.train

evaluate:
	$(PYTHON) -m traffic_forecasting.evaluate

test: prepare
	$(PYTHON) -m pytest

quality:
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke-app: prepare
	@if [ ! -f models/traffic_models.joblib ] || [ ! -f reports/evaluation.json ]; then \
		echo "Run make train evaluate before the app smoke test."; \
		exit 1; \
	fi
	$(PYTHON) scripts/smoke_app.py

app:
	$(PYTHON) -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true
