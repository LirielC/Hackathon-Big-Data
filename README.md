# Hackathon Forecast Model 2025

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WMAPE](https://img.shields.io/badge/WMAPE-2.55%25-orange.svg)](https://github.com/LirielC/Hackathon-Big-Data)

End-to-end machine learning pipeline for weekly sales forecasting at the PDV/SKU level, developed for the **Big Data Corp Hackathon Forecast 2025**.

---

## Overview

The goal of this project is to predict sales volumes for the **first five weeks of January 2023** (weeks 1 through 5), using weekly historical transaction data from the full year of 2022. Predictions are generated at the granularity of individual point-of-sale (PDV) and product (SKU) combinations.

The primary evaluation metric is **WMAPE (Weighted Mean Absolute Percentage Error)**:

```
WMAPE = sum(|actual - predicted|) / sum(actual)
```

This formulation weights errors by the actual sales volume, making it robust to the long tail of low-volume SKUs that would otherwise distort a standard MAPE.

### Results

| Model | WMAPE | Approximate Runtime |
|---|---|---|
| XGBoost (tuned) | **2.55%** | ~6 min |
| Stacking Ensemble | 7.54% | ~30 min |
| Ultra-fast baseline | 24.21% | ~8 sec |

The XGBoost model with Optuna-tuned hyperparameters achieves the best result at **2.55% WMAPE** on the held-out validation set.

---

## Requirements

- Python 3.9 or higher
- At least 8 GB of available RAM
- Approximately 5 GB of free disk space for data and serialized models

---

## Installation

```bash
git clone https://github.com/LirielC/Hackathon-Big-Data.git
cd Hackathon-Big-Data

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Place the raw Parquet files inside `data/raw/` before running the pipeline.

---

## Quick Start

Run the full pipeline end-to-end:

```bash
python main.py --step full --verbose
```

Run individual stages:

```bash
python main.py --step ingestion     # load and validate raw data
python main.py --step preprocessing # aggregate and clean
python main.py --step training      # train all models
python main.py --step experiments   # compare experiment results
```

Use a custom configuration file:

```bash
python main.py --config configs/model_config.yaml --step full
```

The `Makefile` provides shortcuts for common tasks:

```bash
make run-pipeline          # python main.py --step full --verbose
make test                  # run the full test suite
make test-coverage         # generate HTML coverage report
make lint                  # run flake8 / isort checks
make format                # auto-format with black and isort
make validate-submission FILE=path/to/file.csv
```

---

## Repository Structure

```
.
├── configs/
│   ├── model_config.yaml          # primary pipeline configuration
│   ├── experiment_config.yaml     # MLflow experiment settings
│   └── submission_strategies.yaml # named submission strategies
├── data/
│   ├── raw/                       # Parquet source files
│   ├── processed/                 # cleaned and aggregated data
│   └── features/
│       └── cache/                 # cached intermediate feature matrices
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Parquet loading with schema validation
│   │   └── preprocessing.py      # missing-value imputation, aggregation
│   ├── features/
│   │   ├── engineering.py         # temporal, lag, and rolling features
│   │   ├── engineering_optimized.py  # Polars-backed fast path
│   │   └── selection.py           # RFE / correlation-based selection
│   ├── models/
│   │   ├── training.py            # XGBoostModel, LightGBMModel, ProphetModel
│   │   ├── ensemble.py            # weighted average and stacking ensembles
│   │   ├── ensemble_advanced.py   # diversity-aware ensemble utilities
│   │   ├── validation.py          # WalkForwardValidator, ResidualAnalyzer
│   │   ├── prediction.py          # inference pipeline
│   │   ├── prediction_optimized.py
│   │   └── output_formatter.py    # CSV / Parquet output formatting
│   └── utils/
│       ├── experiment_tracker.py  # MLflow wrapper and leaderboard
│       ├── mlflow_integration.py  # autolog setup
│       └── performance_utils.py   # memory and parallelism helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_eda_interactive.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_results_analysis.ipynb
├── scripts/
│   └── submission_cli.py          # CLI for managing submission files
├── tests/                         # pytest test suite
├── final_submission/              # versioned submission CSV files
├── main.py                        # pipeline entry point
├── requirements.txt
└── Makefile
```

---

## Data Pipeline

### 1. Ingestion

Raw data is loaded from Parquet files using PyArrow and optionally Polars for lazy evaluation. Automatic schema detection validates column types before any transformation is applied. Data quality checks flag missing keys, unexpected nulls, and out-of-range values.

### 2. Preprocessing

- Missing numeric values are imputed with the median; categorical values use the mode; time series gaps use forward fill.
- Daily transactions are aggregated to weekly frequency (`W`) by summing quantities.
- Transaction, product, and PDV dimension tables are joined on their natural keys.

### 3. Feature Engineering

All features are generated in `src/features/engineering.py`. A Polars-optimized variant (`engineering_optimized.py`) is used when `performance.use_polars` is enabled.

**Temporal features** (derived from the week start date):
- Week of year, month, quarter
- Boolean holiday flag using the Brazilian public holiday calendar (`holidays` library)
- Cyclical encodings (sine/cosine) for week and month to preserve periodicity

**Lag features** (computed per PDV/SKU group):
- Sales at lags of 1, 2, 4, and 8 weeks
- Week-over-week and year-over-year growth rates

**Rolling statistics** (windows of 4, 8, and 12 weeks):
- Rolling mean, standard deviation, and coefficient of variation
- Rolling min/max to capture seasonal bounds

**Categorical encoding**:
- Target encoding for high-cardinality dimensions (product category, PDV type) with a minimum category size of 10 observations to avoid target leakage on rare levels

**Feature selection**:
- Recursive Feature Elimination (RFE) or correlation-based pruning (threshold 0.95) reduces the feature matrix to at most 50 columns before training.

### 4. Model Training

All model classes extend `BaseModel` (defined in `src/models/training.py`), which enforces a common `fit` / `predict` / `save_model` / `load_model` interface.

**XGBoost** (`XGBoostModel`):
- Gradient-boosted decision trees with `eval_metric=mae` and early stopping at 50 rounds.
- Default hyperparameters: `n_estimators=1000`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.
- Optuna TPE search over `n_estimators` [500, 2000], `max_depth` [3, 10], `learning_rate` [0.01, 0.3], `subsample` [0.6, 1.0], `colsample_bytree` [0.6, 1.0]; 100 trials, 3600 s budget.

**LightGBM** (`LightGBMModel`):
- Same architecture and search space as XGBoost; uses `metric=mae` and `verbose=-1`.
- Leaf-wise tree growth typically offers faster convergence on sparse PDV/SKU matrices.

**Prophet** (`ProphetModel`):
- Additive/multiplicative decomposition for trend, weekly seasonality, and yearly seasonality.
- `changepoint_prior_scale=0.05` (conservative trend flexibility), `seasonality_prior_scale=10.0`.
- Fitted independently per PDV/SKU; parallelised across groups with `joblib`.

### 5. Ensemble

Two combination strategies are available:

**Weighted average** (`EnsembleModel`):
- Default weights: XGBoost 0.4, LightGBM 0.4, Prophet 0.2.
- Weights can be optimized via Optuna (200 trials, 1800 s timeout) to minimize WMAPE on the validation fold.

**Stacking** (`ensemble_advanced.py`):
- Out-of-fold predictions from base learners are used as inputs to a meta-learner (Ridge regression with `alpha=1.0` by default, or Random Forest with `n_estimators=100`, `max_depth=5`).
- Diversity check enforces a maximum pairwise prediction correlation of 0.70 between base models before the ensemble is built.
- A minimum relative improvement of 1% over the best individual model is required to accept the ensemble.

### 6. Validation

`WalkForwardValidator` splits the 2022 data into 5 folds using `TimeSeriesSplit`, with a test window of 4 weeks per fold and no gap. Metrics computed per fold: WMAPE, MAE, RMSE, MAPE. `ResidualAnalyzer` checks for heteroscedasticity, outlier concentration, and normality (Kolmogorov-Smirnov test).

### 7. Post-processing and Output

Predictions are post-processed to:
- Clip negative values to zero (`ensure_positive=true`)
- Cap extreme values at 3x the historical mean for the same PDV/SKU (`max_multiplier=3.0`)
- Round to integers, since the submission format requires integer quantities

The final CSV is written with `;` as the delimiter, UTF-8 encoding, and columns in the order `semana;pdv;produto;quantidade`. Versioned files are stored under `final_submission/`.

---

## Configuration Reference

The primary configuration file is `configs/model_config.yaml`. Key sections:

```yaml
general:
  random_seed: 42
  n_jobs: -1           # -1 uses all available CPU cores

data:
  raw_data_path: "data/raw/"
  aggregation:
    frequency: "W"
    start_date: "2022-01-01"
    end_date: "2022-12-31"
  missing_data:
    numeric_strategy: "median"
    categorical_strategy: "mode"
    time_series_strategy: "forward_fill"

features:
  lag:
    periods: [1, 2, 4, 8]
    rolling_windows: [4, 8, 12]
    include_growth_rate: true
  categorical:
    encoding_method: "target"
    min_category_size: 10
  selection:
    method: "rfe"
    max_features: 50
    correlation_threshold: 0.95

models:
  xgboost:
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
    eval_metric: "mae"

ensemble:
  weighted:
    optimize_weights: true
    optimization_trials: 200
    optimization_timeout: 1800
  stacking:
    meta_learner: "ridge"
    ridge_alpha: 1.0
  validation:
    diversity_threshold: 0.7
    min_improvement: 0.01

hyperparameter_tuning:
  enabled: true
  method: "optuna"
  n_trials: 100
  timeout: 3600

performance:
  use_polars: true
  chunk_size: 10000
  parallel_processing: true
  cache_features: true
  memory:
    dtype_optimization: true
```

Environment variables for optional integrations:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## Experiment Tracking

MLflow is used to log parameters, metrics, and artifacts for every training run.

```bash
mlflow ui      # start the tracking UI at http://localhost:5000
```

Tracked artifacts per run: serialized model files (via `joblib`), feature importance plots, prediction CSV, and fold-level metric tables.

```python
from src.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("hackathon-forecast-2025")
leaderboard = tracker.get_leaderboard(metric="wmape", top_k=10)
```

---

## Performance Optimizations

- **Polars** is used for all aggregation and feature-computation steps when `performance.use_polars=true`. Polars lazy evaluation avoids materializing large intermediate DataFrames.
- **Chunked loading** (`chunk_size: 10000`) keeps memory usage bounded during feature matrix construction.
- **dtype optimization** downcasts numeric columns to the smallest adequate type (`float32`, `int32`) to reduce RAM footprint.
- **Feature caching**: computed feature matrices are persisted to `data/features/cache/` and reused across runs when inputs have not changed.
- **Parallel model training**: `joblib` distributes Prophet per-group fitting and Optuna trial evaluation across all available CPU cores.

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# Unit tests only (fast)
pytest tests/ -v -m "not integration and not performance and not slow"

# Integration tests
pytest tests/test_pipeline_integration.py -v -m integration

# Coverage report
pytest tests/ --cov=src --cov-report=html
# open htmlcov/index.html
```

Via Make:

```bash
make test
make test-unit
make test-integration
make test-coverage
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Sales distribution, temporal patterns, outlier identification |
| `02_eda_interactive.ipynb` | Interactive Plotly visualizations of the same EDA |
| `03_feature_engineering.ipynb` | Feature development, importance analysis, multicollinearity checks |
| `04_model_development.ipynb` | Algorithm comparison, hyperparameter sensitivity, learning curves |
| `05_results_analysis.ipynb` | Residual analysis, per-segment error breakdown, business insights |

---

## Code Style

The project follows these conventions:

- Type hints on all public functions and methods
- Google-style docstrings
- Code formatted with `black` and import order enforced by `isort`
- Linting with `flake8`
- Security scanning with `bandit`

Commit message format:

```
feat: add new ensemble diversity metric
fix: correct weekly aggregation boundary condition
docs: update API documentation for FeatureEngineer
test: add unit tests for lag feature generation
```

---

## Submission Format

The output CSV must satisfy the following constraints:

- Columns: `semana;pdv;produto;quantidade`
- Delimiter: `;`
- Encoding: UTF-8
- Weeks: integers 1 through 5
- All values: non-negative integers

Versioned submission files are written to `final_submission/` with a timestamp suffix. Validate a file before submission:

```bash
make validate-submission FILE=final_submission/hackathon_forecast_submission_corrected_<timestamp>.csv
```

---

## License

Developed for the Big Data Corp Hackathon Forecast 2025. See [LICENSE](LICENSE) for details.

---

**Version**: 1.0.0 | **Python**: 3.9+ | **Release**: January 2025