# Churn Model — Local MLOps Pipeline

A local, reproducible churn-modeling pipeline: train an XGBoost classifier, track and
register it with MLflow, score production-like batches, and monitor data/prediction
drift and performance with Evidently.

## Architecture

```
real_data/train          ──► train.py            ──► XGBoost model
                                                       │
                                                       ▼
                                              MLflow Tracking + Model Registry
                                                       │
                                                       ▼
real_data/batches/features ──► batch_predict.py  ──► reports/predictions/*.parquet
                                                       + metadata.json
                                                       │
                                                       ▼
real_data/batches/labels   ──► monitor.py        ──► reports/monitoring/*_status.json
                                                       + Evidently HTML reports
                                                       │
                                                       ▼
                                              dashboard.py (Streamlit)
```

## Project structure

```
project/
├── real_data/                  # provided dataset (not modified)
├── src/
│   ├── training/train.py       # load, validate, train, track, register
│   ├── inference/batch_predict.py   # score batches via the model registry
│   ├── monitoring/monitor.py   # data quality, drift, performance, status
│   └── utils/config.py         # shared paths, thresholds, constants
├── reports/
│   ├── training/                # feature importance plot
│   ├── predictions/              # batch predictions + metadata
│   └── monitoring/                # status.json + Evidently HTML reports
├── mlruns/                      # local MLflow tracking store (file-based)
├── dashboard.py                 # Streamlit dashboard
├── docker-compose.yml           # optional MLflow tracking server
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.11 (XGBoost/PyArrow wheels are not yet available for 3.13+ on Windows
at the time of writing).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Place the provided `real_data/` folder at the project root (already present in this
submission).

## Running the pipeline

All commands run from the project root, with the virtual environment activated.

### 1. Train and register the model

```bash
python -m src.training.train
```

This loads `features_train.parquet` / `target_train.parquet`, validates row counts /
schema / nulls, does a stratified 80/20 train-test split, trains an XGBoost classifier,
logs parameters/metrics/artifacts to MLflow, and registers the model as `churn_model` in
the local MLflow Model Registry. Each run creates a new model version.

### 2. Score batches

```bash
python -m src.inference.batch_predict
```

Loads the **latest registered version** of `churn_model` from the MLflow registry (not a
pickle file), scores every file in `real_data/batches/features/`, and writes:
- `reports/predictions/<batch_id>_predictions.parquet` (`merchant_id`, `prediction_probability`, `batch_id`)
- `reports/predictions/<batch_id>_metadata.json` (model version, row count, timestamp, prediction mean/std)

### 3. Monitor

```bash
python -m src.monitoring.monitor
```

For each batch: computes data quality, feature drift (PSI vs. training data), prediction
drift (PSI vs. training-set predictions), and — when ground-truth labels are available —
AUC/precision/recall. Writes `reports/monitoring/<batch_id>_status.json` and Evidently
HTML reports.

### 4. Dashboard

```bash
streamlit run dashboard.py
```

Shows model version, batch status (GREEN/AMBER/RED), AUC, prediction distribution and
drift metrics per batch.

### Optional: MLflow tracking server via Docker

By default all scripts use local file-based MLflow tracking (`./mlruns`), so Docker is
**not required** to run the pipeline. If you want a tracking server with a UI:

```bash
docker compose up -d
```

Then point the scripts at it:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # Windows: set MLFLOW_TRACKING_URI=...
```

MLflow UI (file-based mode, no Docker needed):

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Key technical decisions

### Why XGBoost?
The brief explicitly values **interpretability over complexity** for this business
decision. Gradient-boosted trees on tabular features are a pragmatic middle ground:
strong baseline performance on tabular data with limited tuning, native feature
importance, and wide familiarity among risk teams reviewing the model. A simpler logistic
regression was considered but the 180 chronologically-ordered count features have
non-linear, threshold-like effects (e.g. "any activity in the last week") that trees
capture more naturally without manual feature engineering.

### Why MLflow?
Tracking (params/metrics/artifacts) and the Model Registry are both needed by the brief,
and MLflow provides both in one tool, runs entirely locally (file-based backend, no cloud
dependency), and is the de-facto standard so the workflow generalizes to a real
deployment.

### Why Evidently?
Purpose-built for the exact checks required here — data quality, feature drift,
prediction drift, and performance — with sane defaults (PSI, missing-value reports) and
HTML reports that are easy to hand to a non-technical reviewer.

### Critical data finding: feature scale mismatch between train and batch data
`features_train.parquet` is **row-wise max-normalized**: every training row's maximum
feature value is exactly `1.0`. The batch feature files contain **raw, unbounded
transaction counts** (values up to ~30,000). Feeding batch data to the model without
correcting for this would silently produce near-meaningless probabilities, since the
model only ever saw inputs in `[0, 1]` during training.

`batch_predict.py` and `monitor.py` both apply the same row-wise max-rescaling before
calling the model (see `normalize_feature_scale`). This was verified by reproducing the
exact training-data normalization on a few batch rows and confirming the resulting
distribution matches.

### Column name normalization
Batch feature files use `merchant_id`; label files use `MERCHANT_ID`. Both are normalized
to `merchant_id` on load so batches, predictions, and labels join cleanly.

### Loading the model: native xgboost flavor, not pyfunc
`batch_predict.py` and `monitor.py` load the registered model with
`mlflow.xgboost.load_model("models:/churn_model/<version>")` and call `.predict_proba()`
directly. We deliberately avoid `mlflow.pyfunc.load_model(...).predict()` here: for an
`XGBClassifier` logged via `mlflow.xgboost.log_model`, the pyfunc wrapper's `predict()`
returns **hard class labels (0/1)**, not probabilities — it silently calls the
underlying `.predict()`, not `.predict_proba()`. Since the brief explicitly asks for
raw churn probabilities (not just the predicted class), using pyfunc here would have
quietly produced wrong outputs. The model is still loaded by name/version from the
registry either way, so traceability is unaffected.

### Partial label coverage
Batch label files cover ~97% of the corresponding feature rows. Performance metrics
(AUC/precision/recall) are computed on an inner join between predictions and available
labels — `label_coverage` is reported alongside the metrics so a drop in coverage itself
is visible, since it could mask a deteriorating eval sample.

## Model selection logic

A single XGBoost configuration was trained with simple, commonly-used defaults
(`n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`,
`colsample_bytree=0.8`) — no hyperparameter search, per the brief's instruction not to
over-invest in tuning. Model quality was judged primarily on **ROC AUC** (overall ability
to rank churners above non-churners, independent of a probability threshold), with
**precision and recall** reported at the default 0.5 threshold to make the business
trade-off concrete:
- **Precision** — of the merchants flagged as likely churners, how many actually churn.
  Low precision means R2 would withhold credit from merchants who would not have
  churned (lost revenue).
- **Recall** — of the merchants who actually churn, how many the model catches. Low
  recall means churn-prone merchants still get a loan (capital at risk).

Result on the held-out test split: **AUC ≈ 0.815, precision ≈ 0.64, recall ≈ 0.43**. The
threshold (and possibly a cost-sensitive objective) is the natural next tuning step once
the business's relative cost of a false-negative vs. false-positive loan is defined —
intentionally left as a follow-up rather than guessed at here.

## Monitoring strategy and status framework

| Signal | Metric | AMBER | RED |
|---|---|---|---|
| Data quality | missing rate | ≥ 1% | ≥ 5% |
| Feature drift | mean PSI across C1..C180 | ≥ 0.10 | ≥ 0.25 |
| Performance | AUC drop vs. training | ≥ 0.03 | ≥ 0.07 |

PSI (Population Stability Index) thresholds follow the common industry convention
(<0.1 stable, 0.1–0.25 moderate shift, >0.25 significant shift). The status is the worst
(most severe) of the three gated checks above.

Two additional drift signals are computed and surfaced on the dashboard, but kept as
**diagnostic** rather than status-gating, to avoid double-counting the same root cause
through multiple triggers:
- **Prediction drift** (PSI of predicted probabilities vs. training-set predictions) —
  expected to move whenever either features or model behavior change, so it's a
  symptom of the same underlying shift the feature-drift and AUC-drop checks already gate on.
- **Label drift** (PSI of the batch's ground-truth churn rate vs. the training churn
  rate, plus the raw rates themselves) — this is the only signal that reflects the real
  business outcome directly rather than the model's behavior, and is only available once
  ground-truth labels arrive for a batch. It's diagnostic rather than gating because it
  lags batch scoring (labels arrive after predictions are made) and a shift here is
  useful context for *interpreting* an AUC drop, not an independent action trigger.

### Observed results on the provided batches
All three production batches were flagged **AMBER**, driven by an AUC drop of
0.045–0.068 (from 0.815 training AUC down to ~0.75–0.77 on each batch). This is
consistent in direction and magnitude across all three batches, which points to a
mild, structural population shift rather than a one-off batch anomaly — worth keeping
an eye on, though not yet severe enough to halt scoring or force an emergency retrain.
Label drift was mild (batch churn rates of 19–23% vs. a 20% training rate, PSI < 0.01 in
all three batches), so the AUC drop is better explained by the model's ranking ability
degrading on the batch population than by a shift in the underlying churn rate itself.

## How this would evolve in production

- **Orchestration**: replace manual script execution with Airflow or Dagster DAGs
  (train → register → batch infer → monitor → alert), scheduled to match the batch
  arrival cadence.
- **Storage**: move `real_data/`, `reports/`, and MLflow artifacts to S3 (or equivalent),
  with the MLflow backend store on a managed Postgres/MySQL instance instead of SQLite.
- **Observability**: ship status.json metrics to CloudWatch/Datadog/Grafana with alerting
  on RED status, rather than relying on someone opening the dashboard.
- **Feature store**: if features are reused across multiple models or teams, a feature
  store (e.g. Feast) would centralize the normalization logic discovered here so it's
  defined once instead of duplicated in every consuming pipeline.
- **Retraining cadence**: given the consistent, mild AUC drop observed across all three
  batches, retraining should be triggered by status crossing into RED (or AMBER
  persisting across several consecutive batches) rather than on a fixed calendar
  schedule — with a fixed quarterly retrain as a fallback floor even if no drift is
  detected, to avoid silent staleness.
