# Churn Prediction Model — MLOps Pipeline

> *"Simplicity is the ultimate sophistication."* — Leonardo da Vinci

A local, end-to-end churn modeling pipeline built on three principles:

**Observability** — every step is tracked in MLflow: training runs, evaluation gates, batch inference results, per-batch drift status, and SHAP explainability. Nothing happens in a black box; every decision is fully traceable.

**Reproducibility** — the model is versioned in the MLflow Model Registry and always loaded by alias (`@champion`). Any inference or monitoring run can be traced back to the exact model version and training run that produced it.

**Simplicity** — no orchestration framework, no cloud dependencies, no unnecessary abstractions. Five plain Python scripts cover the entire lifecycle. Infrastructure should serve the work, not the other way around.

---

## Quick Start

Requires **Docker** and **Docker Compose**. No local Python environment needed.

```bash
# 1. Build the image
docker compose build

# 2. Start the MLflow tracking server
docker compose up -d mlflow
# → open http://localhost:5000

# 3. Train the model
docker compose run --rm app python -m src.training.train

# 4. Score production batches
docker compose run --rm app python -m src.inference.batch_predict

# 5. Evaluate and gate the release
docker compose run --rm app python -m src.evaluation.evaluate
# → check output: release_decision must be APPROVED

# 6. Promote to @champion  (manual step in MLflow UI)
# http://localhost:5000 → Models → churn_model → [version] → Aliases → add "champion"

# 7. Run monitoring and open the report
docker compose run --rm app python -m src.monitoring.monitor
# → open reports/monitoring/champion_report.html in your browser
```

---

## Architecture

```
real_data/train              ──► train.py           ──► MLflow  churn_model vN
                                                               │
                         evaluate.py ◄──────────────────────── │ ◄── real_data/batches/
                          (release gate)                       │      features + labels
                         human promotes ──────────────► @champion alias
                                                               │
                                                               ▼
real_data/batches/features  ──► batch_predict.py   ──► reports/predictions/
                                                         predictions.parquet
                                                         metadata.json
                                                               │
                                                               ▼
real_data/batches/labels    ──► monitor.py         ──► reports/monitoring/
                                                         champion_report.html
                                                         <batch_id>_status.json
                                                               │
                                                               ▼
                                                    reports/monitoring/champion_report.html
                                                    (open in browser)
```

---

## Project Structure

```
project/
├── real_data/                          # provided dataset (not modified)
│   ├── train/                          # features_train.parquet, target_train.parquet
│   └── batches/
│       ├── features/                   # <batch_id>_features.parquet
│       └── labels/                     # <batch_id>_Ground_Truth.parquet
├── src/
│   ├── training/train.py               # validate → train → log → register
│   ├── evaluation/evaluate.py          # release gate (metrics + latency thresholds)
│   ├── inference/batch_predict.py      # score batches via @champion
│   ├── monitoring/monitor.py           # drift · quality · performance → HTML report
│   └── utils/config.py                 # shared paths, thresholds, constants
├── reports/
│   ├── training/                       # feature_importance.png
│   ├── predictions/                    # <batch_id>_predictions.parquet + metadata.json
│   └── monitoring/                     # champion_report.html + <batch_id>_status.json
├── mlruns/                             # MLflow tracking store (file-based or Docker SQLite)
├── docker-compose.yml                  # MLflow server service
├── Dockerfile
└── requirements.txt
```

---

## Setup

Requires **Docker** and **Docker Compose**. No local Python installation needed — the image uses Python 3.11 with all dependencies pre-installed.

```bash
# Build the app image (once, or after requirements.txt changes)
docker compose build
```

The `real_data/` folder must be present at the project root (included in this submission). All output files (`reports/`, `mlruns/`) are written to the host through Docker volume mounts and can be opened directly.

### Start MLflow

```bash
docker compose up -d mlflow
# → UI at http://localhost:5000
```

`MLFLOW_TRACKING_URI=http://mlflow:5000` is pre-configured in the `app` service — no extra setup needed.

---

## Pipeline Reference

All commands run from the project root with the virtual environment activated.

### 1 · Train

```bash
docker compose run --rm app python -m src.training.train
```

Loads `features_train.parquet` / `target_train.parquet`, validates schema and nulls, runs a stratified 80/20 split, and trains an XGBoost classifier. Logs params, metrics, and the feature importance artifact to the `churn_model_training` experiment, then registers the model as `churn_model vN` in the MLflow Model Registry.

The new version has no alias — it is **not** live until a human sets `@champion`.

### 2 · Evaluate (release gate)

```bash
docker compose run --rm app python -m src.evaluation.evaluate                    # latest version
docker compose run --rm app python -m src.evaluation.evaluate --model-version 3  # specific version
```

Evaluates the model against production-labeled data (`real_data/batches/labels/` joined with the corresponding feature files). Logs results to the `churn_model_evaluation` experiment and sets a `release_decision` tag (`APPROVED` / `REJECTED`). See [Model Release Methodology](#model-release-methodology) for the full threshold table and promotion instructions.

### 3 · Score batches

```bash
docker compose run --rm app python -m src.inference.batch_predict                    # all batches
docker compose run --rm app python -m src.inference.batch_predict --batch 0923_1737  # single batch
```

Loads `@champion` from the registry, applies the same feature normalization used during training, and scores every file in `real_data/batches/features/`. Writes per-batch:
- `reports/predictions/<batch_id>_predictions.parquet` — merchant-level churn probabilities
- `reports/predictions/<batch_id>_metadata.json` — model version, row count, scoring time, prediction stats

Also computes SHAP values (500-row sample) and logs them to the inference run in MLflow.

### 4 · Monitor

```bash
docker compose run --rm app python -m src.monitoring.monitor                       # all batches + HTML report
docker compose run --rm app python -m src.monitoring.monitor --batch 0923_1737     # single batch + report
docker compose run --rm app python -m src.monitoring.monitor --lookback 14         # extend report window to 14 days
```

For each batch: computes data quality, feature drift (PSI, KS D statistic, standardized mean shift vs. training data), prediction drift, and — when ground-truth labels are available — live AUC / Precision / Recall. Saves `reports/monitoring/<batch_id>_status.json` (GREEN / AMBER / RED with reasons) and regenerates the self-contained Plotly report at `reports/monitoring/champion_report.html`.

Requires `@champion` to be set in the MLflow Registry.

### 5 · Open the report

```bash
# Open reports/monitoring/champion_report.html directly in any browser
# The file is fully self-contained — no server needed
```

A self-contained interactive Plotly report with batch trend charts, feature drift (PSI + KS + mean shift), SHAP importance, output distributions, and live performance metrics. Regenerated automatically by `monitor.py` after each run.

---

## Model Release Methodology

A new model version is promoted to production through a three-stage gate: **train → evaluate → human review & promote**. Promotion is always a deliberate human decision; no script auto-promotes a model to `@champion`.

### Stage 1 — Train

```bash
docker compose run --rm app python -m src.training.train
```

Trains an XGBoost classifier on `real_data/train/features_train.parquet` using a stratified 80/20 split. On completion a new version of `churn_model` is registered in the MLflow Model Registry and the run is logged to the `churn_model_training` experiment. Key metrics logged: **AUC, Precision, Recall** on the held-out 20 % test split.

The registered version is in the `None` stage by default — it is **not** live until explicitly promoted.

### Stage 2 — Evaluate

```bash
docker compose run --rm app python -m src.evaluation.evaluate                    # evaluates the latest version
docker compose run --rm app python -m src.evaluation.evaluate --model-version 3  # evaluate a specific version
```

Evaluates the model against **production-labeled data** — all files in `real_data/batches/labels/` joined with the corresponding feature files in `real_data/batches/features/`. This is a harder and more representative test than the training hold-out because it uses real batch data that has already flowed through the scoring pipeline. The same `normalize_feature_scale` transformation applied by `batch_predict.py` is used here, so the model sees inputs in exactly the same range it encounters in production.

Results are logged to the `churn_model_evaluation` experiment in MLflow. The `release_decision` tag records `APPROVED` or `REJECTED` based on the following minimum thresholds:

| Metric | Release floor | Rationale |
|---|---|---|
| AUC | ≥ 0.70 | Minimum ranking ability to separate churners from non-churners |
| Precision (at 0.5 threshold) | ≥ 0.50 | At least half of flagged merchants must actually churn |
| Recall (at 0.5 threshold) | ≥ 0.40 | Model must catch at least 40 % of real churners |
| Training time | ≤ 300 s | A longer run signals unacceptable complexity for the retraining cadence |
| Inference latency | ≤ 10 ms / row | Average across all available scored batches; guards against model bloat |

Training time is read directly from the training run's MLflow metrics (`training_time_seconds`). Inference latency is computed from the `time_spent_seconds` and `row_count` fields in each `reports/predictions/<batch_id>_metadata.json` file — the timer in `batch_predict.py` covers only file reading, normalization, and `predict_proba` (SHAP is excluded). If no metadata files exist for the validation batches the latency check is skipped.

If any threshold is missed the run is tagged `REJECTED` with the failing metrics listed in `release_reasons`. **The script never promotes automatically** — a `REJECTED` result means the model should not proceed to Stage 3.

### Stage 3 — Human Review & Promotion

Open the MLflow UI and navigate to **Experiments → churn_model_evaluation → latest run**.

Check:
- `release_decision` tag → must be `APPROVED`
- `auc`, `precision`, `recall` → compare against the prior champion version (visible in the Registry)
- `validation_batches` param → confirm the evaluation covered the expected batch IDs

If satisfied, go to **Models → churn_model → [version number] → Aliases** and set the alias `@champion`. Both `batch_predict.py` and `monitor.py` load `models:/churn_model/@champion`, so they pick up the new version from the next run onwards without any code change.

**Why human promotion?** The precision-recall trade-off carries a direct, asymmetric business cost: false negatives (missed churners) expose R2 to credit loss, while false positives (wrongly flagged merchants) withhold credit from good borrowers. A human should verify that the new model's specific trade-off is acceptable in the current business context — this judgment should not be automated.

**What to do if REJECTED:**
1. Check the `release_reasons` tag in MLflow to identify the failing metric.
2. Inspect the evaluation run's full metric table for context (e.g. very low recall may indicate class imbalance needs addressing).
3. Consider retraining with more or fresher data, adjusting class weights, or tuning the prediction threshold — then repeat from Stage 1.

---

## Retraining Trigger Methodology

After each batch is scored, `monitor.py` computes a health status for the `@champion` model. This status is the primary signal driving the decision of whether and when to trigger retraining.

### How status is computed

Three independent signals are evaluated after each batch. The composite status is the **worst (most severe)** of the three — a single RED signal makes the batch RED regardless of the others.

#### Signal 1 — Data Quality

| Metric | AMBER | RED |
|---|---|---|
| Average missing rate across C1..C180 | ≥ 1 % | ≥ 5 % |

Computed as the mean fraction of null values per feature across all rows in the batch. A rising missing rate usually indicates an upstream data pipeline issue unrelated to model quality.

#### Signal 2 — Feature Drift (PSI)

| Metric | AMBER | RED |
|---|---|---|
| Mean PSI across C1..C180 | ≥ 0.10 | ≥ 0.25 |

Population Stability Index quantifies how much each feature's distribution has shifted relative to the training set. The mean PSI across all 180 features is the gating metric. The per-feature breakdown (top 20 by PSI, KS D statistic, and standardized mean shift) is available in `champion_report.html` → Feature Distribution Drift section.

**PSI formula:** for each feature, bin the training distribution into 10 equal-width bins, apply the same edges to the batch, then compute `PSI = Σ (cur% − ref%) × ln(cur% / ref%)`. Thresholds follow industry convention: < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift.

#### Signal 3 — Live Performance (AUC drop)

| Metric | AMBER | RED |
|---|---|---|
| AUC drop vs. training AUC | ≥ 0.03 | ≥ 0.07 |

Requires ground-truth labels for the batch (`real_data/batches/labels/<batch_id>_Ground_Truth.parquet`). When available, the model's live AUC is compared to the AUC logged during training. A drop signals degraded ranking ability on the live population. When labels are absent this signal is skipped and status is determined by signals 1 and 2 only.

### Status definitions and recommended actions

#### GREEN — all signals within thresholds

No action required. The model is performing within expected bounds. Continue monitoring after each new batch.

#### AMBER — at least one signal elevated, none RED

Do **not** retrain immediately — a single AMBER batch may be noise (seasonal effects, an atypical cohort). Recommended steps:

1. Open `reports/monitoring/champion_report.html` and identify which signal triggered AMBER.
2. **Feature drift AMBER**: inspect the top-20 PSI chart. A handful of high-PSI features versus all 180 moving together suggests a narrow data issue rather than a population-wide shift.
3. **AUC drop AMBER**: check the Output Distribution section for label drift. If the actual churn rate shifted, the model may simply be calibrated to a different base rate.
4. If AMBER persists for **two or more consecutive batches**, begin preparing a retrain. If the third batch is also AMBER or worse, trigger retraining.

#### RED — at least one signal breached the RED threshold

Immediate action required:

1. Flag RED-scored batches as **low-confidence** where feasible — avoid using scores for automated credit decisions until the model is replaced.
2. Open `champion_report.html`: identify which signal is RED, inspect the feature drift table and performance card.
3. Trigger retraining immediately using the full cycle below.
4. Do **not** skip the evaluation gate even under time pressure — a new model that fails release thresholds should not replace the current champion.

### Full retraining cycle

```
RED status  (or ≥ 2 consecutive AMBER batches)
        │
        ▼
1.  docker compose run --rm app python -m src.training.train             # train on latest available data
2.  docker compose run --rm app python -m src.evaluation.evaluate        # gate: AUC ≥ 0.70 · Precision ≥ 0.50 · Recall ≥ 0.40
3.  Review in MLflow UI  http://localhost:5000                           # confirm release_decision = APPROVED
4.  Set @champion alias on the new version                               # Models → churn_model → Aliases
5.  docker compose run --rm app python -m src.inference.batch_predict    # re-score pending batches with the new model
6.  docker compose run --rm app python -m src.monitoring.monitor         # recompute status — confirm RED/AMBER resolved
```

### Diagnostic signals (not status-gating)

Two additional drift signals are computed by `monitor.py` and surfaced in the report, but do **not** gate the composite status:

- **Prediction drift** (PSI of model output scores vs. training-set scores) — expected to shift whenever features shift, so it is a downstream symptom of the same root cause already captured by Signal 2. Used for diagnosis, not as an independent trigger.
- **Label drift** (batch churn rate vs. training churn rate, plus label PSI) — the only signal that reflects the actual business outcome rather than model behavior. Useful for interpreting an AUC drop. Excluded from gating because labels arrive after predictions are already in use, introducing a lag that makes it unsuitable as a real-time action trigger.

### Calendar retrain floor

Even when status stays GREEN indefinitely, retrain the model **at least quarterly**. Gradual population drift can accumulate below detection thresholds; a periodic forced retrain prevents silent, invisible degradation.

---

## Key Technical Decisions

### Why XGBoost?

The brief explicitly values **interpretability over complexity** for this business decision. Gradient-boosted trees on tabular features are a pragmatic middle ground: strong baseline performance with limited tuning, native feature importance, and wide familiarity among risk teams reviewing the model. A simpler logistic regression was considered but the 180 chronologically-ordered count features have non-linear, threshold-like effects (e.g. "any activity in the last week") that trees capture more naturally without manual feature engineering.

### Why MLflow?

Tracking (params / metrics / artifacts) and the Model Registry are both needed by the brief, and MLflow provides both in one tool, runs entirely locally (file-based backend, no cloud dependency), and is the de-facto standard so the workflow generalizes to a real deployment.

### Critical data finding: feature scale mismatch between train and batch data

`features_train.parquet` is **row-wise max-normalized**: every training row's maximum feature value is exactly `1.0`. The batch feature files contain **raw, unbounded transaction counts** (values up to ~30,000). Feeding batch data to the model without correcting for this would silently produce near-meaningless probabilities, since the model only ever saw inputs in `[0, 1]` during training.

`batch_predict.py`, `evaluate.py`, and `monitor.py` all apply the same row-wise max-rescaling before calling the model (see `normalize_feature_scale`). This was verified by reproducing the exact training-data normalization on a few batch rows and confirming the resulting distribution matches.

### Column name normalization

Batch feature files use `merchant_id`; label files use `MERCHANT_ID`. Both are normalized to `merchant_id` on load so batches, predictions, and labels join cleanly.

### Loading the model: native XGBoost flavor, not pyfunc

All scripts load the registered model with `mlflow.xgboost.load_model("models:/churn_model/@champion")` and call `.predict_proba()` directly. We deliberately avoid `mlflow.pyfunc.load_model(...).predict()`: for an `XGBClassifier` logged via `mlflow.xgboost.log_model`, the pyfunc wrapper's `predict()` returns **hard class labels (0/1)**, not probabilities — it silently calls `.predict()`, not `.predict_proba()`. Since the brief explicitly asks for raw churn probabilities, using pyfunc would have quietly produced wrong outputs. The model is still loaded by name and alias from the registry either way, so traceability is unaffected.

### Partial label coverage

Batch label files cover ~97 % of the corresponding feature rows. Performance metrics (AUC / Precision / Recall) are computed on an inner join between predictions and available labels — `label_coverage` is reported alongside the metrics so a drop in coverage is itself visible, since it could mask a deteriorating evaluation sample.

---

## Model Selection

A single XGBoost configuration was trained with commonly-used defaults (`n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`) — no hyperparameter search, per the brief's instruction not to over-invest in tuning. Model quality was judged primarily on **ROC AUC** (overall ranking ability, independent of a probability threshold), with **Precision and Recall** reported at 0.5 to make the business trade-off concrete:

- **Precision** — of the merchants flagged as likely churners, how many actually churn. Low precision means R2 withholds credit from merchants who would not have churned (lost revenue).
- **Recall** — of the merchants who actually churn, how many the model catches. Low recall means churn-prone merchants still get a loan (capital at risk).

Result on the held-out test split: **AUC ≈ 0.815, Precision ≈ 0.64, Recall ≈ 0.43**. The threshold (and possibly a cost-sensitive objective) is the natural next tuning step once the business's relative cost of a false-negative vs. false-positive loan is defined — intentionally left as a follow-up rather than guessed at here.

### Observed results on the provided batches

All three production batches were flagged **AMBER**, driven by an AUC drop of 0.045–0.068 (from 0.815 training AUC down to ~0.75–0.77 per batch). The consistent direction and magnitude across all three batches points to a mild, structural population shift rather than a one-off anomaly — worth monitoring, but not yet severe enough to halt scoring or force an emergency retrain. Label drift was mild (batch churn rates of 19–23 % vs. 20 % in training, PSI < 0.01 in all three batches), so the AUC drop is better explained by degraded ranking on the batch population than by a shift in the underlying churn rate.

---

## Production Architecture

The POC scripts contain all the business logic that matters — feature normalization, PSI computation, release gates, status rules. Moving to production means swapping the local runtime for managed cloud services; **the logic itself does not change**.

### Target architecture (AWS)

```
                        ┌──────────────────────────────────────────────────────┐
                        │                      AWS                              │
                        │                                                        │
  Batch files ─────────►  S3  (data lake)                                      │
  Ground truth          │   data/batches/features/                              │
                        │   data/batches/labels/                                │
                        │   data/train/                                         │
                        │        │                                              │
                        │        ▼                                              │
                        │  Amazon MWAA  (Airflow)                              │
                        │  ┌─────────────────────────────────────┐             │
                        │  │  scoring_dag  (per batch arrival)   │             │
                        │  │    validate → score → monitor        │             │
                        │  │          → alert / flag RED         │             │
                        │  └─────────────────────────────────────┘             │
                        │  ┌─────────────────────────────────────┐             │
                        │  │  retraining_dag  (on demand)        │             │
                        │  │    train → evaluate → notify human  │             │
                        │  └─────────────────────────────────────┘             │
                        │        │                                              │
                        │        ▼                                              │
                        │  AWS Glue / EMR  (PySpark)                          │
                        │  score_batch.py · monitor.py · train.py             │
                        │        │                                              │
                        │        ▼                                              │
                        │  S3  (outputs)              MLflow Server            │
                        │   predictions/ (Parquet)    ECS Fargate              │
                        │   monitoring/status/ (JSON) RDS PostgreSQL           │
                        │   monitoring/reports/ (HTML)S3 artifact store        │
                        │        │                                              │
                        │   Athena (query layer)      CloudWatch + SNS         │
                        │        │                    (alerts)                  │
                        │        ▼                                              │
                        │   Dashboard  (see options below)                     │
                        └──────────────────────────────────────────────────────┘
```

---

### Storage — S3 + Athena + RDS

**S3** is the single source of truth for all data and outputs:

```
s3://r2-churn-model/
├── data/
│   ├── train/                      # features_train.parquet, target_train.parquet
│   ├── batches/features/           # incoming batch files (partitioned by date)
│   └── batches/labels/             # ground truth (arrives after predictions)
├── predictions/                    # <batch_id>_predictions.parquet + metadata.json
├── monitoring/
│   ├── status/                     # <batch_id>_status.json
│   └── reports/                    # champion_report.html (static site, served via CloudFront)
└── mlflow-artifacts/               # MLflow artifact store (models, plots)
```

**Athena** sits on top of S3 as a serverless query layer — no ETL pipeline needed. Parquet files in `predictions/` are immediately queryable for trend analysis ("what was the mean risk score and batch status over the last 90 days?"). Monitoring JSON files can be registered as an external table via AWS Glue Data Catalog. Pay-per-query, zero infrastructure to maintain.

**RDS PostgreSQL** serves as the MLflow backend store (replacing the local SQLite file). A `db.t3.micro` instance is sufficient for the experiment tracking load of this pipeline.

---

### Processing — PySpark on AWS Glue

Both `batch_predict.py` and `monitor.py` are already structured as pure data transformation pipelines. The migration to PySpark is mechanical:

**Batch scoring** (`score_batch.py` → Glue job):
```python
# The normalize_feature_scale logic maps directly to a Pandas UDF
@pandas_udf(DoubleType())
def normalize_and_score(batch_iter):
    model = broadcast_model.value          # model broadcast to all workers
    for batch in batch_iter:
        row_max = batch[FEATURE_COLUMNS].max(axis=1).replace(0, 1)
        scaled  = batch[FEATURE_COLUMNS].div(row_max, axis=0)
        yield pd.Series(model.predict_proba(scaled.astype("float64"))[:, 1])

df = spark.read.parquet(f"s3://r2-churn-model/data/batches/features/{batch_id}/")
predictions = df.withColumn("prediction_probability", normalize_and_score(*FEATURE_COLUMNS))
predictions.write.parquet(f"s3://r2-churn-model/predictions/{batch_id}/")
```

**Monitoring** (`monitor.py` → Glue job): PSI is computed using `approxQuantile` to build reference bin edges, then a `histogram` aggregation on the current batch. The result is the same JSON written to S3 instead of a local file. SHAP values are computed on a 500-row sample collected to the driver — this does not need to be distributed.

**Training** (`train.py` → Glue or ECS task): XGBoost on 180 features and a dataset of this size trains comfortably on a single machine (a `m5.xlarge` ECS Fargate task completes training in seconds). Distributed training via `xgboost.spark.SparkXGBClassifier` is available if the dataset grows by orders of magnitude, but is unnecessary here. Keep it simple.

AWS Glue is recommended over EMR for this workload: serverless (no cluster management), native S3 integration, and the jobs complete fast enough that the per-run cost is minimal.

---

### Orchestration — Apache Airflow on Amazon MWAA

Two DAGs cover the full lifecycle:

**`scoring_dag`** — triggered by an S3 event when a new batch file lands:

```
S3 event: new batch
      │
      ▼
validate_batch       # schema check, row count, null rate — fail fast
      │
      ▼
score_batch          # Glue job: normalize + predict → S3 predictions/
      │
      ▼
run_monitoring       # Glue job: PSI, quality, perf → S3 monitoring/status/
      │
      ▼
check_status         # Python operator: read status JSON, branch on GREEN/AMBER/RED
      │
      ├── GREEN ──► update_report    # regenerate champion_report.html → S3
      │
      ├── AMBER ──► update_report
      │             notify_amber     # SNS → Slack: "batch X is AMBER, reasons..."
      │
      └── RED ───► update_report
                   flag_low_confidence   # tag batch as unreliable in Athena metadata
                   notify_red            # SNS → PagerDuty: "batch X is RED — action required"
                   trigger_retraining_dag
```

**`retraining_dag`** — triggered manually or by `scoring_dag` on RED:

```
      ▼
train_model          # Glue/ECS task: train.py → new churn_model vN in MLflow
      │
      ▼
evaluate_model       # ECS task: evaluate.py → release_decision tag in MLflow
      │
      ├── APPROVED ──► notify_human   # SNS: "v{N} approved — promote to @champion in MLflow"
      │
      └── REJECTED ──► notify_human   # SNS: "v{N} rejected: {reasons}"
```

Promotion to `@champion` remains a **human step** in the MLflow UI — the DAG never promotes automatically.

---

### MLflow — ECS Fargate + RDS + S3

```bash
# docker-compose.yml → ECS task definition
mlflow server \
  --backend-store-uri postgresql://user:pass@rds-host:5432/mlflow \
  --artifacts-destination s3://r2-churn-model/mlflow-artifacts \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000
```

Deploy the MLflow server as an ECS Fargate service behind an internal Application Load Balancer — accessible within the VPC by Airflow workers, Glue jobs, and the data science team's machines (via VPN or AWS Client VPN). No public internet exposure needed.

All scripts already read `MLFLOW_TRACKING_URI` from an environment variable (`src/utils/config.py`), so no code changes are required — just update the environment variable in the ECS task definitions.

---

### Monitoring Dashboard — Three Options

#### Option A · Static HTML on S3 + CloudFront *(recommended for simplicity)*

The existing `champion_report.html` is already a fully self-contained, interactive Plotly report. After each `scoring_dag` run, the Airflow `update_report` task uploads the regenerated file to S3 and CloudFront invalidates its cache:

```python
s3.upload_file("champion_report.html", "r2-churn-model", "monitoring/reports/champion_report.html")
cloudfront.create_invalidation(DistributionId="...", Paths={"Quantity": 1, "Items": ["/monitoring/reports/*"]})
```

A pre-signed URL or a CloudFront distribution with IP allowlist gives the risk team access. Zero servers, zero maintenance, zero additional cost beyond S3 storage. The report refreshes automatically after every batch.

#### Option B · Streamlit on ECS Fargate *(good for internal teams)*

Build a Streamlit app that reads directly from S3 (`status.json` files and Parquet predictions) and from the MLflow tracking server. Deploy as an ECS Fargate service behind an internal Application Load Balancer with Cognito authentication. Requires one always-on Fargate task but gives a fully interactive, real-time UI without the static-refresh limitation of Option A.

#### Option C · Custom Frontend *(best UX, higher effort)*

A React + FastAPI application reading from Athena and S3. The FastAPI backend exposes endpoints like `/api/batches`, `/api/batch/{id}/status`, `/api/trend` that query Athena for historical metrics and return JSON. The React frontend renders charts with Recharts or Plotly.js. Given current AI coding tools, a functional prototype of this can be produced in a day — Claude Code can generate the boilerplate from the API schema. This option makes the most sense once the stakeholder audience extends beyond the data science team.

---

### Alerting — CloudWatch + SNS

Airflow publishes batch status to a CloudWatch custom metric namespace (`ChurnModel/BatchStatus`, dimension `batch_id`). A CloudWatch Alarm fires on any `RED` value and publishes to an SNS topic wired to:
- **Email** — for the model owner
- **Slack** — via AWS Chatbot, for the on-call team
- **PagerDuty** — for after-hours RED alerts that require immediate action

AMBER alerts go to Slack only (informational). GREEN is silent unless a team member queries the dashboard.

---

### What does not change

The entire business logic layer is portable as-is:

| POC component | Production equivalent |
|---|---|
| `src/utils/config.py` | same file, env vars injected by ECS/Glue |
| `normalize_feature_scale()` | same function, wrapped in a Pandas UDF |
| PSI / KS / status logic in `monitor.py` | same logic, runs on the Glue driver or in a UDF |
| `_decide_release()` in `evaluate.py` | same function, runs in an ECS task |
| MLflow experiment tracking | same API calls, different tracking URI |
| `@champion` alias promotion | same manual step in the same MLflow UI |

The investment in a clean, well-abstracted POC pays off here: there is no rewrite, only a change in where and how each script runs.
