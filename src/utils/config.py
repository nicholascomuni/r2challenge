"""Shared paths and constants used across the training, inference and monitoring scripts."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw data (provided by R2)
REAL_DATA_DIR = PROJECT_ROOT / "real_data"
TRAIN_FEATURES_PATH = REAL_DATA_DIR / "train" / "features_train.parquet"
TRAIN_TARGET_PATH = REAL_DATA_DIR / "train" / "target_train.parquet"
BATCH_FEATURES_DIR = REAL_DATA_DIR / "batches" / "features"
BATCH_LABELS_DIR = REAL_DATA_DIR / "batches" / "labels"

# Outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
MONITORING_DIR = REPORTS_DIR / "monitoring"
TRAINING_DIR = REPORTS_DIR / "training"

# MLflow
# Defaults to local file-based tracking (no server required). Set MLFLOW_TRACKING_URI
# (e.g. http://localhost:5000) to point at the docker-compose MLflow server instead.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn_model_training"
INFERENCE_EXPERIMENT = "churn_model_inference"
EVALUATION_EXPERIMENT = "churn_model_evaluation"
REGISTERED_MODEL_NAME = "churn_model"

# Release gate thresholds (evaluation on labeled batch data)
EVAL_AUC_MIN = 0.70
EVAL_PRECISION_MIN = 0.50
EVAL_RECALL_MIN = 0.40
TRAIN_TIME_MAX_SECONDS = 300.0   # reject if training run took longer than 5 minutes
INFERENCE_MS_PER_ROW_MAX = 10.0  # reject if average batch scoring exceeds 10 ms/row

# Feature schema
FEATURE_COLUMNS = [f"C{i}" for i in range(1, 181)]
LABEL_COLUMN = "LABEL"

# Column name normalization: production batches use lowercase "merchant_id",
# label files use uppercase "MERCHANT_ID". We normalize everything to this name.
MERCHANT_ID_COLUMN = "merchant_id"

# Monitoring thresholds (see README for rationale)
PSI_AMBER_THRESHOLD = 0.1
PSI_RED_THRESHOLD = 0.25
AUC_AMBER_DROP = 0.03  # absolute AUC drop vs training AUC
AUC_RED_DROP = 0.07
MISSING_RATE_AMBER = 0.01
MISSING_RATE_RED = 0.05
