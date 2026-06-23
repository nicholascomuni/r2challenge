"""Monitor a scored batch: data quality, feature drift, prediction drift and performance.

Generates an Evidently HTML report per batch and a status.json with the
GREEN/AMBER/RED diagnosis used by the dashboard.

Usage:
    python -m src.monitoring.monitor
    python -m src.monitoring.monitor --batch 0923_1737
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from mlflow import MlflowClient
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    AUC_AMBER_DROP,
    AUC_RED_DROP,
    BATCH_LABELS_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MERCHANT_ID_COLUMN,
    MISSING_RATE_AMBER,
    MISSING_RATE_RED,
    MLFLOW_TRACKING_URI,
    MONITORING_DIR,
    PREDICTIONS_DIR,
    PSI_AMBER_THRESHOLD,
    PSI_RED_THRESHOLD,
    REAL_DATA_DIR,
    REGISTERED_MODEL_NAME,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
)

BATCH_FEATURES_DIR = REAL_DATA_DIR / "batches" / "features"


def normalize_feature_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Same row-wise max scaling applied during batch inference (see batch_predict.py)."""
    row_max = df[FEATURE_COLUMNS].max(axis=1).replace(0, 1)
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].div(row_max, axis=0)
    return df


def load_reference_data(sample_size: int = 5000) -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and labels as the monitoring reference, sampled together
    (features_train.parquet and target_train.parquet are row-aligned)."""
    features = pd.read_parquet(TRAIN_FEATURES_PATH)
    labels = pd.read_parquet(TRAIN_TARGET_PATH)[LABEL_COLUMN].astype(int)
    if len(features) > sample_size:
        features = features.sample(sample_size, random_state=42)
        labels = labels.loc[features.index]
    return features, labels


def load_latest_model_version():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{REGISTERED_MODEL_NAME}'")
    return max(versions, key=lambda v: int(v.version))


def load_reference_auc(run_id: str) -> float:
    """Read the train/test AUC logged for the registered model's training run, so the
    monitoring baseline always matches whatever model is actually deployed."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics["auc"]


def compute_data_quality(current: pd.DataFrame) -> dict:
    missing_rate = float(current[FEATURE_COLUMNS].isnull().mean().mean())
    return {
        "row_count": len(current),
        "missing_rate": missing_rate,
    }


def compute_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict, Report]:
    """PSI per feature column; report mean and max PSI across all 180 features."""
    report = Report(metrics=[DataDriftPreset(columns=FEATURE_COLUMNS, stattest="psi")])
    report.run(reference_data=reference[FEATURE_COLUMNS], current_data=current[FEATURE_COLUMNS])
    result = report.as_dict()["metrics"][1]["result"]

    psi_scores = {col: v["drift_score"] for col, v in result["drift_by_columns"].items()}
    return {
        "mean_psi": float(np.mean(list(psi_scores.values()))),
        "max_psi": float(np.max(list(psi_scores.values()))),
        "share_of_drifted_features": result["share_of_drifted_columns"],
    }, report


def compute_prediction_drift(reference_scores: pd.Series, current_scores: pd.Series) -> dict:
    report = Report(metrics=[ColumnDriftMetric(column_name="prediction", stattest="psi")])
    report.run(
        reference_data=pd.DataFrame({"prediction": reference_scores}),
        current_data=pd.DataFrame({"prediction": current_scores}),
    )
    result = report.as_dict()["metrics"][0]["result"]
    return {"prediction_psi": float(result["drift_score"])}


def load_batch_labels(batch_id: str) -> pd.DataFrame | None:
    labels_path = BATCH_LABELS_DIR / f"{batch_id}_Ground_Truth.parquet"
    if not labels_path.exists():
        return None
    return pd.read_parquet(labels_path).rename(columns={"MERCHANT_ID": MERCHANT_ID_COLUMN})


def compute_label_drift(reference_labels: pd.Series, batch_labels: pd.DataFrame | None) -> dict | None:
    """Compare the actual churn rate in this batch's ground truth against the training
    label rate. Distinct from prediction drift: this flags a real shift in the business
    outcome itself (e.g. seasonal churn spikes), independent of whether the model's
    predicted probabilities moved."""
    if batch_labels is None or batch_labels.empty:
        return None

    report = Report(metrics=[ColumnDriftMetric(column_name="LABEL", stattest="psi")])
    report.run(
        reference_data=pd.DataFrame({"LABEL": reference_labels}),
        current_data=pd.DataFrame({"LABEL": batch_labels["LABEL"].astype(int)}),
    )
    result = report.as_dict()["metrics"][0]["result"]

    return {
        "label_psi": float(result["drift_score"]),
        "reference_label_rate": float(reference_labels.mean()),
        "batch_label_rate": float(batch_labels["LABEL"].mean()),
    }


def compute_performance(predictions: pd.DataFrame, batch_labels: pd.DataFrame | None) -> dict | None:
    if batch_labels is None:
        return None

    merged = predictions.merge(batch_labels, on=MERCHANT_ID_COLUMN, how="inner")

    if merged.empty:
        return None

    y_true = merged["LABEL"].astype(int)
    y_proba = merged["prediction_probability"]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "label_coverage": len(merged) / len(predictions),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def determine_status(
    data_quality: dict, feature_drift: dict, performance: dict | None, reference_auc: float
) -> tuple[str, list[str]]:
    """GREEN / AMBER / RED based on missing rate, feature PSI, and AUC drop vs. baseline."""
    reasons = []
    status = "GREEN"

    missing_rate = data_quality["missing_rate"]
    if missing_rate >= MISSING_RATE_RED:
        status = "RED"
        reasons.append(f"missing_rate={missing_rate:.4f} >= {MISSING_RATE_RED}")
    elif missing_rate >= MISSING_RATE_AMBER:
        status = "AMBER"
        reasons.append(f"missing_rate={missing_rate:.4f} >= {MISSING_RATE_AMBER}")

    mean_psi = feature_drift["mean_psi"]
    if mean_psi >= PSI_RED_THRESHOLD:
        status = "RED"
        reasons.append(f"mean_psi={mean_psi:.4f} >= {PSI_RED_THRESHOLD}")
    elif mean_psi >= PSI_AMBER_THRESHOLD and status != "RED":
        status = "AMBER"
        reasons.append(f"mean_psi={mean_psi:.4f} >= {PSI_AMBER_THRESHOLD}")

    if performance is not None:
        auc_drop = reference_auc - performance["auc"]
        if auc_drop >= AUC_RED_DROP:
            status = "RED"
            reasons.append(f"auc_drop={auc_drop:.4f} >= {AUC_RED_DROP}")
        elif auc_drop >= AUC_AMBER_DROP and status != "RED":
            status = "AMBER"
            reasons.append(f"auc_drop={auc_drop:.4f} >= {AUC_AMBER_DROP}")

    if not reasons:
        reasons.append("all metrics within expected thresholds")

    return status, reasons


def monitor_batch(
    batch_id: str,
    reference_features: pd.DataFrame,
    reference_scores: pd.Series,
    reference_labels: pd.Series,
    reference_auc: float,
    model_version: str,
) -> dict:
    predictions_path = PREDICTIONS_DIR / f"{batch_id}_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions found for batch '{batch_id}'. Run batch_predict.py first.")

    predictions = pd.read_parquet(predictions_path)

    raw_features = pd.read_parquet(BATCH_FEATURES_DIR / f"{batch_id}_features.parquet")
    raw_features = raw_features.rename(columns={c: MERCHANT_ID_COLUMN for c in raw_features.columns if c.lower() == MERCHANT_ID_COLUMN})
    current_features = normalize_feature_scale(raw_features)
    batch_labels = load_batch_labels(batch_id)

    data_quality = compute_data_quality(current_features)
    feature_drift, drift_report = compute_feature_drift(reference_features, current_features)
    prediction_drift = compute_prediction_drift(reference_scores, predictions["prediction_probability"])
    label_drift = compute_label_drift(reference_labels, batch_labels)
    performance = compute_performance(predictions, batch_labels)

    status, reasons = determine_status(data_quality, feature_drift, performance, reference_auc)

    summary = {
        "batch_id": batch_id,
        "model_version": model_version,
        "status": status,
        "status_reasons": reasons,
        "data_quality": data_quality,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "label_drift": label_drift,
        "performance": performance,
        "reference_auc": reference_auc,
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    status_path = MONITORING_DIR / f"{batch_id}_status.json"
    status_path.write_text(json.dumps(summary, indent=2))

    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=reference_features[FEATURE_COLUMNS], current_data=current_features[FEATURE_COLUMNS])
    quality_report.save_html(str(MONITORING_DIR / f"{batch_id}_data_quality.html"))
    drift_report.save_html(str(MONITORING_DIR / f"{batch_id}_feature_drift.html"))

    print(f"Batch '{batch_id}': status={status} ({'; '.join(reasons)})")
    print(f"  -> {status_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run monitoring checks for one or all scored batches.")
    parser.add_argument("--batch", type=str, default=None, help="Batch id, e.g. 0923_1737 (defaults to all).")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("Loading reference (training) data...")
    reference_features, reference_labels = load_reference_data()

    model_version = load_latest_model_version()
    reference_auc = load_reference_auc(model_version.run_id)
    print(f"Using '{REGISTERED_MODEL_NAME}' version {model_version.version} (train/test AUC={reference_auc:.4f})")

    print("Scoring reference data with the registered model for prediction-drift baseline...")
    # Native xgboost flavor (not pyfunc): pyfunc.predict() on an XGBClassifier returns hard
    # class labels, not probabilities — same reasoning as in batch_predict.py.
    model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_version.version}")
    reference_scores = pd.Series(
        model.predict_proba(reference_features[FEATURE_COLUMNS].astype("float64"))[:, 1],
        index=reference_features.index,
    )

    if args.batch:
        batch_ids = [args.batch]
    else:
        batch_ids = sorted(p.stem.replace("_features", "") for p in BATCH_FEATURES_DIR.glob("*.parquet"))

    if not batch_ids:
        raise RuntimeError(f"No batch feature files found in {BATCH_FEATURES_DIR}")

    for batch_id in batch_ids:
        monitor_batch(
            batch_id, reference_features, reference_scores, reference_labels, reference_auc, model_version.version
        )


if __name__ == "__main__":
    main()
