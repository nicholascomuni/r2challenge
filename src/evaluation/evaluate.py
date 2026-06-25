"""Evaluate the registered churn model on labeled batch data and gate the release.

Aggregates all labeled batch files from real_data/batches/labels/ as the
validation set. Logs an evaluation run to the churn_model_evaluation experiment
and, if the model passes the configured thresholds, promotes it to @champion
in the MLflow Model Registry.

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --model-version 3
    python -m src.evaluation.evaluate --no-promote
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    BATCH_FEATURES_DIR,
    BATCH_LABELS_DIR,
    EVAL_AUC_MIN,
    EVAL_PRECISION_MIN,
    EVAL_RECALL_MIN,
    EVALUATION_EXPERIMENT,
    FEATURE_COLUMNS,
    INFERENCE_MS_PER_ROW_MAX,
    LABEL_COLUMN,
    MERCHANT_ID_COLUMN,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_DIR,
    REGISTERED_MODEL_NAME,
    TRAIN_TIME_MAX_SECONDS,
)

# Promotion to @champion is intentionally manual via the MLflow UI.


def _normalize_feature_scale(df: pd.DataFrame) -> pd.DataFrame:
    row_max = df[FEATURE_COLUMNS].max(axis=1).replace(0, 1)
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].div(row_max, axis=0)
    return df


def load_validation_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Aggregate all labeled batch files into a single held-out validation set."""
    label_files = sorted(BATCH_LABELS_DIR.glob("*.parquet"))
    if not label_files:
        raise RuntimeError(f"No ground truth files found in {BATCH_LABELS_DIR}")

    frames, loaded = [], []
    for label_path in label_files:
        batch_id = label_path.stem.replace("_Ground_Truth", "")
        feature_path = BATCH_FEATURES_DIR / f"{batch_id}_features.parquet"
        if not feature_path.exists():
            print(f"  [skip] No feature file for '{batch_id}'")
            continue

        labels = pd.read_parquet(label_path).rename(
            columns={"MERCHANT_ID": MERCHANT_ID_COLUMN}
        )
        raw = pd.read_parquet(feature_path)
        raw = raw.rename(
            columns={c: MERCHANT_ID_COLUMN for c in raw.columns if c.lower() == MERCHANT_ID_COLUMN}
        )
        merged = raw.merge(labels[[MERCHANT_ID_COLUMN, LABEL_COLUMN]], on=MERCHANT_ID_COLUMN, how="inner")
        if merged.empty:
            print(f"  [skip] Empty join for '{batch_id}'")
            continue
        frames.append(merged)
        loaded.append(batch_id)

    if not frames:
        raise RuntimeError("No labeled batch data available for validation.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalize_feature_scale(combined)

    X = combined[FEATURE_COLUMNS].astype("float64")
    y = combined[LABEL_COLUMN].astype(int)

    print(f"Validation set: {len(combined):,} samples from {len(loaded)} batch(es): {', '.join(loaded)}")
    print(f"Label rate: {y.mean():.4f}")
    return X, y, loaded


def _load_model(version: str | None):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{REGISTERED_MODEL_NAME}'")
    if version is None:
        mv = max(versions, key=lambda v: int(v.version))
    else:
        mv = next((v for v in versions if v.version == str(version)), None)
        if mv is None:
            raise RuntimeError(f"Model version '{version}' not found for '{REGISTERED_MODEL_NAME}'")
    model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")
    return model, mv


def _load_inference_latency(batch_ids: list[str]) -> float | None:
    """Return the average inference latency in ms/row across all available batch metadata files."""
    samples = []
    for batch_id in batch_ids:
        meta_path = PREDICTIONS_DIR / f"{batch_id}_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        t = meta.get("time_spent_seconds", 0.0)
        n = meta.get("row_count", 0)
        if n > 0:
            samples.append(t / n * 1000)  # ms per row
    return float(sum(samples) / len(samples)) if samples else None


def _decide_release(
    metrics: dict,
    training_time: float,
    inference_ms_per_row: float | None,
) -> tuple[str, list[str]]:
    reasons, decision = [], "APPROVED"
    for name, floor in [("auc", EVAL_AUC_MIN), ("precision", EVAL_PRECISION_MIN), ("recall", EVAL_RECALL_MIN)]:
        if metrics[name] < floor:
            decision = "REJECTED"
            reasons.append(f"{name}={metrics[name]:.4f} < {floor}")
    if training_time > TRAIN_TIME_MAX_SECONDS:
        decision = "REJECTED"
        reasons.append(f"training_time={training_time:.1f}s > {TRAIN_TIME_MAX_SECONDS:.0f}s")
    if inference_ms_per_row is not None and inference_ms_per_row > INFERENCE_MS_PER_ROW_MAX:
        decision = "REJECTED"
        reasons.append(f"inference_latency={inference_ms_per_row:.3f}ms/row > {INFERENCE_MS_PER_ROW_MAX:.0f}ms/row")
    if not reasons:
        reasons.append("all metrics above release thresholds")
    return decision, reasons


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the registered churn model and gate the release."
    )
    parser.add_argument(
        "--model-version", type=str, default=None,
        help="Model version to evaluate (default: latest).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EVALUATION_EXPERIMENT)

    print("Loading validation data (labeled batches)...")
    X_val, y_val, loaded_batches = load_validation_data()

    print(f"\nLoading model '{REGISTERED_MODEL_NAME}' version {args.model_version or 'latest'}...")
    model, mv = _load_model(args.model_version)
    print(f"Evaluating version {mv.version} (training run: {mv.run_id})")

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_val, y_proba)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "label_rate": float(y_val.mean()),
        "mean_predicted_proba": float(y_proba.mean()),
        "n_samples": float(len(y_val)),
        "n_batches": float(len(loaded_batches)),
    }

    training_time       = MlflowClient().get_run(mv.run_id).data.metrics.get("training_time_seconds", 0.0)
    inference_ms_per_row = _load_inference_latency(loaded_batches)

    decision, reasons = _decide_release(metrics, training_time, inference_ms_per_row)

    print(
        f"\nAUC={metrics['auc']:.4f}  Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )
    latency_str = f"{inference_ms_per_row:.3f} ms/row" if inference_ms_per_row is not None else "N/A (no metadata files)"
    print(f"Training time: {training_time:.1f}s  |  Inference latency: {latency_str}")
    print(f"Release decision: {decision} ({'; '.join(reasons)})")
    if decision == "APPROVED":
        print("Promote to @champion manually via the MLflow UI (Models → Aliases).")

    with mlflow.start_run(run_name=f"eval_v{mv.version}"):
        mlflow.log_params({
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": mv.version,
            "training_run_id": mv.run_id,
            "validation_batches": ", ".join(loaded_batches),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", training_time)
        if inference_ms_per_row is not None:
            mlflow.log_metric("inference_ms_per_row", inference_ms_per_row)
        mlflow.set_tag("release_decision", decision)
        mlflow.set_tag("release_reasons", "; ".join(reasons))


if __name__ == "__main__":
    main()
