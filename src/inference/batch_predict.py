"""Score batch feature files using the registered churn model.

Usage:
    python -m src.inference.batch_predict
    python -m src.inference.batch_predict --batch 0923_1737
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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    BATCH_FEATURES_DIR,
    FEATURE_COLUMNS,
    MERCHANT_ID_COLUMN,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_DIR,
    REGISTERED_MODEL_NAME,
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize merchant id column naming across batch files (merchant_id / MERCHANT_ID)."""
    rename_map = {col: MERCHANT_ID_COLUMN for col in df.columns if col.lower() == MERCHANT_ID_COLUMN}
    return df.rename(columns=rename_map)


def normalize_feature_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale raw transaction-count features to match the [0, 1] row-wise max-normalization
    used in features_train.parquet (every training row's max value is exactly 1.0).

    Batch files contain unbounded raw counts, so feeding them to the model directly
    would silently produce wrong probabilities. We replicate the same per-row max scaling here.
    """
    row_max = df[FEATURE_COLUMNS].max(axis=1).replace(0, 1)
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].div(row_max, axis=0)
    return df


def load_latest_model_version():
    """Load the latest registered model version using the native xgboost flavor (not pyfunc),
    so we can call predict_proba() directly. mlflow.pyfunc.predict() on a logged XGBClassifier
    returns hard class labels (0/1), not probabilities, which is not what we want here.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    model = mlflow.xgboost.load_model(model_uri)
    return model, latest.version


def score_batch(batch_path: Path, model, model_version: str) -> dict:
    batch_id = batch_path.stem.replace("_features", "")

    df = pd.read_parquet(batch_path)
    df = normalize_columns(df)

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Batch '{batch_id}' is missing expected feature columns: {sorted(missing_cols)}")

    df = normalize_feature_scale(df)
    features = df[FEATURE_COLUMNS].astype("float64")
    probabilities = model.predict_proba(features)[:, 1]

    predictions = pd.DataFrame(
        {
            MERCHANT_ID_COLUMN: df[MERCHANT_ID_COLUMN],
            "prediction_probability": probabilities,
            "batch_id": batch_id,
        }
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"{batch_id}_predictions.parquet"
    predictions.to_parquet(output_path, index=False)

    metadata = {
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "row_count": len(predictions),
        "prediction_mean": float(predictions["prediction_probability"].mean()),
        "prediction_std": float(predictions["prediction_probability"].std()),
    }
    metadata_path = PREDICTIONS_DIR / f"{batch_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Batch '{batch_id}': {len(predictions)} rows scored, mean prob={metadata['prediction_mean']:.4f}")
    print(f"  -> {output_path}")
    print(f"  -> {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Score batch feature files with the registered churn model.")
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Specific batch id to score, e.g. 0923_1737 (defaults to all files in batches/features/).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model, model_version = load_latest_model_version()
    print(f"Loaded '{REGISTERED_MODEL_NAME}' version {model_version}")

    if args.batch:
        batch_files = [BATCH_FEATURES_DIR / f"{args.batch}_features.parquet"]
    else:
        batch_files = sorted(BATCH_FEATURES_DIR.glob("*.parquet"))

    if not batch_files:
        raise RuntimeError(f"No batch feature files found in {BATCH_FEATURES_DIR}")

    for batch_path in batch_files:
        score_batch(batch_path, model, model_version)


if __name__ == "__main__":
    main()
