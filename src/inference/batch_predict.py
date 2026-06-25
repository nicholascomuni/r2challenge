"""Score batch feature files using the registered churn model.

Usage:
    python -m src.inference.batch_predict
    python -m src.inference.batch_predict --batch 0923_1737
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import shap
from mlflow import MlflowClient

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    BATCH_FEATURES_DIR,
    FEATURE_COLUMNS,
    INFERENCE_EXPERIMENT,
    MERCHANT_ID_COLUMN,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_DIR,
    REGISTERED_MODEL_NAME,
)
_SHAP_SAMPLE_ROWS = 500


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


def _log_shap(model, features: pd.DataFrame, batch_id: str) -> None:
    """Compute SHAP values and log bar + beeswarm plots and top-feature metrics to the active run."""
    sample = features.sample(min(_SHAP_SAMPLE_ROWS, len(features)), random_state=42)

    # shap<=0.46 can't parse XGBoost 2+ base_score stored as '[value]'.
    # XGBoost always re-normalises to the bracket format, so patching save_config
    # or load_config on the booster has no effect. The real data path is:
    #   save_raw(raw_format="ubj") → decode_ubjson_buffer() → float(base_score)
    # We patch decode_ubjson_buffer in shap's _tree module namespace so it strips
    # the brackets before the float() conversion.
    import shap.explainers._tree as _shap_tree

    _orig_decode = _shap_tree.decode_ubjson_buffer

    def _fixed_decode(fd):
        result = _orig_decode(fd)
        try:
            bs = result["learner"]["learner_model_param"]["base_score"]
            if isinstance(bs, str) and bs.startswith("["):
                result["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        except (KeyError, TypeError):
            pass
        return result

    _shap_tree.decode_ubjson_buffer = _fixed_decode
    try:
        explainer = shap.TreeExplainer(model)
    finally:
        _shap_tree.decode_ubjson_buffer = _orig_decode
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = (
        pd.Series(abs(shap_values).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        bar_path = Path(tmpdir) / "shap_bar.png"
        plt.savefig(bar_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        mlflow.log_artifact(str(bar_path), artifact_path="plots")

        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.tight_layout()
        beeswarm_path = Path(tmpdir) / "shap_beeswarm.png"
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        mlflow.log_artifact(str(beeswarm_path), artifact_path="plots")

    top20 = mean_abs_shap.head(20)
    mlflow.log_metrics({f"shap_{name}": round(float(val), 6) for name, val in top20.items()})


def score_batch(batch_path: Path, model, model_version: str) -> dict:
    batch_id = batch_path.stem.replace("_features", "")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    t0 = time.perf_counter()

    df = pd.read_parquet(batch_path)
    df = normalize_columns(df)

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Batch '{batch_id}' is missing expected feature columns: {sorted(missing_cols)}")

    df = normalize_feature_scale(df)
    features = df[FEATURE_COLUMNS].astype("float64")
    probabilities = model.predict_proba(features)[:, 1]

    scoring_time = time.perf_counter() - t0

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

    proba = predictions["prediction_probability"]
    metadata = {
        "batch_id": batch_id,
        "run_timestamp": run_timestamp,
        "time_spent_seconds": round(scoring_time, 3),
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(model_version),
        "row_count": len(predictions),
        "prediction_mean": float(proba.mean()),
        "prediction_std": float(proba.std()),
        "prediction_min": float(proba.min()),
        "prediction_max": float(proba.max()),
        "prediction_median": float(proba.median()),
        "high_risk_rate": float((proba >= 0.5).mean()),
    }
    metadata_path = PREDICTIONS_DIR / f"{batch_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    with mlflow.start_run(run_name=f"batch_{batch_id}"):
        mlflow.log_params({
            "batch_id": batch_id,
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": str(model_version),
            "run_timestamp": run_timestamp,
        })
        mlflow.log_metrics({
            "row_count": metadata["row_count"],
            "time_spent_seconds": metadata["time_spent_seconds"],
            "prediction_mean": metadata["prediction_mean"],
            "prediction_std": metadata["prediction_std"],
            "prediction_min": metadata["prediction_min"],
            "prediction_max": metadata["prediction_max"],
            "prediction_median": metadata["prediction_median"],
            "high_risk_rate": metadata["high_risk_rate"],
        })
        _log_shap(model, features, batch_id)

    print(f"Batch '{batch_id}': {len(predictions)} rows, {scoring_time:.2f}s, mean prob={metadata['prediction_mean']:.4f}")
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
    mlflow.set_experiment(INFERENCE_EXPERIMENT)

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
