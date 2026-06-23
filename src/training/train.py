"""Train a churn classifier, track the run in MLflow, and register the model.

Usage:
    python -m src.training.train
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    TRAINING_DIR,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
}


def load_and_validate_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load train features/labels and run basic sanity checks."""
    features = pd.read_parquet(TRAIN_FEATURES_PATH)
    target = pd.read_parquet(TRAIN_TARGET_PATH)

    if len(features) != len(target):
        raise ValueError(f"Row count mismatch: features={len(features)}, target={len(target)}")

    missing_cols = set(FEATURE_COLUMNS) - set(features.columns)
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {sorted(missing_cols)}")

    if LABEL_COLUMN not in target.columns:
        raise ValueError(f"Expected target column '{LABEL_COLUMN}' not found")

    null_counts = features.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Found null values in training features:\n{null_counts[null_counts > 0]}")

    if target[LABEL_COLUMN].isnull().any():
        raise ValueError("Found null values in training target")

    print(f"Loaded features {features.shape}, target {target.shape}")
    print(f"Label rate: {target[LABEL_COLUMN].mean():.4f}")

    return features[FEATURE_COLUMNS], target[LABEL_COLUMN].astype(int)


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], output_path: Path) -> Path:
    importances = model.feature_importances_
    top_n = 20
    order = importances.argsort()[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Feature importance (gain)")
    plt.title(f"Top {top_n} features - churn_model")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_validate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_baseline") as run:
        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        print(f"AUC={auc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        importance_plot_path = TRAINING_DIR / "feature_importance.png"
        plot_feature_importance(model, FEATURE_COLUMNS, importance_plot_path)
        mlflow.log_artifact(str(importance_plot_path))

        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Registered model URI: {model_info.model_uri}")

    print("Training complete.")


if __name__ == "__main__":
    main()
