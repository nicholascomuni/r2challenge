"""Simple Streamlit dashboard for the churn model batch monitoring results.

Usage:
    streamlit run dashboard.py
"""

import json

import pandas as pd
import streamlit as st

from src.utils.config import MONITORING_DIR, PREDICTIONS_DIR, REGISTERED_MODEL_NAME

st.set_page_config(page_title="Churn Model Monitoring", layout="wide")

STATUS_COLORS = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}


def load_batches() -> list[str]:
    status_files = sorted(MONITORING_DIR.glob("*_status.json"))
    return [f.stem.replace("_status", "") for f in status_files]


def load_status(batch_id: str) -> dict:
    with open(MONITORING_DIR / f"{batch_id}_status.json") as f:
        return json.load(f)


def load_predictions(batch_id: str) -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_DIR / f"{batch_id}_predictions.parquet")


def load_metadata(batch_id: str) -> dict:
    with open(PREDICTIONS_DIR / f"{batch_id}_metadata.json") as f:
        return json.load(f)


st.title("Churn Model — Batch Monitoring Dashboard")
st.caption(f"Registered model: `{REGISTERED_MODEL_NAME}`")

batches = load_batches()
if not batches:
    st.warning("No monitored batches found. Run batch_predict.py and monitor.py first.")
    st.stop()

selected_batch = st.selectbox("Batch", batches, index=len(batches) - 1)

status = load_status(selected_batch)
metadata = load_metadata(selected_batch)
predictions = load_predictions(selected_batch)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", f"{STATUS_COLORS[status['status']]} {status['status']}")
col2.metric("Model version", metadata["model_version"])
col3.metric("Row count", metadata["row_count"])
col4.metric("Run timestamp", metadata["timestamp"].split("T")[0])

st.subheader("Status reasons")
for reason in status["status_reasons"]:
    st.write(f"- {reason}")

st.subheader("Key metrics")
col1, col2, col3 = st.columns(3)
performance = status.get("performance")
if performance:
    col1.metric("Batch AUC", f"{performance['auc']:.4f}", delta=f"{performance['auc'] - status['reference_auc']:.4f} vs train")
    col2.metric("Precision", f"{performance['precision']:.4f}")
    col3.metric("Recall", f"{performance['recall']:.4f}")
else:
    st.info("Ground-truth labels not available for this batch yet — performance metrics pending.")

st.subheader("Drift metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean feature PSI", f"{status['feature_drift']['mean_psi']:.4f}")
col2.metric("Max feature PSI", f"{status['feature_drift']['max_psi']:.4f}")
col3.metric("Prediction PSI", f"{status['prediction_drift']['prediction_psi']:.4f}")
label_drift = status.get("label_drift")
if label_drift:
    col4.metric(
        "Label PSI",
        f"{label_drift['label_psi']:.4f}",
        delta=f"{label_drift['batch_label_rate'] - label_drift['reference_label_rate']:.4f} churn rate vs train",
    )
else:
    col4.metric("Label PSI", "n/a")

st.subheader("Data quality")
st.json(status["data_quality"])

st.subheader("Prediction probability distribution")
st.bar_chart(predictions["prediction_probability"].value_counts(bins=20).sort_index())

st.subheader("All batches — status overview")
overview_rows = []
for batch_id in batches:
    s = load_status(batch_id)
    overview_rows.append(
        {
            "batch_id": batch_id,
            "status": s["status"],
            "auc": s["performance"]["auc"] if s["performance"] else None,
            "mean_psi": s["feature_drift"]["mean_psi"],
            "prediction_psi": s["prediction_drift"]["prediction_psi"],
        }
    )
st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

st.caption("Detailed Evidently HTML reports are available in reports/monitoring/.")
