"""Monitor scored batches and generate the champion health HTML report.

Computes data quality, feature drift, prediction drift, and live performance
for each new batch, saves a status.json per batch, then renders a
self-contained HTML report.

Focus: Champion Health · Data Quality · Feature Drift · SHAP

Usage:
    python -m src.monitoring.monitor                       # all batches + report
    python -m src.monitoring.monitor --batch 0923_1737     # single batch + report
    python -m src.monitoring.monitor --lookback 14         # report covers 14 days
    python -m src.monitoring.monitor --output my.html      # custom output path
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from mlflow import MlflowClient
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    AUC_AMBER_DROP,
    AUC_RED_DROP,
    BATCH_LABELS_DIR,
    FEATURE_COLUMNS,
    INFERENCE_EXPERIMENT,
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

_STATUS_COLOR = {"GREEN": "#22c55e", "AMBER": "#f59e0b", "RED": "#ef4444", "UNKNOWN": "#6b7280"}
_MLFLOW_UI    = "http://localhost:5000"


# ══════════════════════════════════════════════════════════════════════════════
# MONITORING — data loading & computation
# ══════════════════════════════════════════════════════════════════════════════

def normalize_feature_scale(df: pd.DataFrame) -> pd.DataFrame:
    row_max = df[FEATURE_COLUMNS].max(axis=1).replace(0, 1)
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].div(row_max, axis=0)
    return df


def load_reference_data(sample_size: int = 5000) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_parquet(TRAIN_FEATURES_PATH)
    labels   = pd.read_parquet(TRAIN_TARGET_PATH)[LABEL_COLUMN].astype(int)
    if len(features) > sample_size:
        features = features.sample(sample_size, random_state=42)
        labels   = labels.loc[features.index]
    return features, labels


def load_latest_model_version():
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions for model '{REGISTERED_MODEL_NAME}'")
    return max(versions, key=lambda v: int(v.version))


def load_reference_auc(run_id: str) -> float:
    return MlflowClient().get_run(run_id).data.metrics["auc"]


def compute_data_quality(current: pd.DataFrame) -> dict:
    return {
        "row_count":    len(current),
        "missing_rate": float(current[FEATURE_COLUMNS].isnull().mean().mean()),
    }


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.histogram_bin_edges(reference, bins=bins)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.maximum(np.histogram(reference, bins=edges)[0] / len(reference), 1e-10)
    cur_pct = np.maximum(np.histogram(current,   bins=edges)[0] / len(current),   1e-10)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """PSI + KS statistic + full descriptive stats per feature."""
    psi_scores: dict[str, float] = {}
    ks_stats:   dict[str, float] = {}
    all_stats:  dict[str, dict]  = {}

    for col in FEATURE_COLUMNS:
        ref_vals = reference[col].values
        cur_vals = current[col].values

        psi_scores[col] = _psi(ref_vals, cur_vals)
        ks_d, _         = scipy_stats.ks_2samp(ref_vals, cur_vals)
        ks_stats[col]   = float(ks_d)

        ref_mean = float(ref_vals.mean())
        ref_std  = float(ref_vals.std())
        cur_mean = float(cur_vals.mean())
        cur_std  = float(cur_vals.std())

        all_stats[col] = {
            "ref_mean":       round(ref_mean, 6),
            "ref_std":        round(ref_std, 6),
            "ref_q25":        round(float(np.percentile(ref_vals, 25)), 6),
            "ref_median":     round(float(np.median(ref_vals)), 6),
            "ref_q75":        round(float(np.percentile(ref_vals, 75)), 6),
            "cur_mean":       round(cur_mean, 6),
            "cur_std":        round(cur_std, 6),
            "cur_q25":        round(float(np.percentile(cur_vals, 25)), 6),
            "cur_median":     round(float(np.median(cur_vals)), 6),
            "cur_q75":        round(float(np.percentile(cur_vals, 75)), 6),
            "mean_diff":      round(cur_mean - ref_mean, 6),
            "mean_shift_std": round((cur_mean - ref_mean) / (ref_std + 1e-10), 4),
            "std_diff_pct":   round((cur_std - ref_std) / (ref_std + 1e-10) * 100, 2),
            "ks_stat":        round(ks_stats[col], 4),
        }

    drifted = sum(1 for v in psi_scores.values() if v >= PSI_AMBER_THRESHOLD)
    top20   = sorted(psi_scores, key=psi_scores.__getitem__, reverse=True)[:20]

    top_drifted_features = [
        {"feature": col, "psi": round(psi_scores[col], 4), "ks_stat": round(ks_stats[col], 4),
         **all_stats[col]}
        for col in top20
    ]

    return {
        "mean_psi":                  float(np.mean(list(psi_scores.values()))),
        "max_psi":                   float(np.max(list(psi_scores.values()))),
        "share_of_drifted_features": drifted / len(FEATURE_COLUMNS),
        "top_drifted_features":      top_drifted_features,
    }


def compute_prediction_drift(reference_scores: pd.Series, current_scores: pd.Series) -> dict:
    return {"prediction_psi": _psi(reference_scores.values, current_scores.values)}


def load_batch_labels(batch_id: str) -> pd.DataFrame | None:
    path = BATCH_LABELS_DIR / f"{batch_id}_Ground_Truth.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).rename(columns={"MERCHANT_ID": MERCHANT_ID_COLUMN})


def compute_label_drift(reference_labels: pd.Series, batch_labels: pd.DataFrame | None) -> dict | None:
    if batch_labels is None or batch_labels.empty:
        return None
    return {
        "label_psi":             _psi(reference_labels.values,
                                      batch_labels["LABEL"].astype(int).values, bins=2),
        "reference_label_rate":  float(reference_labels.mean()),
        "batch_label_rate":      float(batch_labels["LABEL"].mean()),
    }


def compute_performance(predictions: pd.DataFrame, batch_labels: pd.DataFrame | None) -> dict | None:
    if batch_labels is None:
        return None
    merged = predictions.merge(batch_labels, on=MERCHANT_ID_COLUMN, how="inner")
    if merged.empty:
        return None
    y_true = merged["LABEL"].astype(int)
    y_proba = merged["prediction_probability"]
    y_pred  = (y_proba >= 0.5).astype(int)
    return {
        "label_coverage": len(merged) / len(predictions),
        "auc":            float(roc_auc_score(y_true, y_proba)),
        "precision":      float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, zero_division=0)),
    }


def determine_status(
    data_quality: dict, feature_drift: dict, performance: dict | None, reference_auc: float
) -> tuple[str, list[str]]:
    reasons, status = [], "GREEN"

    missing_rate = data_quality["missing_rate"]
    if missing_rate >= MISSING_RATE_RED:
        status = "RED";   reasons.append(f"missing_rate={missing_rate:.4f} >= {MISSING_RATE_RED}")
    elif missing_rate >= MISSING_RATE_AMBER:
        status = "AMBER"; reasons.append(f"missing_rate={missing_rate:.4f} >= {MISSING_RATE_AMBER}")

    mean_psi = feature_drift["mean_psi"]
    if mean_psi >= PSI_RED_THRESHOLD:
        status = "RED";   reasons.append(f"mean_psi={mean_psi:.4f} >= {PSI_RED_THRESHOLD}")
    elif mean_psi >= PSI_AMBER_THRESHOLD and status != "RED":
        status = "AMBER"; reasons.append(f"mean_psi={mean_psi:.4f} >= {PSI_AMBER_THRESHOLD}")

    if performance is not None:
        auc_drop = reference_auc - performance["auc"]
        if auc_drop >= AUC_RED_DROP:
            status = "RED";   reasons.append(f"auc_drop={auc_drop:.4f} >= {AUC_RED_DROP}")
        elif auc_drop >= AUC_AMBER_DROP and status != "RED":
            status = "AMBER"; reasons.append(f"auc_drop={auc_drop:.4f} >= {AUC_AMBER_DROP}")

    if not reasons:
        reasons.append("all metrics within expected thresholds")
    return status, reasons


def monitor_batch(
    batch_id: str,
    reference_features: pd.DataFrame,
    reference_scores:   pd.Series,
    reference_labels:   pd.Series,
    reference_auc:      float,
    model_version:      str,
) -> dict:
    predictions_path = PREDICTIONS_DIR / f"{batch_id}_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No predictions for batch '{batch_id}'. Run batch_predict.py first."
        )

    predictions      = pd.read_parquet(predictions_path)
    raw_features     = pd.read_parquet(BATCH_FEATURES_DIR / f"{batch_id}_features.parquet")
    raw_features     = raw_features.rename(
        columns={c: MERCHANT_ID_COLUMN for c in raw_features.columns if c.lower() == MERCHANT_ID_COLUMN}
    )
    current_features = normalize_feature_scale(raw_features)
    batch_labels     = load_batch_labels(batch_id)

    data_quality     = compute_data_quality(current_features)
    feature_drift    = compute_feature_drift(reference_features, current_features)
    prediction_drift = compute_prediction_drift(
        reference_scores, predictions["prediction_probability"]
    )
    label_drift  = compute_label_drift(reference_labels, batch_labels)
    performance  = compute_performance(predictions, batch_labels)
    status, reasons = determine_status(data_quality, feature_drift, performance, reference_auc)

    summary = {
        "batch_id":       batch_id,
        "model_version":  model_version,
        "status":         status,
        "status_reasons": reasons,
        "data_quality":   data_quality,
        "feature_drift":  feature_drift,
        "prediction_drift": prediction_drift,
        "label_drift":    label_drift,
        "performance":    performance,
        "reference_auc":  reference_auc,
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    status_path = MONITORING_DIR / f"{batch_id}_status.json"
    status_path.write_text(json.dumps(summary, indent=2))

    print(f"Batch '{batch_id}': {status} ({'; '.join(reasons)})")
    print(f"  → {status_path}")
    return summary


def load_monitoring_status(batch_id: str) -> dict | None:
    p = MONITORING_DIR / f"{batch_id}_status.json"
    return json.loads(p.read_text()) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — CSS / JS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0f172a; color: #e2e8f0;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 14px; line-height: 1.5;
}
.header {
  background: #1e293b; border-bottom: 1px solid #334155;
  padding: 14px 28px; display: flex; align-items: center;
  justify-content: space-between; position: sticky; top: 0; z-index: 200;
}
.header-title { font-size: 15px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.01em; }
.header-meta  { font-size: 11px; color: #475569; }
.header-links { display: flex; gap: 12px; }
.header-links a {
  font-size: 11px; color: #60a5fa; text-decoration: none;
  padding: 4px 10px; border: 1px solid #334155; border-radius: 6px;
}
.header-links a:hover { background: #0f172a; }
.status-banner {
  padding: 20px 28px; display: flex; align-items: center;
  gap: 18px; border-bottom: 1px solid #334155;
}
.banner-GREEN  { background: linear-gradient(135deg,rgba(34,197,94,.12),rgba(34,197,94,.03));   border-left: 4px solid #22c55e; }
.banner-AMBER  { background: linear-gradient(135deg,rgba(245,158,11,.12),rgba(245,158,11,.03)); border-left: 4px solid #f59e0b; }
.banner-RED    { background: linear-gradient(135deg,rgba(239,68,68,.14),rgba(239,68,68,.03));   border-left: 4px solid #ef4444; }
.banner-UNKNOWN{ background: linear-gradient(135deg,rgba(107,114,128,.10),rgba(107,114,128,.02));border-left:4px solid #6b7280; }
.status-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
.dot-GREEN  { background: #22c55e; box-shadow: 0 0 10px #22c55e88; }
.dot-AMBER  { background: #f59e0b; box-shadow: 0 0 10px #f59e0b88; }
.dot-RED    { background: #ef4444; box-shadow: 0 0 10px #ef444488; animation: pulse 1.4s infinite; }
.dot-UNKNOWN{ background: #6b7280; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);}50%{opacity:.5;transform:scale(.85);} }
.status-main  { display: flex; flex-direction: column; gap: 3px; }
.status-badge { font-size: 26px; font-weight: 800; letter-spacing: -.02em; }
.badge-GREEN{color:#22c55e;} .badge-AMBER{color:#f59e0b;} .badge-RED{color:#ef4444;} .badge-UNKNOWN{color:#6b7280;}
.status-model   { color: #94a3b8; font-size: 13px; }
.status-reasons { color: #64748b; font-size: 11px; font-style: italic; }
.status-meta  { margin-left: auto; text-align: right; color: #64748b; font-size: 11px; line-height: 1.8; }
.content { max-width: 1500px; margin: 0 auto; padding: 0 24px 60px; }
.cards-grid   { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 24px 0; }
.cards-grid-2 { display: grid; grid-template-columns: repeat(2,1fr); gap: 16px; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px 22px; }
.card-title { font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 12px; }
.card-desc  { font-size: 11px; color: #64748b; margin-bottom: 14px; line-height: 1.6; }
.kv { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; border-bottom: 1px solid #0f172a; }
.kv:last-child { border-bottom: none; }
.kv-k { color: #64748b; font-size: 12px; }
.kv-v { color: #cbd5e1; font-size: 13px; font-weight: 500; }
.kv-v.big   { font-size: 17px; font-weight: 700; color: #f1f5f9; }
.kv-v.mono  { font-family: 'Fira Code', monospace; font-size: 11px; }
.kv-v.green { color: #22c55e; font-weight: 600; }
.kv-v.amber { color: #f59e0b; font-weight: 600; }
.kv-v.red   { color: #ef4444; font-weight: 600; }
.kv-v.blue  { color: #60a5fa; font-weight: 600; }
.card-link { display: block; margin-top: 14px; font-size: 11px; color: #60a5fa; text-decoration: none; text-align: right; }
.card-link:hover { text-decoration: underline; }
.section { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 16px 0; }
.section-title { font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 4px; }
.section-desc  { font-size: 12px; color: #64748b; margin-bottom: 18px; line-height: 1.6; }
.chart-placeholder {
  min-height: 120px; display: flex; align-items: center; justify-content: center;
  color: #334155; font-style: italic; border: 1px dashed #334155; border-radius: 8px; font-size: 12px;
}
.chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
.chart-label  { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.table-wrap { overflow-x: auto; margin-top: 12px; }
.drift-table { width: 100%; border-collapse: collapse; }
.drift-table thead th {
  background: #0f172a; color: #475569; font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em; font-size: 9px;
  padding: 8px 10px; text-align: right; cursor: pointer; user-select: none; white-space: nowrap;
}
.drift-table thead th:first-child { text-align: left; }
.drift-table thead th:hover { color: #94a3b8; }
.drift-table tbody tr { border-bottom: 1px solid #0f172a; }
.drift-table tbody tr:hover { background: rgba(255,255,255,.03); }
.drift-table td { padding: 6px 10px; text-align: right; color: #94a3b8; font-size: 11px; white-space: nowrap; }
.drift-table td.fname { text-align: left; color: #e2e8f0; font-weight: 500; font-family: 'Fira Code', monospace; font-size: 10px; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
.drift-table td.ref-col { color: #60a5fa; }
.drift-table td.cur-col { color: #a78bfa; }
.c-red   { color: #f87171 !important; font-weight: 600; }
.c-amber { color: #fbbf24 !important; font-weight: 500; }
.c-green { color: #4ade80 !important; }
[data-tip] { position: relative; cursor: help; }
[data-tip]::after {
  content: attr(data-tip); display: none; position: absolute;
  bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
  background: #020617; border: 1px solid #334155; color: #cbd5e1;
  font-size: 11px; line-height: 1.5; padding: 8px 12px; border-radius: 8px;
  width: 230px; white-space: normal; z-index: 1000; font-weight: 400;
  text-transform: none; letter-spacing: 0; pointer-events: none;
  box-shadow: 0 4px 16px rgba(0,0,0,.5);
}
.drift-table thead [data-tip]::after { top: calc(100% + 8px); bottom: auto; }
[data-tip]:hover::after { display: block; }
.legend { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #64748b; }
.legend-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.perf-row  { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.perf-card { background: #0f172a; border-radius: 10px; padding: 18px; text-align: center; }
.perf-val  { font-size: 28px; font-weight: 800; color: #f1f5f9; letter-spacing: -.02em; }
.perf-label{ font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: .08em; margin-top: 6px; }
.footer { text-align: center; color: #1e293b; font-size: 11px; margin-top: 40px; padding: 20px; }
@media (max-width: 1100px) {
  .cards-grid { grid-template-columns: 1fr 1fr; }
  .chart-grid-2 { grid-template-columns: 1fr; }
  .perf-row { grid-template-columns: repeat(2,1fr); }
}
@media (max-width: 700px) {
  .cards-grid { grid-template-columns: 1fr; }
  .cards-grid-2 { grid-template-columns: 1fr; }
}
"""

_JS = """
(function(){
  var _d={};
  window.sortTable=function(tid,col){
    var t=document.getElementById(tid); if(!t) return;
    var tb=t.querySelector('tbody');
    var rows=Array.from(tb.querySelectorAll('tr'));
    var key=tid+'_'+col; _d[key]=_d[key]===1?-1:1;
    rows.sort(function(a,b){
      var av=a.cells[col].innerText.replace(/[%+σ]/g,'');
      var bv=b.cells[col].innerText.replace(/[%+σ]/g,'');
      var af=parseFloat(av),bf=parseFloat(bv);
      if(!isNaN(af)&&!isNaN(bf)) return _d[key]*(af-bf);
      return _d[key]*av.localeCompare(bv);
    });
    rows.forEach(function(r){tb.appendChild(r);});
  };
})();
"""


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — MLflow data loaders
# ══════════════════════════════════════════════════════════════════════════════

def get_champion() -> dict | None:
    try:
        mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        return {
            "version": mv.version,
            "run_id":  mv.run_id,
            "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc),
        }
    except Exception:
        return None


def get_training_run(run_id: str) -> dict | None:
    try:
        run = MlflowClient().get_run(run_id)
        return {
            "metrics":       run.data.metrics,
            "run_id":        run.info.run_id,
            "experiment_id": run.info.experiment_id,
        }
    except Exception:
        return None


def get_inference_runs(lookback_days: int) -> list[dict]:
    client = MlflowClient()
    exp    = client.get_experiment_by_name(INFERENCE_EXPERIMENT)
    if exp is None:
        return []
    cutoff_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
    )
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"start_time >= {cutoff_ms}",
        order_by=["start_time ASC"],
    )
    result = []
    for r in runs:
        result.append({
            "batch_id":      r.data.params.get("batch_id", "?"),
            "model_version": r.data.params.get("model_version", "?"),
            "timestamp":     datetime.fromtimestamp(r.info.start_time / 1000, tz=timezone.utc),
            "metrics":       r.data.metrics,
            "run_id":        r.info.run_id,
            "experiment_id": r.info.experiment_id,
            "drift_status":  "UNKNOWN",  # filled in main() from status JSONs
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — chart builders
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(height: int = 380, margin_l: int = 20) -> dict:
    return dict(
        height=height,
        paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif", size=11),
        margin=dict(l=margin_l, r=20, t=20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1,
                    font=dict(color="#94a3b8", size=11)),
        xaxis=dict(gridcolor="#1e293b", linecolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", linecolor="#334155", zerolinecolor="#334155"),
        colorway=["#60a5fa", "#a78bfa", "#34d399", "#f59e0b", "#f87171"],
        hovermode="x unified",
    )


def chart_batch_trend(runs: list[dict]) -> go.Figure:
    if not runs:
        return go.Figure()

    # Keep only the most recent run per batch_id, then sort by inference timestamp
    seen: dict[str, dict] = {}
    for r in runs:
        bid = r["batch_id"]
        if bid not in seen or r["timestamp"] > seen[bid]["timestamp"]:
            seen[bid] = r
    deduped = sorted(seen.values(), key=lambda r: r["timestamp"])

    x_labels   = [r["timestamp"].strftime("%b %d, %Y")           for r in deduped]
    batch_ids  = [r["batch_id"]                                   for r in deduped]
    mean_prob  = [r["metrics"].get("prediction_mean", 0)          for r in deduped]
    high_risk  = [r["metrics"].get("high_risk_rate",  0)          for r in deduped]
    row_counts = [int(r["metrics"].get("row_count",   0))         for r in deduped]
    statuses   = [r.get("drift_status", "UNKNOWN")                for r in deduped]
    mcolors    = [_STATUS_COLOR.get(s, "#6b7280")                 for s in statuses]
    timestamps = [r["timestamp"].strftime("%Y-%m-%d %H:%M UTC")   for r in deduped]

    _axis = dict(
        type="category",
        gridcolor="#334155", linecolor="#334155",
        tickfont=dict(family="Fira Code, monospace", size=11),
        tickangle=-30,
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Mean Prediction Score per Batch", "High-Risk Rate per Batch (score ≥ 0.5)"),
        vertical_spacing=0.20,
    )

    fig.add_trace(go.Bar(
        x=x_labels, y=mean_prob,
        marker_color=mcolors, marker_line_width=0, opacity=0.9,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Mean risk score: %{y:.4f}<br>"
            "Merchants: %{customdata[1]:,}<br>"
            "Scored at: %{customdata[2]}<br>"
            "Drift status: %{customdata[3]}<extra></extra>"
        ),
        customdata=list(zip(batch_ids, row_counts, timestamps, statuses)),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_labels, y=high_risk,
        marker_color=mcolors, marker_line_width=0, opacity=0.9,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "High-risk merchants: %{y:.1%}<br>"
            "Merchants: %{customdata[1]:,}<br>"
            "Drift status: %{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(batch_ids, row_counts, statuses)),
        showlegend=False,
    ), row=2, col=1)

    layout = _base_layout(height=520)
    layout["margin"] = dict(l=20, r=20, t=20, b=80)
    layout["xaxis"]  = {**_axis}
    layout["xaxis2"] = {**_axis}
    layout["yaxis"]  = dict(gridcolor="#334155", linecolor="#334155",
                             tickformat=".3f", title_text="mean risk score")
    layout["yaxis2"] = dict(gridcolor="#334155", linecolor="#334155",
                             tickformat=".0%",  title_text="% high-risk")
    layout["bargap"]  = 0.35
    layout["hovermode"] = "x"
    fig.update_layout(**layout)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#64748b", size=11))
    return fig


def chart_psi_bars(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return go.Figure()

    feats  = [f["feature"]           for f in reversed(top_features)]
    psi    = [f["psi"]               for f in reversed(top_features)]
    ks     = [f.get("ks_stat", 0)    for f in reversed(top_features)]
    shifts = [f.get("mean_shift_std", 0) for f in reversed(top_features)]

    colors = ["#ef4444" if k >= 0.20 else "#f59e0b" if k >= 0.10 else "#22c55e" for k in ks]

    fig = go.Figure(go.Bar(
        y=feats, x=psi, orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "PSI: %{x:.4f}<br>"
            "KS D stat: %{customdata[0]:.4f}<br>"
            "Mean shift: %{customdata[1]:+.3f}σ<extra></extra>"
        ),
        customdata=list(zip(ks, shifts)),
    ))
    fig.add_vline(x=0.10, line_dash="dot", line_color="#f59e0b", line_width=1,
                  annotation_text="AMBER 0.10", annotation_font_color="#f59e0b", annotation_font_size=9)
    fig.add_vline(x=0.25, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="RED 0.25",   annotation_font_color="#ef4444",  annotation_font_size=9)

    layout = _base_layout(height=500, margin_l=160)
    layout["xaxis"]     = dict(gridcolor="#334155", linecolor="#334155", title_text="PSI", zerolinecolor="#334155")
    layout["yaxis"]     = dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(family="Fira Code, monospace", size=10))
    layout["hovermode"] = "y"
    fig.update_layout(**layout)
    return fig


def chart_mean_shift(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return go.Figure()

    srt    = sorted(top_features, key=lambda x: abs(x.get("mean_shift_std", 0)), reverse=True)
    feats  = [f["feature"]               for f in reversed(srt)]
    shifts = [f.get("mean_shift_std", 0) for f in reversed(srt)]
    ref_m  = [f.get("ref_mean", 0)       for f in reversed(srt)]
    cur_m  = [f.get("cur_mean", 0)       for f in reversed(srt)]
    ref_s  = [f.get("ref_std", 0)        for f in reversed(srt)]

    colors = ["#ef4444" if abs(s) >= 1.0 else "#f59e0b" if abs(s) >= 0.5 else "#22c55e" for s in shifts]

    fig = go.Figure(go.Bar(
        y=feats, x=shifts, orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Δμ/σ = %{x:+.3f} std devs<br>"
            "Ref μ: %{customdata[0]:.4f}  σ: %{customdata[2]:.4f}<br>"
            "Cur μ: %{customdata[1]:.4f}<extra></extra>"
        ),
        customdata=list(zip(ref_m, cur_m, ref_s)),
    ))
    for v, c in [(0.5, "#f59e0b"), (-0.5, "#f59e0b"), (1.0, "#ef4444"), (-1.0, "#ef4444")]:
        fig.add_vline(x=v, line_dash="dot", line_color=c, line_width=1)

    layout = _base_layout(height=500, margin_l=160)
    layout["xaxis"]     = dict(gridcolor="#334155", linecolor="#334155",
                                title_text="Standardized Mean Shift  (Δμ / σ_ref)", zerolinecolor="#475569")
    layout["yaxis"]     = dict(gridcolor="#1e293b", linecolor="#334155",
                                tickfont=dict(family="Fira Code, monospace", size=10))
    layout["hovermode"] = "y"
    fig.update_layout(**layout)
    return fig


def chart_shap(run: dict) -> go.Figure:
    shap = {k[5:]: v for k, v in run["metrics"].items() if k.startswith("shap_")}
    if not shap:
        return go.Figure()

    top20  = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:20]
    feats  = [s[0] for s in reversed(top20)]
    vals   = [s[1] for s in reversed(top20)]
    max_v  = max(vals) if vals else 1
    colors = [f"rgba(96,165,250,{0.25 + 0.75 * v / max_v:.2f})" for v in vals]

    fig = go.Figure(go.Bar(
        y=feats, x=vals, orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.6f}<br>Higher = more influence on predictions<extra></extra>",
    ))
    layout = _base_layout(height=480, margin_l=160)
    layout["xaxis"]     = dict(gridcolor="#334155", linecolor="#334155", title_text="Mean |SHAP value|", zerolinecolor="#334155")
    layout["yaxis"]     = dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(family="Fira Code, monospace", size=10))
    layout["hovermode"] = "y"
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — HTML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chart_html(fig: go.Figure, placeholder: str = "No data") -> str:
    if not fig.data:
        return f'<div class="chart-placeholder">{placeholder}</div>'
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displayModeBar": "hover",
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                "displaylogo": False})


def _kv(k: str, v: str, cls: str = "", tip: str = "") -> str:
    tip_attr = f' data-tip="{tip}"' if tip else ""
    return f'<div class="kv"><span class="kv-k"{tip_attr}>{k}</span><span class="kv-v {cls}">{v}</span></div>'


def _mlflow_run_link(label: str, experiment_id: str, run_id: str) -> str:
    url = f"{_MLFLOW_UI}/#/experiments/{experiment_id}/runs/{run_id}"
    return f'<a class="card-link" href="{url}" target="_blank">&#x2197; {label}</a>'


def _mlflow_model_link(version: str) -> str:
    url = f"{_MLFLOW_UI}/#/models/{REGISTERED_MODEL_NAME}/versions/{version}"
    return f'<a class="card-link" href="{url}" target="_blank">&#x2197; View in MLflow Registry</a>'


def _champion_card(champion: dict | None, training: dict | None) -> str:
    if not champion:
        return (
            '<div class="card"><div class="card-title">Champion Model</div>'
            '<p class="card-desc">No @champion alias set.<br>'
            'Go to MLflow → Models → churn_model → Aliases to promote a version.</p></div>'
        )
    m       = (training or {}).get("metrics", {})
    created = champion["created"].strftime("%Y-%m-%d %H:%M UTC")
    auc_cls = "green" if m.get("auc", 0) >= 0.75 else "amber" if m.get("auc", 0) >= 0.65 else "red"
    links   = _mlflow_model_link(champion["version"])
    if training:
        links += _mlflow_run_link("View training run", training["experiment_id"], training["run_id"])
    return f"""
    <div class="card">
      <div class="card-title">Champion Model</div>
      {_kv("Name", REGISTERED_MODEL_NAME, "mono")}
      {_kv("Version", f"v{champion['version']}", "big blue")}
      {_kv("Promoted", created)}
      {_kv("Train AUC", f"{m.get('auc',0):.4f}", auc_cls,
           "AUC on the held-out test split from training. Baseline for drift comparison.")}
      {_kv("Train Precision", f"{m.get('precision',0):.4f}", "",
           "Precision at threshold 0.5 on the training test split.")}
      {_kv("Train Recall", f"{m.get('recall',0):.4f}", "",
           "Recall at threshold 0.5 on the training test split.")}
      {_kv("Training time", f"{m.get('training_time_seconds',0):.1f}s")}
      {links}
    </div>"""


def _data_quality_card(latest_status: dict | None) -> str:
    if not latest_status:
        return (
            '<div class="card"><div class="card-title">Data Quality</div>'
            '<p class="card-desc">Run monitor to compute quality stats.</p></div>'
        )
    dq        = latest_status.get("data_quality", {})
    fd        = latest_status.get("feature_drift", {})
    missing   = dq.get("missing_rate", 0)
    miss_cls  = "red" if missing >= 0.10 else "amber" if missing >= 0.05 else "green"
    mean_psi  = fd.get("mean_psi", 0)
    psi_cls   = "red" if mean_psi >= 0.25 else "amber" if mean_psi >= 0.10 else "green"
    drift_pct = fd.get("share_of_drifted_features", 0)
    drift_cls = "red" if drift_pct >= 0.30 else "amber" if drift_pct >= 0.10 else "green"
    return f"""
    <div class="card">
      <div class="card-title">Data Quality — Latest Batch</div>
      {_kv("Merchants scored", f"{int(dq.get('row_count',0)):,}")}
      {_kv("Missing rate", f"{missing:.2%}", miss_cls,
           "Average fraction of null values across all features. "
           "Above 5% = AMBER, above 10% = RED.")}
      {_kv("Mean PSI (all features)", f"{mean_psi:.4f}", psi_cls,
           "Average Population Stability Index across all features. "
           "PSI < 0.10 = stable · 0.10–0.25 = moderate drift · > 0.25 = significant drift.")}
      {_kv("Max PSI (worst feature)", f"{fd.get('max_psi',0):.4f}")}
      {_kv("Features above AMBER (≥ 0.10)", f"{drift_pct:.1%}", drift_cls,
           "Share of features whose PSI exceeds 0.10. "
           "Indicates how widespread the drift is across the feature space.")}
      {_kv("Reference AUC (training)", f"{latest_status.get('reference_auc',0):.4f}", "",
           "Baseline AUC from the training run. Used to measure AUC drop when ground truth is available.")}
    </div>"""


def _latest_batch_card(run: dict | None, batch_id: str) -> str:
    if not run:
        return (
            '<div class="card"><div class="card-title">Latest Batch</div>'
            '<p class="card-desc">No recent inference runs found.</p></div>'
        )
    m         = run["metrics"]
    ts        = run["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
    risk_cls  = "red" if m.get("high_risk_rate", 0) > 0.20 else "amber" if m.get("high_risk_rate", 0) > 0.10 else "green"
    link      = _mlflow_run_link("View inference run", run["experiment_id"], run["run_id"])
    return f"""
    <div class="card">
      <div class="card-title">Latest Batch</div>
      {_kv("Batch ID", batch_id, "mono")}
      {_kv("Model version", f"v{run['model_version']}")}
      {_kv("Scored at", ts)}
      {_kv("Mean risk score", f"{m.get('prediction_mean',0):.4f}", "",
           "Average predicted churn probability across all merchants in this batch.")}
      {_kv("High-risk merchants (≥ 0.5)", f"{m.get('high_risk_rate',0):.1%}", risk_cls,
           "Fraction of merchants with predicted churn probability ≥ 0.5.")}
      {_kv("Score range", f"{m.get('prediction_min',0):.3f} – {m.get('prediction_max',0):.3f}")}
      {link}
    </div>"""


def _output_distribution_section(latest_status: dict | None, latest_run: dict | None) -> str:
    ld   = (latest_status or {}).get("label_drift") or {}
    pd_s = (latest_status or {}).get("prediction_drift") or {}
    m    = (latest_run or {}).get("metrics", {})

    ref_label = ld.get("reference_label_rate")
    cur_label = ld.get("batch_label_rate")
    label_psi = ld.get("label_psi")
    pred_psi  = pd_s.get("prediction_psi")

    def _psi_cls(v):
        if v is None: return ""
        return "red" if v >= 0.25 else "amber" if v >= 0.10 else "green"

    if ref_label is not None and cur_label is not None:
        delta     = cur_label - ref_label
        delta_cls = "red" if abs(delta) >= 0.05 else "amber" if abs(delta) >= 0.02 else "green"
        label_card = f"""
        <div class="card">
          <div class="card-title">Label Distribution — Actual Churn Rate</div>
          <p class="card-desc">
            The <strong>real-world churn rate</strong> observed in ground truth labels.
            Measures whether the business outcome itself is shifting —
            independent of what the model predicts.
            A rising rate means more merchants are actually churning compared to when the model was trained.
          </p>
          {_kv("Reference churn rate (training)", f"{ref_label:.2%}")}
          {_kv("Current batch churn rate", f"{cur_label:.2%}")}
          {_kv("Delta", f"{delta:+.2%}", delta_cls,
               "How much the churn rate shifted. ±2% = AMBER (watch), ±5% = RED (investigate).")}
          {_kv("Label PSI", f"{label_psi:.4f}" if label_psi is not None else "—", _psi_cls(label_psi),
               "PSI on the binary churn label. High value = significant shift in the proportion of churners.")}
        </div>"""
    else:
        label_card = """
        <div class="card">
          <div class="card-title">Label Distribution — Actual Churn Rate</div>
          <p class="card-desc">
            The <strong>real-world churn rate</strong> vs training reference.
            Not available for this batch — add a
            <code style="color:#60a5fa">*_Ground_Truth.parquet</code>
            file to enable label drift monitoring.
          </p>
        </div>"""

    pred_median = m.get("prediction_median", m.get("prediction_mean", 0))
    pred_card = f"""
    <div class="card">
      <div class="card-title">Prediction Score Distribution</div>
      <p class="card-desc">
        How the <strong>model's output scores</strong> are distributed across merchants in this batch.
        Always available — does not require ground truth.
        A shift here may reflect feature drift, genuine business change, or both.
      </p>
      {_kv("Mean", f"{m.get('prediction_mean',0):.4f}", "",
           "Average churn probability across all merchants.")}
      {_kv("Median", f"{pred_median:.4f}", "",
           "50th percentile. Less sensitive to extreme scores than the mean.")}
      {_kv("Min / Max", f"{m.get('prediction_min',0):.4f} / {m.get('prediction_max',0):.4f}")}
      {_kv("High-risk rate (≥ 0.5)", f"{m.get('high_risk_rate',0):.1%}",
           "red" if m.get("high_risk_rate",0) > 0.20 else "amber" if m.get("high_risk_rate",0) > 0.10 else "green",
           "Fraction of merchants above the 0.5 threshold. > 10% = AMBER, > 20% = RED.")}
      {_kv("Prediction PSI vs reference", f"{pred_psi:.4f}" if pred_psi is not None else "—", _psi_cls(pred_psi),
           "PSI between prediction score distributions: training reference vs this batch.")}
    </div>"""

    return f"""
    <div class="section">
      <div class="section-title">Output Distribution</div>
      <div class="section-desc">
        Two complementary views of model outputs.
        <strong>Label distribution</strong> measures the ground truth churn rate (requires labels).
        <strong>Prediction distribution</strong> measures the model's scored risk — always available.
        Hover metric names for definitions.
      </div>
      <div class="cards-grid-2">{label_card}{pred_card}</div>
    </div>"""


def _descriptive_stats_table(features: list[dict]) -> str:
    if not features:
        return '<p style="color:#334155;font-style:italic;font-size:12px;">No drift data — run monitor first.</p>'

    def _psi_cls(v):  return "c-red" if v >= 0.25 else "c-amber" if v >= 0.10 else "c-green"
    def _ks_cls(v):   return "c-red" if v >= 0.20 else "c-amber" if v >= 0.10 else "c-green"
    def _sh_cls(v):   a=abs(v); return "c-red" if a >= 1.0 else "c-amber" if a >= 0.5 else ""
    def _std_cls(v):  a=abs(v); return "c-red" if a >= 20  else "c-amber" if a >= 10  else ""

    rows = "".join(f"""
        <tr>
          <td class="fname" title="{f['feature']}">{f['feature']}</td>
          <td class="{_psi_cls(f['psi'])}">{f['psi']:.4f}</td>
          <td class="{_ks_cls(f.get('ks_stat',0))}">{f.get('ks_stat',0):.4f}</td>
          <td class="ref-col">{f.get('ref_mean',0):.4f}</td>
          <td class="cur-col">{f.get('cur_mean',0):.4f}</td>
          <td class="{_sh_cls(f.get('mean_shift_std',0))}">{f.get('mean_shift_std',0):+.3f}σ</td>
          <td class="ref-col">{f.get('ref_std',0):.4f}</td>
          <td class="cur-col">{f.get('cur_std',0):.4f}</td>
          <td class="{_std_cls(f.get('std_diff_pct',0))}">{f.get('std_diff_pct',0):+.1f}%</td>
        </tr>""" for f in features)

    TIP = {
        "PSI":   "Population Stability Index. < 0.10 stable · 0.10–0.25 moderate drift · > 0.25 significant drift.",
        "KS":    "KS D statistic: max absolute difference between reference and current CDFs (0–1). Unaffected by sample size. < 0.10 small · 0.10–0.20 moderate · > 0.20 large.",
        "SHIFT": "Standardized mean shift = (cur μ − ref μ) / ref σ. How many reference std devs the mean moved. |·| < 0.5 stable · 0.5–1.0 moderate · > 1.0 significant.",
        "DSTD":  "Relative change in spread = (cur σ − ref σ) / ref σ × 100%. Positive = more dispersed, negative = more concentrated.",
    }

    def _th(label, col, tip=""):
        ta = f' data-tip="{tip}"' if tip else ""
        return f'<th onclick="sortTable(\'stats-table\',{col})"><span{ta}>{label}</span> ↕</th>'

    return f"""
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#60a5fa"></span>Reference (training)</span>
      <span class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span>Current batch</span>
      <span class="legend-item"><span class="legend-dot" style="background:#f87171"></span>High drift</span>
      <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Moderate drift</span>
      <span style="font-size:11px;color:#475569;margin-left:auto">Hover column headers for definitions · Click to sort</span>
    </div>
    <div class="table-wrap">
      <table class="drift-table" id="stats-table">
        <thead><tr>
          {_th('Feature', 0)}
          {_th('PSI', 1, TIP['PSI'])}
          {_th('KS Stat', 2, TIP['KS'])}
          {_th('Ref μ', 3, 'Reference mean (training dataset)')}
          {_th('Cur μ', 4, 'Current mean (this batch)')}
          {_th('Δμ/σ', 5, TIP['SHIFT'])}
          {_th('Ref σ', 6, 'Reference standard deviation')}
          {_th('Cur σ', 7, 'Current standard deviation')}
          {_th('Δσ%', 8, TIP['DSTD'])}
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _perf_section(latest_status: dict | None) -> str:
    if not latest_status or not latest_status.get("performance"):
        return ""
    p = latest_status["performance"]
    def _sc(v): return _STATUS_COLOR.get("GREEN" if v >= 0.75 else "AMBER" if v >= 0.60 else "RED")
    return f"""
    <div class="section">
      <div class="section-title">Performance vs Ground Truth — Latest Batch</div>
      <div class="section-desc">
        Computed by joining predictions with the actual churn labels.
        The most direct signal of live model quality.
        Only available when a <code>*_Ground_Truth.parquet</code> file exists for this batch.
      </div>
      <div class="perf-row">
        <div class="perf-card">
          <div class="perf-val" style="color:{_sc(p.get('auc',0))}">{p.get('auc',0):.3f}</div>
          <div class="perf-label">AUC</div>
        </div>
        <div class="perf-card">
          <div class="perf-val">{p.get('precision',0):.3f}</div>
          <div class="perf-label">Precision</div>
        </div>
        <div class="perf-card">
          <div class="perf-val">{p.get('recall',0):.3f}</div>
          <div class="perf-label">Recall</div>
        </div>
        <div class="perf-card">
          <div class="perf-val">{p.get('label_coverage',0):.1%}</div>
          <div class="perf-label">Label Coverage</div>
        </div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — assembler
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    champion:       dict | None,
    training:       dict | None,
    inference_runs: list[dict],
    latest_status:  dict | None,
    lookback_days:  int,
) -> str:
    status     = (latest_status or {}).get("status", "UNKNOWN")
    reasons    = "; ".join((latest_status or {}).get("status_reasons", []))
    latest_run = inference_runs[-1] if inference_runs else None
    batch_id   = latest_run["batch_id"] if latest_run else "—"
    batch_ts   = latest_run["timestamp"].strftime("%Y-%m-%d %H:%M UTC") if latest_run else "—"
    model_ver  = f"v{champion['version']}" if champion else "—"
    gen_ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top_features = (latest_status or {}).get("feature_drift", {}).get("top_drifted_features", [])
    fd           = (latest_status or {}).get("feature_drift", {})

    fd_desc = (
        f"Comparing feature distributions between this batch and the training reference. "
        f"Mean PSI across all features: <strong>{fd.get('mean_psi',0):.4f}</strong> · "
        f"Features above AMBER (PSI ≥ 0.10): <strong>{fd.get('share_of_drifted_features',0):.1%}</strong>. "
        f"Showing top 20 by PSI. Hover column headers for metric definitions."
    ) if top_features else "Run monitor to compute feature drift."

    mlflow_links = (
        f'<a href="{_MLFLOW_UI}" target="_blank">MLflow UI</a>'
        f'<a href="{_MLFLOW_UI}/#/models/{REGISTERED_MODEL_NAME}" target="_blank">Model Registry</a>'
        f'<a href="{_MLFLOW_UI}/#/experiments" target="_blank">Experiments</a>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Champion Health — {REGISTERED_MODEL_NAME}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
<div class="header">
  <span class="header-title">&#9679; Champion Model Health — {REGISTERED_MODEL_NAME}</span>
  <div class="header-links">{mlflow_links}</div>
  <span class="header-meta">Generated {gen_ts}</span>
</div>
<div class="status-banner banner-{status}">
  <div class="status-dot dot-{status}"></div>
  <div class="status-main">
    <div class="status-badge badge-{status}">{status}</div>
    <div class="status-model">{model_ver} &nbsp;·&nbsp; Batch {batch_id}</div>
    <div class="status-reasons">{reasons}</div>
  </div>
  <div class="status-meta">
    Last scored: {batch_ts}<br>
    Lookback: {lookback_days} days ({len(inference_runs)} batch runs)
  </div>
</div>
<div class="content">
  <div class="cards-grid">
    {_champion_card(champion, training)}
    {_data_quality_card(latest_status)}
    {_latest_batch_card(latest_run, batch_id)}
  </div>
  <div class="section">
    <div class="section-title">Batch Score Trend — Last {lookback_days} Days</div>
    <div class="section-desc">
      One bar per batch, colored by drift status (green = stable · amber = watch · red = investigate).
      When multiple inference runs exist for the same batch, the most recent one is shown.
      A rising mean score or high-risk rate may indicate genuine churn increase or feature drift.
    </div>
    {_chart_html(chart_batch_trend(inference_runs), f"No inference runs in the last {lookback_days} days.")}
  </div>
  {_output_distribution_section(latest_status, latest_run)}
  <div class="section">
    <div class="section-title">Feature Distribution Drift — Top 20 by PSI</div>
    <div class="section-desc">{fd_desc}</div>
    <div class="chart-grid-2">
      <div>
        <div class="chart-label">PSI · bar color = KS D stat (red ≥ 0.20 · amber ≥ 0.10 · green &lt; 0.10)</div>
        {_chart_html(chart_psi_bars(top_features), "Run monitor to compute feature drift.")}
      </div>
      <div>
        <div class="chart-label">Standardized mean shift (Δμ / σ_ref) · sorted by |shift| · dashed lines at ±0.5σ and ±1.0σ</div>
        {_chart_html(chart_mean_shift(top_features), "Run monitor to compute feature drift.")}
      </div>
    </div>
    {_descriptive_stats_table(top_features)}
  </div>
  <div class="section">
    <div class="section-title">SHAP Feature Importance — Latest Batch</div>
    <div class="section-desc">
      Top 20 features by mean absolute SHAP value across up to 500 sampled merchants.
      SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to individual predictions.
      Higher value = more influential on the model's output for this batch.
    </div>
    {_chart_html(chart_shap(latest_run) if latest_run else go.Figure(), "No SHAP data. Run batch_predict.py to generate SHAP values.")}
  </div>
  {_perf_section(latest_status)}
  <div class="footer">{REGISTERED_MODEL_NAME} &nbsp;·&nbsp; {gen_ts}</div>
</div>
<script>{_JS}</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monitor batches and generate the champion health report."
    )
    parser.add_argument("--batch",    type=str, default=None,
                        help="Single batch ID to monitor (default: all).")
    parser.add_argument("--lookback", type=int, default=7,
                        help="Days of inference history to include in the report (default: 7).")
    parser.add_argument("--output",   type=str, default=None,
                        help="HTML output path (default: reports/monitoring/champion_report.html).")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── 1. Monitoring ──────────────────────────────────────────────────────
    print("Loading reference (training) data...")
    reference_features, reference_labels = load_reference_data()

    model_version = load_latest_model_version()
    reference_auc = load_reference_auc(model_version.run_id)
    print(f"Using '{REGISTERED_MODEL_NAME}' v{model_version.version} (train AUC={reference_auc:.4f})")

    print("Scoring reference data for prediction-drift baseline...")
    model = mlflow.xgboost.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{model_version.version}"
    )
    reference_scores = pd.Series(
        model.predict_proba(reference_features[FEATURE_COLUMNS].astype("float64"))[:, 1],
        index=reference_features.index,
    )

    batch_ids = (
        [args.batch] if args.batch
        else sorted(p.stem.replace("_features", "") for p in BATCH_FEATURES_DIR.glob("*.parquet"))
    )
    if not batch_ids:
        raise RuntimeError(f"No batch feature files found in {BATCH_FEATURES_DIR}")

    for batch_id in batch_ids:
        monitor_batch(batch_id, reference_features, reference_scores,
                      reference_labels, reference_auc, model_version.version)

    # ── 2. Report ──────────────────────────────────────────────────────────
    print("\nBuilding HTML report...")
    champion = get_champion()
    if not champion:
        print("WARNING: No @champion alias set — champion card will be empty.")

    training       = get_training_run(champion["run_id"]) if champion else None
    inference_runs = get_inference_runs(args.lookback)
    print(f"  Found {len(inference_runs)} inference run(s) in the last {args.lookback} days.")

    # Enrich inference runs with drift status from local status JSONs
    for run in inference_runs:
        status_data = load_monitoring_status(run["batch_id"])
        run["drift_status"] = (status_data or {}).get("status", "UNKNOWN")

    latest_batch_id = inference_runs[-1]["batch_id"] if inference_runs else None
    latest_status   = load_monitoring_status(latest_batch_id) if latest_batch_id else None

    html = build_report(champion, training, inference_runs, latest_status, args.lookback)

    out = Path(args.output) if args.output else MONITORING_DIR / "champion_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Report saved → {out}")


if __name__ == "__main__":
    main()
