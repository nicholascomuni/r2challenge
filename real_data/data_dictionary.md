# Data Dictionary (Real Data)

## Overview
- This package contains real, anonymized data for the MLE assessment.
- Feature columns are anonymized as `C1`..`C180`.
- Each row in train data represents one merchant.
- Each feature column (`C1`..`C180`) is one point in the merchant transaction-count time series.
- Time granularity is 12 hours (2 observations per day).
- Feature order is chronological (`C1` oldest, `C180` most recent).

## Train files

### `train/features_train.parquet`
- Grain: one row per training merchant.
- Shape: `87232 rows x 180 columns`.
- Columns:
  - `C1`..`C180` (`float64`): anonymized time-indexed transaction-count features.
- Notes:
  - No explicit ID column is present in this file.
  - Feature list example: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, ... , C180

### `train/target_train.parquet`
- Grain: one row per training merchant, aligned by row order with `features_train.parquet`.
- Shape: `87232 rows x 1 columns`.
- Columns:
  - `LABEL` (`float64`): binary target label (`1 = churn`, `0 = non-churn`).

## Batch feature files

Files under `batches/features/`:
- `0923_1737_features.parquet`: `30762 rows x 181 columns`
  - `merchant_id` (`int64`): merchant identifier
  - `C1`..`C180`: anonymized model features
- `1023_2038_features.parquet`: `30817 rows x 181 columns`
  - `merchant_id` (`int64`): merchant identifier
  - `C1`..`C180`: anonymized model features
- `1127_1528_features.parquet`: `28519 rows x 181 columns`
  - `merchant_id` (`int64`): merchant identifier
  - `C1`..`C180`: anonymized model features

## Batch label files

Files under `batches/labels/`:
- `0923_1737_Ground_Truth.parquet`: `29828 rows x 2 columns`
  - `MERCHANT_ID` (`int64`): merchant identifier
  - `LABEL` (`int64`): observed target (`1 = churn`, `0 = non-churn`)
- `1023_2038_Ground_Truth.parquet`: `29836 rows x 2 columns`
  - `MERCHANT_ID` (`int64`): merchant identifier
  - `LABEL` (`int64`): observed target (`1 = churn`, `0 = non-churn`)
- `1127_1528_Ground_Truth.parquet`: `27854 rows x 2 columns`
  - `MERCHANT_ID` (`int64`): merchant identifier
  - `LABEL` (`int64`): observed target (`1 = churn`, `0 = non-churn`)

## Join keys and conventions
- To join batch predictions with labels, use `merchant_id` (features) == `MERCHANT_ID` (labels).
- `LABEL` is binary for model evaluation/monitoring (`1 = churn`, `0 = non-churn`).
- Feature names are intentionally anonymized; semantic mapping is not included in this package.
