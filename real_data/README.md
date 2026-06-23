# Real Data Package

This folder contains the real anonymized dataset used for the MLE assessment.

## Folder structure
- `train/features_train.parquet`: training feature matrix (`87232 x 180`)
- `train/target_train.parquet`: training labels (`87232 x 1`)
- `train/metadata.json`: schema metadata for train files
- `batches/features/*.parquet`: batch feature files with `merchant_id` + `C1..C180`
- `batches/labels/*.parquet`: batch ground-truth labels with `MERCHANT_ID` + `LABEL`
- `dataset_summary.csv`: row/column summary and label coverage by file
- `data_dictionary.md`: column-level definitions and join conventions

## How to use
1. Train using `train/features_train.parquet` and `train/target_train.parquet`.
2. Score each file in `batches/features/`.
3. Join predictions with corresponding files in `batches/labels/` using merchant ID to evaluate batch performance.

## Important notes
- Each train row is one merchant.
- `C1..C180` are transaction-count time-series points at 12-hour frequency (`C1` oldest, `C180` most recent).
- Feature columns are anonymized (`C1..C180`).
- Batch files include merchant identifiers; train features do not include explicit IDs.
- Labels are provided per batch and may have lower row count than feature files (partial coverage).
- Label convention: `LABEL=1` means churn, `LABEL=0` means non-churn.
