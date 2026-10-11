# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle competition project: score each transaction as fraudulent or not, based on deviation from a customer's past behavior. Evaluation metric is ROC-AUC on a hidden test set. See `notes.md` for EDA findings (highly imbalanced data — 1.64% positive rate in train; the `timestamp` column is a per-user sequence number, not a real timestamp).

## Commands

```bash
pip install -r requirements.txt   # install dependencies (no lockfile/pyproject; requirements.txt is the source of truth)
python -m src.features.make_features  # rebuild combined_new_features.csv from the active FEATURES
python src/models/train.py        # run baseline training (logistic regression), logs to MLflow
mlflow ui                         # view experiment runs (reads ./mlruns, set via configs/params.yaml tracking_uri)
```

There is no test suite, linter, or notebook-execution command configured in this repo.

Run scripts as modules from the repo root (`python -m src.data.load_data` or `python src/models/train.py`) since `src/*` uses absolute imports like `from src.data.load_data import load_dataset`.

## Data

- `data/raw/train.csv`, `data/raw/test.csv` — raw Kaggle files (columns: `transaction_id, user_id, timestamp, amount, merchant_category, country, device_id, channel, hours_since_prev_txn, label`).
- `data/raw/sample_submission.csv` — Kaggle submission format.
- `data/processed/combined.csv` — train+test merged (see notes.md: all train users appear in test, and the two sets overlap in the `timestamp` ordering, so features are engineered on the combined frame before splitting back out).

## Architecture

- `src/data/load_data.py` — `load_dataset()` reads a CSV (defaults to `data/raw/train.csv`, resolved relative to the file's own location via `Path(__file__).resolve().parents[2]`, not cwd). Also has missing-data reporting helpers.
- `src/data/preprocess.py` — `build_preprocess_pipeline()` returns a sklearn `Pipeline` (median imputer + `StandardScaler`) for numeric features only. `prepare_features()` splits a DataFrame into `X`/`y`, currently keeping only numeric columns — categorical fields (`merchant_category`, `country`, `device_id`, `channel`) are dropped and not yet encoded.
- `src/features/build_features.py` — all feature engineering lives in this one file (amount/gap deviation vs. the user's own prior history, categorical novelty, device sharing). Each feature is a plain self-contained function `f(df) -> Series` that derives everything it needs from the raw columns, so features never depend on one another. The module-level `FEATURES` list holds the ones that are switched on; `build_feature_frame(df, features=)` sorts by `["user_id", "timestamp"]` (a per-user sequence number, see notes.md), computes each function into a column named after it, and restores the input row order. `feature_columns()` lists those column names. 27 feature functions are defined; `FEATURES` currently holds 5 (the four `*_changed_from_prev` flags and `is_short_gap_since_prev`) — add more to the list, or pass `features=[...]` for a one-off run. Per-user baselines use strictly prior transactions, so the first txn of a user is NaN for those (imputed downstream). Run `python -m src.features.build_features` for summary stats of whatever is active.
- `src/features/make_features.py` — writes the combined frame plus the active feature columns to `combined_new_features.csv`, next to whichever `combined.csv` it finds. Drops stale feature columns first, so rerunning after editing `FEATURES` always yields exactly what is switched on. Run it as `python -m src.features.make_features`.
- `src/models/train.py` — end-to-end baseline: loads raw data, splits train/test (stratified, `test_size=0.2`, `random_state=42`), fits `preprocess pipeline -> LogisticRegression`, and logs params/metrics/model to MLflow under the `fraud_detection_baseline` experiment. Note: it does **not** use `src/features` at all — it trains on the raw numeric columns that `prepare_features()` keeps.
- `configs/params.yaml` — central config (experiment name, MLflow tracking URI, random_state, test_size, model type). Note: `train.py` does not currently read this file; splitting/model params are hardcoded inline and duplicate what's in `params.yaml`.
- `notebooks/exploration.ipynb` — EDA notebook.
