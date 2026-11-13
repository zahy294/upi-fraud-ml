# UPI Fraud Rings

Reproducible pipeline for the 12-class ML Gradient Wars 2026 task. Current team public score is **0.5034** by team report; the current visible leader was **0.5487** when implementation began. The working target is 0.70 macro-F1.

## What is implemented

- Raw-data integrity audit with hashes, schemas, joins, time tails, labels and submission-ID checks.
- Row, metadata, interaction, multi-window relational and strictly prior-month label-history features.
- July-August fitting, September 1-23 calibration and September 24-30 assessment.
- LightGBM CPU, XGBoost GPU and CatBoost GPU configurations.
- Class-offset tuning for macro-F1, with all 12 per-class metrics and confusion matrices.
- Final refit, aligned test probabilities, strict id,label submission validation and persistent artifacts.
- Unit tests for event windows, temporal history isolation, decisions and submission writing.
- Thin Colab A100 driver in notebooks/colab_train.ipynb.

The canonical strategy is [docs/TARGET_070_IMPLEMENTATION_PLAN.md](docs/TARGET_070_IMPLEMENTATION_PLAN.md). The exact coder-to-A100 handoff is [docs/CODING_HANDOFF.md](docs/CODING_HANDOFF.md).

## Setup

Use Python 3.11+ and place the five organizer CSVs in data/raw/: train.csv, test.csv, accounts.csv, merchants.csv, and sample_submission.csv.

Install dependencies with:

    python -m pip install -r requirements.txt

Raw data, caches, run artifacts, credentials, models and submission CSVs are excluded from Git.

## Local commands

    python run.py audit --config configs/baseline_lgb.yaml
    python -m unittest discover -s tests -v
    python run.py smoke --config configs/smoke.yaml
    python run.py features --config configs/baseline_lgb.yaml
    python run.py train --config configs/baseline_lgb.yaml
    python run.py predict --config configs/baseline_lgb.yaml --run-dir runs/<validated-run>
    python run.py blend --config configs/baseline_lgb.yaml --run-dirs runs/<xgb-run> runs/<cat-run>

Train writes a unique run directory containing the model, aligned validation probabilities, predictions and a manifest with environment, commit, feature columns, split sizes, class weights and all metrics. Predict verifies the feature fingerprint from that run, refits on all supplied labels, saves the fitted model and test probabilities, and writes a validated submission.

For the A100, use configs/xgb_a100.yaml first, then configs/catboost_a100.yaml as a diversity model. Run validation before final fitting. A high leaderboard target is not a measured result until an authorized upload scores it.

## Earlier EDA

The original EDA remains under scripts/eda/ and reports/eda/. The production pipeline does not depend on positional concatenation from those scripts; derived features are joined by txn_id.
