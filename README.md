# UPI Fraud Rings: ML Gradient Wars 2026

48-hour live round, 13 Nov 10:00 → 15 Nov 10:00 IST. Classify each October UPI transaction into 1 of 12 labels. Scored by macro-F1.

## Layout

```
upi-fraud-ml/
├── data/
│   ├── raw/          original CSVs from the organisers (read-only; git-ignored)
│   ├── interim/      parquet copies of raw, made by scripts/eda/00_load_and_overview.py
│   ├── processed/    engineered features (relational_v0.parquet)
│   └── archive/      the original zip
├── docs/
│   ├── PROBLEM_STATEMENT.md                 official task description
│   ├── IMPLEMENTATION_PLAN.md               ← modelling plan: steps, validation, Colab runbook, research
│   ├── hackathon/research.md                first notes on the event
│   ├── hackathon/analysis.md                in-depth analysis: rules, strategy, legitimacy, similar contests
│   └── past_runs_amazon_ml_postmortem.md    lessons from the Amazon ML Challenge run
├── notebooks/00_starter_notebook.ipynb      organisers' baseline (DATA path → ../data/raw)
├── reports/eda/
│   ├── EDA_REPORT.md                        ← main findings
│   ├── figures/                             charts referenced by the report
│   └── tables/                              raw script outputs (markdown / txt)
├── scripts/eda/                             EDA scripts, run in order 00 → 07
├── src/                                     (pipeline code goes here)
├── models/                                  (git-ignored)
└── submissions/                             upload files (git-ignored); rules_v0.csv = rules-only baseline
```

## Reproduce the EDA

```bash
python -I scripts/eda/00_load_and_overview.py .     # csv -> parquet + structure (~25 s)
python -I scripts/eda/01_time_probe.py .
python -I scripts/eda/02_labels_and_row_profiles.py .
python -I scripts/eda/03_drift.py .                  # includes adversarial validation (~1.5 min)
python -I scripts/eda/04_relational.py .             # builds data/processed/relational_v0.parquet (~3 min)
python -I scripts/eda/05_signatures.py .             # needs 04
python -I scripts/eda/06_offpattern.py .             # needs 04
python -I scripts/eda/07_baselines.py .              # rules + LightGBM ablation, writes submissions/rules_v0.csv
```

Requires Python 3.11 with pandas, numpy, pyarrow, scikit-learn, lightgbm, scipy and matplotlib.

`kaggle.json` (credentials) is git-ignored. Never commit it.
