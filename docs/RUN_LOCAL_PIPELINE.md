# Run the measured local pipeline

This implementation follows E0 → E1 → E2 in `SCORE_IMPROVEMENT_PLAN.md`. It reproduces the research baseline and adds probability-gated use of the existing fraud rules. The supplied `../s1_v1.csv` scored 0.5034 publicly, but its generating run is still unidentified. It is preserved as a comparison artifact, not used as training labels.

## Install and run

Run from the repository root with Python 3.12 or later. `pyproject.toml` pins the direct dependencies used in the research environment; every run records its actual versions and source hashes.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run validate \
  --data-dir ../upi-fraud-rings --output-dir models/measured_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run refit \
  --data-dir ../upi-fraud-rings --output-dir models/measured_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run predict \
  --data-dir ../upi-fraud-rings --output-dir models/measured_v1 \
  --reference-submission ../s1_v1.csv --submission submissions/measured_v1.csv
.venv/bin/python -m upi_fraud.run check-submission \
  --data-dir ../upi-fraud-rings --submission submissions/measured_v1.csv
```

All commands are local. They contain no submission-upload or Git-write operations. An existing submission is never overwritten. Run directories and CSV outputs are under the already ignored `models/`, `data/processed/` and `submissions/` paths. Raw files and the existing reports are read only.

Use `--resume` on `validate` or `refit` to verify an already completed identical run. The command rejects changed data, code, configuration, dependency versions or altered artifacts. For another configuration use a new output directory and `--config path/to/config.json`. Interrupted runs can reuse verified feature caches; model fits without a completed manifest are recomputed. The default cache directory is `data/processed/measured_v1`; use `--cache-dir` to change it.

## What is fitted and when

The only model uses the measured 61 features: 23 numeric row features, 33 relational features and five categorical features. Category dictionaries are persisted alongside each model. IDs remain metadata and join keys, not model predictors. Training labels, app version and IP risk are excluded from feature construction. No historical target rates, pseudo-labels, label cleaning, event propagation or failed interaction packs are enabled.

| Stage | Model-training labels | Label-free transaction context | Decisions |
|---|---|---|---|
| August | Before 1 August | Before 1 September | Fit shared/class weights and rule thresholds on August |
| September | Before 1 September | Before 1 October | Apply August decisions unchanged |
| Final | All supplied train rows, including October tails | All supplied train and test rows | Apply the same saved August decisions |

The LightGBM settings, 20% legitimate-row sampling with weight 5, and fraud-row weight 1 are fixed in `configs/measured.json`. Features are computed before sampling, and validation includes every labelled row in its month. Tree count stays at the measured 200. The final model is not tuned against test predictions or the public score.

Within a context boundary, windows may use earlier and later payments because this competition allows batch connections. Counts include the query transaction and have inclusive endpoints. Flow links exclude transactions at the exact query timestamp. Original loader ordinal resolves equal-time candidate ties and remains attached when frames are shuffled. Feature caches are one-to-one keyed by `txn_id` and checked against the data/code hashes.

## Artifacts for review

`models/measured_v1/` contains:

- `validation_manifest.json`: raw-file hashes, implementation hashes, Git revision, dependency versions, training counts/cutoffs, category mappings, cached-feature hashes, baseline probability reproduction results and artifact checksums.
- `august/` and `september/`: model text, schema, named probability columns, truth/predictions keyed by transaction ID, all 12×12 confusion matrices and per-class metrics.
- `decisions.json`: August-fitted shared/class weights, probability gates, class order and the hash of the tuning transaction IDs.
- `comparison.md` and `validation_metrics.json`: model/rules/combined comparisons, per-class changes, September halves and paired day-resampling diagnostics.
- `final_model.txt`, `final_schema.json`, `refit_manifest.json`: the final fit and its connection to validation.
- `test_probabilities.parquet` and its schema: all test probabilities and model/rule/combined predictions in sample order.
- `prediction_manifest.json`: submission/model/probability hashes, prediction counts by month, seen/unseen entity diagnostics, and disagreement with `s1_v1.csv`.
- `differences_from_s1_v1.csv`: keyed prediction disagreements, for reproducibility and aggregate comparison. These are predictions, not hand-assigned labels.

The requested candidate is `submissions/measured_v1.csv`. The prediction manifest records its exact hash. To reproduce it, retain the model/schema/decision files, install the recorded environment and rerun `predict` with identical raw data and a new submission path. Feature caches can be rebuilt; models are portable LightGBM text, without pickled Python objects.

## Correctness checks

`tests/test_pipeline.py` verifies closed window boundaries, no-neighbour and empty-event cases, same-time exclusion, merchant/account separation, shuffled keyed joins, unknown category encoding, validation-label perturbation, chronological training filters, probability class order, model save/load equality, explicit 12-class macro-F1 against scikit-learn, and rejection of malformed submissions.

Historical-label donor checks are inapplicable because that feature family is absent. Validation-label changes leave both the feature matrix and fitted earlier-month model predictions unchanged in the isolation test. On the full dataset, validation also compares its keyed probability arrays against the prior bounded research artifacts when those local artifacts exist.

September is a development comparison: the rules and previous experiments have already used it. Local noisy-label gains do not establish a gain over 0.5034 on the clean, unknown October targets. Review this limitation along with row alignment, label order, source boundaries and the per-class regression table before choosing a leaderboard upload.
