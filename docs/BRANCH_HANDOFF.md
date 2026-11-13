# Experimental branch handoff

Branch: `thanmay/score-improvement`. This branch packages the local implementation and research for teammate review. It is based on main at `f2a9e923ff42898de4333ddad5d144a9b9a75768`; the teammate's generating code for the 0.5034 submission was not available in this checkout. Reconcile this work with that code before adopting changes.

## Actual outcome

| Run | September macro-F1 | Reported public macro-F1 |
|---|---:|---:|
| Team's original `s1_v1.csv` | Generating run not reproduced | **0.5034** |
| Reproduced research model | 0.423480 | Not uploaded |
| Model plus probability-gated rules | **0.448954** | **0.4719** |

The public result is a regression of **0.0315** against the original team submission. A local gain over the research baseline did not establish a gain over the scored team pipeline. Neither the local score nor agreement with the old CSV reveals which October predictions are correct. Preserve the team's stronger submission and use it as the baseline once its generating code is available.

The candidate has 260,582 valid sample-ordered predictions and differs from `s1_v1.csv` on 524 rows. Its SHA-256 is `65dead93dac984edd59e7eacbc17a5846023f1350cc4d08095205e4db52a47dd`; the original scored file's SHA-256 is `1d5b1b383132b490fdde0751da0aaa08827c7503d971a9c1acda7791174951e7`. Both CSVs stay local under the existing ignore policy. No October ground truth is available to diagnose the public regression directly.

## Read in this order

1. [RUN_SUMMARY.json](../reports/implementation/RUN_SUMMARY.json): configuration, aggregate metrics, data/source hashes and actual public-score outcome. All paths in this summary are repository-relative.
2. [MEASURED_V1.md](../reports/implementation/MEASURED_V1.md): historical implementation report, per-class changes and feature-boundary details. It was written before the candidate's public result; the outcome above supersedes its statement that the score is unknown.
3. [RUN_LOCAL_PIPELINE.md](RUN_LOCAL_PIPELINE.md): setup, commands, artifact contracts and targeted checks.
4. [SCORE_IMPROVEMENT_PLAN.md](SCORE_IMPROVEMENT_PLAN.md): implementation specification. Its recommendations predate the public regression and should not be treated as evidence that this candidate is better than 0.5034.
5. [CANDIDATE_EVIDENCE.md](../reports/research/CANDIDATE_EVIDENCE.md) and [primary-source research](hackathon/method_research_2026-11-13.md): successful and failed local ablations, Kaggle author writeups, and limitations. The corresponding numeric evidence is in [RESULTS.json](../reports/research/RESULTS.json).

## Components worth inspecting

- `src/upi_fraud/data.py`: unique-ID validation, keyed alignment and sample-order submission checks.
- `features.py`: existing 23 numeric row features, 33 relational features and five categorical features, with explicit time/group boundaries. Labels, IP risk and app version are absent from the feature layer.
- `model.py`: the measured 200-tree LightGBM and explicit training cutoffs. All fraud is retained; 20% of legitimate training rows is retained with inverse-sampling weight 5. The final model used 358,263 sampled rows from the supplied 1,564,060-row training set.
- `decisions.py`: shared/class-specific probability multipliers and probability-gated original rules. August selects decisions; September and test receive frozen decisions. The public result means this rule/model combination is unproven as an improvement to the team's actual model.
- `artifacts.py` and `run.py`: content-hashed feature caches, twelve named probability columns, explicit class/category schemas and validation/refit/prediction manifests.
- `tests/test_pipeline.py`: ten targeted tests for temporal isolation, window ties/boundaries, namespaces, keyed shuffles, label perturbation, category codes, model save/load, macro-F1 and invalid submissions.

No historical-target features, label cleaning, test pseudo-labels or event propagation are enabled in the delivered pipeline. Earlier experiments adding a broad historical-label pack or the tested interaction pack reduced validation scores, so those packs remain research scripts rather than production features.

## Known review findings

The review found no validation-label leakage, incorrect month boundary, wrong rule priority, category mapping mismatch or submission alignment error in the measured configuration. All ten tests and the complete candidate submission check passed. Two actionable issues remain **unfixed**:

1. **Saved-model portability:** `run.py:159–165` checks the current Git revision and all validation artifact paths before prediction. The manifests record absolute local paths. Committing these unchanged sources or moving the old artifact bundle prevents reuse through the documented command. Keep Git revision as provenance, use source/data/config hashes for compatibility, and use relative artifact paths with only inference dependencies required for prediction. The original pre-commit local manifests will not resume unchanged after this branch is committed. A fresh run from source in a new output directory remains supported.
2. **Decision-search ties:** `decisions.py:56` scores coordinate candidates using strict `>`, while final `argmax` selects the lowest class index on ties. These can choose different labels for tied probabilities. Make the search match inference and add a tie regression test. No tied maxima were observed in the original August shared/joint outputs, so this is not an established explanation for the public score drop.

The historical per-class comparison also shows a phishing regression of about 0.01034 F1. Keep that tradeoff visible when considering the rules.

## Reproduce from source

The extracted competition data is not committed. Supply the directory containing the five original CSVs through `--data-dir`; the example uses the original local layout. Python 3.12+ is required by the pinned dependencies.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run validate \
  --data-dir ../upi-fraud-rings --output-dir models/teammate_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run refit \
  --data-dir ../upi-fraud-rings --output-dir models/teammate_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m upi_fraud.run predict \
  --data-dir ../upi-fraud-rings --output-dir models/teammate_review \
  --submission submissions/teammate_review.csv
```

These commands create a fresh run for the current checkout, avoiding the old manifest's Git/path coupling. They do not upload predictions. Optional `--reference-submission ../s1_v1.csv` adds prediction-agreement diagnostics when the original CSV is available locally. An existing submission path is never overwritten.

## Research helper status

`scripts/research/` preserves the ablation and audit code. Its original feature caches/probability arrays are ignored; earlier EDA scripts can rebuild them following the evidence report. `reproduce_s1_baseline.py` attempts to identify the original 400-tree EDA model by prediction agreement with `s1_v1.csv`; it is **not** an accuracy experiment and never trains on that CSV's predictions. Its last run was interrupted at the user's request while waiting for teammate code, so no successful source identification is claimed.

The original EDA files and reports remain intact. Use source/data hashes and complete run configurations when comparing this branch with the teammate's scoring pipeline; a passing submission-format check and local score improvement alone are insufficient acceptance criteria for another leaderboard upload.
