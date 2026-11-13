# Local candidate: measured_v1

Implemented the smallest confirmed path from the score-improvement plan: the 61-feature LightGBM research baseline, August-fitted class decisions, and probability-gated use of the original fraud rules. The generating run of the supplied public-0.5034 CSV remains unidentified. No target-history features, extra interaction packs, label propagation or label changes are included.

## Deliverables

Paths below are relative to the repository root:

| Artifact | Path |
|---|---|
| Local candidate, ready for user review | `submissions/measured_v1.csv` |
| Complete validation provenance | `models/measured_v1/validation_manifest.json` |
| Final fit provenance | `models/measured_v1/refit_manifest.json` |
| Candidate, model and prediction checksums | `models/measured_v1/prediction_manifest.json` |
| Per-class comparison and date splits | `models/measured_v1/comparison.md` |
| All confusion matrices and metrics | `models/measured_v1/validation_metrics.json` |
| August/September keyed probabilities | `models/measured_v1/{august,september}/probabilities.parquet` |
| Test keyed probabilities | `models/measured_v1/test_probabilities.parquet` |
| Research prediction reproduction check | `models/measured_v1/reproduction_check.json` |
| Run instructions and reviewer handoff | `docs/RUN_LOCAL_PIPELINE.md` |

The candidate has **260,582 unique IDs**, sample order, valid labels and header, and is **4,459,129 bytes**, below the 15 MB limit. Its SHA-256 is:

```text
65dead93dac984edd59e7eacbc17a5846023f1350cc4d08095205e4db52a47dd
```

It differs from `s1_v1.csv` on **524 transactions**, with 99.7989% overall agreement and 86.9392% agreement among the 4,012 rows called fraud by either file. These are prediction comparisons; they do not establish which file is more accurate. The original CSV's hash remains `1d5b1b383132b490fdde0751da0aaa08827c7503d971a9c1acda7791174951e7`.

## Validation

| Variant | August macro-F1 | September macro-F1 |
|---|---:|---:|
| Model, raw argmax | 0.415025 | 0.416366 |
| Model, shared August multiplier | 0.421078 | 0.421432 |
| Model, class-specific August weights | 0.424725 | 0.423480 |
| Rules only, bounded transaction context | 0.379674 | 0.390984 |
| Model/rule combination | **0.442360** | **0.448954** |

The September gain over the same model's class-specific decision is **+0.025474 macro-F1**. Its exploratory paired-day resampling interval is **[0.021060, 0.029992]**. The earlier research also checked a second random seed; this implementation reproduces its seed-137 configuration without a new parameter search against September.

| Class | Model F1 | Combined F1 | Delta |
|---|---:|---:|---:|
| legit | 0.986503 | 0.986508 | +0.000005 |
| mule_account | 0.361809 | 0.371848 | +0.010039 |
| velocity_bot | 0.470810 | 0.482611 | +0.011801 |
| layering_chain | 0.497590 | 0.512728 | +0.015138 |
| sim_swap | 0.282873 | 0.339811 | +0.056938 |
| account_takeover | 0.275362 | 0.335329 | +0.059967 |
| phishing_collect | 0.414857 | 0.404516 | -0.010341 |
| qr_swap | 0.456235 | 0.456805 | +0.000570 |
| refund_abuse | 0.423975 | 0.465686 | +0.041712 |
| fake_cashback | 0.296296 | 0.351792 | +0.055495 |
| loan_app_extortion | 0.300683 | 0.357871 | +0.057188 |
| merchant_collusion | 0.314766 | 0.321942 | +0.007177 |

Both calendar halves improve: **1–15 September 0.419461 → 0.447182**, and **16–30 September 0.425684 → 0.448988**. These masks use the actual calendar day. The older research helper used a numeric day offset that split after 14 September despite naming the segments 1–15/16–30; the corrected calendar split remains positive.

The reproduced macro-F1 values match the earlier bounded-context research. **Raw and class-weighted predicted labels match the research artifacts exactly for every August and September transaction.** Raw probability equality is not claimed: three August transactions show small probability differences (maximum 0.001455), and September's maximum difference is approximately 2.9×10⁻²¹. The separately saved keyed reproduction check records zero prediction differences in both folds.

## Training and boundaries

- August model: 120,684 sampled fit rows, labels strictly before 1 August; label-free context strictly before 1 September. Validation includes all 526,852 August rows.
- September model: 241,344 sampled fit rows, labels strictly before 1 September; label-free context strictly before 1 October. Validation includes all 509,996 September rows.
- Decisions are fitted on August and reused unchanged for September and test. Original labels are retained.
- Final model: 358,263 sampled fit rows from all 1,564,060 supplied training rows, including the 147 October tails. The complete 1,824,642-row train/test context supplies label-free connections. All 46 November test rows receive predictions.
- Sampling retains all fraud and a fixed 20% of legitimate training rows; legitimate sample weight is 5, fraud weight is 1. Features precede sampling. Tree count is the measured 200 throughout.

Data/config/source hashes, category schemas, class order, row counts, environment, model hashes and probability hashes are recorded in the manifests. Keyed feature joins and sample-ordered submission assembly avoid reliance on CSV row positions.

## Verification and review

**10 targeted tests passed** using `python -m unittest discover -s tests -v`. They exercise validation-label perturbation, chronological cutoffs (including October tails), shuffled features/predictions, equal-time and closed-window handling, empty events, account/merchant namespace separation, unknown categories, saved-model prediction equality, class ordering, metric agreement with scikit-learn, and malformed submission rejection.

The `check-submission` command passed on the full candidate. `validate --resume` and `refit --resume` verified the completed runs and their checksums. Raw-data hashes remain equal to those recorded before implementation, and `s1_v1.csv` remains unchanged. No existing EDA/research report was modified.

For review, prioritise `src/upi_fraud/features.py` (window and label-free contracts), `model.py` (training cutoffs and sampling), `decisions.py` (August-only fitting), and `run.py` (artifact/provenance and output order). Follow `docs/RUN_LOCAL_PIPELINE.md` to reproduce the run. Models, caches and CSVs remain in ignored local directories; implementation, tests, configuration and this report are new local files.

The candidate has no measured public/private score. September has already informed the EDA rules and prior experiments, and its labels are noisy. These validation results support the implementation's measured local gain; they do not demonstrate a gain over the public 0.5034 submission. No leaderboard upload, commit or push was performed.
