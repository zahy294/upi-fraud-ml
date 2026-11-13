# Candidate evidence: 13 November 2026

## Decision

The best supported next implementation is **a LightGBM on the existing relational features, jointly tuned class decision weights, and probability-gated use of the existing rules**. It improved a later-month validation score after decisions were fitted on an earlier month. A separate run confirmed the gain after removing later-month transaction context.

This does **not** establish an October score above 0.5034. The supplied `s1_v1.csv` contains hard test predictions, not test truth or validation probabilities. No public submission was made during this research. A 0.55+ result is a candidate to test, not a measured or guaranteed result. Evidence does not currently support forecasting 0.60–0.70.

Read [the implementation plan](../../docs/SCORE_IMPROVEMENT_PLAN.md) for the exact algorithm and work sequence. Read [the source research](../../docs/hackathon/method_research_2026-11-13.md) for primary competition writeups and method limitations.

## What was inspected and reproduced

- Local `main` and fetched `origin/main` both point to `f2a9e92`. Working files were not pulled, reset, committed or pushed.
- Read EDA reports, source scripts, the previous implementation plan, supplied problem statement and starter notebook.
- Verified the extracted CSV row counts, schema, transaction-ID integrity and month tails. SHA-256 hashes are recorded in the result manifest.
- Recomputed the original relational features in an isolated ignored directory. The original rules' September macro-F1 reproduced at **0.39220837**, matching the saved 0.3922.
- Ran six initial model ablations, two additional contextual models, a two-model chronological-context confirmation, and a two-model second-seed confirmation: **12 fits total**. Decisions were selected on August and held fixed for September.
- Audited the seed-inclusive purity statistic and ran independent revealed-label topology checks over two months and three random seeds.
- Training used only the provided competition data. External sources informed methods, not training examples.

## Scored submission identification

The user identifies `../s1_v1.csv` (relative to repository root) as the **0.5034 public submission**. Its SHA-256 is `1d5b1b383132b490fdde0751da0aaa08827c7503d971a9c1acda7791174951e7`.

It has the correct `id,label` columns, 260,582 unique rows, exactly the test IDs, and the sample's row order. Its labels are in the official vocabulary. It contains 256,794 legitimate predictions and 3,788 fraud predictions.

Comparison by ID with the exact `07_baselines.py` rules output:

| Check | Result |
|---|---:|
| Different predictions | 1,834 |
| Overall agreement | 99.296% |
| Rows called fraud by either file | 4,667 |
| Agreement restricted to that fraud union | 60.703% |

The high overall agreement is dominated by legitimate transactions. **The scored CSV is not the rules-only output.** The exact generating model/configuration remains unidentified. The user says its code was in main; this revision includes both EDA code and the starter notebook, but the CSV alone cannot identify which training run generated it. Neither the main code nor the CSV supplies that run's held-out probabilities. Preserve the CSV and compare future candidates to it without treating its predictions as truth.

## Experimental protocol

Fixed label order comes from `scripts/eda/common.py`. Run A trains on July timestamps and validates all August rows; run B trains before September and validates all September rows. Labels and validation populations remain unchanged.

LightGBM: 200 trees, 31 leaves, learning rate 0.05, minimum 40 observations per leaf, max bin 127, subsample/column sample 0.8, four threads. Seed 137 initially. All labelled fraud was retained; 20% of legitimate training rows were randomly retained with weight 5, versus fraud weight 1. Counts were 120,684 training rows in A and 241,344 in B. These are cheap, weighted ablations, not identical to the original full-data 400-tree EDA model. No validation sampling was used.

Three decision variants were measured: raw argmax; one shared fraud multiplier chosen on August; eleven jointly searched class weights chosen on August. For the rule combination, all override thresholds were also fitted on August and frozen for September.

The first ablation used the original EDA's label-free train+test context. This includes transactions beyond each simulated prediction month and is not a faithful availability simulation. The confirmation rebuilds A context through August only and B context through September only. Thus the confirmed candidate does not rely on October transactions during September evaluation.

The EDA's existing hand-built rules were developed with knowledge of this dataset, including September. These runs are controlled development comparisons, **not a pristine untouched holdout**. Day-block intervals do not account for all adaptive experiment selection or all dependence between days.

## Initial controlled ablations

All rows below use August-fitted class weights. The contextual pack contains 18 extra features: previous/next gap and ±2h/±24h amount sums and amount/mean ratios for sender, receiver and device.

| Feature/model variant | August macro-F1 | September macro-F1 | Interpretation |
|---|---:|---:|---|
| Existing rules, no fitted model | 0.38071 | 0.39221 | Reproduced reference |
| Existing row + relational features | 0.42175 | 0.41499 | Research model reference |
| Add rule flags and simple interactions | 0.41316 | 0.38719 | Reject this tested pack |
| Add broad delayed historical label rates to that pack | 0.41103 | 0.33350 | Reject this tested pack |
| Add label-free gaps/local amount context to reference | 0.42629 | 0.42089 | Small promising gain; separate bounded confirmation still needed |
| Reference + fitted rule/model combination | 0.44134 | 0.44590 | Strongest cheap decision change |
| Contextual model + fitted rule/model combination | 0.44187 | 0.45008 | Best initial score; context-pack confirmation remains a gate |

The reference's shared multiplier scored 0.41218 on September; class-specific weights scored 0.41499. Thus decision weights alone added **0.00281**. The larger improvement came from combining model and rules, not from weights alone.

The paired day-resampled interval for the interaction pack's delta versus reference was **[-0.03455, -0.02107]**. Adding histories to that interaction pack yielded **[-0.06040, -0.04545]**. These specific implementations failed across September, so a literature recommendation is not sufficient reason to include them.

## Confirmation with chronological transaction context

All models and rules were rebuilt on transactions available through the end of each simulated prediction month. The initial model settings and seed were retained; weights and rule thresholds were reselected on bounded August, then frozen for bounded September.

| Variant | August macro-F1 | September macro-F1 |
|---|---:|---:|
| Model, ordinary argmax | 0.41503 | 0.41637 |
| Model, shared August multiplier | 0.42108 | 0.42143 |
| Model, joint August weights | 0.42472 | 0.42348 |
| Model + August-selected rule combination | **0.44236** | **0.44895** |

The confirmed combination adds **0.02547** over the same model with tuned decisions. A 1,000-draw paired day bootstrap gives a delta interval of **[0.02106, 0.02999]**. This is an exploratory stability interval, not a public-score confidence interval.

| September segment | Model | Combination |
|---|---:|---:|
| 1–15 September | 0.41666 | 0.44377 |
| 16–30 September | 0.42710 | 0.45125 |

The combination corrected 413 previously incorrect predictions and broke 209 previously correct predictions. Macro-F1 improves because the changes help several rare classes; total accuracy alone would hide much of that value.

| Class | Model F1 | Combined F1 | Delta |
|---|---:|---:|---:|
| legit | 0.9865 | 0.9865 | ~0.0000 |
| mule_account | 0.3618 | 0.3718 | +0.0100 |
| velocity_bot | 0.4708 | 0.4826 | +0.0118 |
| layering_chain | 0.4976 | 0.5127 | +0.0151 |
| sim_swap | 0.2829 | 0.3398 | +0.0569 |
| account_takeover | 0.2754 | 0.3353 | +0.0600 |
| phishing_collect | 0.4149 | 0.4045 | -0.0103 |
| qr_swap | 0.4562 | 0.4568 | +0.0006 |
| refund_abuse | 0.4240 | 0.4657 | +0.0417 |
| fake_cashback | 0.2963 | 0.3518 | +0.0555 |
| loan_app_extortion | 0.3007 | 0.3579 | +0.0572 |
| merchant_collusion | 0.3148 | 0.3219 | +0.0072 |

These class-level changes explain the gain and its tradeoff. They also give the implementation reviewer concrete regression targets.

### Second-seed confirmation

Repeated the bounded-context pair with seed **281**, changing both the negative sample and the model randomness. Training retained 121,099 rows for A and 241,687 for B. All decisions were fitted anew on that run's August predictions.

| September result | Seed 137 | Seed 281 |
|---|---:|---:|
| Model with August-fitted class weights | 0.42348 | 0.41824 |
| Model + August-fitted rule combination | **0.44895** | **0.44243** |
| Combination delta | **+0.02547** | **+0.02419** |
| Paired day-resampled delta interval | [0.02106, 0.02999] | [0.01805, 0.02981] |

Seed 281's combination also improved both September halves: 0.41515 → 0.43584 and 0.41882 → 0.44597. The score level varies by seed, but the combination's contribution remained positive. This supports implementing the combination, not promising a particular October score.

Class-specific weights alone were less consistent: on seed 281 they scored 0.41824 versus 0.41950 for a shared August multiplier. Thus the plan retains both decision baselines rather than assuming eleven weights always help. The confirmed rule/model combination has stronger evidence than class weighting by itself.

## Re-examining entity history and event purity

Earlier-month fraud history is associated with later fraud, but it is not a reliable class label:

| Historical entity flag | August any-fraud precision | September any-fraud precision |
|---|---:|---:|
| Sender had earlier fraud | 4.77% | 4.69% |
| P2P receiver had earlier fraud | 24.86% | 23.18% |
| Device had earlier fraud | 4.22% | 3.93% |

The receiver flag covered 487 September fraud rows not caught by the existing rules, but many false positives and wrong-type ambiguities remain. This is a diagnostic association, not evidence that a receiver-history model adds value. The tested broad pack's failure overrides the tempting standalone statistic. A receiver-only alternative remains unmeasured.

The original purity probes selected 6,048 mule-neighbour rows, of which 5,745 were the very labelled mule seeds used in selection. For bots, 8,106 of 8,627 selected rows were labelled bot seeds. That makes the quoted ratios unsuitable as independent propagation accuracy.

To probe topology more fairly, randomly reveal 20% of a month's labels, select true mule/bot seeds only from those revealed rows, and evaluate neighbours exclusively among the hidden 80%. Repeat with seeds 137, 281 and 419 in August and September. Results:

- Mule neighbours: **80.1–87.4% precision**, recovering about **9.9–12.3% of hidden P2P mule rows**.
- Bot neighbours: **83.2–84.9% precision**, recovering about **29.2–34.0% of hidden bot rows**.

This is evidence that some real clustering exists. It is conditional on true seed labels, uses labels only for this diagnostic, and does not measure propagation from model-generated seeds. No test row was labelled by hand. No labels were changed for model fitting or evaluation.

The teammate's reported eight-candidate flow and aggregate-payout negative results were incorporated as prior session evidence, not independently rerun. There was no reason to repeat these lower-priority probes before testing cheaper decision combinations.

## What changed in the plan

- Promote the measured rule/model combination ahead of additional feature families or ensembles.
- Keep joint class decisions; their gain is useful but small.
- Keep the timing/amount pack as a separate candidate, not a prerequisite for the confirmed result.
- Reject the tested interaction and broad history packs; leave narrower variants explicitly unmeasured.
- Replace seed-inclusive purity with independent masked-seed diagnostics, and defer propagation until nested validation exists.
- Preserve original validation labels. Remove the older plan's model-derived “clean validation” acceptance criterion, arbitrary tree-count multiplier, forced class predictions, October rate matching and public leaderboard ablation probes.
- Correct the starter notebook's `~is_sep` training mask, which includes October-labelled tails when evaluating September.

## Reproduction and artifacts

Research scripts are in `scripts/research/`. They neither generate a competition submission nor refit a final model on all labels. The original scripts and reports remain untouched. Research workspaces, parquet caches and probability arrays are under ignored `data/processed/`; `.venv` is also ignored.

[RESULTS.json](RESULTS.json) contains the metric outputs, per-class results, seed diagnostics, submission comparison, data hashes and installed dependency versions. It contains no fitted models or transaction-level probability arrays.

Environment: Python 3.14, NumPy 2.5.4, pandas 3.1.1, LightGBM 4.7.0, scikit-learn 1.9.1, SciPy 1.18.1, PyArrow 26.0.1, matplotlib 3.11.2. Full environment and raw-data hashes are in the result manifest. The initial six fits plus preparation took about 268 seconds after cached EDA features were available; the contextual pair took about 112 seconds. These are observed laptop runtimes, not runtime guarantees.

The initial root `data/processed/research_probe` contains a `data/raw` symlink to the extracted supplied data. Create `reports/eda/{figures,tables}` under that root before the original EDA scripts. Run commands from the repository root:

```bash
.venv/bin/python scripts/eda/00_load_and_overview.py data/processed/research_probe
.venv/bin/python scripts/eda/04_relational.py data/processed/research_probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/research/validate_candidates.py data/processed/research_probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/research/validate_candidates.py data/processed/research_probe --context-only
.venv/bin/python scripts/research/audit_evidence.py data/processed/research_probe
.venv/bin/python scripts/research/prepare_bounded_context.py data/processed/research_probe data/processed/research_aug 2026-09-01
.venv/bin/python scripts/research/prepare_bounded_context.py data/processed/research_probe data/processed/research_sep 2026-10-01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/research/validate_candidates.py data/processed/research_aug --base-only --only-month 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/research/validate_candidates.py data/processed/research_sep --base-only --only-month 9 --weights-from data/processed/research_aug/candidate_results.json
.venv/bin/python scripts/research/validate_rule_combination.py data/processed/research_aug data/processed/research_sep base
.venv/bin/python scripts/research/identify_submission.py data/processed/research_probe ../s1_v1.csv
```

The bounded-context and submission-identification helpers execute only the feature/rule-building prefix of the existing EDA files, with an asserted unique marker; they skip plotting, training and submission-export sections. They are reproducibility helpers, not the architecture proposed for production.

For the second seed, create separate output roots with `data` symlinks to the already-built bounded August/September roots, rerun `validate_candidates.py` with `--seed 281`, and point September's `--weights-from` to that seed's August results. Run `validate_rule_combination.py` on that pair. Separate roots preserve the first seed's probability artifacts.
