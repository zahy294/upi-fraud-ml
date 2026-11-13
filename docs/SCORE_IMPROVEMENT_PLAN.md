# UPI fraud: evidence-led implementation handoff

Prepared 13 November 2026 for the implementation agent. User-reported public score: **0.5034**; leader: **0.5487**. The requested outcome is a reproducible improvement with a credible chance of crossing 0.55. **No method in this document has a verified October score.** Local validation has noisy labels; October labels are unavailable and described as clean. A 0.60–0.70 result remains an unverified ambition.

This document supersedes conflicting recommendations in `IMPLEMENTATION_PLAN.md`. Read this file first, then the [measured evidence](../reports/research/CANDIDATE_EVIDENCE.md). Consult the [primary-source research](hackathon/method_research_2026-11-13.md) only for a method's rationale. The old EDA remains useful for feature definitions, not for its claimed ceiling or propagation accuracy. **Initial implementation path: E0 → E1 → E2.** The bounded-context confirmation improved September from 0.42348 to 0.44895. Add E3 only as a separate ablation; skip the failed E4 broad pack.

A second seed confirmed the combination's gain: 0.41824 → 0.44243. Joint class weights alone were less consistent than the rule/model combination; retain the shared-multiplier comparison. These local scores cannot be numerically added to the public score.

## 1. Scope and first checkpoint

Implement and evaluate locally. **Do not push to origin.** Preserve raw files, existing reports and the best submission. This task's research scripts are disposable ablations, not the final training pipeline.

Actual raw data is in `../upi-fraud-rings/`, outside the repository. Accept `--data-dir` explicitly; do not require moving or duplicating 280 MB of CSVs. Train contains 1,564,060 rows; test 260,582; accounts 420,000; merchants 30,000. Read the supplied `PROBLEM_STATEMENT.md` for labels and submission requirements.

**Scored artifact:** `../s1_v1.csv`, user-reported public macro-F1 0.5034. It has valid unique test IDs in sample order; SHA-256 `1d5b1b383132b490fdde0751da0aaa08827c7503d971a9c1acda7791174951e7`. Preserve it unchanged. The user confirmed the code came from `main`; a fetch confirmed local and remote main both point to `f2a9e92`. The CSV differs from the exact existing rules output on **1,834 rows**. This revision also contains an organiser starter notebook, but the generating pipeline/configuration is not yet identified. Hard test labels alone cannot recover its probabilities or September performance. Do not claim an improvement over 0.5034 by beating an older local baseline. Proceed against the measured research baseline, and identify the scored pipeline when its run information becomes available. Do not use this CSV as ground-truth labels or blindly overwrite its predictions with rules.

Completion: write a run manifest with code revision, data hashes, label order, model settings, features, seed, training rows and date cutoffs. Reproduce the available baseline's predictions or clearly record why exact reproduction is unavailable.

## 2. Validation contract — implement before new models

Use explicit timestamp filters and the supplied train/test identity. There are 147 labelled training rows in October and 46 test rows in November. Include all test IDs in the final output. Do not accidentally treat the October training tails as September validation data.

| Run | Labels permitted for fitting | Prediction population | Use |
|---|---|---|---|
| A | Train rows timestamp < 2026-08-01 | All labelled August rows | Feature exploration and decision-weight fitting |
| B | Train rows timestamp < 2026-09-01 | All labelled September rows | Later-month comparison with A's decision weights frozen |
| Final | All supplied train labels | All supplied test rows | Fit selected configuration; no test labels exist |

For final **historical label features**, freeze the history donors to train rows before 2026-10-01 throughout October/November. The full classifier may use all supplied labelled training tails, but excluding those tails is a useful sensitivity check. Record this distinction.

Label-free transaction context for A ends before 2026-09-01; for B before 2026-10-01; final context includes all supplied train+test rows. Within those boundaries, future transactions in the prediction month may inform features: the rules explicitly allow batch connections. Keep this offline availability contract consistent. The first research ablation reused EDA features built on all four months; E0/E1/E2 were subsequently confirmed with these boundaries. **The extra context pack still needs its own bounded-context confirmation before adoption.**

Build graph/session structure from label-free columns. For outer September evaluation, only pre-September labels may influence model fitting, decision tuning, historical label features or teacher models. September labels are used only to compute evaluation results; they cannot influence features, cleaning or predictions. Fix the starter notebook's `~is_sep` fit filter if reusing it: that filter includes the 147 October training tails; the correct pre-September filter is `timestamp < 2026-09-01`.

For every experiment save probabilities and predictions indexed by `txn_id`, the 12×12 confusion matrix, per-class precision/recall/F1/support, macro-F1, predicted counts, elapsed time and configuration. Always score the full validation population against its original labels. Report September 1–15 and 16–30 separately. Repeated use of September makes it a development set; disclose that selection bias.

Use paired resampling of validation **days**, with the same resampled days for baseline and candidate, to assess delta stability. This is a sensitivity diagnostic, not a guarantee: accounts and long events can cross day boundaries. For finalists also check a second model seed. Never call a relabelled validation score clean ground truth.

Completion: aligned saved baseline probabilities for A/B, independently fitted decisions for A, frozen evaluation on B, and verified absence of validation labels in all feature donors.

## 3. Minimal code structure and data contracts

Keep the implementation small. Prefer these modules over a generic experiment framework:

| Module | Contract |
|---|---|
| `src/upi_fraud/data.py` | Load typed data, validate IDs, preserve source split and original test order; join accounts/merchants |
| `features.py` | `build_features(context, query_ids) -> frame[txn_id, label_free_features]`; sorted windows and bounded sessions |
| `history.py` | `build_history(queries, labelled_donors, donor_cutoffs) -> frame[txn_id, history_features]` |
| `model.py` | Fit/predict a configured LightGBM; optional second model only after a useful feature delta |
| `decisions.py` | Fit/apply joint class weights; compute macro-F1 from a 12×12 confusion matrix |
| `run.py` | Explicit `validate`, `refit`, `predict`, `check-submission` commands; manifests and output directories |

One configuration file should specify data/output directories, seed, temporal boundaries, enabled feature packs, model parameters, negative sampling and decisions. Save outputs under a unique `runs/<run_id>/` or ignored model/submission directory; keep large arrays out of Git. Preserve a fixed label order from `scripts/eda/common.py`. Store that order with every probability array.

Feature joins must use `txn_id` with one-to-one validation. Existing positional concatenation in some EDA scripts is unsafe if either frame is reordered. Maintain stable integer group codes, keeping account recipients and merchant recipients distinct. Retain raw IDs for joins and diagnostics; exclude transaction IDs and raw entity codes from the initial predictors.

The research environment is Python 3.14 in `.venv`; exact installed versions are in the evidence report. A production runtime may use another compatible Python, but record and pin the environment actually used. This laptop has 12 GiB RAM with other applications active; the research runs used 4 threads and sampled training negatives. Avoid simultaneous full-data fits.

Completion: one command runs each fold, writes keyed probabilities and can resume from a cached feature pack whose data/config hash matches.

## 4. Experiment queue and stopping rules

The evidence report determines which branches survived the research probes. Treat literature-backed but unmeasured packs as candidates, not guaranteed gains. Change one family at a time and retain a result ledger including failures.

### E0. Match the available baseline and sampling policy

Reuse the existing 23 numeric row features, five categorical columns and 33 relational columns. Exclude `app_version` and `ip_risk_score` for the first model; compare them only in an explicit drift sensitivity run. Drift alone does not prove a feature has zero predictive value. Keep raw amounts alongside relative features.

Research settings: 200 trees, learning rate 0.05, 31 leaves, minimum 40 rows per leaf, max bin 127, row/column subsampling 0.8, seed 137, 4 threads. Retain all labelled fraud and a fixed random 20% of legitimate **training** rows, assigning retained legitimate rows weight 5 and fraud rows weight 1. Build features before sampling. This preserves the original class proportions in the weighted objective in expectation; it does not make the model identical to full-data training. Leaf-size constraints still depend on sampled counts. Validation is never sampled.

On an adequate machine, compare the winning configuration against full-data fitting once. Do not silently combine inverse sampling weights with inverse class-frequency weights. If the actual 0.5034 pipeline differs, it becomes E0 and every later experiment is an ablation from it.

### E1. Joint decision weights — cheapest candidate

Use `prediction = argmax_c(P[c] * w[c])`, with `w[legit] = 1`. This is equivalent to offsets on log probabilities. It is a practical search, not a proven globally optimal multiclass F1 decision rule.

Start from the best shared fraud multiplier in `[1,2,3,5,8,12,20]` on A. Then coordinate-search eleven weights over a small fixed logarithmic grid; two sweeps were used in research. For each candidate recompute all 12 class F1s, since changing one weight moves predictions out of another class. Include the existing weight in the grid. Prefer a stable plateau and small change over a narrow peak. Store the chosen weights and apply them unchanged to B.

Compare ordinary argmax, shared multiplier and class-specific weights. A class having no predictions is a diagnostic to investigate, not a reason to fabricate one prediction. Do not apply the binary calibrated-probability `F1/2` formula separately to twelve competing classes.

Completion: both shared and joint decision results are saved, with per-class regressions documented. Prefer the variant whose contribution survives the temporal and seed checks; the second research seed showed a small regression from joint weights alone. If this already exists in the 0.5034 model, skip duplicate implementation.

### E2. Rule/model decision combination — main measured candidate

This was the strongest low-cost decision change in research. Use the **existing** priority-ordered rules in `scripts/eda/07_baselines.py` to produce one label `r` per row (or `legit` if none fires). Compute model decision `m = argmax(P * w)` using E1's weights. Return `r` only when `r != legit` and the model's **unweighted** probability `P[r] >= tau[r]`; otherwise return `m`.

Initialize every rule threshold to infinity, which means the pure model. On A only, make two coordinate sweeps over fraud classes with threshold candidates `[0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]`. Include the current setting as the fallback. Change it only if the whole 12-class macro-F1 improves by more than 0.0002. Store thresholds and their label order; freeze them for B. A zero threshold means the rule wins whenever it fires; infinity means that class never overrides. Singleton rule rows are allowed because these are the pre-existing transaction rules, not event propagation.

The original-context probe improved September from 0.41499 to 0.44590; with the timing/amount pack it reached 0.45008. The subsequent bounded-context base model improved from **0.42348 to 0.44895**. These are local raw-label scores, not October scores. Prefer the bounded, confirmed variant initially. August-selected thresholds are a reproducible starting configuration, not universal constants to copy onto a different model.

Implementation needs only the existing rules, probability arrays and a small decision function. Keep it ahead of target history and new model families. Do not infer a hybrid score by choosing the largest per-class F1 from separate models; actually evaluate the single resulting prediction vector, including conflicts.

Completion: B gains persist after feature-context boundaries are matched, on both September halves, with per-class tradeoffs and paired day-resampling results reported. If the received 0.5034 CSV already uses this exact combination, compare against its original probabilities/configuration instead of repeating it.

### E3. Compact label-free timing/amount context

Primary winner writeups support this family; the local evidence report records the small tested version. Add features incrementally, starting with the measured pack rather than every conceivable aggregate.

| Pack | Exact initial features | Why it could help |
|---|---|---|
| Gaps and local amounts | For sender/receiver/device: previous and next strictly different timestamp gap, capped at 7 days; ±2h and ±24h amount sum; row amount divided by window mean | Distinguishes rapid repeated activity and unusual amounts within local activity |
| Burst composition, optional next | Device×merchant-category and P2P receiver×transaction-type counts/distinct participants over ±1h/±24h; new-payee and whole-rupee shares | Separates ordinary merchant traffic from coordinated events |
| Sender drain, optional next | Sender amount sum divided by current balance, fraction of new payees/night transactions, presence of a prior ₹500 payment within 1h | Combines takeover/SIM clues with activity structure |

For optional packs explicitly specify whether the query row is included, handle zero denominators/missing neighbours, and retain support counts. A balance is a row-level attribute, not a verified account ledger; a sum/balance ratio is a feature, not a money-conservation proof.

Use sorted group/time arrays with binary-search window boundaries and prefix sums; avoid per-row Python scans. Group boundaries must be checked for predecessor/successor lookup. Identical timestamps require documented tie handling. Add a missing indicator where useful; no-neighbour and a true zero gap are different states. Start with fixed windows; postpone graph components and multi-hop traversal.

The tested rule/interaction pack adds existing rule flags, exact ₹500, over-balance, short/long activity ratios and simple SIM/device combinations. **Its individual results must pass the same gate as any new feature**; manually plausible conditions may add no information to trees or may hurt them.

Completion: isolated delta on A and frozen-decision B. Keep the smaller pack when extra features do not add a stable gain.

### E4. Historical entity rates — rejected broad pack, optional narrow branch

The broad sender/receiver/device pack failed: September fell from 0.38719 with the interaction pack to 0.33350 with histories added. **Exclude this pack from the initial implementation.** A receiver-only variant is an optional later hypothesis, because the raw-data probe finds more useful prior fraud concentration for P2P receivers than for senders/devices. Prior fraud is an input clue, never a permanent fraud label.

History features: log count of earlier labelled transactions; log count of earlier fraud; class-specific smoothed rates; unseen-entity flag; time since earlier fraud. Preserve support. A tested conservative rate is `class_count / (history_count + 20)`; this uses a fixed zero shrinkage prior. A nonzero-prior variant must estimate that prior only from permitted donors.

**Training rows:** donors must have timestamps strictly before the start of the query day minus seven days. The seven-day exclusion reduces same-event label transfer and gives July training rows meaningful histories. **Validation rows:** additionally cap donors strictly before the fold's training cutoff. Thus August never sees August labels, and September never sees September labels. Keep an audit of the maximum donor timestamp per query. Evaluate a stricter fixed pre-month snapshot if adopting the pack.

A simple previous-month map for all rows is not sufficient: all July training rows would have empty histories, so a July-trained model cannot learn how to use August's nonempty maps. Conversely, mapping each entity's full training labels back onto its own rows leaks the target.

Completion: receiver history adds a reproducible gain over the strongest label-free model on B, including a seen/unseen-entity breakdown. If not, leave it out. A high standalone fraud rate among previously flagged recipients is not proof of incremental model value.

### E5. Small ensemble — after useful base models

Use saved aligned probabilities to test a small predetermined blend grid, tuning its mix and decisions on A and freezing both on B. Add a second seed before introducing another model library. CatBoost is a reasonable later comparison for categorical diversity; its existence is not evidence it improves this dataset.

Retune E1/E2 decisions on the August blend before comparing it on September. The raw component probabilities and the combined decision must have matching label order and transaction IDs.

Completion: improvement over the best component after the same decision-tuning budget. Otherwise retain the single model.

### E6. Propagation / noise correction — gated, lower priority

Read the independent seed-masking diagnostics in the evidence report first. They measure topology conditional on **true** seed labels, not accuracy when a model supplies imperfect seeds. The original 94–95% includes seed rows and cannot justify copying labels.

If earlier steps plateau, try one narrow relation at a time: same-device short sessions for bots or same-P2P-receiver short sessions for mules. Bound total session duration as well as adjacent gap. Keep singletons unchanged and exclude each row's own prediction from neighbour statistics. Measure newly correct rows and newly introduced false positives beyond the base model.

For outer B, all models, histories and decisions used to create stage-two training features must be fit inside July–August. Generate July–August features through inner forward blocks; score September using only teachers trained before September. A global OOF array can leak September targets indirectly into July/August teacher predictions even if each row's own label was omitted. If this nesting is too costly, defer stage two.

Do not modify validation labels. Training-noise experiments may conservatively downweight a small audited set of temporal-OOF contradictions, but confidence alone is not proof of mislabelling. Preserve minority support and original labels. Compare against untouched training and report the ambiguity introduced by noisy validation. Omit wholesale cleaning, automatic event relabelling and test pseudo-labels from the first implementation.

## 5. Acceptance and final refit

Suggested gates are engineering choices, not significance theorems:

1. A candidate is exploratory until evaluated on B with A-selected decisions.
2. Prefer B macro-F1 delta ≥0.003 and consistent signs on September halves; inspect every class losing >0.02 F1 rather than hiding losses in the average. Small cheaper decision-only gains may still be useful.
3. Use paired day resampling and a second seed to reject fragile improvements. Validate feature availability boundaries before final acceptance. A public score increment cannot be predicted by adding the local delta to 0.5034.
4. Inspect October feature/prediction distributions by class, time and seen/unseen entities. Differences are alerts, not proof of failure. Do not force October fraud counts to match September, and do not drop useful features merely to minimise a train/test discriminator's AUC.
5. Once the configuration is frozen, retrain on all supplied labels, build all permitted test connections and apply stored decision parameters. Select tree count from validation; do not multiply it by an arbitrary 1.15.

Submission checks: exactly `id,label`; 260,582 unique IDs matching the sample; same order as sample; no missing predictions; labels in the official vocabulary; under 15 MB. Compute the CSV hash and record the run/config/model hashes that produced it. Inspect class counts, including any zero count, without forcing artificial predictions.

The competition automatically chooses the best public upload for private scoring. Keep a submission ledger; no leaderboard probing or uploads are part of this implementation handoff. The user can review the candidate file and evidence before choosing an upload. Do not push code to origin.

## 6. Targeted checks and review handoff

Required correctness checks should test failure modes rather than duplicate implementation:

- Shuffle input rows and confirm features/predictions still align by `txn_id`.
- For a tiny hand-constructed graph, verify same-timestamp ties, no-neighbour cases, merchant/account separation and exact window boundaries.
- Perturb all validation labels and verify features, history donors and predictions are unchanged.
- Verify each training history excludes its own row and the specified seven-day interval; each validation history excludes the entire validation month.
- Verify categorical codes and class-column order are identical across fit/predict artifacts.
- Recompute macro-F1 with scikit-learn using the explicit 12 labels and compare with the fast confusion-matrix calculation.
- Reject duplicate IDs, missing IDs, invalid labels and wrong joins before writing a submission.

Give the reviewer the diff, manifest, A/B results, per-class deltas, context/history cutoff definitions, probability hashes and exact reproduction command. Ask them to prioritise leakage, train/test feature mismatch, label order, row alignment and score-selection bias. Resolve those issues before spending effort on extra seeds.

## Paste this to GPT Sol High

> Read `docs/SCORE_IMPROVEMENT_PLAN.md` and `reports/research/CANDIDATE_EVIDENCE.md`. Implement the smallest measured improvement path, preserving the original data and reports. First locate and reproduce the 0.5034 pipeline if available; otherwise explicitly use the research baseline. Implement chronological validation and keyed probability artifacts before changing models. Use the evidence table to skip failed packs. Fit decisions on August and freeze them for September. Correct the research feature-context boundary before adopting a gain. Run the targeted leakage/alignment checks, then provide a local candidate submission, run manifest and per-class comparison. Keep the changes local; do not push to origin or upload to the leaderboard.
