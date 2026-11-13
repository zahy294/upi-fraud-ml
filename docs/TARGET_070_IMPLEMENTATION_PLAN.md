# UPI Fraud Rings: implementation plan targeting 0.70 macro-F1

Updated 13 November 2026. Primary target: **0.70 public macro-F1**, with **0.55 as an intermediate milestone** and robust private-split performance as the final objective. Latest team public score: **0.5034**, reported by the user during this revision. This is an experiment plan, not a claim that the target has been achieved or can be guaranteed. All 12 classes remain in every reported macro-F1 calculation.

This is the canonical local implementation plan. It supersedes the modelling recommendations in [the original plan](reference/FRIEND_IMPLEMENTATION_PLAN.md), while preserving that file for provenance. Research and small diagnostic probes are complete for this revision; production feature engineering and model training start after the user's go-ahead.

## 1. Starting evidence and what changes

The user supplied screenshots showing `rules_v0_ordered.csv` scored **0.4061** on the public split, with one upload used. The user later reported that the team scored **0.5034** and the leader reached **0.5487**. The newer team's filename, code revision, model, features and validation results are not yet available. Treat 0.5034 as the current reported team baseline; do not attribute that gain to any feature in this document. Leaderboard position and upload quota need a fresh check before future submissions.

The [friend's EDA](reference/FRIEND_EDA_REPORT.md) reports September macro-F1 of 0.339 for a row model, 0.391 with relational features, and 0.392 for rules. We have inspected its source but have not reproduced those trained-model results. Its shared fraud multiplier was selected on the same September rows that were scored, so these are tuned development results.

Two claims must not constrain the new search:

- The 0.454 signature oracle measures a restricted detector using those signatures; it is not an information-theoretic ceiling. Weak individual-feature AUC also cannot exclude useful feature interactions.
- The approximate 0.426 obtained by choosing the better per-class F1 from two predictors is not a measured ensemble. Their predictions can conflict and changing one class changes other classes' precision/recall.

Public scores and September scores use different periods and label quality. There is no justified fixed conversion between them. With `legit` F1 around 0.985, macro-F1 0.55 requires average fraud-class F1 around **0.510**, and 0.70 requires around **0.674**. Broad gains in fraud coverage are therefore the main objective; extra random seeds alone are a low-priority route.

The main change is to **measure additional detection beyond the existing rules before investing in larger models or complex graph machinery**. The first discovery budget goes to feature combinations, prior-month entity evidence, supported episode features and class-specific residual errors.

## 2. Evidence from this planning pass

The raw CSVs are already under `upi-fraud-rings/`. Narrow diagnostic scripts are under `scripts/research/`, with results under `reports/research/`. They preserve raw files and do not train a model or generate a submission.

### Negative findings that save implementation time

[The flow probe](../scripts/research/probe_flow_matches.py) compares the nearest transfer against the first 3 and 8 candidate transfers within 30 minutes. It retains the existing large, whole-rupee, P2P and 0.8–0.9999 amount-ratio conditions. In August and September, broadening the search added **zero** layering positives; July added one. September's standalone nearest-match mask found 474 layering rows among 653 flagged rows, with 2,182 layering positives in the month. This is a mask diagnostic, not a full 12-class model score.

A second fixed probe required at least three incoming P2P payments in 30 minutes followed by an outgoing payment of at least INR5,000, equal to 70–105% of the combined inflow. It found one row in July and none in August or September. This narrow definition has no evidence of material added coverage. It does not rule out longer horizons, split flows or continuous aggregate features, but those ideas must earn further work with a cheap probe first.

The flow probe uses context within each calendar month and excludes the 147 October-dated training rows from these diagnostics. It records the data hash, script hash, definitions and package versions in [flow_signal_probe.json](../reports/research/flow_signal_probe.json). Month boundaries are censored. These aggregates have now been inspected; do not call them untouched validation.

### Source-code findings

- The baseline calculates some metadata, including KYC, OS and bank, but its model feature list omits them. Their standalone distributions look weak; their conditional value is untested.
- The flow features use only the first future transfer or the last incoming transfer. The negative probe above means fixing that limitation is a correctness/completeness improvement, not a demonstrated large score gain.
- The 94–95% event-purity probes select neighbourhoods using known fraud labels and include the seed rows themselves. They do not establish that independently selected neighbouring transactions have 94–95% fraud precision.
- Relational features are concatenated by row position in the baseline. The implementation must join by unique `txn_id` and assert coverage and ordering.

See [conditional_signal_probe.md](../reports/research/conditional_signal_probe.md) for the separate fixed row/episode and historical-entity probes. Treat promising precision as a feature lead, not proof of an overall F1 improvement.

### Conditional findings and their limits

The conditional probe evaluated 38 fixed masks, including sequential follow-up overlap checks; it did not train models or inspect test transactions. These findings inform priority:

- Prior-month phishing history at a receiver plus a current COLLECT payment found 54 phishing positives among 137 flagged August rows and 117 among 283 September rows. After excluding the existing phishing rule, this leaves **11/82 in August and 18/143 in September** (13.4% and 12.6% precision). This is a small, repeatable source of positives outside that particular rule, useful as a model feature or specialist candidate. It is not outside every existing rule, and is too weak to use as an unconditional override.
- Drain/fresh-SIM and drain/new-device interactions reach 56.4% SIM-swap precision and 50.8% takeover precision in September, but sharpen subsets of existing rules. Their value is resolving subtype confusion, not demonstrated new coverage.
- The INR500-then-drain probe found only three September rows, including one takeover. Slow-login bot candidates added no bot positives beyond the original bot mask in August or September.
- At least two previous refund labels plus a current refund claim found 23 positives among 31 September rows, but none of those positives survived exclusion of the existing repeated-refund mask. Generic historical device-bot flags deteriorated across months; QR receiver histories were diluted by popular merchants.

**Conclusion:** these probes do not establish a route to 0.70 by adding simple rules. The next substantive experiment is a reproducible multivariate residual model, using joint context and carefully isolated history. Each larger feature family must demonstrate incremental detection before earning more compute.

## 3. Evaluation contract

### Forward labels and matched observation horizons

1. Use July labels to fit, and August to develop features and decision rules. Split August chronologically into calibration and subsequent checking blocks where support permits.
2. Freeze a small candidate set, fit on July–August and evaluate September. Keep the final September week out of repeated candidate comparisons from now on. Existing EDA and this research have already exposed aggregate September results, so this is a less-reused confirmation segment, not a previously unseen dataset.
3. Repeated ablations may inspect August and September 1–23 only. Reserve September 24–30 from further candidate comparisons. After model, feature and offset choices freeze, reveal final-week and full-September metrics together: all 12 precision/recall/F1/support values, macro-F1, prediction counts and confusion. Do not repeatedly expose the reserved week indirectly through a full-month score. Save probabilities keyed by `txn_id` and an explicit class order.
4. Backtests use only labels before their training cutoff. Do not assign October-dated training rows to September merely because they may be event tails. Report their coverage separately. Final fitting may use all officially supplied training labels, including tails; document this distinction.
5. Label-free future connections are permitted by the competition. Main August and September backtests stop feature context at the end of the simulated target month, with available-history and available-future coverage flags. Compare an extended-tail context only as a separately labelled sensitivity run. For final inference, all supplied test rows, including November tails, are available.

Use supplied validation labels as the headline metric. Never optimize against labels changed to agree with the model. Label-noise estimates and alternative targets are sensitivity analyses only.

### Prove that a feature adds information

Freeze the existing rule and signature definitions. For every new feature block, compare the same model, training rows and decision-tuning procedure with and without it. Report:

- Overall 12-class macro-F1 and per-class changes.
- Additional correctly detected fraud outside the old rules, plus added false positives and class-confusion costs on the full population.
- Recall inside and outside each class's original signature. Distinguish absence of the own-class signature from absence of every signature.
- Performance on known/new entities and high/low observation coverage.
- Runtime and memory cost; consistency across August and the development portion of September, followed by one frozen later check.

Use ordinary argmax and calibrated class offsets as separate reported results. Tune offsets on earlier predictions and apply frozen offsets to the later check. Do not apply binary F1 threshold identities as exact rules for exclusive multiclass prediction.

Prefer consistent improvements across dates. For close decisions, use day/episode-block uncertainty summaries; do not claim a universal 0.003 noise threshold. Always retain an unchanged baseline.

## 4. Prioritized search for additional signal

### A. Joint behavioural context and omitted metadata

Start here because the row model and broad rules leave many interactions untested.

- Add sender and receiver KYC, account ages, bank and OS, plus missingness. Use sender/receiver differences and categorical interactions with transaction type, merchant category, new-payee status, device novelty and home-city mismatch.
- Retain continuous `amount/balance`, residual-balance fraction, SIM age, device age and login delay. Add log scales, whole-rupee/price-ending features, zero-balance handling and `amount > balance`. The last flag is an empirical anomaly, not proof of fraud or a valid account ledger.
- Compute amount percentile and robust deviation within receiver-type/transaction-type/category peers. Fit historical references within the fold; compare separately with permitted label-free target-month peer normalization.
- Add joint event summaries: fraction of new payees, recent SIMs, unfamiliar devices and home-city mismatches among an account/device's bounded neighbours. Exclude the current transaction from behavioural reference statistics.
- Preserve missing QR-check results separately from checked matches. Do not derive geographic distances without coordinates.

Experiment A0 is a fixed global multiclass GBDT with these features. A1 assesses the residual positives outside old signatures. Feature selection and any specialist candidates must be decided from earlier folds, not by repeatedly searching September.

### B. Prior-month entity evidence, with explicit label isolation

The EDA suggests receiver histories can carry fraud information, but a history of any fraud is not a reliable subtype label. Test smoothed prior counts/rates for sender, P2P receiver, merchant and device: any-fraud rate, per-class rates, number of independent episodes, time since last flagged episode, and label diversity.

Combine these with current payment semantics: prior receiver QR/phishing evidence with current QR/COLLECT, prior sender refund evidence with a current REFUND claim, and prior device bot evidence with current merchant activity. These are **features**, not unconditional labels for reused entities.

For every training block, use only earlier blocks' labels; for validation, use only its training prefix. Keep population priors and unseen-entity flags for sparse histories. Cap counts and smooth rates to avoid a single noisy label dominating. Use entity IDs as join keys; start without feeding raw IDs to the model.

This pack must improve both known-entity and whole-population evaluation. A high-precision subgroup with a few examples is insufficient. If it helps only recurring entities, preserve a complementary behavioural model for new entities.

### C. Supported episode shapes and role-aware features

Use short bounded sessions and multiple time scales, rather than expanding every window/key combination.

- **Bot behaviour:** repeated-amount share, inter-payment gap quantiles and variability, merchant/category concentration, counterparty entropy, device switching and burst intensity relative to longer-term activity. Include slow bursts as well as 5-minute bursts.
- **Refund and collusion:** distinct transaction days, runs of refund claims, share of observed payments that are claims, customer concentration, round-amount repetition and purchase/claim order. REFUND is a claim; do not reverse its direction automatically.
- **Loan extortion:** recurrence across evenings, interval regularity, receiver/category concentration and deviations from ordinary AUTOPAY schedules. Preserve legitimate lending customers as hard negative examples.
- **SIM swap versus takeover:** ordered SIM/device novelty, new payees, repeated drain-like amounts and changing city. The exact INR500-then-drain pattern needs adequate support before becoming a priority.
- **Cashback versus phishing:** direction and position in a small-transfer/return-payment episode, amount ratio, lag, repeated targets and competing reverse links. Score both legs when their evidence supports the documented labels.
- **Mule versus layering:** incoming/onward role, fan-in/out balance and ambiguity, bounded path depth in each direction, amount decay and time consistency. Begin with continuous summaries; defer expensive capacity matching and deeper paths until they show additional coverage beyond the negative probes above.

Require adequate personal history for distribution-change features; otherwise use peer fallbacks and support flags. Sender histories are sparse, so global sequence transformers and large per-account time series are poor first investments.

Primary-source mechanisms and computational bounds are documented in [ADDITIONAL_SIGNAL_RESEARCH.md](ADDITIONAL_SIGNAL_RESEARCH.md). Local evidence governs priority when it disagrees with a paper-inspired idea.

### D. Residual specialists and event predictions

Only after A–C produce a trustworthy global baseline, test a small set of specialists: mule/layering, SIM/takeover, cashback/phishing, and the weakest merchant-related classes. Include ordinary legitimate transactions and confusing legitimate lookalikes. A hard rule gate must not prevent recovery of all off-pattern positives; if screening is used, report its recall ceiling.

Blend specialist probabilities or log-score corrections into a global 12-class decision, then evaluate the complete multiclass output. A specialist's binary F1 is not an overall improvement.

For event propagation, first measure non-seed neighbours of events selected by frozen rules or earlier-trained predictions. Exclude seeds from scored neighbours, distinguish row-weighted from event-weighted results, and control high-degree hubs. Do not propagate labels across lifetime connected components.

Start with a small probability blend over role-specific neighbours. A learned second stage needs **nested forward prediction generation**: for a target training block, score it AND all of its neighbours using a base model that excludes that target block's labels. Ordinary OOF predictions stored separately for each neighbour are insufficient because a neighbour's model may have trained on the target row's label. Early blocks without a valid earlier model retain missing-coverage indicators or are excluded from meta-training. Every outer validation label remains excluded throughout.

### E. Noise robustness and ensembles last

Preserve original labels. Compare unchanged training with modest downweighting of a bounded set of disagreements, identified entirely within the outer training partition. Use time-held-out model predictions plus independent behavioural evidence; keep minority support and a nonzero weight floor. Automatic relabelling, cleaned-validation optimization and repeated self-training are not default steps.

Standard positive-unlabelled assumptions are not established: fraud labels can have the wrong subtype, and reported cases need not be randomly selected. Any PU-style correction requires a separate assumption/sensitivity study.

Train complementary XGBoost/CatBoost/LightGBM models only after a useful feature set exists. Blend saved probabilities if gains survive later checks. Add seeds after model-family diversity; skip extra fits when improvements are smaller than uncertainty or merely repeat existing errors. Neural embeddings and sequence-template alignment are optional residual experiments, not prerequisites.

## 5. Work sequence and decision gates

1. **Reproduce the starting point.** Obtain the second CSV's recipe/commit and saved probabilities from the teammate if available, and reproduce the **0.5034 candidate's local validation** before replacing it. If unavailable, proceed with the documented rule/GBDT baselines and mark the newer recipe unknown. Inspect actual hardware and Python compatibility, audit IDs/dates/joins/labels and match submitted files by hash when available. Record environment differences; do not assume the friend's RTX4050 or Python3.11 describes this computer.
2. **Build the evaluation harness.** Version splits, context horizons and fixed class mapping. Produce all 12 metrics, saved probabilities, rule-overlap diagnostics and a valid candidate-writing path.
3. **Run A and B first.** Compare behavioural/metadata interactions and time-safe entity evidence one block at a time. Use the probe results to choose a small number of candidates. If gains only sharpen already-caught rows, shift to residual recall discovery.
4. **Add supported parts of C.** Prioritize the weakest classes by additional true positives at controlled false-positive cost. Stop low-support motif branches quickly.
5. **Test D and E selectively.** Specialists or neighbour aggregation should solve a measured confusion. Noise weighting and ensembles must improve the complete later-period result.
6. **Freeze and reproduce.** Refit selected models on all supplied labelled data with a documented tree-count rule. Rebuild features and predictions, save model artifacts and verify the candidate. Track the 0.55 milestone and 0.70 target separately; public scores are confirmed only by actual authorized uploads.

If progress stalls below the target, report what limits it and revise hypotheses; do not turn the target into fabricated metrics or a reason to tune to hidden labels. Preserve the best reproducible candidate throughout. The supplied deadline is 15 November 2026, 10:00 IST; recheck remaining time when implementation starts and reserve time for reproduction/code submission.

## 6. Local development and Colab A100 execution

Use one repository, one command-line entry point and configuration files. Keep notebooks thin: environment setup, paths and invocation. Run a small end-to-end smoke check locally and on the friend's Colab early, before expanding the pipeline.

The user's chosen workflow is **GPT-6.1 Sol on high for coding on the user's branch**, followed by the teammate pulling that code and training on an A100. This document records that preference; it does not change the current model or create/publish a branch. The concrete coding and training handoff is in [CODING_HANDOFF.md](CODING_HANDOFF.md).

Local work covers data audit, feature development, validation checks and a CPU LightGBM reference. Colab runs larger GPU XGBoost/CatBoost fits after a short benchmark confirms the benefit. Measure feature-building CPU/RAM separately from GPU fit time. An A100 helps compute throughput; it does not create missing signal.

Pin tested dependencies for both environments; verify Python/CUDA compatibility instead of copying version guesses. Copy working data from Drive to runtime-local storage, then persist models, logs and probabilities back to durable storage. Colab's hardware availability and lifetime are variable ([official FAQ](https://research.google.com/colaboratory/faq.html)); retain a local/CPU fallback. Preserve actual fitted models because GPU training, including CatBoost, may not be deterministic ([CatBoost documentation](https://catboost.ai/docs/en/features/training-on-gpu)).

Every run records data hashes, code revision/source snapshot, config, package versions, seed, class mapping, feature names, split/context definition, device, runtime, validation probabilities and final decision weights. Cache keys include data, code, config and horizon hashes; file existence alone does not validate a cache. Derived features join by `txn_id` with uniqueness/coverage assertions.

Sampling for a smoke test must not be mistaken for valid evaluation. Compute features on complete allowed context before selecting scored rows, or select bounded complete episodes and report what was truncated. Sampling senders alone can break receiver fan-in and shared-device events.

## 7. Submission and scope boundaries

Keep both the accepted 0.4061 rules candidate and the newer reported 0.5034 candidate with their metadata. At least two CSVs have now been reported, but the live quota is unknown. Check it before any future upload. Upload only a candidate we would accept as the final entry, because the supplied rules select the best public upload for private scoring.

Do not use class-removal leaderboard probes: they change `legit` F1 too, can increase the score, and do not isolate one class's F1. Do not force October label proportions to match September. The frequency of every predicted class is a diagnostic, not a mandatory validity condition.

Submission checks: exact `id,label` header; one row per test ID; no missing, extra or duplicate IDs; legal labels; sample-compatible order; no saved index; file under 15 MB. Investigate absent classes without inventing predictions merely to satisfy a count check.

This revision authorizes no upload, repository publication or communication with the teammate. The referenced repository was public when inspected; its own plan calls for private collaboration during the round. Local planning files are the deliverable for this step.

## Provenance

- Friend's source snapshot: [`f2a9e923ff42898de4333ddad5d144a9b9a75768`](https://github.com/zahy294/upi-fraud-ml/tree/f2a9e923ff42898de4333ddad5d144a9b9a75768), specifically `reports/eda/EDA_REPORT.md`, `docs/IMPLEMENTATION_PLAN.md`, `scripts/eda/04_relational.py`, `07_baselines.py` and `common.py`.
- Original cloud documents: [HANDOFF.md](reference/HANDOFF.md), [RESEARCH_PLAN.md](reference/RESEARCH_PLAN.md), from the user's handoff archive.
- Competition definitions: [local problem statement](../upi-fraud-rings/PROBLEM_STATEMENT.md).
- New diagnostic evidence: [flow probe JSON](../reports/research/flow_signal_probe.json) and [conditional probe report](../reports/research/conditional_signal_probe.md). These are exploratory observations, not achieved model gains.

