# Additional signals for UPI Fraud Rings

Research date: 13 November 2026. Scope: revise the experiment plan; no training, inference, upload, or test hand-labelling was performed. Latest user update: team public score 0.5034; primary target 0.70, with 0.55 an intermediate milestone. See the canonical implementation plan and coding handoff for the latest decisions.

## Recommendation

**Local evidence update:** use the ordering in [the revised implementation plan](IMPLEMENTATION_PLAN.md). Subsequent fixed probes found no additional August/September layering coverage from searching the first eight transfers, and no hits from a narrow 30-minute aggregate-payout rule. Those observations lower the immediate priority of the flow ideas below. They do not test every possible flow definition. Start with conditional context, time-safe entity evidence and residual-coverage measurements; expand graph work only when supported.

The literature-derived ordering below originally favored **aggregate temporal flow matching**, then **bounded multi-hop paths**, because they address specific blind spots of nearest-forward-transfer rules. These remain mechanism-based hypotheses, not measured gains. Defer neural sequence models, full graph mining frameworks, global all-pairs sequence search, and automatic relabelling.

The sources below support the mechanisms, not an expected competition score. Neither a score above 0.55 nor the 0.70 target can be inferred from unrelated papers. All feature designs and experiment ordering below are proposals for this dataset, not claims that its hidden generator uses a particular formula.

The local [problem statement](../upi-fraud-rings/PROBLEM_STATEMENT.md) explicitly permits label-free test connections, describes train-label omissions and class mistakes, and says that every incoming payment in a mule burst and the mule's onward transfers receive the mule label. That last detail makes **transaction-role coverage** particularly important: detecting a suspicious account is insufficient if its other participating transactions remain unlabelled.

Evidence grades: **A** = authoritative source establishes the method or its assumptions; **B** = primary empirical study supports the mechanism in a different setting; **C** = dataset-specific hypothesis requiring an ablation. No option below has direct evidence of improving this competition yet.

## 1. Aggregate fan-in/fan-out conservation and episode roles

**Priority: first. Evidence: B for flow conservation; C for the proposed features.**

FlowScope models multi-step flows and penalizes imbalance at intermediary accounts. This is stronger evidence for tracking money through an account than for using static degree alone. Its original objective is dense-flow discovery, so borrowing its conservation idea does not require implementing its entire optimization algorithm. [FlowScope, author-hosted paper](https://shenghua-liu.github.io/assets/pdf/papers/aaai2020cr-flowscope.pdf)

SMoTeF separately shows why temporal ordering matters: outgoing money that predates an incoming transfer cannot be attributed to that transfer merely because the amounts and graph structure match. Its method combines ordered events with flow capacities. [SMoTeF, publisher full text](https://link.springer.com/article/10.1007/s10489-024-05545-4)

**Proposed features.** For each account and candidate episode, build both incoming and outgoing event lists. Initially use 10-, 30-, and 120-minute windows. Preserve raw incoming/outgoing sums and counts, then add:

- Normalized residual `abs(in_sum - out_sum) / max(in_sum, out_sum, epsilon)` and signed retention `(in_sum - out_sum) / max(in_sum, epsilon)`.
- Chronologically attributable throughput: maintain remaining capacity for each preceding incoming event; allocate outgoing amounts only against earlier available capacity. Report total attributable amount, attributable fraction of inflow and outflow, and unmatched amount. FIFO is a deterministic heuristic attribution, not proof of money provenance or a general maximum-flow solver.
- Time to forward 25%, 50%, and 90% of an episode's observed incoming amount; missing/censored flags when those levels are not reached.
- Distinct source and destination counts, largest-source/destination amount share, and the fractions attributable to one-to-many, many-to-one, and one-to-one transfer patterns.
- For every participating transaction: incoming/outgoing role, amount share of its episode, episode conservation score, and whether it participates in a high-throughput interval. Do not spread an episode flag to all lifetime transactions of the account.

For example, incoming 2,000 + 3,000 followed by outgoing 4,850 can be informative even though neither individual incoming payment matches the onward amount. Conversely, a large unrelated outgoing transfer should not be reused as full evidence for every incoming payment. Capacity accounting prevents this double counting.

**Cost.** Sorting and compact per-account arrays dominate preparation. Fixed-window sums use prefix sums and binary search or two-pointer scans. A chronological matching pass is linear in episode events once sorted. Avoid all-pairs pandas joins. Session boundaries should be bounded by both an inactivity gap and a maximum duration so continuously active merchants do not become month-long episodes.

**Risks.** Legitimate collection accounts also forward money; absolute conservation alone is not a label. Opening balances and unobserved external transactions make exact provenance impossible. Keep volume, role, timing, and coverage features separate. Handle equal timestamps as ambiguous rather than inventing a causal order from shuffled rows. Merchant IDs are endpoints unless actual outgoing merchant events exist in the data.

**Minimal experiment.** Add only conservation, forwarding quantiles, and role features to the frozen baseline. Report mule and layering precision/recall separately, plus performance on incoming versus onward legs. Compare raw aggregate ratios against capacity-limited matching. This tells us whether the extra matching work earns its cost.

## 2. Bounded temporal paths, including the first and last legs

**Priority: second. Evidence: B for temporal transaction graphs; C for the proposed features.**

FaSTMAN represents sequential transactions with temporal relationships and uses higher-order graph information to prune insignificant edges before expensive flow queries. Its setting differs from this challenge, but it supports representing transaction succession explicitly instead of relying only on account communities. [FaSTMAN, authors' full text](https://arxiv.org/html/2309.13662v1)

**Proposed features.** Make transactions the vertices of a temporary directed graph. Add candidate continuation `e -> f` when `receiver(e) == sender(f)`, `time(f) > time(e)`, and the gap lies within a bounded horizon. Keep the best few continuations by time and amount compatibility, alongside the total candidate count so pruning remains visible.

Start with paths of 2, 3, and 4 transfers and per-hop horizons of 10 and 30 minutes. Record longest compatible path, count of retained paths, elapsed time, terminal/start amount ratio, mean and maximum log-amount decrement, monotonic-decrease fraction, and dispersion of decrements. Also record preceding-path and following-path depth separately: a first leg has no predecessor and a final leg has no successor.

Amount compatibility must be a continuous feature or broad candidate filter; a hard near-equality constraint would lose split/merge flows and larger commissions. Keep the aggregate-flow features from option 1 for those cases. Bounded cycles can be a later collusion-specific addition, but a cycle should not be a prerequisite for layering.

**Cost.** With at most `k` successors per transaction and fixed depth `d`, dynamic programming summaries on the time-directed graph can avoid enumerating every path; complexity is approximately `O(N*k*d)` after candidate retrieval. Explicit path enumeration grows exponentially in depth and should be avoided. Start `k=4`, `d=4`, recording how often the cap truncates candidates. These are proposed engineering bounds, not optimal values.

**Risks.** Busy accounts create accidental paths. Stable gaps and amount relationships are more informative than existence alone. Path overlap can duplicate evidence. Same-second transfers are temporally ambiguous; retain a separate tied-time candidate flag rather than assert causality. In validation, build connections from the permitted feature-only period, never from held-out labels.

**Minimal experiment.** Compare one-hop matching with 2-4-hop summaries on the same folds and model. Inspect layering-versus-mule confusion and first/last-leg recall. Reject additional depth if it raises false positives without stable class benefit.

## 3. Event distributions relative to personal and peer baselines

**Priority: third, inexpensive version only. Evidence: A for distance definition; C for fraud utility.**

SciPy supports Wasserstein-1 distance between empirical one-dimensional distributions, including weighted histograms. It measures distributional displacement, so it can distinguish event histories that share a mean but have different shapes. It does not preserve event order. [SciPy Wasserstein documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)

**Proposed features.** For sender, device, receiver, and sender-receiver pair, retain a recent event block and an earlier non-overlapping reference block. Compare distributions of log amount, log positive inter-event gap, amount/balance ratio, and hour-of-day where support exists. For time of day, use circular summaries or a circular histogram distance; ordinary Wasserstein on hours incorrectly treats 23:59 and 00:01 as far apart.

Start with histogram-CDF distances using 8-16 frozen training-derived bins, rather than calling a general distance routine for every row. Add recent/reference support counts and a missing-history indicator. Require an initial minimum of eight observations for a personal comparison; otherwise use a peer reference defined by receiver type, merchant category, and broad activity level. The minimum is a tuning hypothesis, not a statistical guarantee.

Useful cheap companions include gap coefficient of variation, repeated-amount share, entropy of counterparties, and the share of events concentrated in one short interval. Their value here is the **change relative to the same entity or a matched peer**, beyond ordinary total counts. Retain both absolute amount level and normalized shape: normalization alone would hide account draining.

**Cost.** Fixed histograms and rolling updates are bounded per event; support-aware peer aggregates are small. Compute only for entities with usable histories. Do not materialize an account-by-minute dense matrix.

**Risks.** Sparse entities dominate many transaction datasets; personal references may be unavailable. October's festival spending can shift legitimate amounts. Label-free monthly peer normalization is permitted but should be simulated identically in each validation month and compared with a historical-only version. A large distance indicates change, not fraud class.

**Minimal experiment.** Add approximately 10-20 supported change features, not hundreds. Track benefits for velocity_bot, refund_abuse, loan_app_extortion, and merchant_collusion. Check seen versus new entities and support-count strata.

## 4. Short symbolic sequences first; template alignment only if needed

**Priority: fourth. Evidence: A for alignment mechanics; C for fraud utility.**

DTW aligns multivariate sequences, and a Sakoe-Chiba band constrains how far the alignment may warp. Unconstrained pairwise DTW uses quadratic time in sequence lengths. [tslearn DTW documentation](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html) Shapelet methods turn short discriminative subsequences into distance features, with documented examples in STUMPY. [STUMPY shapelet tutorial](https://stumpy.readthedocs.io/en/stable/Tutorial_Shapelet_Discovery.html)

**Proposed first version.** Encode recent events with direction, transaction type, relative amount bucket, gap bucket, and whether the counterparty repeats. Compute a small fixed set of two- and three-event pattern counts plus position within a burst. Examples to test include repeated same-size merchant payments, repeated refund claims across merchants, repeated evening payments to one receiver, and small reverse-direction payment followed by a larger return payment. Existing fake-cashback pair features already cover the last pattern; only extend them if multi-event context adds value.

**Optional second version.** For one weak class family, extract 8-16-event sequences and compare them with a small frozen bank of training-only prototypes, including legitimate lookalikes. Use event index as the sequence axis and include log elapsed time as a channel; irregular payments should not silently become uniformly sampled time series. Keep raw and relative amounts. Use a narrow warp band and features for minimum class-prototype distance and the margin over legitimate-prototype distance.

**Cost.** Symbolic patterns are linear after sorting. A bounded prototype bank costs approximately `O(C*P*L*b)` for candidate count `C`, prototype count `P`, sequence length `L`, and warp-band width `b`, using an implementation that actually restricts visited cells. Restrict comparison to one candidate family; global all-pairs sequence matching is unnecessary. Prototype selection and any normalizers must be fitted inside each training fold.

**Risks.** Templates selected from noisy labels can encode mistakes. Excessive warping can make distinct fraud types appear similar. Templates from a repeated entity can leak identity through validation. Use grouped episode evaluation as a diagnostic and retain a simpler symbol-only baseline.

**Minimal experiment.** Run symbolic patterns first. Try at most one template family only if class-level residual errors still suggest repeated sequences that the trees fail to separate.

## 5. Class specialists and cautious learning from noisy labels

**Priority: after structural features; standard PU is not a default. Evidence: A for assumptions; C for proposed weighting.**

Cleanlab requires out-of-sample predicted probabilities for label-issue detection; training-set confidence is unsuitable. Its standard workflow estimates label issues before pruning and retraining. [Cleanlab cross-validation guidance](https://docs.cleanlab.ai/stable/tutorials/pred_probs_cross_val.html), [Cleanlab classification API](https://docs.cleanlab.ai/stable/cleanlab/classification.html)

**Proposed experiment.** Train a global model first, then a small number of binary or family-specific specialists for persistent confusions: mule/layering, sim_swap/account_takeover, and a weak merchant/refund family. Give specialists the relevant structural features plus diverse legitimate lookalikes. Avoid hard rule gates that prevent a specialist from recovering an unusual positive; if computational screening is needed, measure its held-out recall separately.

Generate disagreement scores with rolling time splits, keeping the outer validation labels untouched. A candidate high-confidence anchor needs agreement between an out-of-fold model and independent structural evidence, but remains a **noisy anchor**, not known truth. Compare unchanged training with capped downweighting of strongly disputed rows. Retain a nonzero minimum weight and class support. Do not automatically delete every difficult minority example or relabel entire connected components. Any specialist blending or class offsets must be fitted on inner out-of-fold predictions, with an untouched outer month for selection.

**Why ordinary PU assumptions do not automatically hold.** Elkan and Noto assume labelled positives are truly positive and selected randomly from the positive class for their constant probability correction. [Original PU paper](https://cseweb.ucsd.edu/~elkan/posonly.pdf) This dataset explicitly has wrong fraud types, so a reported `mule_account` is not necessarily a clean positive for a mule-versus-rest task. Selecting only easy motif-consistent positives also creates feature-dependent selection. nnPU addresses negative empirical risk and overfitting; its paper assumes samples from the positive marginal and a known class prior, so it does not remove these identification problems. [nnPU paper](https://papers.nips.cc/paper_files/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)

**Conservative optional PU branch.** Collapsing fraud types to binary fraud reduces within-fraud type contamination only if errors do not also introduce legitimate false positives; the statement does not establish that. If testing PU, treat positives as potentially contaminated, vary plausible class priors and positive-sample weighting, and keep results as a sensitivity experiment. Do not call its outputs calibrated fraud probabilities. Do not impose a hidden fraud prevalence derived from the approximate 96% statement or public leaderboard feedback.

**Cost and risk.** Cross-fitting is the major cost; reuse existing held-out predictions where valid. For 1.824 million rows and 12 classes, one float32 probability matrix alone is about 88 MB, before metadata and model memory. Model agreement can amplify shared bias, especially under festival drift. Ordinary noisy-label validation cannot certify clean-test precision.

**Minimal experiment.** Compare baseline, specialists only, and specialists plus modest disagreement downweighting. If all improvement depends on aggressive relabelling or a finely tuned class prior, deprioritize it.

## Evaluation contract for the revised plan

1. Use at least two rolling time checks where data permits; freeze windows, prototypes, weights, and offsets on earlier data before later confirmation. Existing EDA and planning diagnostics have already inspected aggregate September results, so do not describe the historical September data as previously unseen. Limit future reuse as specified in the revised plan.
2. Reproduce the allowed label-free forward context in each held-out month and stop it at the simulated observation boundary. Add a feature for available forward horizon near month end. Structural features use no labels. Optional historical label features require strictly earlier training labels and must exclude all held-out labels.
3. Save per-class precision, recall, F1, predicted prevalence, and confusion, together with incoming/onward/first/last-leg diagnostics. Bootstrap by bounded episode or entity group when practical; row bootstrap understates dependence.
4. Remember that validation labels are noisy: report scores against supplied labels as such. Supplement with stability across months and structurally supported diagnostics; do not pretend a cleaned validation set is ground truth.
5. Use one feature-block ablation at a time, cached typed arrays, and one fixed model configuration initially. Record wall time, peak memory, and candidate truncation rates. Promote complexity only after a repeatable benefit.
6. Do not optimize public submissions as a search loop. These experiments can justify an improved candidate; they cannot establish a promised public score before evaluation.

