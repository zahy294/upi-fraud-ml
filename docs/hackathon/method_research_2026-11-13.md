# UPI fraud methods: evidence and next experiments

Researched 2026-11-13. Scope: primary sources, the supplied problem statement, EDA, and baseline code; no training or implementation. Published IEEE-CIS results below concern **binary ROC-AUC**, not this competition's 12-class macro-F1. Their methods are candidates to test, not evidence of a particular UPI score gain.

**Subsequent local evidence:** the parent investigation ran twelve fits and found a stable gain from probability-gated rules, while the tested broad historical-label pack failed. The [measured evidence](../../reports/research/CANDIDATE_EVIDENCE.md) and [implementation plan](../SCORE_IMPROVEMENT_PLAN.md) supersede the literature-only priority order below. The scored CSV is now available as `../s1_v1.csv` relative to the repository root, but its generating run is not yet identified.

**Recommendation:** first recover the reproducible pipeline and probabilities behind the reported 0.5034 submission. Then test joint class-specific decision tuning, richer label-free entity/session summaries, and carefully isolated historical label features. Prediction propagation and label cleaning come later. The leader's reported 0.5487 does not establish which method is missing.

## What the local evidence actually supports

- The [problem statement](../PROBLEM_STATEMENT.md) explicitly allows connections built from unlabelled test transactions. Thus future transactions within a held-out month may inform **label-free** features in this offline task. That permission does not make validation labels available for feature construction.
- [Baseline code](../../scripts/eda/07_baselines.py) tunes one common fraud multiplier over seven values. Its reported 0.391 relational model is not the unavailable 0.5034 pipeline. Repeating an already-used feature family would not necessarily improve the latter.
- The [EDA](../../reports/eda/EDA_REPORT.md) reports 0.339 → 0.391 when relational features are added. This is direct local evidence for that feature family, subject to reproducing its validation and feature alignment.
- **0.454 is not a score ceiling.** The code labels only true positives hit by manually selected signatures and treats other rows as legitimate. That evaluates those signatures with perfect precision, not all information in the data. Weak univariate separation also cannot rule out multivariate signal.
- **94–95% event purity is not independent propagation accuracy.** Label-selected seed transactions are included in these neighborhoods. Recompute precision on added, non-seed transactions and report incremental recall, false positives, number of events, and singleton rate before claiming value.
- The current investigation reports that eight forward-flow candidates recovered no extra layering coverage over the nearest candidate, and aggregate-payout matches were nearly empty. This is a session finding, not independently reproduced here; deprioritise further variants until a concrete missed pattern appears.

## What strong primary sources establish

| Source | Published finding | Transfer limit |
|---|---|---|
| [McDonald and Deotte's first-place IEEE-CIS explanation](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/) | Frequency encodings and entity group statistics were important. They summarize amount and other variables within reconstructed identities. The final solution combined XGBoost, CatBoost and LightGBM. | UPI already supplies identities; copying the anonymous-card reconstruction is unnecessary. Model variety alone is not evidence of complementary errors. |
| [Kaggle's hosted adaptation of the winners' writeup](https://www.kaggle.com/solution-write-up-documentation) | The winners used aggregate features to generalize across identities, checked features across time, and evaluated known versus unseen clients. Averaging predictions within clients helped their submission. | UPI accounts can participate in legitimate activity and several fraud types. A permanent entity-level fraud label is unjustified. |
| [CPMP's second-place author writeup, including aggregation code](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/2-uncles-and-3-puppies-2nd-solution-cpmp-view) | Entity counts, next-transaction gaps, gap statistics, next amounts and amount statistics improved his public AUC from about 0.942 to 0.952. Entity/chain target encodings subsequently reached about 0.9606. He evaluated multiple forward-time splits. | His lagged-average prediction stacking increased CV but reduced leaderboard AUC by 0.01. Reusing predictions can fail even when internal scores look persuasive. |

Original first-place posts: [part 1](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284), [part 2](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308). Winner-owned notebooks: [XGB Fraud with Magic](https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600), [RAPIDS feature engineering](https://www.kaggle.com/cdeotte/rapids-feature-engineering-fraud-0-96/). These notebook locations were verified, but their executable cells were not readable through the research interface; the directly inspected code is in CPMP's writeup. The NVIDIA page was read beyond its AI-generated summary. Do not copy its suggestion that month-grouped `GroupKFold` automatically enforces forward-only training: fold membership must be checked explicitly.

## Ordered experiment queue

### 1. Joint macro-F1 decisions on saved probabilities

**Evidence:** [Lipton, Elkan and Narayanaswamy](https://arxiv.org/abs/1402.1892) show that F1-optimal decisions depend on the score distribution; the familiar threshold of half the achievable F1 applies to calibrated **binary** probabilities. It is not an independent threshold rule for mutually exclusive multiclass predictions. The [scikit-learn threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html) separates probability estimation from decision selection and warns against tuning on the classifier's training data.

**UPI experiment:** predict `argmax_c(log(max(p_c, epsilon)) + b_c)`, fixing `b_legit = 0`. Start from the existing shared multiplier, then coordinate-search a small, bounded grid of eleven class offsets against macro-F1. Recompute the whole 12-class confusion matrix for every change, because one class takes predictions from others. This is a practical search, not a globally optimal F1 algorithm.

Use July-trained August predictions for initial tuning and July–August-trained September predictions for a later check; also inspect early/late September sensitivity. Report per-class precision, recall, support and prediction counts. Prefer stable ranges over a narrow best point. Refit probability models without retuning on the public leaderboard. If 0.5034 already used this method, move on rather than assuming another threshold search closes the gap. Uncalibrated scores are acceptable for empirical offset search; do not interpret the resulting scores as posterior probabilities.

### 2. Label-free entity and event context in the existing GBDT

**Transfer hypothesis from the winner evidence:** retain the existing row features and add context that distinguishes an unusual transaction from an unusual entity. Suggested compact families:

| Group | Candidate summaries |
|---|---|
| Sender, receiver, device | Previous/next time gap; count and distinct counterparties over 5 minutes, 1 hour, 1 day and 7 days; amount median/spread; current amount relative to group median; short-window share of longer-window activity |
| Device × merchant category, receiver × transaction type | Burst size, distinct senders, transaction-type mix, new-payee fraction, amount dispersion and gap dispersion |
| Ordered sender–receiver pair | Recurrence gaps, reverse-direction occurrence, amount rank within pair, refund fraction and round-amount repetition |
| Bounded receiver/device session | Duration, transaction position, distinct participants, sum/median amount, proportions of row-level signatures |

Preserve both raw counts and relative measures; normalization is not automatically robust when its denominator is tiny. Use missingness/support indicators and explicitly separate P2P recipients from merchants. Build windows from sorted timestamps, not shuffled file order, and align joins by `txn_id`. Use identical availability rules for validation and test; omit October from September simulation. Short-window context can use the entire unlabelled held-out month because test connections are allowed.

Avoid unlimited connected components: a popular merchant or a sequence of overlapping windows can merge unrelated events. Bound duration as well as adjacent gaps. Test feature families individually and compare both August and September; a feature is not useful merely because its training gain importance is large. Given the reported flow-search result, prioritize contextual summaries over another broad money-matching pass.

### 3. Historical class rates, only with faithful label availability

**Evidence:** [scikit-learn's executable target-encoding example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html) demonstrates overfitting when a category's training target mean includes the encoded observations. [The CatBoost paper](https://arxiv.org/abs/1706.09516) motivates ordered statistics by target leakage and prediction shift. Neither establishes that randomly cross-fitted encodings respect this competition's monthly prediction setting.

**UPI experiment:** for each entity and class, use a smoothed historical rate `(class_count + alpha * prior) / (history_count + alpha)`, accompanied by support, recency and unseen-entity flags. Start with receiver/device and compare sender separately. The prior must also use only the permitted historical labels.

Construct monthly snapshots: August features use July labels; September uses July–August labels. Freeze those lookups throughout the held-out month. Training rows need analogous earlier-block histories; never map the full training target mean back onto the same training rows. No September label may reach any component producing September predictions, including a teacher model, relabeling rule or second-stage feature. Handle timestamp tails explicitly when defining folds. An entity's previous fraud type is a fallible feature, not an override. Sparse entities, reused victims and fraud-type changes may erase the gain.

### 4. Event prediction aggregation after the preceding checks

**Transfer hypothesis, not verified event purity:** for a narrow label-free session, add the mean/max of neighboring class probabilities and neighbor count to a small second-stage model, or test a conservative interpolation with the row probabilities. Exclude the row itself when measuring neighbor evidence. Keep singleton rows unchanged and compare receiver sessions versus device sessions separately.

Every probability supplied to a training second stage must be genuinely out of fold. For outer September validation, all first-stage training predictions must be regenerated inside July–August; a globally produced OOF table can still leak September labels into the other rows used to fit the second stage. Score additions beyond the first-stage predictions, not the seed-inclusive event purity. Reject propagation if minority-class false positives outweigh new correct predictions. The published CPMP failure above makes this a lower-priority ablation.

### 5. Label-noise handling as a sensitivity experiment

**Evidence:** [Northcutt, Jiang and Chuang's Confident Learning paper](https://arxiv.org/abs/1911.00068) estimates noisy/true-label relationships under a class-conditional noise model. [Cleanlab's API documentation](https://docs.cleanlab.ai/stable/cleanlab/filter.html) recommends out-of-sample probabilities for identifying label issues. The assumptions and trustworthy class probabilities are not established for this synthetic, highly imbalanced dataset.

First rank suspicious rows using temporal OOF disagreement plus independent label-free event evidence. Compare unchanged training with a small, bounded downweighting of only high-confidence contradictory rows. Preserve all original labels and keep a per-class audit of affected counts. Avoid wholesale relabeling, deleting all hard minority examples, or treating a model's confidence as proof of error. These actions can remove the cases the model most needs to learn.

September labels are themselves noisy, so local F1 cannot settle every correction intended for clean October labels. Keep this limitation explicit; no clean-validation score is available. The existing 4–5% legitimate labels in seed-selected bursts do not prove all are annotation mistakes.

## Decision rule for the remaining competition time

Use the same reproducible baseline, folds and saved probabilities for each comparison. Record class-level changes, temporal stability and unseen-entity behavior. Try one genuinely different GBDT only after the feature/decision experiments, and blend only if its held-out errors complement the baseline. Do not replace event reasoning with a new GNN or long sequence model without evidence that the current features miss accessible structure. None of these sources supports a guaranteed gain from 0.5034 to 0.5487; the next useful result is an isolated, repeatable improvement over the actual submitted pipeline.
