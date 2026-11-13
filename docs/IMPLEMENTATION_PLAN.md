# Implementation plan: training the UPI Fraud Rings model

Written 2026-11-13, 11:30 IST (round day 1, about 46 hours left). Builds on [the EDA report](../reports/eda/EDA_REPORT.md), [the problem statement](PROBLEM_STATEMENT.md) and [the rules analysis](hackathon/analysis.md). Read those for the data findings; this file only says what to build, in what order, and why.

Expected gains in this file are my estimates from the EDA numbers and the research below. They are not measurements. The only measured scores so far are the EDA baselines (rules 0.392, LightGBM + relational features 0.391, both on September).

---

## 0. Summary

**What we optimise.** The score is macro-F1 over 12 labels on the private split, not accuracy. Predicting `legit` everywhere is 96% accurate and scores 0.08. Every decision below is judged by macro-F1 on a time-based validation month.

**What wins on this data.** Every source I found agrees with the EDA: on transaction fraud with relational structure, the score comes from features built across rows, a validation scheme that matches the test month, and a decision rule tuned for the metric. Model family matters much less. Gradient-boosted trees with graph-pattern features beat graph neural networks on the closest public benchmark, and nothing in the 2026 literature changes that for a 48-hour round.

**The build, in order.** Each step is a complete, uploadable pipeline. Stop wherever the clock runs out.

| # | Step | Est. September macro-F1 | Effort |
|---|---|---|---|
| – | Today's baselines (rules only / LightGBM + relational) | 0.392 / 0.391 (measured) | done |
| 1 | **v1:** rule flags and profile flags as features, multiclass GBDT, per-class decision weights | 0.42–0.43 | 2–3 h |
| 2 | **v2 features:** multi-window bursts, chain depth, drain bursts, entity profiles, merchant concentration, then a time-consistency filter | +0.01–0.02 | 4–6 h |
| 3 | **Label cleaning:** relabel unreported fraud inside events, fix wrong-type labels | +0.002–0.008 (on clean labels) | 1–2 h |
| 4 | **Stage 2:** second model that sees the first model's predictions for each row's event neighbours | +0.01–0.02 | 3–4 h |
| 5 | **Ensemble:** XGBoost + LightGBM + CatBoost, several seeds, refit on Jul–Sep | +0.003–0.01 | 2 h, mostly machine time |
| 6 | **October hedges** for `velocity_bot` and `loan_app_extortion`: rate-matched thresholds, self-training on test | unknown on September; the leaderboard decides | 2 h |

A realistic target is **0.44–0.46 on September**. The EDA's soft ceiling is 0.454 for a perfect detector of signature rows, and the organisers say nobody is expected to reach 0.50.

**What not to build during the round:** a GNN, a sequence transformer, a cross-encoder, SMOTE or other resampling, or a tabular foundation model as the core model. Section 2 gives the evidence for each. They are listed as future iterations in section 9.

**Do this first (10 minutes):** if the rules-only file has not been uploaded yet, upload `submissions/rules_v0_ordered.csv` as a prediction upload on the competition site. It is `rules_v0.csv` with rows put in the same order as `sample_submission.csv`, in case the site scores by row position instead of by id. Its leaderboard score is the first measurement of how September validation transfers to October (section 4, gate 0).

---

## 1. Constraints that shape the plan

From the rules ([details](hackathon/analysis.md)):

- Round closes **15 Nov 10:00 IST**. Ten uploads per day, resetting at midnight IST, so three upload days.
- **The best public upload becomes the final entry**, scored on the private 70%. You cannot pick. Public is about 78k rows with roughly 100–400 positives per fraud class, so a public difference under about 0.01 is noise.
- The code link (GitHub or notebook) must be submitted before the close, and winners must reproduce their best upload from it.
- Train only on the provided data. Open-source libraries and public pretrained weights are allowed. No hand-labelling of test rows; rules written in code are fine.
- Sharing code or predictions between teams means disqualification. Keep the GitHub repo **private** until you submit the link, and don't share the Colab notebook or Drive folder.

From the EDA ([report](../reports/eda/EDA_REPORT.md)):

- Fraud is 3.69% of rows. Each fraud type is about 25% rows with a crisp signature and about 75% rows that look like ordinary payments and link to nothing. Recall is capped near 25–35% per class, so **precision is the lever**.
- The signal is relational: receiver bursts, money forwarded within 30 minutes, device bursts, reverse pairs, repeat refunds. Events are 94–95% pure in label.
- `app_version` and `ip_risk_score` are traps. Drop them.
- October has half the rows per day, with fraud events intact. `velocity_bot` and `loan_app_extortion` signals fire at about half the September rate in October.
- Training labels are noisy (about 4–5% of rows inside bursts are labelled `legit`; some fraud is filed under the wrong type). Test labels are clean.

**Checked today, nothing there.** I tested whether fraud rows break cross-row consistency, which is where synthetic datasets often leak. They don't. Balances are not a running ledger for any label, SIM and device ages are not continuous across a sender's rows for any label, the account-age column matches `accounts.csv` for every label, and `txn_id` is random. The best single signal for off-pattern fraud is still "whole-rupee amount" at AUC 0.555. Don't spend time looking for a simulator tell. One small find: `balance_before < amount` holds for 0.01% of legit rows, 1.2% of off-pattern fraud and 8% of on-pattern fraud. Add it as an explicit flag (the model partly has it through `amt_to_balance`).

---

## 2. What the research says

I read competition write-ups, papers, and practitioner threads on Kaggle, Reddit, LinkedIn, Hugging Face and vendor blogs. Searches on X returned only pointers to the same NVIDIA and Stripe material, with no independent practitioner threads worth citing. Links are in section 10.

| Source | What they found | What we take from it |
|---|---|---|
| **IEEE-CIS Fraud Detection, 1st place** (Deotte & Yakovlev, Kaggle 2019) | The task was really about entities, not rows. They rebuilt a client ID, then **never fed the ID to the model**: only group aggregates over it, so the model generalises to unseen clients. They ran a "time consistency" test (train on the first month with one feature group, score the last month) and dropped about 5% of features. They post-processed predictions per entity. XGBoost, LightGBM and CatBoost each won on different subsets, and the blend won overall. | Aggregate over sender, receiver, device and pair. No raw IDs as features. Run the time-consistency test on every feature group (step 2). Post-process per event (step 4). Blend three GBDT libraries (step 5). |
| **TalkingData AdTracking Fraud, 1st place** (Kaggle 2018) | Features computed on the **full data**, not the down-sampled training rows. Brute-force counts and "time to next / previous event" for every key combination. Trained on all positives plus down-sampled negatives, bagged over five samples, which kept each fit under 30 minutes with hundreds of features. Picked the tree count on a later day, then refit on everything with that count. | Time-delta features for each key. Down-sample `legit` with bagging when fits get slow. Same refit recipe for the final model. |
| **IBM Graph Feature Preprocessor** (Blanuša et al., ICAIF 2024) and **AMLworld** (Altman et al., NeurIPS 2023) | On synthetic laundering data with 0.1% positives, plain LightGBM scored 21% minority-class F1. Adding fan-in/fan-out features gave about 50%, and adding vertex statistics and cycle/scatter-gather features gave 63%. That beat the GIN and PNA graph networks. Windows were 6 hours and 1 day. | Graph-pattern features feeding a GBDT are the right architecture. Fan-in and fan-out carry most of the gain; multi-hop patterns add a few points. |
| **Multi-GNN** (Egressy et al., AAAI 2024) | GNNs only catch up with trees on this task after adding reverse message passing, port numbering and ego IDs. | A GNN needs careful, task-specific design. Future iteration only. |
| **r/MachineLearning thread, May 2026** | A team's GCN/GraphSAGE/GAT on IEEE-CIS reached PR-AUC 0.52; they confirm "boosted tree is heavily outperforming our GNN". Commenters point to over-smoothing and graph construction, and say paper numbers are often 2–20% inflated. | Same conclusion from practitioners. |
| **fraud-synth-bench** (one author's LinkedIn write-up, synthetic data, Oct 2026) | LightGBM with shared-device, fan-in and component-size features was never beaten by GraphSAGE or FT-Transformer. When the simulated fraudsters switched tactics, every supervised model lost 88% or more of its detection quality until it was refit. | Weak evidence on its own, but it matches the rest. The second finding is the October risk for our two drifting classes (step 6). |
| **NVIDIA Kaggle Grandmasters playbook** (2025) | Fast experiments and trustworthy validation first. Then many features, hill-climbing blends, stacking, pseudo-labelling with soft labels (which can also clean noisy training labels), multiple seeds, and a refit on 100% of the data. | Steps 3–6 follow this order. |
| **Lipton, Elkan & Narayanaswamy 2014** | For calibrated probabilities, the F1-optimal threshold is half the best achievable F1. | With per-class F1 around 0.3–0.5, sensible thresholds sit near 0.15–0.25, far below 0.5. Use this as a sanity check on tuned thresholds. |
| **Wang, Li, Yan & Koyejo 2019** | Optimising a metric built from the confusion matrix is equivalent to learning a weighted classifier, and it can be done as a post-processing step on predicted probabilities. | The decision rule in step 1: `argmax(weight[c] * prob[c])` with weights tuned on validation. |
| **Elor & Averbuch-Elor 2022, "To SMOTE, or not to SMOTE?"**; an r/DataScientist benchmark of nine imbalance techniques | Balancing the data does not help strong classifiers once the threshold is tuned. In the benchmark SMOTE-ENN finished sixth and a mild class weight won. | No resampling for balance. Unweighted (or mildly weighted) training plus tuned decision weights. |
| **Label noise in GBDTs** (Eisenbürger et al. 2024; Ponti et al. 2022; Robust-GBDT 2023) | Noisy labels can be found from out-of-fold predictions or training dynamics, then relabelled or removed. | Step 3. |
| **Transaction foundation models** (Sardine 2026; Stripe, Nubank, Revolut, NVIDIA blueprint) | Sequence-model embeddings fed to XGBoost caught 24–35% more card-issuing fraud. Sardine expects little gain where accounts have short histories and rich per-event features. | Our senders average 2.4 rows a month. Skip. |
| **Tabular foundation models** (TabArena 2026; TabPFN-3.5; TabICLv2) | In-context models now lead small and mid-size benchmarks. TabPFN-3.5 accepts up to 1M rows and has its own licence. TabICLv2 is BSD-3. | Possible ensemble member on a down-sampled candidate set. Unproven at our size and imbalance. Stretch only. |
| **Kaggle Playground S6E11** (this month's card-fraud playground; one public experiment log) | GBDT with value encodings, then a logistic stacker: AUC 0.9452 → 0.9509. | Stacking adds a little. Do it last. |

### The Reddit cross-encoder idea

The quote describes a retrieve-then-rerank design for text: TF-IDF and LightGBM narrow the candidates ("locking the keys"), then a cross-encoder with a classification head reads each candidate pair. That fits text matching, which is what the Amazon ML entity-resolution task was. **It does not apply here as written:** this dataset has no text, only numbers, categories and IDs, and a pretrained language cross-encoder has nothing to read.

The part that transfers is the cascade. A cheap first stage scores all 1.8M rows; a heavier second stage looks only at the few percent that are candidates, together with their context. Step 4 is exactly that, with a GBDT as the second stage. The neural version (a small transformer that reads a candidate row together with its neighbour rows) is the first item in section 9.

---

## 3. Architecture

```
data/interim/*.parquet (train + test together, labels never used for features)
        │
        ▼
 FEATURES  row flags · relational windows · per-type packs · rule flags · entity profiles
        │   └─ time-consistency + adversarial filter drops what drifts
        ▼
 STAGE 1   multiclass GBDT (XGBoost GPU / LightGBM / CatBoost), month-wise out-of-fold
        │   └─ label cleaning uses these out-of-fold probabilities, then stage 1 is refit
        ▼
 STAGE 2   GBDT on: stage-1 probs of the row + sums/means of stage-1 probs over its
        │   event neighbours (same receiver ±1h, device ±5min, sender ±24h, pair, money flow)
        ▼
 BLEND     weighted average of model probabilities (weights by hill-climbing on OOF)
        ▼
 DECIDE    label = argmax(weight[c] · prob[c]); weights tuned for macro-F1 on OOF;
        │   optional rate-matching for the two drifting classes
        ▼
 submissions/<name>.csv  + one line in submissions/LOG.md
```

Code layout. Keep it this small; the Amazon run died of over-engineering.

```
src/upi/
  events.py     Events class from scripts/eda/04_relational.py, plus sum / gap / index lookups
  features.py   every feature builder; writes data/processed/features_v{N}.parquet
  cv.py         month folds and the two validation folds
  model.py      fit/predict wrappers for xgb | lgb | cat, with a --device flag
  noise.py      relabel / drop rules
  stage2.py     neighbour aggregates of OOF probabilities, stage-2 fit
  decide.py     class-weight search, rate matching, submission writer and checker
run.py          one entry point:  python run.py --stage features|stage1|stage2|decide|all --config configs/v1.yaml
configs/        one small YAML per uploaded version
notebooks/colab_train.ipynb   thin driver: mount Drive, clone repo, call run.py
requirements.txt
```

Reuse, don't rewrite: `add_row_features` in [scripts/eda/common.py](../scripts/eda/common.py), the `Events` class and all 33 features in [scripts/eda/04_relational.py](../scripts/eda/04_relational.py), and the `RULES` and `SIG` masks in [scripts/eda/07_baselines.py](../scripts/eda/07_baselines.py).

---

## 4. Validation protocol

This matters more than any model choice. Three rules.

**Rule 1: validate forward in time, on whole months.**

| Fold | Fit on | Score on | Use |
|---|---|---|---|
| A (primary) | Jul + Aug | Sep | every decision |
| B (check) | Jul | Aug | confirm any gain between 0.003 and 0.006 on fold A |
| OOF (for stacking and thresholds) | the other two months | each of Jul, Aug, Sep in turn | stage-2 inputs, blend weights, decision weights |

The 147 October-dated training rows are tails of September events; put them in the September fold. Never use random K-fold: rows of one event would land in different folds and the model would recognise the event from near-identical features, which inflates every score.

September holds about 19k fraud rows (645 to 3,370 per class). Treat fold-A differences under **0.003** as noise.

**Rule 2: report two scores, raw and cleaned.** The September labels carry the same noise as the training labels, but October's are clean. A model that correctly flags unreported fraud inside a burst is penalised on raw September labels. So compute macro-F1 twice: against the raw labels, and against labels with the step-3 cleaning rules applied to September. Adopt a change when the cleaned score improves and the raw score does not fall by more than 0.003.

**Rule 3: the public leaderboard calibrates, it does not select.**

- **Gate 0 (first upload, `rules_v0.csv`, September 0.392).**
  - Leaderboard within about ±0.03 of 0.39: validation transfers. Proceed as planned.
  - Far higher (above 0.50): October has much less ordinary-looking fraud than the training labels suggest. Tune decision weights on the cleaned-label score and lean toward recall.
  - Far lower (below 0.33): drift is larger than the EDA measured. Compare predicted counts per class between September and October before building more.
- After that, upload only models whose fold-A score is the best so far or within 0.005 of it. Any upload can become the final entry, so every upload must be one you'd accept.
- **Safe single-class probes.** To measure one class on October, upload your best file with that class's predictions changed to `legit`. The score drops by about that class's public F1 divided by 12. Since the probe scores below the file it was made from, it will not become your final entry. Budget two of these on day 2, for `velocity_bot` and `loan_app_extortion`.

**Acceptance rule for any change:** fold-A cleaned macro-F1 up by at least 0.003, no single class down by more than 0.02, and the per-class predicted counts on October within a factor of 1.5 of the September rate (the class mix is stable to ±0.03 percentage points month to month, so a big swing in predicted counts means drift, not luck).

Log every experiment in `reports/experiments.md`: date, config, fold-A raw and cleaned score, per-class F1, decision.

---

## 5. Build steps

### Step 0. Setup (45 minutes)

1. `git init`, first commit, push to a **private** GitHub repo. The `.gitignore` already excludes data, models, submissions and `kaggle.json`.
2. Upload `data/interim/*.parquet` and `data/processed/relational_v0.parquet` (about 150 MB in total) to a Drive folder, e.g. `MyDrive/upi-fraud/data/`, keeping the `interim/` and `processed/` subfolders. Do not upload `kaggle.json`.
3. Move the `Events` class and the feature code into `src/upi/`, and get `python run.py --stage all --sample 0.02` running end to end on the laptop. Sample **2% of senders**, not 2% of rows, so events stay whole.
4. Run the same command in Colab (section 6) before any real run. The Amazon post-mortem's main lesson: a two-minute smoke test on the target machine would have saved the run.

Done when: the sampled pipeline writes a valid submission file on both machines.

### Step 1. v1 model (2–3 hours) → upload

**Features.** Model C's columns from the EDA (row features, categoricals, the 33 relational features) plus:

- One 0/1 column per rule in `RULES` and per signature in `SIG`, and a count of how many fire.
- Profile flags: whole-rupee amount, new payee, P2P, not the home phone, SIM under 168 h, device under 7 days, just under a round thousand, round ₹500 and ₹1,000, ₹10 or less, exactly ₹500, `balance_before < amount`.
- Drop `app_version`, `app_major`, `ip_risk_score`. Never use `sender_id`, `receiver_id`, `device_id` or `txn_id` as features.

**Model.** Multiclass, unweighted. Starting points:

```python
# XGBoost on the Colab GPU (primary)
xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", device="cuda",
                  n_estimators=1500, learning_rate=0.05, max_depth=8, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.5, max_bin=256, reg_lambda=5,
                  enable_categorical=True, early_stopping_rounds=100, random_state=seed)
# LightGBM on the laptop CPU (the EDA setting, slower learning rate)
lgb.LGBMClassifier(objective="multiclass", n_estimators=1500, learning_rate=0.03, num_leaves=63,
                   min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                   reg_lambda=5, n_jobs=12, random_state=seed)
```

Early stopping on log-loss does not pick the best tree count for macro-F1. Save probabilities at several tree counts (for example every 200) and choose the count by fold-A macro-F1 after the decision weights are fitted.

**Decision rule.** Predict `argmax(w[c] * prob[c])` with `w[legit] = 1`. Fit the eleven fraud weights by coordinate ascent on validation probabilities:

```python
def macro_f1(y, pred, k=12):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm); f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return f1.mean()

def fit_weights(P, y, sweeps=4, grid=np.geomspace(0.1, 60, 45)):
    w = np.ones(P.shape[1])
    for _ in range(sweeps):
        for c in range(1, P.shape[1]):
            s = np.array([macro_f1(y, (P * np.r_[w[:c], g, w[c+1:]]).argmax(1)) for g in grid])
            plateau = np.where(s >= s.max() - 0.0005)[0]
            w[c] = grid[plateau[len(plateau) // 2]]      # middle of the plateau, not the spike
    return w
```

This runs in under a minute on September. If a weight lands on the edge of the grid, the score is flat for that class; cap the weight rather than trusting it.

Guards against overfitting the weights: take the middle of the plateau as above; once month-wise OOF exists (step 4), fit on the pooled OOF of all three months, which triples the positives; and check that each class's implied threshold is in the range the Lipton result predicts.

Also test one variant: after the model decides, let the three rules with over 80% September precision (`refund_abuse`, `velocity_bot`, `loan_app_extortion` in `RULES`) override the model. Keep it only if fold A improves.

**Final fit and file.** Refit on Jul–Sep with the chosen tree count × 1.15 (half again as much data), three seeds averaged, same weights. The submission writer must assert: same ids in the same order as `sample_submission.csv`, labels spelled exactly as in the statement, **every one of the 12 labels predicted at least once** (an empty class costs 0.083), file under 15 MB.

Done when: fold A reaches about 0.42 and the file is uploaded.

### Step 2. v2 features (4–6 hours) → upload

Add two methods to `Events` first: a windowed **sum** (prefix sums over the sorted values) and the **gap to the previous / next event** in the group.

```python
# in __init__:  self.cs = np.r_[0.0, np.cumsum(self.v)]  when vals are given
def sum(self, qg, qt, lo, hi):
    b = qg.astype(np.int64) * SCALE + qt + OFF
    r = np.searchsorted(self.K, b + hi, "right"); l = np.searchsorted(self.K, b + lo, "left")
    return np.where(qg >= 0, self.cs[r] - self.cs[l], 0.0)
```

Build features in packs so each can be ablated on its own. All are label-free and computed on train and test together.

| Pack | Aimed at | Features |
|---|---|---|
| **Time deltas** | all | Seconds to the previous and next event for sender, receiver, device and pair. Minimum and median gap within ±1 h. |
| **Multi-window bursts** | `velocity_bot`, `mule_account` | Counts and amount sums at ±1 min, 5 min, 1 h, 6 h, 24 h, 7 d for device, sender and receiver. Ratios between windows (5 min ÷ 24 h) so the model sees burstiness, not volume. Distinct merchants and merchant categories per device per hour and per day. |
| **Fan-in / fan-out** | `mule_account`, `qr_swap`, `phishing_collect` | Distinct senders per P2P receiver per hour, 6 h and day; share of them that are new payees; the same restricted to `COLLECT`, to `QR`, and to `qr_vpa_mismatch = 1`. Receiver's outgoing count and amount in the following 30 min, 2 h, 24 h. Pass-through ratio: money out ÷ money in over 24 h. |
| **Chains** | `layering_chain` | Chain length forward and backward: follow "receiver sends 80–100% of the amount within 30 minutes" up to five hops in each direction (iterate with the index of the next outgoing event; check the next three events, not only the first). Position in the chain, total decay from first hop to this one. |
| **Drain** | `sim_swap`, `account_takeover` | For the sender within ±24 h: payments out, distinct receivers, amount sum ÷ `balance_before`, share to new payees, share at night. "₹500 then a larger payment within an hour". PIN retries summed over the window. Device seen with this account for the first time. |
| **Pairs** | `fake_cashback`, `refund_abuse`, `merchant_collusion`, `loan_app_extortion` | Seconds between a ₹10-or-less bait and the payback, and the amount ratio. Refunds ÷ purchases per sender and per sender–merchant pair; distinct merchants refunded. Distinct days a pair transacts on; share of those in the late evening. |
| **Entity profiles** | all | Per sender, receiver, device and merchant, over ±15 days rather than the whole dataset: row count, distinct counterparties, median and spread of amount, share whole-rupee, share P2P, share at night. This row's amount relative to the sender's median. For merchants: concentration of senders (share of volume from the top three), share of round-₹500 payments, share of refunds, merchant age. |

Two rules for robustness to October:

1. **No whole-dataset totals unless normalised by exposure.** Entities that appear only in October have one thin month of history. Prefer symmetric windows of 7–15 days and ratios. The existing `*_total` features stay only if they pass the checks below.
2. **`velocity_bot` and `loan_app_extortion` must not depend on one tempo.** Before building their packs, compare the distribution of same-device gaps on non-home phones between September and October, and of evening repeat-pair gaps. If October shows mass at longer gaps, make sure a window covers it. This uses aggregate statistics of test rows to design features in code, which the rules allow.

**Filter what drifts** (the IEEE-CIS winners' test, adapted):

- *Time consistency.* For each pack, fit a small model on July with only that pack and score September one-vs-legit AUC per class. Drop features that look strong on July and weak on September.
- *Adversarial validation.* Train a classifier to tell September rows from October rows using the engineered features. Any feature near the top of its importance list is replaced by a windowed or ratio version, or dropped. The target is an AUC near 0.55, as the EDA reached for row features.
- *Conditional check.* Among rows where any signature fires, compare each feature's distribution in September and October. A feature that shifts there is risky even if it looks stable overall.

Done when: fold A improves by at least 0.01 over v1, and predicted counts per class on October are within 1.5× of the September rate for every class except possibly the two known drifters.

### Step 3. Label cleaning (1–2 hours)

Use month-wise out-of-fold probabilities from step 1 or 2. Apply the same rules to the September labels to produce the "cleaned" validation labels.

| Noise | Rule (start conservative) | Action |
|---|---|---|
| Unreported fraud in a mule burst | Row is `legit`, P2P, and at least two other rows paid the same receiver within ±1 h and are labelled `mule_account`, and the row's OOF `mule_account` probability is at least 0.5 | Relabel to `mule_account` |
| Unreported fraud in a bot burst | Row is `legit`, at least two rows on the same device within ±5 min are labelled `velocity_bot`, OOF probability at least 0.5 | Relabel to `velocity_bot` |
| Refund abusers' other refunds | `REFUND` rows labelled `legit` from a sender with three or more `refund_abuse` rows | Drop from training (weight 0); too ambiguous to relabel |
| Wrong fraud type | Row has fraud label *c*, OOF probability of *c* under 0.1 and of another fraud type over 0.7 | Relabel to the other type |

The EDA puts unreported fraud at about 4–5% of burst rows, a few hundred rows per class. The gain is small but it removes "negative" examples that sit inside events and teach the model to hold back there. Refit stage 1 after cleaning.

Done when: cleaned fold-A score is up and raw fold-A score has not dropped by more than 0.003.

### Step 4. Stage 2: event propagation (3–4 hours) → upload

Events are 94–95% pure, so a row's neighbours say a lot about it. The IEEE-CIS winners got their last gains this way.

1. Produce stage-1 out-of-fold probabilities `P1` for every training row (leave one month out) and for test (average of the three month models, or the full refit).
2. For each neighbour link and each class, compute the **sum and mean of `P1` over the row's neighbours, excluding the row itself**, with `Events(..., vals=P1[:, c]).sum(...) - P1[:, c]`:
   - same receiver within ±1 h (P2P and merchant kept separate)
   - same device within ±5 min and ±1 h
   - same sender within ±24 h
   - same sender–receiver pair within ±7 d, and the reverse pair
   - money flow: the receiver's outgoing rows in the next 30 min, and the sender's incoming rows in the previous 30 min
3. Train the stage-2 GBDT on the stage-1 features, the row's own `P1`, and these aggregates, with the same month folds. Optionally train it only on candidates (any rule fires, or any fraud probability above 0.01, or any neighbour above 0.2), which is a few percent of rows and makes many seeds cheap.
4. Refit the decision weights on stage-2 OOF.

Try the simple version first, since it takes 30 minutes: blend each row's probability with its neighbours' mean, `p' = (1 − a)·p + a·mean`, and tune `a` per link type on fold A. If that already gives most of the gain, the stage-2 model can wait.

A variant to compare once stage 2 works: **decide per event**. Group rows linked by high-probability neighbour links into events, pick one fraud type per event by summing log-probabilities, then include each row if its fraud probability clears a lower threshold than a standalone row would need.

Done when: fold A improves by at least 0.005 over step 3.

### Step 5. Ensemble and final refit (2 hours, mostly machine time) → upload

- Train XGBoost (GPU), LightGBM (CPU) and CatBoost (`task_type="GPU"`, `loss_function="MultiClass"`) on the same features, five seeds each.
- Blend probabilities by hill-climbing: start from the best single model, add the model and weight that most improves the OOF score, stop when nothing improves by 0.0005. Optimise OOF log-loss for the blend, then refit the decision weights on the blended OOF.
- If fits are slow, train on all fraud rows plus a random 20% of `legit`, with a different sample per seed (the TalkingData recipe). Validation rows are never down-sampled, and the decision weights absorb the changed prior.
- One diversity member worth an hour if time allows: eleven one-vs-rest binary models instead of one multiclass model.
- Final models are refit on Jul–Sep with the tree count from validation × 1.15.

Done when: the blend beats the best single model on fold A by at least 0.003. If it doesn't, ship the single model.

### Step 6. October hedges (2 hours, day 2 evening onward)

The rule predictions already show `velocity_bot` at half its September rate on October. The class mix is otherwise stable month to month, so half as many predictions most likely means missed fraud, not less fraud.

1. **Measure first.** Run the two single-class probes from section 4. If either class's public F1 is far below its September value, apply the fixes below to that class only.
2. **Rate-matched threshold.** For a drifting class, instead of the September probability cut, predict the top *n* October rows by probability, where *n* is the class's September predicted share times the number of test rows. Blend halfway between the two cut-offs if unsure. Apply only when the October predicted count falls below 70% of the expected count.
3. **Self-training on October.** Add October rows with fraud-class probability above 0.9 (and a sample of rows with `legit` probability above 0.999) to the training set with their predicted labels, at weight 0.5, and refit. This lets the model learn October's tempo from rows it already recognises by other features. Check it first with September standing in as unlabelled test (fit on Jul–Aug, pseudo-label September, refit, score September): it must not lower the score there.
4. Upload a hedge only if its fold-A score is unchanged and the probe says the class needs it.

### Step 7. Freeze, reproduce, submit

- For every uploaded file, save in Drive and note in `submissions/LOG.md`: file name, git commit, config, seeds, fold-A raw and cleaned scores, leaderboard score, and the `pip freeze` and GPU model of the run.
- GPU training can differ in the last decimals between runs. Keep the saved model files and probabilities for the best upload so the exact file can be regenerated, and make sure `python run.py --stage all --config configs/<best>.yaml` rebuilds it from the raw CSVs.
- Write a short README section: how to run, what each stage does, expected runtime.
- Submit the code link **by 15 Nov 09:00 IST**, an hour early. Check what access the Submission tab expects for a private repo before then.

---

## 6. Colab runbook

**Budget (updated 13 Nov, 12:15 IST): about 350 compute units, A100 available.** That is more than this plan needs. The whole build is roughly 10–15 GPU-hours, because the data is small (1.8M rows).

- **Units burn while the runtime is connected, working or not.** Third-party measurements from March 2026 put an A100 at about 5.4 units an hour and a T4 at about 1.2; older figures are roughly double. Read the real rate in Colab's Resources panel at the start. Leaving an A100 connected for the rest of the round could use most of the budget, so disconnect when you stop working.
- **Use the A100 runtime while you are actively iterating.** Time is the scarce resource, not units. An L4 or T4 is enough for the XGBoost fits themselves if you want to stretch the budget.
- **Check the CPU side with `nproc`.** Paid GPU runtimes usually come with more cores and RAM than the free tier's 2 cores. With 8 or more cores, run LightGBM and feature building in Colab too and keep everything in one place. With 2 cores, keep LightGBM and feature development on the laptop's 14 threads.
- What the bigger GPU buys: more seeds per library in step 5 (ten instead of five), stage 2 trained on all rows instead of candidates only, and room for one neural member (TabM or RealMLP) if step 5 finishes early. It does not change the order of the steps. Features and validation still decide the score.

Google does not guarantee the GPU model or session length; sessions end after at most 12 hours and idle ones sooner. So everything is cached to Drive and every stage is resumable.

```python
# Cell 1: what did we get?
!nvidia-smi -L; nproc; free -g | head -2

# Cell 2: Drive + code. Put a fine-grained GitHub token in Colab's Secrets panel as GH_TOKEN.
import os
from google.colab import drive, userdata
drive.mount('/content/drive')
os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
!git clone -q https://$GH_TOKEN@github.com/<user>/upi-fraud-ml.git && cd upi-fraud-ml && git log -1 --oneline
%cd upi-fraud-ml
!pip -q install -r requirements.txt
!ln -s /content/drive/MyDrive/upi-fraud/data data
!mkdir -p /content/drive/MyDrive/upi-fraud/runs

# Cell 3: smoke test on 2% of senders (a few minutes). Watch nvidia-smi: GPU use must be above 0.
!python run.py --stage all --sample 0.02 --device cuda --out /content/drive/MyDrive/upi-fraud/runs/smoke

# Cell 4: the real run. Each stage skips work whose output file already exists.
!python run.py --stage all --config configs/v1.yaml --device cuda --out /content/drive/MyDrive/upi-fraud/runs/v1
```

- Pin versions in `requirements.txt` to what the laptop has: xgboost 3.2.0, lightgbm 4.7.0, catboost 1.2.10, scikit-learn 1.7.2, pandas 2.3.3. Record `pip freeze` with each run.
- Memory: the feature matrix at about 300 float32 columns × 1.82M rows is about 2.2 GB. Store features as float32 parquet, load only the columns a config needs, and pass XGBoost a `QuantileDMatrix` so the GPU holds binned data (well under 1 GB).
- Write OOF and test probabilities as `.npy` next to each model. Stage 2, blending and decision tuning then run in seconds on either machine.
- LightGBM on GPU: the pip wheel's OpenCL mode (`device_type="gpu"`) may work in Colab but I have not verified it for this version; the CUDA mode needs a source build. Don't depend on either. Run LightGBM on CPU: in Colab if the runtime has 8 or more cores, otherwise on the laptop.
- **Fallback if Colab cuts the GPU:** the laptop's RTX 4050 runs `device="cuda"` XGBoost with the same code, and the EDA's LightGBM fits took about two minutes on its CPU. Nothing in steps 1–5 requires Colab.

---

## 7. Schedule and upload budget

Times are IST. Durations assume one person; with a team, split features (step 2) from modelling (steps 1, 4).

| When | Work | Uploads |
|---|---|---|
| 13 Nov, now → 13:00 | Upload `rules_v0.csv` if not done. Step 0: repo, Drive, smoke tests. | 1 (gate 0) |
| 13 Nov, 13:00 → 17:00 | Step 1: v1 model. | 1 |
| 13 Nov, 17:00 → 24:00 | Step 2: feature packs with the drift filter. Upload the best state before midnight. | 1–2 |
| 14 Nov, morning | Finish step 2. Step 3: label cleaning. | 1 |
| 14 Nov, afternoon | Step 4: stage 2. Two single-class probes. | 1 + 2 probes |
| 14 Nov, evening | Step 5: ensemble and seeds on Colab. Step 6 if the probes call for it. | 1–2 |
| 14 Nov, 22:00 → 24:00 | Freeze. Full refit. Push code. README. | 1 |
| 15 Nov, 00:00 → 08:00 | Only small validated changes (seeds, a hedge). Sleep. | 0–2 |
| 15 Nov, by 09:00 | Code link submitted. Final check of the leaderboard. | – |

That is 10–13 of the 30 available uploads. The rest are reserve; unused uploads cost nothing.

If you fall behind, cut in this order: step 6, step 5 down to seeds of one library, the stage-2 model (keep the 30-minute neighbour blend), the smaller feature packs. Never cut the validation protocol or the submission checks.

---

## 8. Guardrails

- **Reproducibility is a prize condition.** One entry point, fixed seeds, configs in git, a log line per upload.
- **No leakage through folds.** Month folds only. Any feature that uses labels (for example "this receiver had fraud before") must be computed for each row from other months' labels only, and must pass the time-consistency test before it is kept. Treat that pack as optional.
- **No leakage through events in stage 2.** Neighbour aggregates always exclude the row itself and always use out-of-fold probabilities for training rows.
- **Every class gets predictions.** Assert it in the submission writer.
- **Don't tune on the leaderboard.** Probes diagnose; fold A decides.
- **Don't move infrastructure mid-run.** If Colab fails, fall back to the laptop with the same command. Don't start a third environment.
- **Secrets.** No tokens in notebook cells or git. `kaggle.json` stays out of the repo and off Drive.
- **Stop adding, start shipping at 14 Nov 22:00.**

---

## 9. Future iterations (after the round, or for the December contest)

Ordered by how much I'd expect them to add on data like this.

1. **Event-set reranker** (the Reddit cascade, done for tabular data). For each candidate row from stage 1, gather its neighbour rows and train a small set transformer that reads the candidate with its neighbours and outputs the class. It replaces hand-built neighbour aggregates with learned ones. It only has to run on the few percent of rows that are candidates, so it fits a free Colab GPU.
2. **GNN embeddings into the GBDT.** Model each payment as an edge between accounts, train an edge-classification GNN with the Multi-GNN adaptations (reverse message passing, ports, ego IDs), and feed its edge embeddings to XGBoost alongside the hand-built features. The public evidence says the GNN alone will not beat the trees; the combination might.
3. **Label-free graph embeddings.** SVD or node2vec on the sender–receiver and device–account graphs, and the TalkingData trick of LDA/NMF over co-occurrence counts. Cheap to try as one more feature pack.
4. **A tabular foundation model as an ensemble member.** TabICLv2 (BSD-3) with all fraud rows plus a few hundred thousand `legit` rows as context, predicting only candidate rows. Read the licence before using TabPFN weights in a prize competition.
5. **AutoGluon as a second opinion.** Run its best-quality preset on the final feature table with a time-based validation split, as a check that the hand-built ensemble is not leaving easy gains.
6. **A robust loss for noisy labels** (Robust-GBDT's focal-style loss) instead of rule-based cleaning.
7. **A transaction sequence model.** Only worth it on data with long per-account histories; not this dataset.

Reusable from this round for any next tabular contest: the `Events` window helper, the month-fold OOF scaffolding, the weighted-argmax decision tuner, the drift filter, and the submission log.

---

## 10. Sources

Competition write-ups
- IEEE-CIS Fraud Detection, 1st place, part 1: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284
- IEEE-CIS Fraud Detection, 1st place, part 2 (time consistency, validation): https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- NVIDIA summary of the IEEE-CIS solutions: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- TalkingData AdTracking Fraud Detection, 1st place: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475 and code at https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
- Kaggle Playground S6E11 experiment log: https://www.kaggle.com/datasets/rajibbahar/ps-s6e11-experiment-results
- The Kaggle Grandmasters Playbook (NVIDIA, 2025): https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

Papers
- Blanuša et al., Graph Feature Preprocessor (2024): https://arxiv.org/abs/2402.08593
- Altman et al., Realistic Synthetic Financial Transactions for AML Models (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf
- Egressy et al., Provably Powerful GNNs for Directed Multigraphs (2023): https://arxiv.org/abs/2306.11586
- Lipton, Elkan & Narayanaswamy, Thresholding Classifiers to Maximize F1 Score (2014): https://arxiv.org/abs/1402.1892
- Wang, Li, Yan & Koyejo, Consistent Classification with Generalized Metrics (2019): https://arxiv.org/abs/1908.09057
- Elor & Averbuch-Elor, To SMOTE, or not to SMOTE? (2022): https://arxiv.org/abs/2201.08528
- Eisenbürger et al., Training GBDTs on Tabular Data Containing Label Noise (2024): https://arxiv.org/abs/2409.08647
- Ponti et al., Improving Data Quality with Training Dynamics of GBDTs (2022): https://arxiv.org/abs/2210.11327
- Luo, Quan & Xu, Robust-GBDT (2023): https://arxiv.org/abs/2310.05067
- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986 · TabICLv2: https://arxiv.org/abs/2602.11139 · PRAGMA (Revolut): https://arxiv.org/abs/2604.08649

Practitioner posts
- r/MachineLearning, "GNN Model For Fraud Detection Isn't Performing Well": https://www.reddit.com/r/MachineLearning/comments/1tovj42/rgnn_model_for_fraud_detection_isnt_performing/
- r/DataScientist, nine imbalance techniques compared: https://www.reddit.com/r/DataScientist/comments/1vjzrib/we_tested_9_techniques_for_handling_extreme_class/
- r/MLQuestions, graph-based mule and network fraud: https://www.reddit.com/r/MLQuestions/comments/1qgv7bv/graphbased_fraud_detection_ip_mule_network_how_do/
- fraud-synth-bench write-up (LinkedIn): https://www.linkedin.com/posts/thanatchon_when-the-simulated-fraudsters-switched-tactics-activity-7515781493428768768-v1Ie
- Sardine, foundation model for issuer fraud (2026): https://www.sardine.ai/blog/foundation-model-issuer-fraud-detection
- NVIDIA, Build Your Own Transaction Foundation Model: https://developer.nvidia.com/blog/build-your-own-transaction-foundation-model-for-financial-intelligence/
- NVIDIA, fraud detection with GNN embeddings into XGBoost: https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/
- Molnar, The state of Tabular Foundation Models (2026): https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models

Hugging Face and tooling
- TabICL weights (BSD-3): https://huggingface.co/jingang/TabICL · TabPFN-3.5 weights (custom licence): https://huggingface.co/Prior-Labs/tabpfn_3_5 · TabPFN repo and limits: https://github.com/PriorLabs/tabpfn
- Colab limits are not guaranteed; a summary of what Google documents: https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations
