# Review of v1 features and stage 1; nested neighbour stage 2

Reviewed 13 November 2026 against repository HEAD `d4ecf6caef06c08fc43b965f14fbabb998a3a34a`. Scope: `features_v1.py`, `common.py`, `exp_stage1.py`, their `Events` implementation and the builders of their input caches. This review concerns the default v1 features and raw training labels; it does not validate the newer band rules, cleaned labels or their benchmark. No pre-existing files were edited. All Python commands used isolated mode (`-I`).

**Conclusion:** I found no direct label feature in the current default v1 inputs. The September weight search uses the full validation population, so keeping 25% of legitimate *fit* rows does not itself bias that search's confusion matrix. There are real cache/artifact correctness risks, approximate windows with misleading names, and an empty-input bug in two `Events` helpers. Naively aggregating one global table of month-OOF probabilities would introduce cross-month label feedback; the new stage-2 implementation uses nested, query-specific cross-fitting to prevent it.

## Findings, in priority order

| Priority | Location | Finding and consequence | Proposed correction |
|---|---|---|---|
| High | [features_v1.py](../src/upi/features_v1.py), lines 16–18, 33–34 | The ratio cache is checked only for equal length, then attached by position. An equally long cache from a reordered/rebuilt base silently assigns another transaction's ratios. The final feature file receives current IDs, so the later ID assertion cannot detect this upstream corruption. | Include `txn_id` in the ratio cache, validate uniqueness and the exact sequence, and fail or join one-to-one by ID. Version the source cache. |
| High | [exp_stage1.py](../src/upi/exp_stage1.py), lines 47–54, 66–70 | `sep_proba.npy` and `weights.npy` correspond to the chosen quarter/half/full tree count, but `lgb.txt` saves the full fitted booster. Loading that model with default prediction can fail to reproduce the selected result. The selected count is printed but not saved in structured metadata. | Save with `num_iteration=n` and record the chosen count, feature list, class order, sampling fraction and trap setting. The new runner saves explicitly specified iterations. `fit_submit.py --trees N` currently relies on the operator supplying the same count. |
| Medium | [common.py](../src/upi/common.py), lines 25–27 | Extra feature files bypass `DROP`: every column except `txn_id` is concatenated. A numeric `label`, trap column or target-derived statistic could become a model input. Duplicate feature names are also accepted here. | Validate added columns against a schema; reject forbidden names and duplicates. The new loader adds these checks, but a name blacklist cannot prove an arbitrarily named feature is label-free. |
| Medium | [exp_stage1.py](../src/upi/exp_stage1.py), lines 47–54; [common.py](../src/upi/common.py), lines 56–80 | September labels tune 11 decision multipliers, select among three tree counts, and then score the chosen result on those same rows. That is a development score with selection optimism, not an independent estimate. | Report untuned outer-month scores; select hyperparameters/weights inside outer training data if an independent tuned score is required. The new runner separates untuned OOF metrics from optional development-tuned metrics. |
| Medium | [features_v1.py](../src/upi/features_v1.py), lines 69–77 | `dev_uniq_snd_10m` and `dev_uniq_rcv_10m` are the maximum distinct count in two offset fixed 600-second bins. They are not exact counts in the same ±600-second window used by `dev_cnt_10m`. The analogous 1h features have the same approximation. | Rename them as bin approximations, or implement exact rolling distinct counts before treating their ratios as consistent window statistics. |
| Low, latent | [events.py](../src/upi/events.py), lines 30–44 | `prev_dt` and `next_dt` index `self.K[0]`/`self.K[-1]` before masking an empty source. Both raise `IndexError` on an empty event set instead of returning NaN. V1 uses `count`/`sum`, so this does not currently break the reviewed feature build. | Return a NaN array immediately when `self.n == 0`. |
| Low on this data | [events.py](../src/upi/events.py), lines 4, 16, 21–23 | Packed keys assume entity time ranges plus window reach cannot overlap adjacent entity blocks and multiplication cannot overflow. There is no guard. Violating this produces incorrect counts and possibly a non-sorted search array. | Assert these bounds or use structured `(group, time)` keys. The new neighbour index uses structured keys. |

The existing ratio cache is **not demonstrated corrupt**: a read-only recomputation of all rows for `bal/amt`, and each of sender/receiver/device `dt_prev` and `bal/prevbal`, matched the stored float32 values including missingness. These are seven of the 26 cache columns, not an exhaustive proof of every ratio. Feature-cache IDs also match the base, and base transaction IDs are unique.

Other issues worth preserving in the audit:

- `--traps` explicitly puts `ip_risk_score` and `app_old = (app_major == "3")` into stage 1. That flag is unsuitable for the requested model. Default loading excludes these, `app_major`, `app_version`, labels and IDs. The trap-based recall/purity calculations after prediction are diagnostics only. A trap experiment can still overwrite a normal experiment's tag, and its weights could later be used by `fit_submit.py`, which loads without traps. Keep experiment configurations distinct and recorded.
- `load_xy` maps missing/unknown labels to `-1` without validating training targets. Its documented claim that test targets are `-1` relies on the base cache actually having missing test labels. Existing fitting masks are split-based, so populated test labels would not enter those fits. The new loader explicitly validates training targets and forces test targets to `-1`.
- `fit_weights` is a heuristic coordinate search, not guaranteed monotone ascent: it selects the midpoint of a near-best plateau (within 0.0003), and the previous coordinate need not be on the grid. The candidate-row shortcut is safe for the default fixed legit weight of 1 and fraud weights bounded by the grid maximum. An arbitrary `w0` outside those bounds violates the shortcut's assumptions. No such nondefault use is needed by the new runner.
- Invalid `--legit-frac` values are not rejected by the existing experiment; tiny tree counts can produce a candidate `num_iteration=0`. The new runner validates positive counts and a fraction in `(0, 1]`.

## Label and window audit

`features_v1.py` constructs features from amounts, balances, entity IDs, transaction types and timestamps. There is no label-based group or mask in it. The ratio builder, [15_spike_finder.py](../scripts/explore/15_spike_finder.py), constructs `R` before its label/trap diagnostics; those diagnostics do not change `R`. The older relational builder similarly keeps its target-dependent diagnostics separate from its numeric relational columns. The current default `load_xy` produces **161 features** with no forbidden or duplicate names.

This distinction matters: discovering a recipe by examining labelled training examples is supervised feature design, not a label value used at prediction time. Repeatedly using September to discover recipes can still make September scores optimistic. Fold-safe stage 2 does not undo that historical feature-selection effect.

All current v1 event indexes and ratio sequences are built over **train and test together**. There is no test-specific branch that clips a window, resets history at a split boundary, or removes future neighbours for training. Category vocabularies are also constructed jointly, without labels. This is transductive batch feature construction, appropriate to the repository's supplied-whole-test setup. It is not a causal, transaction-at-arrival deployment simulation.

Identical code does **not** mean identical observation coverage:

| Split / timestamp month | Rows |
|---|---:|
| Train July | 527,065 |
| Train August | 526,852 |
| Train September | 509,996 |
| Train October tails | 147 |
| Test October | 260,536 |
| Test November tails | 46 |

September has October observations available on its future side. October has only 46 November tail observations rather than a complete next month; July has no June history. Symmetric long windows and all-history totals therefore have edge censoring, while test also has lower observed traffic density. The November tail maximum is not evidence of full November coverage. Counts/sums, distinct bins and the older all-period totals can shift for these reasons even with identical code. Before making a causal claim, separate past/future counts, account for observed exposure, or evaluate equally censored batches. Do not multiply October counts to force a September match.

`common.py` deliberately maps the 147 October-dated **training** tails into the September fold. Thus September validation has **510,143** rows; all **260,582** test rows, including November tails, remain unlabelled context. The new runner preserves that convention.

### Exact boundary semantics

- `Events.count/sum(qg, qt, lo, hi)` cover **closed** intervals `[qt+lo, qt+hi]`: left search uses `side="left"`, right search uses `side="right"`. Exact endpoints and all timestamp duplicates are included. Negative query groups yield zero. An empty masked source works for `count` and `sum`.
- V1 symmetric queries include the query transaction itself if it satisfies the source mask. This is not label leakage. Keep this in mind when interpreting thresholds or converting counts to a number of *other* transactions.
- V1 forward `[1, w]` and backward `[-w, -1]` queries exclude **all** rows in the same second, not only the current transaction. Current timestamps are whole seconds, so these are strict-after/strict-before queries. With fractional timestamps, the integer cast would introduce rounding semantics.
- Nonempty `prev_dt`/`next_dt` exclude all equal-time rows, and include a previous/next gap exactly equal to `within`. The empty-source crash above is separate from endpoint handling.
- The ratio sequence builder permits adjacent rows at zero time difference and up to 7,200 seconds inclusive. Its stable tie order is row order, not a causal order among simultaneous transfers. This differs intentionally from the strict-before/after queries.
- An illustrative distinct-bin discrepancy: one device with different senders at `t=[0, 599, 600]`. For the row at 599, exact ±600 seconds has three senders; each relevant offset bin contains only two. The v1 distinct feature returns two.
- Naming is inconsistent across feature generations: the inherited `snd_refund_cnt_30d` in [04_relational.py](../scripts/eda/04_relational.py), line 126, means ±15 days (30-day total span), whereas v1 `pair_refund_cnt_30d` means ±30 days (60-day total span).

There is **no count/sum endpoint off-by-one on the current data**. The actual `t` range is 0 through 11,369,444 seconds. Even adding the largest v1 radius, 2,592,000 seconds, leaves 53,147,420 seconds below `SCALE = 2**26`; the current packed group values also fit int64. Outside those bounds, `(group=0, t=SCALE)` and `(group=1, t=0)` collide. The new implementation avoids that encoding entirely.

## What 25% legitimate sampling does to class weights

`keep = fit & ((y != 0) | Bernoulli(0.25))` only changes July/August fitting rows. Predictions and `fit_weights(P, yv)` use **all September validation rows**, including all legitimate ones. Consequently the weight search sees the correct validation prevalence and false-positive denominator. There is no need to inverse-weight its validation confusion matrix.

There is, however, a posterior shift. If `p_c(x)` is the full-population posterior, `s_0=0.25` for legit and `s_c=1` otherwise, then ideal sample-trained probabilities satisfy:

```text
q_c(x) = s_c p_c(x) / sum_k s_k p_k(x)
p_c(x) = (q_c(x) / s_c) / sum_k (q_k(x) / s_k)
```

Thus an uncorrected model overstates fraud relative to legit. If desired decision multipliers on population probabilities are `a` with `a_0=1`, the equivalent multipliers on sample probabilities are `w_0=1`, `w_c=0.25*a_c` for fraud. Searching on full September can absorb much of this global shift. Finite grid limits, model misspecification, lost legitimate examples and calibration differences still matter; changing the training fraction or model configuration calls for retuning, not reusing weights blindly.

For neighbour sums, correct and renormalize each row **before** aggregation: the normalization factor varies by row, so correcting an aggregate afterward is not equivalent. The new runner applies this theoretical prior correction once to each stage-1 prediction before any aggregation, and once to each stage-2 output after its own sampled fit. It does not claim to solve arbitrary LightGBM miscalibration. Graph rows and validation rows are never downsampled.

## New code and precise feature definitions

- [stage2_neighbors.py](../src/upi/stage2_neighbors.py): label-free entity/time indexes, sums/means, sampling correction and explicit probability routing by month.
- [exp_stage2.py](../src/upi/exp_stage2.py): standalone nested leave-one-month-out fitting, exact-subset stage-1 caches, OOF metrics, optional final fit and test predictions.
- [test_stage2.py](../tests/test_stage2.py): boundary, routing, loader, sampling and complete 12-class synthetic pipeline tests.

For query row `i`, let `N_r(i)` contain **all observed rows `j != i`** satisfying the relation and time condition below. Whole string IDs are retained, so merchant and account namespaces do not alias.

| Relation name | Entity condition | Closed time condition |
|---|---|---|
| `receiver_24h` | `receiver_j == receiver_i` | `abs(t_j-t_i) <= 86400` |
| `device_10m` | `device_j == device_i` | `abs(t_j-t_i) <= 600` |
| `sender_3h` | `sender_j == sender_i` | `abs(t_j-t_i) <= 10800` |
| `reverse_pair_7d` | `sender_j == receiver_i AND receiver_j == sender_i` | `abs(t_j-t_i) <= 604800` |

Let `P_j^(A,q,c)` denote the corrected stage-1 probability of class `c` for source row `j`, constructed below for allowed fitting months `A` and query month `q`. For each of the 12 classes, append:

```text
s1_{relation}_sum_{class}(i)  = sum_{j in N_relation(i)} P_j^(A,q,class)
s1_{relation}_mean_{class}(i) = sum / |N_relation(i)|, or 0 if empty
s1_{relation}_count(i)       = |N_relation(i)|
s1_self_{class}(i)           = P_i^(A,q,class)
```

That is 96 neighbour probability features, four counts and 12 self probabilities: **112 additional columns**, for **273 total inputs** with the current v1 pack. The query row is subtracted exactly once where it is a member; equal-time *other* transactions stay. Reverse pairs do not automatically subtract one: only a genuine self-transfer belongs to its own reverse pair. An absent reverse pair has count/sum/mean zero. A transaction may legitimately belong to multiple relation sets. Prefix sums accumulate in float64, outputs are float32, and empty means are zero.

### Why ordinary global OOF probabilities are insufficient

Suppose July row `i` has an August neighbour `j`. A conventional OOF prediction for `j` is trained on July and September. It can therefore encode `i`'s label through July training data, even though `j` itself was out of fold. Aggregating that prediction back into July creates label feedback. Moreover, precomputing such features once and then holding out September at stage 2 lets September labels influence stage-2 *training features* through the original stage-1 models.

The implementation requires this stricter routing:

```python
fit_months = available_outer_training_months - {query_month, source_month}
source_probability = stage1_fitted_only_on(fit_months).predict(source_row)
# If fit_months is empty: fixed uniform vector (1/12, ..., 1/12).
```

An unlabelled test row has no training month to remove. Each requested nonempty subset is trained once and cached. Cached files contain predictions for *all* rows, some in-sample; they must only be consumed through the routing above. They are not individually global OOF arrays.

With `J=July`, `A=August`, `S=September`, representative routes are:

| Situation | Query month | Source month | Stage-1 fitting months |
|---|---|---|---|
| Outer holdout S, stage-2 training | J | J | A |
| Outer holdout S, stage-2 training | J | A | None: fixed uniform |
| Outer holdout S, stage-2 training | J | S or test | A |
| Outer holdout S, stage-2 validation | S | S or test | J, A |
| Outer holdout S, stage-2 validation | S | J | A |
| Outer holdout S, stage-2 validation | S | A | J |
| Final stage-2 training | J | A | S |
| Final test prediction | test | J | A, S |
| Final test prediction | test | test | J, A, S |

For each outer month, build both stage-2 training and validation features anew with that month's labels unavailable everywhere upstream, fit stage 2 on the other two months, and score every outer validation row. Rotate across all three months. Fixed tree counts avoid selecting an iteration on the outer validation labels. The stage-1 baseline uses exactly the corresponding two-month fit. July/August LOMO uses later-month labels and therefore measures held-month generalization; the September fold is the forward-label comparison.

**Three-month limitation:** during nested fitting only two training months remain. Excluding both a training query's month and a different training neighbour's month leaves no labels. The implementation uses a fixed uniform probability vector in that case. It does not estimate a fallback prior from excluded labels, drop those neighbours, or secretly reuse an in-sample prediction. Counts remain exact; sums/means at these cross-month links partly encode neutral probabilities. With all three months available for the final model, this fallback is never needed. Models trained on one, two or three months can also differ in calibration. These are explicit train/inference distribution differences, and a reason to measure the experiment rather than promise a gain.

The primary reported scores use argmax of corrected probabilities. Optional `--tune-weights` fits on full pooled OOF rows and records its result as `development_tuned_f1_NOT_independent`; it must not be presented as an independently validated tuned score. Obtain that by another calibration split inside the outer training data if needed. No label cleaning or band overrides are applied.

## Running and artifacts

Use a **new, nonexistent output directory**. Existing input files are read only; the runner refuses to reuse an output directory. Example:

```powershell
python -I tests/test_stage2.py
python -I src/upi/exp_stage2.py . --out models/codex_s2_nested_v1 --stage1-trees 500 --stage2-trees 300 --threads 12
```

In this workspace the PATH `python.exe` is a Windows alias, so the successful test invocation used:

```powershell
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I tests/test_stage2.py
```

Use that same interpreter prefix for the experiment here. `--no-final` runs only the outer evaluation. `--legit-frac 1` disables class sampling and its numerical prior correction. `--features` defaults to `features_v1.parquet`; added files require their own semantic leakage audit.

A complete run fits six nonempty proper-subset stage-1 models for evaluation, one full-data stage-1 model for final test context, three outer stage-2 models and one final stage-2 model. It writes model files with explicit iteration counts, subset metadata, `all_rows.parquet` mapping cache positions to IDs/folds, `fold_provenance.json`, configuration/class order, separate stage-1/stage-2 OOF arrays and their row manifest, metrics, and final test probabilities/IDs. The seven all-row float32 probability arrays alone take about **613 MB decimal** on this dataset. Training memory and end-to-end runtime have not been benchmarked.

`predictions_by_id.csv` is in base test-row order; it is an ID-keyed prediction artifact, not a sample-submission-ordered upload. Join it one-to-one to `data/raw/sample_submission.csv` before any upload. The runner does not upload or modify the existing submission pipeline.

## Verification actually performed

1. Ten tests pass, including brute-force comparison of all four relation windows at exact endpoints and one second outside; duplicate timestamps, self/reverse transfers, absent pairs, shuffled rows and long timestamps; original nonempty `Events` strict-neighbour behavior and masked count/sum behavior; fold tails; excluded-label perturbation tests; theoretical sampling correction; rejection of a target column in an added feature file; and a full 12-class nested synthetic run, including all seven stage-1 subsets, final predictions, row manifests and refusal to overwrite output.
2. Real default input load succeeded: **1,824,642 rows × 161 features**, no duplicate columns, no infinities after loader handling, and test targets explicitly `-1`. Fold sizes were 527,065 / 526,852 / 510,143 training and 260,582 test.
3. Built the new graph over **every real row**. Queried 15 rows distributed over the file and fold starts, using fixed normalized probabilities independent of labels. All **60 relation queries** matched direct brute-force counts and the corresponding sums/means. The result had 112 columns. Load, graph build and these checks took approximately 35 seconds locally; this is not a model-training timing.
4. Checked current cache alignment and seven recomputed ratio columns as described above. Separately reproduced both empty-source `Events` helper crashes. No existing feature builder was executed, because it would overwrite its cache.

**No full competition-data stage-2 fit was run, and no stage-2 F1 improvement is claimed.** The synthetic fit validates mechanics only. The next performance decision should use the runner's actual September outer score and pooled raw-label OOF score, with its three-month fallback and historical feature selection kept in view.
