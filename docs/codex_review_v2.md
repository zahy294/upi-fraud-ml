# Pipeline review: validation, test inference and reproducibility

Reviewed 14 November 2026, working tree at `d4ecf6caef06c08fc43b965f14fbabb998a3a34a`. Review only: no models or decision weights were fitted, no pre-existing files were edited, and nothing was written under `data/`, `models/` or `submissions/`. All Python used `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I`; numerical and Arrow worker limits were four. The existing modification to `docs/IMPLEMENTATION_PLAN.md` was left alone.

**The largest benchmark problem is that “cleaned” training labels are still incomplete. The clearest measured train/test asymmetry is oracle training bands, worth about 0.002 in a frozen-prediction check. The reported 69-label/16-prediction difference is not evidence of two identical training runs being nondeterministic: the raw-model feature recipe changed.**

`final_local.csv` is the latest candidate. Its saved artifacts in `models/oof_final_local` exactly match the fresh clone's `models/oof_clean`. I located that clone under `C:/Users/ASUS/AppData/Local/Temp/claude/be1d3d9ececbd068/919cf462-8892-436a-b600-fdd1e5e3eb47/scratchpad/repro` and read its logs and arrays. Its score is **0.559706662**, versus **0.511344101** against raw labels. The older `s5_c3` scores **0.559136411**. Neither is the uploaded `s3_clean_v4.csv`, whose public score is reported as 0.5400. The latest file differs from that upload in **98 rows**. Thus 0.5597 minus 0.5400 is not a paired local/public comparison; the approximately 0.017 gap in the plan refers to the earlier upload and benchmark recipe.

**Ranked impact on the apparent local/public gap.** Estimates below are explicitly different kinds of evidence; they must not be added.

| Rank | Finding | Measured effect or conditional estimate | Attribution to the 0.017 gap |
|---|---|---|---|
| 1 | Undetected, unreported fraud remains labelled legit in the local reference | Assuming 8% missing labels among undetected fraud lowers `s5_c3` by **0.01676**; using the 12.20% observed on band rows lowers it by **0.02623** | Potentially the whole gap; the missing rate among undetected rows is unobserved |
| 2 | October has a different detected-event mixture, especially fewer bots and collusion rows | Holding missed-fraud density and conditional prediction precision fixed gives **−0.01980** | Potentially the whole gap; a distribution scenario, not a measured test-score loss or a code bug |
| 3 | Label-derived training bands versus blind test bands | Replacing train bands with blind recovery, rebuilding their propagation, holding the final model/weights/targets fixed: **−0.002004** for `final_local` | Approximately 0.002 is a useful proxy; retraining and true test edges were not observed |
| 4 | Prediction-dependent reference cleaning and reuse across outer folds | 873 model-fallback labels; reverting just these labels changes the old benchmark by **−0.01029**. Cross-month votes reach 93 relabelled rows | A sensitivity, **not** 0.01029 of demonstrated inflation; many corrections are real fraud |
| 5 | Decision weights selected on the evaluated OOF rows, before overrides | Saved weights improve latest post-rule OOF score by **0.001311** versus unit weights | Optimism/transfer loss not identifiable without nested calibration; no support for assigning 0.017 to this |
| 6 | Event overrides and imperfect chain construction | Event overrides add **0.001201** locally and change 26 latest test predictions beyond the own-device rule | Small observed scope; some apparent local errors are reference-label errors |
| 7 | The 147 October-dated training tails | **Zero** changed test band features/chain bands/predictions when their band sources are removed; some ordinary windows do overlap | No demonstrated score loss; remaining model effect cannot be rescored without saved fitted models |

Public sampling also matters. A **scenario**, drawing 78,175 independent rows from the old local confusion table, gives macro-F1 SD **0.00982**, with a central 95% range **0.53989–0.57798** over 2,000 simulations. This is conditional on the local distribution and a roughly 30% public sample; it is not a confidence interval for the actual leaderboard, and it does not model clustered sampling. It shows why a 0.017 difference does not identify one bug. The missing-label and October-mixture scenarios together would predict too large a drop if their strongest assumptions were both imposed. That is evidence against treating them as an additive explanation; October's unknown missed-fraud prevalence, label process and public sample can offset them.

**1. Cleaning only detectable unreported fraud makes the benchmark incomplete.**

Locations: `src/upi/clean_labels.py:84–89`; `src/upi/make_submission.py:37,66–70`.

The cleaner promotes a band-free raw-legit row only when raw-model fraud probability is at least 0.85. Ordinary-looking, unreported fraud therefore stays legit. All **35,483** old-model fraud false negatives have identical raw and cleaned labels; cleaning recovers none of the additional fraud hidden among the approximately 1.5 million legit-labelled rows. If test labels are complete, evaluating against this reference overstates recall and macro-F1 even with perfect OOF discipline.

Concrete counterexample: a real fraud row drawn from the ordinary-looking generator, assigned raw label `legit`, on its home device with `pf=0.02`. Neither `r_unrep`, `r_type` nor `r_band` applies. The model's legit prediction is counted as a true negative locally and would be a false negative against complete labels. Such a row cannot be named individually without unavailable ground truth; claiming to locate it from these features would contradict the indistinguishability evidence.

Band rows provide a diagnostic estimate: **802/6,573 = 12.2014%** carry raw label legit, with monthly rates **12.275%, 12.537%, 11.796%** in July, August and September. This estimates missingness in band-device fraud, not necessarily in ordinary-looking fraud. Under the additional assumption that the same missingness applies to undetected fraud, for each fraud class:

```python
# C: cleaned-label confusion table; rows=true class, columns=prediction.
extra_fn = C[1:, 0] * q / (1 - q)
C_complete = C.copy()
C_complete[1:, 0] += extra_fn
C_complete[0, 0] -= extra_fn.sum()
```

| Assumed missing fraction q | Additional hidden false negatives | Expected complete-label macro-F1, old `s5_c3` |
|---|---:|---:|
| 0 | 0 | 0.55914 |
| 0.03 | 1,097 | 0.55304 |
| 0.05 | 1,868 | 0.54885 |
| 0.08 | 3,085 | 0.54238 |
| 0.12 | 4,839 | 0.53337 |
| 0.122014 | 4,931 | 0.53291 |

The 8% row illustrates sensitivity; it was not estimated by matching the public score. The full band-based extrapolation gives a larger drop than the observed gap, so it cannot be asserted as the actual attribution.

Proposed fix: call this an **event-corrected, incomplete-label benchmark**. Keep raw, event-corrected and expected complete-label metrics separately. Report sensitivity to missingness; obtain an independent clean validation target if available. Do not “fix” individual ordinary rows with model-generated labels. No threshold change can recover information absent from the input.

**2. Lower October event rates change attainable recall even when predictions are correct.**

Locations: `src/upi/run_oof.py:64–71`; `src/upi/make_submission.py:66–80,89–92`. These implement pooled historical calibration and a full-train test model; there is no adjustment for changes in the mixture of detectable and undetectable fraud.

The exact bot signature independently falls from **209.02 September rows/100k to 103.23 test rows/100k**, with no supported replacement shape; see [the earlier bot investigation](codex_bot_drift.md). In the saved `s5_c3` predictions, per-row test/OOF prediction-rate ratios are **0.4962 for bots**, **0.6379 for merchant collusion**, **1.2452 for mules**, **1.1468 for takeover**. Fewer easy positives relative to ordinary false negatives lowers class F1. It is not repaired by inflating weights until predicted class counts match September.

Concrete scenario, reproduced in script 26: scale each fraud-prediction column of the OOF confusion table by its observed test/OOF rate ratio; leave its conditional true-class mixture unchanged, retain the per-row density of fraud predicted legit, and adjust true-legit/predicted-legit mass to preserve the total. Score becomes **0.53934**, a **0.01980** decrease. Changing only the bot column gives **−0.01445**; changing only collusion gives **−0.00881**. Other classes partly compensate, and these isolated differences are not additive.

Proposed fix: monitor recipe/event rates and report this explicit shift scenario alongside pooled OOF. Use forward-time validation and independently fixed cleaning/calibration to assess transfer. Do not infer test truth from predicted frequencies or report the scenario as a measured public-score explanation. The model cannot classify an absent observable bot event; lower rates may reflect fewer generated patterns, different ordinary-fraud prevalence, or calibration changes.

**3. The training band features leak held-out labels, and the blind self-check understates the independence problem.**

Locations: `src/upi/bands.py:75–85,126–145,158–162,183–190`; `src/upi/features_v5.py:64–70`; `src/upi/run_oof.py:38,64–66`; `src/upi/clean_labels.py:86–89`.

The training segmentation scores use **all raw training labels** at line 82 before the LOMO split. `band`, sender/receiver band counts, and v5 `ch_band` inherit those labels. The held-out model consequently receives label-derived information unavailable when October's new device block is placed blind. `features_v5.py` is not wholly label-free despite its docstring. Cleaning then sets the evaluation target on every band row to that same band, and the submission rule forces the prediction to it. Perfect agreement there is partly by construction.

Concrete failure: blind recovery places the SIM/takeover cut at **425001**, versus the label-derived **425033**. A held-out SIM row with device number in `[425001,425033)` is supplied the correct SIM band locally but receives takeover under the blind analogue. Reconstructing all sender/receiver counts and chain bands changes **95 latest final predictions** and reduces **0.559706662 → 0.557702218**. For the old model the corresponding figures are 94 predictions and **−0.002003**. This is more informative than quoting only **72/3,231** directly mis-banded rows: errors propagate to other rows.

The calculation freezes model probabilities, weights and the original reference labels. It cannot measure the effect of training on blind rather than oracle features. Re-cleaning the reference to the perturbed bands would conceal the error and is deliberately not done.

The advertised blind check is also not an independent validation: its prior uses test edges that were themselves regularized toward all-train label-derived proportions; anchor-density estimates use the same oracle train edges. Simply excluding the held-out month's labels moves only **1 July, 6 August and 8 September** held-out band rows, because all training months share one device block. That easy interpolation does not replicate inference on a separately generated October block.

Proposed fix: build all supervised band features inside each outer fold; exclude held-out labels from edge fitting and every downstream feature. Separately validate blind block placement using a genuinely withheld block or block perturbations with independently specified priors. Keep the evaluation labels fixed. Report an uncertainty region around test cuts instead of forcing a boundary prediction that has no measured confidence. No test labels are needed for this fix.

Two smaller band issues are currently inactive:

- `bands.py:131,133` estimates a **device-presence probability by counting rows**, whereas line 121 records unique-device presence. Here the counts coincide: mule **115 rows/115 devices**, SIM **42/42**. Duplicating an anchor row would artificially increase the estimate and can eventually make `log(1-lambda)` invalid. Fix with unique-device counts and validate `0 <= lambda < 1`.
- `bands.py:40–45` returns `lo` when no 40-number occupied run exists, because `argmax(all_false)` is zero. Fail explicitly when there is no run, and assert strictly increasing edges/non-empty expected bands. Current train/test block endpoints and all test edges reproduced exactly.

**4. Event cleaning is prediction-dependent, is not a consistent event consensus, and crosses outer folds.**

Locations: `src/upi/clean_labels.py:47–49,61–76,84–89`; `src/upi/run_oof.py:42,64–66`.

The original target changes **3,930 rows**: 2,202 unreported, 737 type corrections, 991 own-band corrections. Of the unreported corrections, **873 have no accepted labelled-neighbour vote** and take the model's type. The final model agrees with **858/873**. Reverting only those 873 targets to raw labels lowers the old score by **0.010288**, while reverting all unreported corrections lowers it by 0.032061. These are target-definition sensitivities, not measured inflation: the unreported group has app-3 share **0.5958**, and the fallback subset **0.5704**, strongly supporting substantial real pattern content. The trap is used only for this diagnostic.

OOF predictions do not make this a fully held-out reference. Labels from the held-out month contribute directly to cross-month neighbour cleaning: **2,162 rows** receive another month's detected-fraud votes; **93** of them are relabelled. More generally, the raw OOF model that produces a training row's pseudo-target can have trained on the eventual outer validation month. A second-stage outer fit on these global pseudo-targets is not nested validation.

The cleaner's “event majority” is also a weighted local vote. A neighbour sharing sender, receiver and device counts three times. Forwarding links from one row's receiver to the next row's sender are missing. Excluding the query row means two differently labelled rows can exchange types instead of reaching consensus. Broad same-sender links across seven days can join unrelated events.

Concrete observed failure: this entire forwarding chain has raw mule labels and a mule-band anchor, but its fourth row is changed to SIM swap:

| Transaction | Time on 10 August | Sender → receiver | Amount | Raw label | Clean label / band |
|---|---|---|---:|---|---|
| TX73a2abce | 20:49:35 | AC387175 → AC228409 | 39,211 | mule | mule |
| TX1496e1b7 | 20:58:17 | AC228409 → AC091069 | 38,267 | mule | mule |
| TXb92e85d0 | 21:10:10 | AC091069 → AC166908 | 37,363 | mule | mule |
| **TX50a807d8** | **21:16:53** | **AC166908 → AC293833** | **36,438** | **mule** | **SIM swap** |
| TX3791fee9 | 21:29:31 | AC293833 → AC231921 | 35,859 | mule | mule, device DV0423213 |
| TX866d9bd3 | 21:39:02 | AC231921 → AC276537 | 34,841 | mule | mule |

Another example, `TX8abd6dba`, is the middle of a three-row mule chain with band anchor `TX58850cec`; raw legit becomes layering via the fallback. The chain submission rule repairs it to mule and is penalized by the current benchmark. Full contexts are in [the label audit](codex_review_v2_labels.txt).

As a diagnostic, applying the vote cleaner to band rows **without the direct band overwrite** reduces errors from 991 to 125, but also makes **58 originally correct rows wrong**. The direct overwrite hides those failures in the actual cleaned reference.

Proposed fix: use conservative, label-free physical event links, including compatible forwarding links; deduplicate neighbour rows; assign a shared event consensus with explicit ambiguity handling. Clean training targets using only outer-training information and inner OOF probabilities. Freeze validation targets independently of the evaluated model; distinguish model-fallback pseudo-labels from independently supported corrections. Keep sibling-shaped events' supported raw event type. Do not count reusing a pseudo-label as independent validation.

The 25% legit subsample also means `pf` is not a calibrated fraud posterior. Under ideal posterior calibration before label noise, undoing fourfold fraud odds gives `p = pf / (4 - 3*pf)`: 0.85 corresponds to approximately 0.586, and 0.5 to 0.2. This does not invalidate the empirical purity checks, but “model sure” is an unsupported probability interpretation. Calibrate inside training folds if probability thresholds are intended to have that meaning.

**5. Class-weight search uses the benchmark labels and optimizes the wrong final objective.**

Locations: `src/upi/make_submission.py:66–80`; `src/upi/common.py:56–85`; `src/upi/run_oof.py:47–56,66,71`.

Fitting weights on training OOF probabilities and applying them to test is normal; it does not use test labels. The optimism arises from scoring on the same OOF rows that selected the weights. The objective also evaluates `argmax(P*w)` **before** band rules, although the published metric and test predictions apply the rules afterwards. Rows whose class is forced by a band can influence weight selection despite their pre-rule type being irrelevant to the final metric. A weight can also cross the legit/fraud gate that controls propagation, so simply excluding all band-neighbour rows is not equivalent to optimizing the final pipeline.

Concrete observed size: latest post-rule score with saved weights is **0.559707**, versus **0.558395** with all weights one: a gain of **0.001311**. For the old model it is 0.001702. This is not an estimate of how much was overfit. There is no measured evidence here that removing weights would recover the public gap.

There is also a distribution difference: each OOF model trains on two months; the test model trains on all three. For default stage 1, `--seeds` is one for OOF but test averages at least three seeds (`run_oof.py:71`). The final default is three for both. The 25% legit sampling changes posterior odds, but validation itself is not downsampled, so the search's confusion matrix is not biased by a 25% evaluation sample. A second arbitrary fourfold correction after weight search would double-correct the prior effect.

Proposed fix: select weights on inner OOF predictions of outer-training rows only, using the **complete post-rule objective**, then evaluate once on the untouched outer month. At deployment, select weights on all training OOF data, record them, and apply unchanged to test. Match seed counts and record training-size/calibration differences. No weights were refitted in this review.

```python
# Proposed objective only; not executed here.
def decision_score(weights, probabilities, target, row_indices, rule_state):
    pred = (probabilities * weights).argmax(axis=1)
    pred = apply_band_rules(pred, row_indices, rule_state)
    return macro_f1(target, pred)

# Each outer fold needs its own inner OOF data, band construction and cleaning.
# Do not tune on P[other_months] from one global OOF file: those fits can include
# the outer-held-out month's labels.
```

**6. Event overrides are broader than their event descriptions; chain construction has real misses.**

Locations: `src/upi/make_submission.py:45–62`; `src/upi/features_v5.py:24–35,38–47,64–70`.

The same rule function is applied to train and test. There is no reversed train/test mask or hidden label read inside it. The risk is the rule's evidence: “one band type in this time window” is not the same as “one event.” Merchant propagation does not require `S_flag`, amount compatibility or a shared fixed-amount session. Drain propagation does not test the balance sequence. Chain propagation accepts any fraud prediction and overrides earlier suggestions even if they conflict.

Measured incremental changes for **latest `final_local`**, in execution order:

| Rule | Train predictions changed | Correct against cleaned / raw labels | Test predictions changed | Local score after rule |
|---|---:|---:|---:|---:|
| Own device band | 14 | 14 / 10 | 1 | 0.558506 |
| Drain | 0 | — | 0 | 0.558506 |
| Fan-in | 79 | 78 / 68 | 8 | 0.559469 |
| Merchant | 2 | 2 / 1 | 7 | 0.559522 |
| Forwarding chain | 28 | 21 / 23 | 11 | 0.559707 |

These are changes from the preceding rule, not all eligible rows. The seven chain disagreements with the cleaned target have concrete event-label problems illustrated above; they are not proof of seven bad chain decisions. In the older run all five changed training merchant rows and all seven changed test merchant rows had `S_flag=1`, so the missing eligibility guard is presently latent in those examples. Test example `TXfee0b1a5` borrows refund band 8 from `TX62df1b07`, the same merchant **15 h 1 min 52 s** later. This may be the same session, but the function verifies only merchant/time/type, not that hypothesis.

v5 first chooses the nearest incoming/outgoing transaction, then checks its amount ratio. It can miss a later compatible transaction, and independently constructed parent/child pointers disagree. Concrete train example:

```text
03:05:05 TX2bc6561e  AC348909 -> AC355256  65,985  layering_chain
03:13:37 TX8fbd2722  AC355256 -> AC214825     720  legit
04:30:39 TX26449413  AC355256 -> AC289081  65,145  layering_chain
```

For the first row the nearest outgoing candidate is 720, so its child is rejected. The later 65,145 payment is within two hours and has ratio 0.98727, and its parent correctly points back. There are **13 non-reciprocal parent links and four non-reciprocal child links**, among 2,076/2,067 links. No chain exceeds the eight-hop cap. Forty chain-member rows are not `PAY`; transaction type is not filtered in this implementation. These counts establish imperfect event reconstruction but not a score gain from changing it.

Proposed fix: enumerate compatible amount/time candidates before selecting links, resolve ambiguous successors/predecessors explicitly, and derive both chain directions and propagated bands from the same accepted graph. Use stable `(time, txn_id)` tie-breaking. Decide eligibility from a supported transaction recipe. For overrides, require membership in the same physical session and agreement between independent band suggestions; retain the model prediction on conflict. Keep valid own-device evidence strongest. The current hard cutoff at eight hops should become either a documented bounded feature or traversal to termination; it is not truncating this dataset.

**7. Joint train/test windows: real overlaps, no demonstrated tail-driven mistake.**

Locations: `src/upi/bands.py:175–190`; `src/upi/features_v5.py:16–35`; `src/upi/common.py:35`; supporting windows in `features_v1.py:84–147`, `features_v2.py:62–76`.

The 147 October-dated training rows remain in joint event indexes. Remapping their fold to September changes their validation assignment, not their timestamps. They include 74 raw refund rows and 18 raw-legit rows; they are not literally all raw-labelled fraud. A train/test split is not an entity namespace, so independently generated rows on the same account can meet window predicates.

Actual test rows touching these tails:

| Link / radius | Test rows |
|---|---:|
| Same sender ±3 hours | 0 |
| Same receiver ±24 hours | 9 |
| Same device ±10 minutes | 0 |
| Same sender ±7 days | 38 |
| v5 parent/child link crossing the split | 0 / 0 |

Only two tails have positive bands. Removing all tail band sources and recomputing sender/receiver band counts and chain bands changes **zero** test features of those kinds and **zero** final predictions in both saved candidates. Ordinary long-window counts can still change; without fitted model artifacts, changing those inputs cannot be propagated through the saved `.npy` probabilities. Claiming zero effect on every feature/model would overstate this audit.

Bidirectional raw-transaction windows are not automatically label leakage in this offline competition: unlabeled test covariates are available. Band-derived windows need the separate supervised-feature treatment in finding 3. Blanket exclusion of cross-split history would also remove legitimate observable history.

Proposed fix: make the intended semantics explicit. Keep generic historical covariates separate from generator-event links; for the latter, separate train/test generation batches when they represent independent runs. Audit month-boundary events before assigning folds, and group complete physical events so their targets do not straddle training and validation. Preserve the October-tail fold mapping; do not call remapping alone a leakage fix.

**8. The 69-label/16-prediction difference starts at a changed raw-model recipe, not a demonstrated RNG failure.**

Locations: `run_all.py:37,61–65`; `src/upi/run_oof.py:38,42,55–57`; `src/upi/clean_labels.py:47,63,84–89`.

The two compared artifacts are a manually built candidate and the fresh-clone pipeline, not two verified clean executions of an identical model recipe. Evidence:

- Main and clone source text is identical for the reviewed driver/model/cleaner/band files after newline normalization. Byte hashes differ because checkout line endings differ.
- Stage-1 OOF probabilities are **bit-identical** across all 1,564,060 rows; stage-1 test probabilities are also bit-identical across all 260,582 rows. Both band edge arrays match.
- The first numerical divergence is `models/oof_raw/oof.npy`: **18,768,708 probability entries** differ, maximum absolute difference **0.569863**, and **357 argmax predictions** differ.
- Main raw OOF was written **13 Nov 21:01:16**, cleaned labels **21:02:29**. `features_v5.py` was created **21:06:06** and its table **21:06:14**. The original raw model/targets predate v5. The fresh run's log explicitly includes **v1,v2,v3,v4,v5** in the raw fit (`run_all_resume.log:67`). The old tag's feature list contains v1–v4. This is strong artifact/history evidence for a feature-configuration change; no original model feature manifest survives to provide a separate definitive fit-time record.
- Of 69 changed targets, **60 were raw-legit and nine raw-fraud**. **53** changed rows cross the 0.85 threshold; eight cross the 0.5 detection threshold. Remaining changes involve event votes/fallback types. `TX07307288` moves from `pf=0.901437` to **0.727073**, changing cleaned layering to legit. This is not a one-ULP threshold wobble.
- The copied fresh cleaned labels/final probabilities/weights exactly match the clone. Their changed training targets feed the final model and calibration; the resulting CSV differs from `s5_c3.csv` in **16 predictions**.

Holding predictions fixed, the target-version change contributes only **+0.00009469** to the old model's score, or **+0.00002091** to the fresh model's score. The full run difference is **+0.00057025**. These are far smaller than 0.017.

**Smallest recipe correction for reproducing the historical `s5_c3` workflow:** give raw-label fitting its own feature list rather than silently adding v5 when the final model's list changes:

```python
# Proposed change in run_all.py; NOT applied.
FEATS_RAW = 'features_v1.parquet,features_v2.parquet,features_v3.parquet,features_v4.parquet'
FEATS = FEATS_RAW + ',features_v5.parquet'
# Use FEATS_RAW at the raw OOF invocation (line 61).
# Keep FEATS at the final-model invocation (line 64).
```

Alternatively, define the latest all-v5 raw/final recipe as the canonical release and rebuild all dependent targets/models consistently. Do not mix these two recipes through existing caches. Matching the historical artifact after the proposed change remains to be verified by an authorized fit; this review did not train a model. There is no evidence here that a random-seed change is needed to explain the observed differences.

There is nevertheless a **latent deterministic-fit weakness**: `run_oof.py:55–56` specifies `random_state` but neither deterministic CPU reductions nor a fixed histogram strategy. For future repeated runs with identical inputs/versions, the smallest documented hardening is:

```python
# Additional LGBMClassifier arguments; proposed only.
deterministic=True,
force_col_wise=True,
# Also use --jobs 4 consistently; expose and forward it from run_all.py.
```

LightGBM documents that these settings stabilize CPU results for the same data and parameters, and warns that version/compiler/system differences can still change results. Without a forced strategy it benchmarks row-wise versus column-wise construction. See [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic). This is a portability safeguard, not a proven diagnosis of these 69 differences. It may itself change an older artifact, so archive the winning recipe before applying it. No repeat-fit claim is made under this task's no-fitting constraint.

**9. Existence-only caching can silently submit a different recipe, and the current repo does not reproduce the best upload by default.**

Locations: `run_all.py:40–46,52–67`; `src/upi/run_oof.py:42,69–72`; `src/upi/make_submission.py:30–41,88–93`.

Concrete cache failure: after one completed run, invoking `run_all.py --trees 300 --seeds 5 --name final_local` can skip every step, including the CSV, because outputs exist. Changing feature code, input labels or seed count likewise does not invalidate downstream artifacts. A different output name merely reruns submission processing on stale models. A partial stage with `test_proba.npy` present but `oof.npy` absent is also considered complete; `features_v3.parquet` alone is enough to skip writing required `band_edges.json`. `--force` avoided stale outputs in the fresh build, but this is a real operational failure mode.

Probability/label files contain no transaction IDs or class-order metadata. Several consumers load F1/F3/F5 positionally without checking IDs. A same-length probability file from a different base order can silently label the wrong transactions. `common.load_xy` checks feature IDs, and final submission merging correctly validates sample IDs/order; neither check validates IDs attached to an unlabelled probability array.

The best-upload problem is separately demonstrated: combining currently saved `oof_clean_v4/test_proba.npy` and `weights_s3_clean_v4.npy` with the current **own-band** rule differs from the actual `s3_clean_v4.csv` in **24 rows**; with current event rules it differs in **52 rows**. Thus merely pointing a judge at these arrays plus current band/rule code does not reproduce the winning file. The tag `s3_clean_v4` is `645494f3dee17478000d40acb4b2ed5e59ab5aae`; its cleaner and submission rules differ from HEAD. The tag is useful, but the hand-run commands and artifacts must also match it.

Proposed fix: cache a manifest hash over input hashes, source/version, feature order, target hash and full CLI parameters; require all outputs. Invalidate descendants when a dependency changes. Store IDs, class order and manifests alongside probabilities, and validate them at every consumer. For the release, provide a separate explicit command/config for the **actual best uploaded file**, run it in an empty output directory, and compare the result with its checksum. Do not use `--name s3_clean_v4` to imply that the current recipe produces that historical upload.

**What a judge still needs.**

The fresh-clone result establishes that this latest pipeline can run end-to-end. It does not yet establish byte-for-byte reproduction of the best public upload. The repo lacks a complete pinned environment and fit-time artifact manifests. `README.md:44` names packages without versions; the Colab notebook pins LightGBM/XGBoost but leaves other packages floating and uses five seeds, an optional GPU model and a different output recipe.

Provide these with the release:

1. Immutable commit plus exact per-stage commands for the chosen upload, including raw versus final features, label-cleaning version, stage-specific seeds, 25% legit sampling, trees, thread count, band/rule versions and saved class weights. Default `run_all.py` currently selects a newer candidate.
2. A fully resolved requirements lock, Python version, OS/architecture and LightGBM wheel/build identification. The audited environment is Python **3.11.0**, Windows AMD64; direct versions are below. They are an environment inventory, not a complete transitive lock.
3. Raw input filenames and checksums, row-order/feature-order/class-order hashes, source checksums, output probability/target hashes, and the expected winning CSV checksum. Data can remain organizer-provided and git-ignored.
4. If permitted by the competition's reproduction process, archived fitted models and their feature/category metadata in addition to code. Currently `run_oof.py` saves only probabilities, so a judge cannot rerun inference after changing features or inspect the fitted model recipe. If weights/artifacts must be regenerated, verify that exact path before release.
5. An explicit `--jobs` driver argument forwarded to every model subprocess. `run_oof.py` defaults to **12**; setting only `OMP_NUM_THREADS=4` does not override that explicit `n_jobs=12`. Arrow/BLAS limits should also be recorded. This review did not invoke those fitting subprocesses.
6. A clean-directory replay check that validates all output hashes and exact submission IDs/order. Once the same recipe is frozen, perform a second replay to test determinism; the current manual-versus-fresh comparison is not that test.

```text
numpy==2.4.6
pandas==2.3.3
pyarrow==21.0.0
scipy==1.16.3
scikit-learn==1.7.2
lightgbm==4.7.0
matplotlib==3.10.7
joblib==1.5.2
threadpoolctl==3.6.0
# Optional Colab/XGBoost branch only: xgboost==3.2.0
```

| Artifact | SHA-256 |
|---|---|
| raw/train.csv | `da48ff9b2823b8460511d8f5bdc660b4a92b74cc2a64395bddb32b89710f61c5` |
| raw/test.csv | `e3094af55340afc3239ac58f40c2e48b2040de9deb57463ac4d197e3aa0ec08b` |
| raw/accounts.csv | `97efcd06601dda981c1d1993b7b35202223c5f9f2be5fd7d13703df18a212539` |
| raw/merchants.csv | `8a2dbe1654b9c47a13385ee4313464a414cb7155eb4b8c2d80fc9b637c6c7d57` |
| raw/sample_submission.csv | `a0e0b01263dfccc7269221d54e27a675d12b10dabf03f10b7f7727f1c4aac4e1` |
| Winning uploaded s3_clean_v4.csv | `1fb3a2efbec085b52d610bf7213669964096e7f0de79315c1a3f093ddee0de62` |
| Older candidate s5_c3.csv | `4959af39b2c6dd8f881e6dbb1a3579249c05c3a4d390e47155ee8fe543dc5720` |
| Latest candidate final_local.csv | `810e432740170546ef2172d7b6550e5001edb1a824275d9aa3bd195c377a29be` |

**New files created for this review and how to reproduce the analysis.**

The report is this file. The four scripts below only read artifacts and print results; their saved outputs are the four linked text files. They do not call model fitting, weight fitting, or the writing entry points of the reviewed pipeline. Script 27 runs script 26 first, so its transcript also includes the earlier audit output; its saved transcript predates the additional latest-candidate summary added to script 26.

| New runnable file | New saved output | Contents |
|---|---|---|
| [26_codex_review_v2.py](../scripts/explore/26_codex_review_v2.py) | [codex_review_v2_audit.txt](codex_review_v2_audit.txt) | Frozen predictions, rules, blind bands, tails, chain checks, mixture and sampling scenarios |
| [27_codex_review_v2_labels.py](../scripts/explore/27_codex_review_v2_labels.py) | [codex_review_v2_labels.txt](codex_review_v2_labels.txt) | Exact cleaning reproduction, branch counts, cross-month votes and full chain examples |
| [28_codex_review_v2_provenance.py](../scripts/explore/28_codex_review_v2_provenance.py) | [codex_review_v2_provenance.txt](codex_review_v2_provenance.txt) | Clone/main probability comparisons, threshold crossings, source hashes, run logs and CSV differences |
| [29_codex_review_v2_noise.py](../scripts/explore/29_codex_review_v2_noise.py) | [codex_review_v2_noise.txt](codex_review_v2_noise.txt) | Missing-label sensitivity, upload reconstruction check, nearest-link failure and normalized source equality |

Run from the repository in PowerShell; these commands print to stdout and do not overwrite any report:

```powershell
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/26_codex_review_v2.py
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/27_codex_review_v2_labels.py
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/28_codex_review_v2_provenance.py C:/Users/ASUS/AppData/Local/Temp/claude/be1d3d9ececbd068/919cf462-8892-436a-b600-fdd1e5e3eb47/scratchpad/repro
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/29_codex_review_v2_noise.py C:/Users/ASUS/AppData/Local/Temp/claude/be1d3d9ececbd068/919cf462-8892-436a-b600-fdd1e5e3eb47/scratchpad/repro
```

Script 28 needs the comparison clone; change that argument if it is moved. Script 29's clone argument is optional. Scripts 26/27 and the sensitivity calculations use the saved artifacts named in their source and this report; they intentionally fail instead of fitting replacements when those inputs are absent.
