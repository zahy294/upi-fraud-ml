# Extra Trees with event composition

This is an experimental diversity candidate, not a replacement for the team's
0.5411 public baseline and not evidence for a promised0.642 score. Training is
run by the user. The existing dependencies suffice; no GPU is required.

**Measured update, 14 November:** the user's completed forest scored
**0.501632** on Sep24-30, below the saved 0.520807 reference. Neither the
bounded forest/hierarchy blends nor the alternative threshold search cleared
the selection margin. Do not refit this forest for a new submission based on
these results. The next isolated feature experiment is documented in
[EVENT_BOOST_V11.md](EVENT_BOOST_V11.md).

Extra Trees randomizes split thresholds across an ensemble of trees, providing
a different model family from the LightGBM/XGBoost/CatBoost boosting models.
That difference is a reason to test it, not proof of a blending gain.
[Pinned scikit-learn documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

## New features

`src/upi/event_composition.py` builds39 label-free features using sender±2h,
receiver±1h and device±10min windows. Each query excludes itself. The columns
measure peer counts, observed support and peer means for new payees, SIM age
at most24h, device age at most7days, payment/balance fraction and log amount.
Amount and payment/balance deviations compare the current payment with peers.

These test whether several weak row signals become informative when clustered
in the same event. They read neither labels nor model probabilities, and
windows cannot cross calendar months or train/test splits. Missing values have
their own observed-support denominator rather than being treated as safe or
suspicious. Receiver namespaces distinguish merchants from personal accounts.
This is permitted offline context, not a causal streaming feature calculation.
New predictive utility remains unmeasured until the model/ablation is evaluated.

## Training and evaluation

The runner reuses the completed `runs/hierarchy_v9/validation/cache` and the
teacher source `runs/challenge_v8`. It verifies source signatures, hashes and
feature IDs. It does not rerun the hierarchy models or train teachers.
The 248 cached features are supplemented with the39 new columns. Categories
are one-hot encoded using frozen levels; numeric imputation fits only on the
fitting rows. Predictions are processed in bounded chunks.

The forest configuration is `configs/extra_trees.json`:128 trees, depth at
most20, at most4096 leaves per tree, minimum8 samples per leaf, four CPU threads,
60% bootstrap samples,35% of features available per split, and seed642.
It retains the cache's inverse sampling weights. No new class weighting is
applied. These fixed parameters are a first experiment, not tuned optima.

Fit July–August, calibrate original-label class offsets on Sep16–23, report
Sep24–30 and all12 class metrics. There is no hyperparameter search on assessment
labels. These assessment rows have been reused during the project. Report
against local0.519038 and0.520807 references, not against public0.5411 as if they
were the same metric sample. Architecture and feature block both change, so
any gain needs a later feature-block ablation before attributing its cause.

## Commands

From `C:\Users\jesto\OneDrive\Desktop\hack4hack\upi-fraud-ml`:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe forest_experiment.py --stage prepare
rtk proxy .\.colab-venv\Scripts\python.exe forest_experiment.py --stage validate
```

`prepare` computes features only. `validate` fits **one** forest and saves
`runs/forest_v10/RUN_REPORT.md`, the model/imputer, probabilities, offsets and
checksums. It creates no competition submission. Keep the terminal open; the
forest prints progress. Its measured score is recorded at the top of this document.

Review that report before running:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe forest_experiment.py --stage refit
```

Final fitting rebuilds full-training bands/cleaning from cached teachers into
the new experiment folder, uses the frozen input/category definitions and
offsets, and writes `runs/forest_v10/refit/submission.csv` with ID/order/read-back
checks. Original runs remain unmodified. A repeated complete stage checks its
outputs and skips work; an interrupted fit restarts that stage, not individual
trees. Changed source/config/dependencies require a fresh `--output-dir`.

The friend-bundle reproduction/tuning workflow is separate:
[V4_MASTER_INTEGRATION.md](V4_MASTER_INTEGRATION.md). Do not blindly append this
forest's September probabilities to full-length month-held-out arrays. Align
IDs, prediction provenance and class order before any blend experiment.

## Verification

Synthetic tests exercise leave-self-out aggregation, missingness, row ordering,
label independence, split/month boundaries, receiver namespaces, category
mapping, fitting-only imputation, fixed12-class probability alignment and the
validation-to-refit CSV flow. Estimator training is mocked in tests. No real
forest or other estimator is trained by the coding agent.

Real-data preparation completed on14November:1,824,642 rows and39 new columns.
An exploratory binary-fraud ranking check on original July/August labels found
only modest standalone signal: sender-neighbour new-payee mean had AUC0.5414/
0.5358 and device-neighbour new-payee mean0.5456/0.5357. Receiver-neighbour means
were approximately chance. Missing peer evidence was set to zero for this
diagnostic only. These are not macro-F1 scores, clean-label estimates, or proof
of improvement beyond the existing feature packs. The unchanged feature block
remains a hypothesis to test with the new model, not a claimed breakthrough.
The local numeric diagnostic is saved as `runs/forest_v10/feature_audit.json`.
