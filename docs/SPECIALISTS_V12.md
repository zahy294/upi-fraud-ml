# Six-candidate specialist sweep

The event-feature ablation scored 0.519411, below the saved local reference
0.520807. It does not justify refitting. This new experiment changes the
learning problem and representation rather than claiming those features won.
No new real model was trained by the coding agent; the user runs training.

## Candidates

Two learners each run on three fixed feature views:

- Eleven independent LightGBM binary classifiers, one for each fraud type.
  Each stops on average precision for its own type, rather than a shared
  multiclass log loss. Maximum 600 rounds, patience 75, 31 leaves, learning
  rate 0.04, four CPU threads, fixed seeds. Existing inverse sampling weights
  are retained; there is no additional class weighting.
- Multinomial logistic regression with L2 regularization, C=1, SAGA, maximum
  200 iterations, tolerance 0.001. Numeric inputs use signed log compression,
  fitting-only median imputation and scaling; categorical inputs use
  fitting-only one-hot encoding. Inverse sampling weight ratios are retained.
  Convergence warnings are saved. Unconverged candidates cannot win selection.

Views are the original 248 features; those plus 39 event features; and that
full set with names containing `band` plus `dev_ext_legit` removed. The last
view tests whether reliance on band-derived columns obscures other patterns.
Other relational/chain features remain. Existing label-free inference rules
are held constant across candidates, including the saved reference.

The independent binary outputs do not form a joint probability distribution.
Their explicit adapter sets legitimate mass to max(1-sum(fraud scores), 1e-9)
and normalizes all twelve columns. These are decision scores that still need
held-out calibration, not a claim of calibrated probabilities.

LightGBM's binary objective and stopping interface are documented in its
[official API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).
The linear learner uses the installed scikit-learn 1.6.1 implementation.
No new dependencies are required. XGBoost and CatBoost were considered but
are not included in this sweep; XGBoost is absent from this local environment.
This is six bounded candidates, not an exhaustive search over algorithms.

## Run locally

From the repository folder:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe specialist_sweep.py --stage check
rtk proxy .\.colab-venv\Scripts\python.exe specialist_sweep.py --stage validate
```

The first command only verifies inputs. The second trains **33 binary models
and three linear models**, sequentially. This is substantially more work than
the last single-model run; runtime is not yet measured. Every binary specialist
and every completed candidate is checksummed and checkpointed. Run the same
command to resume; only an interrupted individual fit restarts. Source/config/
dependency changes require a new `--output-dir`.

It reuses the completed hierarchy cache, forest event features, challenge
teachers, local_podium probabilities and fraud_gate_comparison artifacts.
These data/model artifacts are ignored by Git: teammates need the existing
artifact folders, not just a checkout. Existing runs are never modified.

Fit July-August cleaned labels; binary early stopping uses original Sep1-15
labels. Fit offsets on original Sep16-19 labels and select individual members
or the equal blend on Sep20-23 with a 0.002 minimum gain. Refit the chosen
offsets on Sep16-23, then report Sep24-30. The reference's 0.520807 assessment
must reproduce. Reports include precision/recall/F1/support for all 12 classes.
Neither selection nor stopping uses the assessment labels. Those dates have
already been reused elsewhere in this project, so this is not a new untouched
test set. The result cannot guarantee the public 0.63 target.

Read `runs/specialists_v12/RUN_REPORT.md`. Final fitting is permitted only if
selection chooses a new member/blend and assessment improves over the reference
by more than 0.002. Otherwise keep the existing submission. When eligible:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe specialist_sweep.py --stage refit
```

Only selected new members are refitted; existing reference test probabilities
are reused if its blend weight is positive. Binary tree counts are frozen from
validation, without a stopping set in refit. Linear preprocessing is fitted on
final fitting rows only. Outputs include `runs/specialists_v12/refit/submission.csv`
with strict class/ID/order checks and CSV read-back. No upload is automatic.

## Verification

Synthetic tests mock the estimators and cover feature views, score normalization,
binary class mappings, chronological stopping inputs, fixed refit counts,
per-class checkpoint resume/tamper detection, and fitting-only categorical
preprocessing. Existing chronological selection/submission tests also apply.
Real-artifact preflight passed without model fitting. All changes belong to
`codex/target-070`; no friend's branch is modified.
