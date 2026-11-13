# Recall experiment after the reported 0.5297 public score

Branch: `codex/target-070`. The friend's `modelling` branch is unchanged.

The previous run reported original-label assessment macro-F1 0.515393 and
public macro-F1 0.5297. The friend reported 0.5400; the target to beat is
0.5505. None of these establishes the new experiment's score. This revision
has synthetic/mock verification only; the teammate must measure its results.

## What changes

The original A/B/C ensemble, label cleaning, feature construction, sampling
and direct-band rules are retained. A fourth LightGBM member R uses the same
fit rows and features, seed211, 47 leaves, min_child_samples30 and feature
fraction0.8. No new dependency or GPU requirement is introduced. Expect one
additional model fit in both validation and refit, plus calibration work;
this is not a measured runtime estimate.

R multiplies the existing inverse-sampling weights by class weights computed
from unsampled, cleaned, eligible fit labels only. Legit stays at 1. For
each present fraud class k, the weight is
`clip(sqrt(n_legit / (number_of_present_fraud_classes * n_k)), 1, 6)`.
Absent classes keep weight1. All counts and weights are saved in R.json.
Validation stopping continues to use raw Sep1-15 labels and multi-logloss.
Final refit uses the selected validation round count without a stopping set.

The old local offset search can stall on a flat macro-F1 region: no prediction
changes after a small step, even though a larger step would cross a useful
decision boundary. The enhanced search first considers joint fraud shifts,
then coarse per-class offsets, then the existing fine coordinate search.
The grid is [-1,0,0.5,1,1.5,2,2.5,3], the bound is +/-3, and the existing
offset penalty remains. This explores recall/precision tradeoffs instead of
forcing more fraud predictions. Direct-band overrides remain inside the
objective and at inference. Enhanced conditional offsets are disabled to
limit additional tuning degrees of freedom.

## How the choice is made

Candidates are the unchanged A/B/C policy and enhanced thresholds applied to
blends with R shares 0, 0.25, 0.5 and 1.0. Zero means threshold-only improvement;
one means R alone. Each candidate fits thresholds using Sep16-19 and is
scored on Sep20-23 with all 12 classes and original labels. An enhanced
candidate must beat the legacy policy by more than 0.001 macro-F1. The best
eligible candidate wins; ties favor the incumbent. If no candidate improves
or an inner date block is absent, the legacy policy is retained.

Only the selected policy recalibrates on the full Sep16-23 block. Sep24-30
is reported afterward; its labels do not select the blend or thresholds.
The saved blend and offsets are reused in the full-data refit. R class
weights are recomputed from that refit's eligible training rows, which can
change probability calibration; the unseen public score remains uncertain.
September has already informed this project, so the assessment is not an
untouched competition holdout.

The runner reports the selected policy, its inner comparison, the later
assessment change versus the legacy ensemble, and precision/recall/F1
changes for every class. Per-class metrics distinguish fraud predicted as
legit from fraud assigned the wrong fraud type. Low raw-label recall can
also reflect noisy labels or wrong types; increasing recall alone is not a
success if macro-F1 falls. A favorable inner gate can still regress on later
assessment or the leaderboard.

## Morning training handoff

Follow [COLAB_PODIUM.md](COLAB_PODIUM.md) to pull this branch, mount Drive and
install the pinned environment. Set `TAG = "recall_v7"`. Use its complete
`--stage all` command for one run, or validate first and then `--resume`.
Always use a fresh tag for changed source/configuration. Keep the previous
submissions and their known scores; this run does not overwrite their tags.

For Windows, from the repository folder with the five CSVs in the sibling
`upi-fraud-rings` directory:

```powershell
py -3.12 colab_train_submit.py --root . --data-dir ..\upi-fraud-rings --output-dir "$env:USERPROFILE\upi-training-output" --tag recall_v7 --profiles ensemble --device cpu --install-deps --stage all
```

To resume this **new run**, using its existing environment:

```powershell
.\.colab-venv\Scripts\python.exe colab_train_submit.py --root . --data-dir ..\upi-fraud-rings --output-dir "$env:USERPROFILE\upi-training-output" --tag recall_v7 --profiles ensemble --device cpu --stage all --resume
```

The final file is `runs/recall_v7/ensemble/refit/submission.csv`; its persistent
copy is `OUTPUT/recall_v7/ensemble/refit/submission.csv`. Read
`runs/recall_v7/RUN_REPORT.md` and
`runs/recall_v7/ensemble/validation/recall_selection.json` before submission.
The report's legacy comparator is our A/B/C model, not the friend's 0.5400
CSV. Supply that CSV with `--baseline-csv` only for a distribution comparison.

To disable the upgrade for a separate controlled run, copy configs/podium.json,
set `recall.enabled` to false and pass `--config` with a fresh tag. This keeps
the original training and calibration behavior.

## Verification without training

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe tests/test_recall_policy.py
rtk proxy .\.colab-venv\Scripts\python.exe tests/test_podium_predictions.py
rtk proxy .\.colab-venv\Scripts\python.exe tests/test_podium_features.py
rtk proxy .\.colab-venv\Scripts\python.exe tests/test_colab_runner.py
```

Tests cover a flat-threshold regression, weighting bounds, gate acceptance
and fallback, date restrictions, direct-band precedence, JSON round trips,
mock four-member validation/refit with saved blend reuse, all-class reporting,
submission reconstruction and dependency-preserving resume. Synthetic scores
printed by tests are fixture checks, not estimates of competition performance.
