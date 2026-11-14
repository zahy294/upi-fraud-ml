# Challenge run targeting the reported 0.6338 leader

The screenshot supplied by the user shows team best0.5411 and leader0.6338.
This configuration is an experiment, with no measured new-model/public score.
All changes are on `codex/target-070`; training is performed by the user.

## Evidence and changes

The previous A/B/C models selected only26/28/25 rounds of a2000-round budget.
Ordinary multiclass log loss is dominated by legitimate rows; this suggests
the stopping objective deserves testing, not that longer training is proven
better. Original-label assessment macro-F1 outside direct device bands was
0.426429 (118660 rows). Labels are noisy, so raw-label recall is not a direct
estimate of clean-test recall.

The new config is `configs/challenge.json`:

- A retains ordinary log-loss stopping and unweighted labels as the reference.
- B uses class-balanced log loss and a minimum100 eligible boosting rounds.
- C uses macro-F1 across all12 classes, minimum150 rounds, and modest fraud
  weights (power0.35, cap4).
- R uses macro-F1, minimum150 rounds, fraud weights (power0.5, cap6), and
  training-label reliability weights. Unchanged unbanded raw fraud rows with
  an available forward teacher fraud probability below0.1 receive weight0.35.
  Promoted unreported rows receive weight0.6. Missing teacher evidence and
  direct bands do not trigger the suspicious-label downweighting.

All are LightGBM multiclass models, with up to1500 rounds and150-round
patience. B/C/R use learning rate0.035. A uses0.05. The warmup callback starts
round selection at the stated minimum; it does not retrospectively choose
an earlier round. Explicit custom metrics disable LightGBM's default
log-loss stopping metric. Final fitting reuses the selected round counts
without a validation stopping set. This can require substantially more CPU
time than the old25-round models. No measured runtime estimate is available.

Every model receives36 new peer features: support counts and11 fraud-class
teacher means for sender/hour, receiver/hour and device/10-minute cells.
The current row is subtracted from the group means. Receiver types are
namespaced. Groups never cross a calendar month or train/test boundary.
Training evidence uses forward-fold teachers; unavailable warmup queries
have missing means and zero support. This prevents the full-training test
teacher from feeding training features. Features use teacher probabilities,
not validation labels. Fixed time cells can split events at cell boundaries;
this is a deliberate bounded-computation first experiment.

The same existing event/band features, chronological folds and direct-band
rules remain in effect. Hard band edges can still be wrong; this revision
does not establish their correctness from the public leaderboard.

## Model selection

The new run's reference ensemble is A alone. Candidates are the reference
with broader offsets, each of A/B/C/R individually, and an equal four-way
blend. Their thresholds are fitted on Sep16-19 and evaluated on Sep20-23.
The reference and equal blend also test a separate fraud-detection decision:
sum the11 adjusted fraud probabilities, compare that mass against legit,
and then select the strongest fraud class. Candidate log-odds thresholds are
-0.5,0,0.5,1. They pass through the same temporal selection and saved-offset
checks. The original max-class decision remains a candidate. Direct-band
rules still apply after either decision method.
A candidate must beat the reference by more than0.001 macro-F1. Only the
chosen policy is recalibrated on all Sep16-23; the selected member weights
are reused for the final refit. Sep24-30 is reported afterward, with original
labels and metrics for all12 classes. Assessment data has already informed
the project and is not claimed to be pristine.

In the report, `legacy_baseline` refers to A in this new feature configuration,
not the original0.5297 submission or friend's0.5400 CSV. Compare the new
assessment additionally against the old0.515393 and saved-comparison0.519201.
Recall weighting, reliability weighting and the new peer features can all
regress; inner-gate selection is evidence, not a guarantee of test improvement.

## Run locally

From the project folder, use the existing pinned environment. Use a fresh tag;
source/configuration changes cannot resume `local_podium` or `recall_v7`.

```powershell
.\.colab-venv\Scripts\python.exe colab_train_submit.py --root . --config configs/challenge.json --data-dir ..\upi-fraud-rings --output-dir "$env:USERPROFILE\upi-training-output" --tag challenge_v8 --profiles ensemble --device cpu --stage all
```

This runs preparation, teacher fitting, validation, final fitting and CSV
verification. Keep the terminal open. To resume this exact run at the same
code revision, append `--resume` to that command. Do not add `--install-deps`
to a working local environment. A fresh machine can use the documented
installer workflow in [COLAB_PODIUM.md](COLAB_PODIUM.md), specifying this
config and tag. CPU is supported; a GPU request still requires the runner's
successful backend probe.

Outputs:
- `runs/challenge_v8/RUN_REPORT.md`
- `runs/challenge_v8/ensemble/validation/recall_selection.json`
- `runs/challenge_v8/ensemble/refit/submission.csv`

Persistent copies are under `USERPROFILE/upi-training-output/challenge_v8`.
Keep the whole run folder and the team's proven0.5411 submission. Inspect the
selected weights, new round counts, assessment change and all12 class F1s
before using a competition upload.

## Completed zero-training comparison

The agent ran `compare_saved_models.py` on the original saved probabilities
and verified the original training-label hash. Equal A/B/C weights with broad
offset search won the inner gate (0.505850 vs0.499521). Later assessment rose
from0.515393 to0.519201 (+0.003808). The new candidate is at
`runs/challenge_saved_comparison/submission.csv`, with full diagnostics and
provenance beside it. This is a local gain; its public score is unknown.
It does not include any newly trained challenge model.

A subsequent17-policy saved comparison tested the separate fraud gate.
Equal weights with a-0.5 gate won the inner comparison (0.506567) and raised
later assessment to0.520807 (+0.005415 versus the original0.515393).
That candidate and all12-class reports are at `runs/fraud_gate_comparison`.
Its public score is unmeasured. Repeated use of the same September assessment
reduces its independence; these small gains do not establish top-three odds.

## Verification

Synthetic checks cover leave-self-out evidence, missing teachers, receiver
namespaces, temporal/split isolation, order invariance, reliability weights,
class-balanced metrics, the real LightGBM callback with fake iteration events,
mock estimator argument wiring, and full mocked validation/refit/submission
with saved selection weights. No estimator was trained by the agent.

The peer feature builder also completed a read-only check on all1824642
existing base rows:36 features and no infinite values. Using the saved forward
teachers, available training peer support was28.94% for receiver/hour,0.93%
for sender/hour and0.36% for device/10-minute cells. Coverage demonstrates
that these features exist; it does not measure their predictive value.
Validation training metrics log every100 rounds in
`runs/challenge_v8/logs/validate_00.log`.
