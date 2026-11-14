# Event features with LightGBM

The forest scored 0.501632 locally. This experiment returns to the same
LightGBM learner, seed, cleaned training rows, negative sampling weights,
categorical encodings and stopping settings as the existing 0.519038 flat
baseline. The only feature change is appending the 39 cached event-composition
columns. This isolates whether the features help. No performance improvement
is claimed before the user runs validation.

The added columns measure nearby sender/device/receiver activity: new-payee
frequency, fresh SIM/device frequency, payment-to-balance ratios, amount
deviations and available peer support. Their builder excludes the query row,
separates months and train/test splits, and does not read labels. These are
offline competition features, not streaming features.

## What the saved-model checks found

The immutable local experiment records are `runs/diversity_v11` and
`runs/diversity_gate_v11`. They use original labels and reproduce the saved
reference: Sep20-23 selection macro-F1 0.506567; Sep24-30 assessment 0.520807.

First, a bounded search tested decision-interval calibration (with and without
offset regularization) and 10%, 25%, 50% forest/hierarchy blends. It retained
the reference. Second, the same blend shares were tested with the existing
fraud-gate calibration to separate blending from changing the decision rule.
The best gate candidate, 10% forest, scored 0.506879: +0.000312, below the
predeclared 0.001 required gain. It was rejected. No candidate assessment
scores were used to select blends. Both checks retained assessment 0.520807;
neither generated a submission, trained a model or consumed an upload.

`src/upi/boundary_calibration.py` remains an experimental, tested diagnostic;
the new training runner does not enable it. It scans class decision intervals
with rule-aware predictions. Equal margins move together, and actual predictions
are rescored to protect against floating-point boundary errors. It is not a
global optimizer, and its sharper calibration did not generalize better here.

The assessment week and selection dates have already been used in this project.
These numbers are comparative diagnostics, not a fresh unbiased test or public
score forecast. The team's 0.5411 public score is on a different dataset.

## Local commands

Run from the repository directory. The existing environment and artifacts
are sufficient; there is no dependency installation or full pipeline restart.

```powershell
cd C:\Users\jesto\OneDrive\Desktop\hack4hack\upi-fraud-ml
rtk proxy .\.colab-venv\Scripts\python.exe event_boost_train.py --stage check
rtk proxy .\.colab-venv\Scripts\python.exe event_boost_train.py --stage validate
```

`check` is read-only and verifies cache, source and teacher checksums. It passed
on the existing real artifacts. `validate` trains **one** LightGBM model using
248 existing features plus 39 new columns, with the baseline's six CPU threads.
It does not retrain the forest, hierarchy, teachers or old ensemble.

Fit July-August; stop on original Sep1-15 labels. Fit class offsets on Sep16-19
and choose between baseline/candidate on Sep20-23, requiring +0.001. Freeze
that choice, refit offsets on Sep16-23, then report Sep24-30 macro-F1 and
precision/recall/F1/support for all 12 classes. The saved baseline must reproduce
exactly. Original query labels stay unchanged.

Outputs are under `runs/event_boost_v11`, including `RUN_REPORT.md`, model,
ID-labelled probabilities, comparison, offsets, model rounds, input/code hashes
and completion checksums. Completed stages are verified and skipped. An
interrupted fit restarts that one stage. Changed inputs/code/environment require
a fresh `--output-dir`.

## Final refit

Only if the report says `Eligible for refit: True`:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe event_boost_train.py --stage refit
```

Eligibility requires both the selection gain and assessment exceeding the
stronger of the flat baseline and the 0.520807 saved reference by 0.001.
Otherwise the runner refuses refitting. This is a compute/submission guard,
not evidence that passing guarantees a leaderboard gain.

Final fitting rebuilds fitting-only bands/cleaning from cached teachers in the
new output folder, freezes category mappings, uses the selected number of rounds
without a stopping set, and reuses validation offsets. It trains one model and
writes `runs/event_boost_v11/refit/submission.csv`, checking exact IDs, sample
order, 12-class labels and CSV read-back. No competition upload is automatic.

## Reproduction and handoff

These commands need the completed local `runs/hierarchy_v9`, prepared
`runs/forest_v10` features, `runs/challenge_v8` teacher artifacts,
`runs/fraud_gate_comparison` report and underlying processed/raw data.
They are intentionally ignored by Git. A teammate must copy these artifacts
alongside this branch or reproduce the prerequisite runs; pulling code alone
does not download trained models or competition data. The friend's five v4
probability folders are still a separate missing artifact bundle; see
[V4_MASTER_INTEGRATION.md](V4_MASTER_INTEGRATION.md).

To repeat the saved-probability checks without training, choose new folders:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe compare_diverse_models.py --mode boundaries --output-dir runs/diversity_repeat_boundaries
rtk proxy .\.colab-venv\Scripts\python.exe compare_diverse_models.py --mode gate --output-dir runs/diversity_repeat_gate
```

Tests cover ID alignment, missing/duplicate features, checksum/provenance
failures, calibration date restrictions, tie handling, protected band rules,
selection rejection, frozen refit rounds and sample-ordered submission output.
All estimator fits are mocked in tests. The coding agent ran no real estimator
training. Training and the new model's local score remain for the user to run.

Verification on 14 November: **117 tests passed**, Python compilation passed,
and real-artifact preflight passed. The older test suite imports `upi` from
`src` and needs PyYAML, which is absent from the training environment. For the
full test run, PyYAML 6.0.2 was installed into an ignored test-only folder,
leaving `.colab-venv` dependencies unchanged:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe -m pip install --no-deps --target runs/test_dependencies/pyyaml_602 PyYAML==6.0.2
rtk proxy .\.colab-venv\Scripts\python.exe -c "import sys,unittest; sys.path[:0]=['src','tests','runs/test_dependencies/pyyaml_602']; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests')); sys.exit(not result.wasSuccessful())"
```

All implementation commits/pushes belong only to `codex/target-070`.
