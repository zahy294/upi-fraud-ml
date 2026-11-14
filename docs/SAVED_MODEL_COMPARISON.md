# Compare existing models without training

Use the original `local_podium` run and its original training CSV. This script
reads saved probabilities and original labels; no estimator is loaded or fit.
It uses the existing Python environment, without reinstalling dependencies.

From the repository directory in PowerShell, run this as **one line**:

```powershell
.\.colab-venv\Scripts\python.exe compare_saved_models.py --profile-dir runs/local_podium/ensemble --labels-csv ..\upi-fraud-rings\train.csv --output-dir runs/saved_model_comparison
```

The labels file SHA256 must match the original run manifest. Only its ID and
label columns are read, in chunks, retaining validation IDs. Saved probability
IDs, class order and stage provenance must agree. The script needs the
`local_podium` run manifest plus validation/refit A/B/C probabilities, original
ensemble probabilities, metadata, groups and band features. It does not need
the trained model TXT files or prepared feature matrices.

## Selection and evaluation

Nine policies are compared: the old calibrated ensemble, and broader
threshold searches for A, B, C, equal weights, AB/AC/BC equal pairs and the
original50/30/20 weights. The broad search matches the recall upgrade's
threshold search but does not require the new fraud-weighted member.

For each policy, offsets are fitted on Sep16-19 and scored on Sep20-23.
Selection must improve macro-F1 over the legacy policy by more than0.001;
ties retain the incumbent. The selected policy is frozen before examining
Sep24-30 assessment labels, then recalibrated on Sep16-23. Final predictions
use the same chosen weights and offsets on the existing refit probabilities.
Direct bands and event typing rules remain inside calibration and inference.

The report compares the selected policy's Sep24-30 metrics against the
original saved ensemble and original saved offsets, including all12 classes.
The separate candidates.csv additionally reports each candidate's later
assessment using its inner Sep16-19 offsets. Those columns are explicitly
different from the selected policy's final Sep16-23 calibration. They do not
change selection automatically. September has been used before in this
project, so this is not an untouched competition holdout.

If no policy passes the selection gate, the exact saved legacy probabilities
and offsets are reused. The output reports whether its CSV is identical to
the original. Do not resubmit an identical file expecting an improvement.
If the selected policy's later assessment regresses, that is evidence against
submitting it even though it won on the inner gate. Public F1 remains unknown.

## Outputs

- `runs/saved_model_comparison/COMPARISON.txt`: the concise result to share.
- `comparison.json`: selected weights, full12-class reports, gate scores,
  decision settings and input/source hashes.
- `candidates.csv`: all nine candidate comparisons.
- `per_class_metrics.csv`: selected and legacy assessment metrics.
- `submission.csv`: the selected candidate in original submission order.
- `test_proba.npz` and `offsets.json`: its saved probabilities and decisions.

The script prints progress for each policy. Calibration does CPU work and
can take minutes; no hardware-specific runtime is promised. It does not
retrain A/B/C and will not benefit from selecting a Colab GPU by itself.
An existing output directory is refused; choose a new name such as
`runs/saved_model_comparison_v2` for another deliberate run. Existing training
run files are preserved. Do not use the training runner's `--resume` for this.

Implementation verification uses only small synthetic probability artifacts:

```powershell
rtk proxy .\.colab-venv\Scripts\python.exe tests/test_saved_comparison.py
```

The command can be run by the user or teammate against the saved run.

Optional `--fraud-gate-search` adds eight policies testing total adjusted
fraud mass versus legit, on original/equal blends with log-odds thresholds
-0.5,0,0.5,1. Fraud type is chosen only after the fraud gate. Use a new output
directory. The agent has now run this comparison at the user's request:
`runs/fraud_gate_comparison/COMPARISON.txt` reports assessment0.520807 with
the selected equal-weight,-0.5-gate policy. No models were trained.
