# Colab handoff: codex/target-070

All code is on [codex/target-070](https://github.com/zahy294/upi-fraud-ml/tree/codex/target-070).
The modelling branch is unchanged. The Python runner is the entry point;
the older notebook and run_all.py use the previous training interface.

For the recall upgrade after the reported 0.5297 submission, use a **fresh
`recall_v7` tag** and the current branch checkout. Do not resume `local_podium`
or an older `podium_v6` run with this revision. See
[RECALL_V7.md](RECALL_V7.md) for the experiment, comparison report and tests.

The teammate supplies the five competition CSVs and, optionally, the
submission reported to score 0.5400. That public score is user-reported;
this pipeline has not reproduced it or established a leaderboard improvement.

## 1. Open a fresh supported Python runtime and mount Drive

Use Python 3.11, 3.12 or 3.13. The pinned stack is not specified for Python3.14.
A GPU is optional: CPU is the default LightGBM backend.

```python
import sys
assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version
from google.colab import drive
drive.mount("/content/drive")
```

Put these files directly in a Drive folder, for example
/content/drive/MyDrive/upi_fraud_data:

- train.csv
- test.csv
- accounts.csv
- merchants.csv
- sample_submission.csv

The runner compares existing local files by SHA256 and refuses to overwrite
different inputs. Raw data and generated models are ignored by git.

## 2. Clone the correct branch and optionally pin a commit

```python
from pathlib import Path
import subprocess

ROOT = Path("/content/upi-fraud-ml")
REPO = "https://github.com/zahy294/upi-fraud-ml.git"
PIN = ""  # Optional: paste the full implementation commit SHA from the handoff.

if not ROOT.exists():
    subprocess.run(
        ["git", "clone", "--branch", "codex/target-070", "--single-branch", REPO, str(ROOT)],
        check=True,
    )
subprocess.run(["git", "fetch", "origin", "codex/target-070"], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "codex/target-070"], cwd=ROOT, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", "codex/target-070"], cwd=ROOT, check=True)
if PIN:
    subprocess.run(["git", "checkout", "--detach", PIN], cwd=ROOT, check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True)
```

The optional detached checkout permits an exact revision. The runner records
the commit and source hashes. It never commits or pushes from Colab.

## 3. Configure paths and inspect the plan

```python
import sys

DATA = Path("/content/drive/MyDrive/upi_fraud_data")  # EDIT
OUTPUT = Path("/content/drive/MyDrive/upi_fraud_runs")  # EDIT
TAG = "recall_v7"
BASELINE = None  # Optional Path("/content/drive/MyDrive/friend_05400.csv")

COMMON = [
    sys.executable, str(ROOT / "colab_train_submit.py"),
    "--root", str(ROOT),
    "--data-dir", str(DATA),
    "--output-dir", str(OUTPUT),
    "--tag", TAG,
    "--profiles", "ensemble",
    "--device", "cpu",
    "--install-deps",
]
if BASELINE is not None:
    COMMON += ["--baseline-csv", str(BASELINE)]

subprocess.run(COMMON + ["--stage", "all", "--dry-run"], check=True)
```

Dry-run only prints commands. It does not install dependencies, access data,
mount Drive, import ML packages or create outputs. --mount-drive can be
added to the runner instead of mounting Drive in cell1.

--install-deps bootstraps pinned virtualenv20.31.2, creates ROOT/.colab-venv,
pins pip25.1.1 and installs requirements-colab.txt there. Preparation, training, GPU probes
and validation all use that environment. pip check runs inside the isolated
environment, avoiding conflicts with unrelated Colab packages. Without this
flag, the current interpreter must already have the exact declared versions.
Repeat --install-deps when resuming so the same interpreter is selected.
On resume, an existing private environment is reused without recreating it or
installing packages; pip check and the complete version comparison still run.
A missing environment is installed before the strict comparison.

All required package versions are recorded, along with the complete installed
distribution set and requirements-resolved.txt. These pins were chosen for
compatibility; installation on an actual Colab runtime is externally untested.

## 4. Run validation, then finish the refit

Validation-only includes preparation and the teacher bootstrap; it produces
September reports without fitting the final submission ensemble:

```python
subprocess.run(COMMON + ["--stage", "validate"], check=True)
```

After reviewing validation, continue the same run with verified checkpoints:

```python
subprocess.run(COMMON + ["--stage", "all", "--resume"], check=True)
```

Or run everything immediately in a fresh tag:

```python
subprocess.run(COMMON + ["--stage", "all"], check=True)
```

Use one of the initial-run commands, not both without --resume.
The stages prepare, bootstrap, validate and refit include their prerequisites.
For example --stage refit also plans preparation/bootstrap/validation, skipping
completed stages only when --resume verifies them.

To run the four comparisons, replace the --profiles value with
control,fusion,temporal,ensemble before starting a fresh tag. This adds
external training time. Profile selection must match when resuming.
The runner does not select a winning profile using the assessment block.

## GPU requests

The installed LightGBM wheel is not assumed to support CUDA/OpenCL just
because the runtime has an A100. CPU defaults remain usable on a GPU runtime.
For an explicit GPU request, replace --device cpu with --device gpu or
--device cuda. The runner fits a tiny 24-row, two-round backend probe externally
before preprocessing. Unsupported backends stop with a logged error.

If CPU fallback is acceptable, also add --allow-cpu-fallback. A failed GPU
probe then verifies CPU fitting and records the actual CPU backend. The
effective configuration used by every training stage reflects that result.
A CPU fallback does not claim A100 acceleration. No GPU probe or training was
performed during local implementation.

## Preparation and temporal evaluation

The exact sequence is EDA00 -> EDA04 -> build_base -> ratio-ID check ->
v1 -> v2 -> v4 -> forward teacher -> fold-specific bands/chain/cleaning/v6 ->
validation/calibration -> full refit -> submission verification.
Report directories are created before EDA04. The inherited EDA helper defines
t as integer seconds from 2026-07-01; the trainer uses original timestamps
for calendar splits. Full-context sequence features precede row sampling.

v3 is bands.py output. v5 ch_band is rebuilt independently in validation and
refit contexts. Global data/processed band artifacts are not used for those fits.
Training labels are cleaned within eligible fitting contexts; reported
September metrics use original labels for all12 classes.

Splits: July-August fitting; Sep1-15 stopping; Sep16-23 calibration;
Sep24-30 assessment. The report also shows full September and dated tails.
September has already informed this project; the assessment block is not
claimed to be pristine. Cleaned-label agreement is never the headline score.

## Outputs, resume and verification

Local run: ROOT/runs/TAG.
Persistent copy: OUTPUT/TAG.
Both include manifest.json, environment information, source/config/input
hashes, backend, timings, stage dependencies, logs and RUN_REPORT.md.

Preparation caches are copied into OUTPUT/TAG/prepared_cache after preparation.
They contain the interim and processed parquet files, not additional raw CSVs.
This can consume substantial Drive space. The five raw CSVs must still be
available separately when resuming in a new runtime.

On failure the runner stops, attempts to copy partial logs/artifacts, and
preserves the original nonzero exit even if Drive copying also fails.
Resume verifies commit/source/config/input hashes, Python/dependency versions,
backend and all saved stage artifact hashes and upstream dependencies.
Changes require a fresh tag. Use the same local checkout path when restoring
from Drive, because effective configuration paths are part of provenance.
Logs append on retried stages.

For each profile:
- validation/metrics.json and per_class_metrics.csv contain raw-label metrics.
- validation/assessment_confusion_matrix.csv contains the 12x12 matrix.
- validation/offsets.json contains global offsets, conditional offsets and gate.
- validation/recall_selection.json records the Sep20-23 comparison and chosen
  share of the extra fraud-weighted member, when recall is enabled.
- validation/metrics.json includes the legacy ensemble comparison for every
  partition and all 12 classes, including misses as legit versus wrong fraud type.
- refit/submission.csv is the final CSV.
- refit/submission_metrics.json includes SHA256 and prediction counts.
- refit/prediction_distribution.json and validation_report.json contain checks.
- Member models, selected rounds, provenance, probabilities and matching IDs
  remain alongside those reports; bootstrap and contexts have their own folders.

The runner verifies sample row count/order/ID equality, unique IDs, exact
id,label columns, valid nonmissing labels, saved decision consistency, and
finite nonnegative normalized 12-class probabilities. Missing predicted
classes are reported rather than forcibly inserted.

With --baseline-csv it aligns IDs and reports fraud rates, class shares and
changed rows versus the provided 0.5400 CSV. This compares distributions,
not scores: neither submission supplies true test labels. Without that file,
the comparison is explicitly unavailable. No prediction-rate matching occurs.

## 5. Read metrics and download the submission

```python
import json
RUN = ROOT / "runs" / TAG
report = json.loads((RUN / "validation_report.json").read_text())
for profile, values in report["profiles"].items():
    print(profile, values["original_label_macro_f1"])
    for label, metrics in values["assessment_per_class"].items():
        print(label, metrics)
print((RUN / "RUN_REPORT.md").read_text())

from google.colab import files
files.download(str(RUN / "ensemble/refit/submission.csv"))
```

The persistent copy is OUTPUT/TAG/ensemble/refit/submission.csv.
Training runtime, Colab installation, hardware compatibility and leaderboard
performance must be established by this external run.

## Local implementation checks

Only syntax compilation, dry-run planning and standard-library mock tests
are run locally. No dependencies are installed and no data, model artifacts,
preprocessing, CV or real training are executed:

```text
rtk proxy python -m py_compile colab_train_submit.py tests/test_colab_runner.py
rtk proxy python tests/test_colab_runner.py
rtk proxy python colab_train_submit.py --root . --data-dir X:/example-data --output-dir X:/example-output --stage all --dry-run
```
