# Saved-probability diagnostics and candidate submissions

Use `postprocess_saved.py` from branch `codex/target-070` with the existing
`.colab-venv` interpreter. It never fits a model or reads training labels.
Existing run files are preserved; each output CSV needs a new name and gets
a JSON report with input/code hashes, settings and prediction counts.

Our submitted 0.5297 CSV has 260,582 rows and 3,901 fraud predictions (1.497%).
All 12 classes are predicted. Fraud means `label != "legit"`, not numeric1.
The saved probabilities reproduce this CSV byte-for-byte with its existing
offsets and rules. Its SHA256 is
`5a227fa64e0297a426c0bca4c6753bb0ffebbb382aa09c3e8137c7c80c85897d`.
The friend's exact 0.5400 file/probabilities were not identified locally when
this diagnostic was implemented, so the relative fraud counts are unknown.

## Compare submissions

Replace the friend path with the exact CSV that achieved 0.5400:

```powershell
.\.colab-venv\Scripts\python.exe postprocess_saved.py --profile-dir runs/local_podium/ensemble --baseline-csv "C:\path\friend_05400.csv"
```

Without `--baseline-csv`, this prints our counts alone. With it, the tool
checks identical unique ID sets, aligns by ID and reports per-class counts,
total fraud counts, label disagreement and fraud/legit disagreement. Numeric
binary submissions are rejected because the competition requires 12 named
classes. A difference in positive counts does not establish which model has
better recall or precision; that requires true labels.

## Blend probabilities without training

```powershell
.\.colab-venv\Scripts\python.exe postprocess_saved.py --profile-dir runs/local_podium/ensemble --baseline-csv "C:\path\friend_05400.csv" --friend-proba "C:\path\friend_05400_proba.npz" --friend-weight 0.65 --zero-offsets --output runs/zero_training/blend_065_035.csv
```

The probability file must contain all 12 classes and explicit transaction
IDs. Accepted formats:

- NPZ arrays `ids` (unique nonmissing string IDs), `proba` (N,12), `classes`
  (12 class names or numeric0..11 in the project's fixed class order).
  NumPy object arrays requiring pickle are rejected; export IDs as strings.
- CSV with `id` plus one numeric probability column for each class name.

Columns and rows are aligned explicitly. Probabilities must be finite,
nonnegative, normalized and within [0,1]. Hard-label CSVs and bare NPY arrays
cannot substitute for probabilities with known class/ID ordering.

This command averages friend's probabilities at0.65 and ours at0.35, uses
zero learned offsets and then applies our saved event-band rules.
Omit `--zero-offsets` to apply our learned offsets to the blend instead;
those offsets were fitted to our original ensemble, so transfer to a blend
is an unvalidated hypothesis. Our saved refit band/chain context is used,
not the friend's context. The 0.65 weight is user-specified and has not been
selected on paired validation probabilities. Pipeline differences alone do
not establish useful error diversity or a predictable improvement.

## Optional count matching

Add `--match-baseline-count` to that command to force the total fraud count
to match the supplied friend CSV. Or apply count matching to our own saved
probabilities:

```powershell
.\.colab-venv\Scripts\python.exe postprocess_saved.py --profile-dir runs/local_podium/ensemble --baseline-csv "C:\path\friend_05400.csv" --match-baseline-count --output runs/zero_training/count_matched.csv
```

`--top-k 3100` is also available for an explicitly chosen count. Top-K ranks
the best fraud-class log score minus the legit log score, including the
selected offsets. Fraud types are retained as multiclass decisions. Direct
bands remain protected and count toward K; a K below their count is rejected.
Other rows are ranked with transaction-ID tie breaking, and existing typing
rules apply after selection. K=0 and K=N are supported when feasible.

Count matching is optional and changes the decision boundary, not the raw
probabilities. It can increase or decrease macro-F1 and cannot diagnose a
refit threshold shift by itself. Our learned offsets are not refitted to
test-label guesses. Matching one scalar total also says nothing about the
11 separate fraud class counts.

## Adjust the saved boundary without matching a count

```powershell
.\.colab-venv\Scripts\python.exe postprocess_saved.py --profile-dir runs/local_podium/ensemble --fraud-log-offset 0.25 --output runs/zero_training/offset_plus025.csv
```

A positive shift lowers the fraud/legit boundary; a negative shift raises
it. This is a candidate setting, not a measured improvement. Prefer selecting
blend weights and shifts using aligned chronological validation probabilities
from both models, then inspect precision/recall/F1 for all 12 classes. Do not
repeatedly tune against a small public leaderboard subset.

## Refit and morning training

Rerunning the same saved-model decisions reproduces the same submission;
rerunning an unchanged deterministic refit is not a score-improvement method.
The postprocessor works independently of the old run's source fingerprint and
does not resume or alter it. No training minutes are needed for these outputs.

For the new fraud-weighted model, the teammate still needs the fresh
`recall_v7` training run described in [COLAB_PODIUM.md](COLAB_PODIUM.md).
It trains the extra member and selects its blend chronologically. In Colab,
use `ROOT / '.colab-venv/bin/python'` in place of the Windows interpreter and
pass the actual profile/Drive paths. The five raw CSVs are needed for training;
postprocessing needs only the saved profile artifacts and optional friend files.

No candidate here has measured public F1 or a guaranteed gain of0.012–0.018.
Crossing0.5505 must be established by evaluation and an actual submission.
