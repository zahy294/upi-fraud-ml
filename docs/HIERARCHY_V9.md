# Two-stage fraud experiment

Target: public macro-F1 0.66. This is an objective, not a forecast. No real
hierarchy model has been fitted by the coding agent. The user runs training.

## Why change direction

`challenge_v8` assessed at **0.517891**, below the prior saved-policy result
**0.520807**. A/B/C/R stopped at 26/102/205/169 rounds; selection kept only A.
The stronger weights and longer stopping budgets did not establish a gain.
The same run's reported legacy score, 0.516398, refers to its own A model.

The next experiment separates two tasks:

1. A binary LightGBM learns fraud versus legitimate. Stop on original-label
   September 1–15 average precision, rather than ordinary multiclass log loss.
2. An eleven-class LightGBM learns the type **only on fraud-labelled fitting
   rows**. Stop on class-balanced log loss on fraud-labelled September 1–15.
3. Combine as `P(legit)=1-P(fraud)` and
   `P(type=k)=P(fraud)*P(type=k | fraud)`.

This can prevent the large legitimate class from dominating type learning;
it does not remove label noise or prove better ranking. All class mappings
are explicit. The existing fitting-only cleaned labels are reused. Stopping,
calibration and assessment labels remain original.

A flat twelve-class model is trained on the **same features and same sampled
rows** as a control. The feature set is base plus v1–v5, excluding the previous
teacher-peer and temporal target-encoding additions. This is a new controlled
architecture comparison, not an isolated ablation against challenge_v8.
Legitimate sampling is restored by inverse sampling weights. No new aggressive
fraud weights or automatic relabelling are introduced.

Fit months are July–August. Offsets fit September16–19; selection evaluates
September20–23. Candidates are flat, hierarchy and their equal blend. Duplicate
flat entries in the shared selector are equivalent, not extra hypotheses.
Hierarchy/blending must improve the inner gate by over 0.002 macro-F1. Freeze
selection, recalibrate on September16–23, then report September24–30 and all
twelve class metrics. This assessment period has been reused in this project.
The flat comparator's assessment is reported separately, as are reference
scores from previous experiments. Do not equate this score with clean-test F1.

## Local execution

Use the existing `.colab-venv`; there are **no new dependencies**. Run from the
repository on `codex/target-070`.

```powershell
.\.colab-venv\Scripts\python.exe hierarchical_train.py --stage check
.\.colab-venv\Scripts\python.exe hierarchical_train.py --stage prepare
.\.colab-venv\Scripts\python.exe hierarchical_train.py --stage validate
```

`check` verifies prepared data, source implementation and cached teacher/context
artifacts. `prepare` builds features only, with no model fitting. Preparation
works on complete transaction context before negative sampling and saves a
checked matrix for reuse. `validate` performs three fits (flat, detector, types),
using CPU with six threads. The type fit uses only the much smaller fraud set.
No measured real training runtime or hierarchy F1 is available yet.

Read `runs/hierarchy_v9/RUN_REPORT.md` and `validation/selection.json`. A repeated
command verifies and skips a completed stage. If a training stage is interrupted
before completion, it repeats that stage's three fits; it is not a per-tree
checkpoint. Do not change code, dependencies or config within a resumable run.
For a new experiment, pass a fresh `--output-dir`.

Only after reviewing validation, request final fitting:

```powershell
.\.colab-venv\Scripts\python.exe hierarchical_train.py --stage refit
```

Final fitting uses all training rows, fit-specific bands/cleaning, frozen
categorical schema, selected validation round counts and frozen offsets/blend.
It has no early-stop evaluation set. Output:
`runs/hierarchy_v9/refit/submission.csv`. The script checks IDs, class names,
sample ordering and CSV read-back and writes `submission_report.json`.
It never uploads to the competition. Preserve the team's proven submission.

The default source is `runs/challenge_v8`, which is never modified. Source and
new output folders must be separate. To use another path, supply `--source-run`.
Large datasets, caches and models are intentionally not in Git. A teammate
needs the same `data/processed` files plus source `inputs.json`, `bootstrap`
and `contexts/validation` folders, or their own compatible completed challenge
run. Pulling code alone cannot recreate these already-trained teacher artifacts.
Copy these private artifacts within the team, not into a public repository.
No branch other than `codex/target-070` is used.

## New-pattern investigation

Run `audit_motifs.py --output runs/motif_audit_fresh.json` for the predeclared
July/August-only coverage probes. It fits no models and saves input/code hashes.

- Looking up to four subsequent eligible transfers exposed two August rows
  with a quarter-balance skipped link; **one** was outside the existing nearest
  drain match. No July hits; no skipped 0.35-balance hits in either month.
- The strict device timing-regularity probe (at least six events, gap CV at most
  0.25 in a bounded 30-minute cell) had zero hits in either month.
- Repeated near-identical amounts within a sender/payee week selected three
  refund-labelled July rows and three collusion-labelled August rows. This tiny
  sample does not establish a useful new general rule or unseen coverage.

These findings do not justify enabling new hard rules. Thirteen continuous
motif features remain available behind `include_motifs`, **off by default**.
To test them later, copy the config, enable the flag and use a separate output
directory; compare against this experiment before spending a submission.

A separate saved-A event propagation probe used only probabilities and bounded
device, sender, payee and pair windows. On September20–23, before calibration,
raw-plus-existing-rules F1 was 0.500922; propagation strengths0.25/0.5 did not
change it, and0.75/1 gave0.500993. This is an exploratory selection-window
result, not later assessment or a clean-test score. Propagation is not enabled
in the training runner. `src/upi/event_rescue.py` retains the tested method.

## Validation and limitations

Synthetic tests cover binary/type mappings, legitimate-row exclusion from the
type fit, refit without stop labels, chronological selector use, final CSV
ordering, skipped links, row-order invariance, label independence and separation
between months/splits. Estimator fits in these tests are mocks; a perfect toy
score is not a measured competition result. Real cached-file preflight and
feature preparation are separate from model training.

This experiment might regress. A different architecture, more code or a Kaggle
idea cannot establish 0.66 without actual measured results.

Completed local checks on 14 November: input/cache preflight passed; validation
preparation produced **804,512 rows and 248 features**, and the second prepare
invocation verified/reused the cache. The six new synthetic tests and 53 existing
feature/prediction/runner tests passed (59 total). No real estimator was fitted.
On this checkout, only the `--stage validate` command above is needed next.
