# Coding and A100 training handoff

Prepared 13 November 2026. Read [TARGET_070_IMPLEMENTATION_PLAN.md](TARGET_070_IMPLEMENTATION_PLAN.md) first. Primary target is 0.70 macro-F1; current team public score is 0.5034 by user report. The newer submission's recipe is unknown. Do not equate local validation with public performance or claim a target score before measuring it.

## Starting instruction for GPT-6.1 Sol, high

Implement the revised UPI Fraud Rings plan in this workspace, on the user's feature branch. Begin with hardware/environment inspection, dataset audit and a reproducible chronological baseline. Then implement prioritized feature blocks with ablations, reporting all 12 class metrics. Preserve raw inputs. Use the same package and configurations locally and in a thin Colab driver. The teammate will pull an exact code commit and train on an A100. Research probes are already complete; use their positive and negative findings. Do not rerun the literature search as the first implementation step.

The branch name and repository arrangement have not been established. Inspect Git status/remotes before any checkout or initialization, preserve existing work, and choose an isolated descriptive feature branch if needed. Do not nest a new repository inside an existing checkout. This workspace currently contains supplied CSVs and planning artifacts, not a cloned copy of the friend's full implementation repository. Use the referenced source snapshot for comparison; check the remote's current state before integrating it. Publishing/pushing or uploading competition predictions requires the user's authorization for that action.

## Existing files

- `upi-fraud-rings/`: original train/test/accounts/merchants/sample CSVs, problem statement and starter notebook. Keep these bytes unchanged and exclude raw data from version control before adding files.
- `docs/IMPLEMENTATION_PLAN.md`: canonical plan and evaluation contract.
- `docs/reference/`: frozen cloud handoff and friend's plan/EDA at source commit `f2a9e923ff42898de4333ddad5d144a9b9a75768`.
- `docs/ADDITIONAL_SIGNAL_RESEARCH.md`: primary-source mechanism notes; the canonical plan overrides its literature-based priority ordering.
- `scripts/research/` and `reports/research/`: reproducible narrow probes, not production feature builders or trained-model results.

The current shell is PowerShell. User instructions require shell commands prefixed with `rtk`; `rtk proxy` worked during planning. Installed Python observed during the probes was 3.14.7 with pandas 3.0.6 and NumPy 2.5.2. Inspect supported Python/library combinations for production rather than assuming these are the desired pinned training versions. Hardware has not yet been fully inspected.

## Small proposed implementation layout

Use `src/upi/` with modules for data/audit, splits, row/history/event features, model fitting, metrics/decisions and artifact/submission writing. Add specialist/stacking modules only when those experiments are justified. Put configurations in `configs/`, a single entry point in `run.py`, and the thin driver in `notebooks/colab_train.ipynb`. Keep dependency pins or a lockfile in Git; keep datasets, feature caches, models, credentials and large prediction files outside Git.

Proposed CLI contract to implement (these commands do not exist yet):

```text
python run.py audit --config configs/baseline.yaml
python run.py features --config configs/baseline.yaml
python run.py train --config configs/baseline.yaml --device cpu
python run.py evaluate --run-dir runs/<run_id>
python run.py predict --run-dir runs/<run_id> --output submissions/<candidate>.csv
python run.py smoke --config configs/smoke.yaml --device cuda
```

Configurations specify data/output paths, immutable class mapping, train/calibration/check dates, feature-context horizon, feature families, model/library/device, seed and decision-calibration policy. Treat inference-only target data separately from labels. `train` on a final-fit configuration must be distinguishable from validation training.

## First coding milestone

1. Inspect CPU/RAM/GPU/disk/Python and locate all required files; hash inputs. Ask for the 0.5034 candidate's recipe only if it cannot be found in the available project/remote context, and continue independent baseline work while waiting.
2. Audit schema, label values, timestamp ranges/tails, duplicates, join cardinalities and sample-submission ID alignment. Resolve paths through config, not hardcoded laptop locations.
3. Reproduce the documented rules and fixed row/relational baseline. When reusing EDA code, convert top-level scripts into side-effect-free functions and join features by ID.
4. Save chronological split manifests, full 12-class metrics, raw and calibrated probabilities, decision parameters, model files and configuration/environment manifests. Keep calibration and later assessment separate.
5. Add meaningful checks for time-horizon boundaries, equal timestamps, current-row exclusion, absent classes, ID joins/order and label isolation. In particular, changing validation labels must not alter any training features, selected weights or predictions.
6. Run a small complete smoke test locally and on Colab before large fits. Small-sample performance is a wiring check, not a competitive result.

## Subsequent experiments

Start with interactions and omitted metadata, then strictly earlier-month entity histories. Compare feature packs with identical training/tuning protocols. Track additional true positives outside old signatures and the false-positive cost; the measured historical-phishing residual is small and cannot be treated as a strong override. Test bounded episode summaries and specialists only when residual errors support them. Preserve an ordinary global classifier so candidate gates do not discard off-pattern positives.

The latest 0.5034 model may already contain some proposed features. Once its recipe is known, compare against that implementation and avoid duplicating completed experiments. Preserve its predictions for error-overlap and complementary-model analysis, with validation provenance checked before blending.

## Branch-to-Colab handoff

Each training handoff must identify a **full commit SHA**, config name/hash, dependency lock, input hashes and expected output paths. The branch is the collaboration channel; the SHA identifies the code actually trained. Commit the run entry point, package, config and notebook together so a partial branch update cannot start an incoherent run.

The teammate fetches the branch, checks out the agreed commit in a clean Colab checkout, verifies `git rev-parse HEAD`, installs the pinned environment and checks the GPU with `nvidia-smi`. Use a secure supported Git authentication mechanism if the repo is private; avoid embedding tokens in notebook code or Git remote URLs.

Copy input/feature files to runtime-local storage for execution. Compare input hashes, run the smoke command, then run the specified validation experiment. Save run artifacts to durable storage before disconnecting. Only after validation selection freezes should the teammate use the final-fit config and create a submission candidate.

Each returned run bundle contains:

- Commit SHA, config, environment/device details and input/feature fingerprints.
- Validation IDs, untouched labels, aligned probabilities, fixed class order, metrics for all 12 classes and confusion matrix.
- Training logs, chosen iteration count, timing/memory and fitted model files.
- Test probabilities and CSV when produced, with their hashes and ID-validation results.
- A short experiment note explaining the feature change and its measured result.

Use unique run directories; never overwrite a prior best run. Resume only when input/code/config/horizon fingerprints match. GPU reproducibility relies on retaining fitted artifacts as well as retraining instructions. A disconnected Colab session must not erase the only copy of a model or candidate.

## Done criteria and remaining information

The first implementation milestone is done when the audit, chronological baseline and all 12 metrics are reproducible on a specified commit, and the teammate can launch the same pipeline from the Colab notebook. Subsequent milestones are measured improvements, the 0.55 intermediate public milestone, then the 0.70 target; none is predeclared achieved.

Still to obtain during coding: recipe/commit/file for the 0.5034 upload; actual local hardware; chosen branch/remote; actual Colab device and environment; shared artifact location; current upload quota. These do not block completing this plan. Do not send messages to the teammate or change remote access as part of planning.

