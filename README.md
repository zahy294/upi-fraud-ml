# UPI Fraud Rings: ML Gradient Wars 2026

Our entry for the 48-hour round (13 Nov 10:00 to 15 Nov 10:00 IST): label every October UPI transaction with one of 12 labels, scored by macro-F1.

**Best public score: 0.5472** (`final_s2_typ_all.csv`). The command below rebuilds that file from the five raw CSVs.

## Reproduce the best upload

1. Python 3.11, then `pip install -r requirements.txt`.
2. Put the organisers' files in `data/raw/`: `train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`, `sample_submission.csv`.
3. Run:

```bash
python -I run_all.py --force --name final
```

This writes three files:

| File | What it is | Uploaded as | Public |
|---|---|---|---:|
| `submissions/final_typ_all.csv` | second-stage model + type specialist (the final entry) | `final_s2_typ_all.csv` | 0.5472 |
| `submissions/final.csv` | second-stage model | `final_s2_v7edges.csv` | 0.5471 |
| `submissions/final_base.csv` | one-stage model | `final_base_v7edges.csv` | 0.5468 |

The run takes about three and a quarter hours on a 16 GB laptop with 20 threads and needs no GPU. `--no-stage2` stops after the one-stage file (about 80 minutes). Without `--force` the script skips steps whose outputs already exist.

All three uploaded files came from one from-scratch run of this pipeline in a clean folder on 14 Nov. LightGBM results can differ in a handful of rows between machines with a different thread count; on the same machine the files come out identical.

## Method

Everything is built across rows; no model input is a raw ID.

1. **Recipe features** (`features_v1.py`, `features_v2.py`, `features_v4.py`). Fraud events in this data follow exact numeric recipes across rows: drains that shrink the balance by a fixed ratio per payment, paybacks and forwards at a fixed ratio of the amount, bursts of small whole-rupee payments from one phone, fixed-amount merchant sessions, bait-then-payback timing.
2. **Device-number bands** (`bands.py`). New devices are numbered in the order the data was generated, one contiguous block per fraud type. A row on a device inside a block is fraud of that block's type. Train blocks are found from the labels; test blocks are placed without labels, from the train proportions, a first model's probabilities and how densely each block is occupied.
3. **Cleaned training labels** (`clean_labels.py`). The statement says some training fraud is unreported or filed under the wrong type, and test labels are clean. A row inside a detected event takes its event's majority label.
4. **One-stage model** (`run_oof.py`). LightGBM, 12 classes, legit rows down-sampled, three seeds. Validation is leave-one-month-out over July, August and September.
5. **Second stage** (`features_v6.py`, then `run_oof.py` again). Adds what the same accounts do in other fraud events, using the first model's out-of-fold predictions. A fraud type belongs to an account's episode, not to one event.
6. **Type specialist** (`exp_typing.py`). The second stage keeps deciding fraud or legit. A small model trained only on detected fraud rows decides which of the 11 types, so its capacity goes to telling look-alike types apart.
7. **Decision rule** (`make_submission.py`). Per-class weights chosen for macro-F1 on the out-of-fold predictions, then band rules: one band row in an event gives the whole event that type.

`app_version` and `ip_risk_score` change behaviour in October and are never model inputs. No test row was labelled by hand.

Local scores (leave-one-month-out macro-F1 against the cleaned labels): 0.5625 for the final file and 0.5630 for the second stage alone. The public score runs about 0.016 lower. The three uploaded files are within 0.0004 of each other on the public board, which is inside the noise.

## What limits the score

About 60% of fraud rows are ordinary-looking payments drawn like any legit row. We measured that nothing in the files separates them: their share is 2.3% of ordinary rows in every bucket of every column and account attribute. With every patterned row detected and typed correctly the local score would be about 0.61. An independent search asked to break this claim (`docs/redteam_stealth.md`) tested 26 million column-combination cells and a cross-validated model, and found none (AUC 0.5006 on home-device rows). The remaining errors are fraud types that the data emits in a sibling type's shape. Findings are in `docs/IMPLEMENTATION_PLAN.md` (update boxes at the top) and `reports/eda/EDA_REPORT.md`.

## Uploads

| File | Public | Code state |
|---|---:|---|
| `rules_v0_ordered.csv` | 0.4061 | `scripts/eda/07_baselines.py` (hand rules) |
| `s1_v1.csv` | 0.5034 | recipe features, one model |
| `s3_clean_v4.csv` | 0.5400 | git tag `s3_clean_v4` |
| `final_base_v7edges.csv` | 0.5468 | this branch, `run_all.py` |
| `final_s2_v7edges.csv` | 0.5471 | this branch, `run_all.py` |
| `final_s2_typ_all.csv` | **0.5472** | this branch, `run_all.py` |

Five `probe_*.csv` files were diagnostic uploads and score lower by design. The last, `probe_simband_shape.csv` (0.5325), retyped 73 rows in the SIM-swap device band by the shape of their event; the drop confirms that test labels follow the device band.

## Layout

```
run_all.py            the single entry point (raw CSVs to submissions)
requirements.txt      package versions the uploads were produced with
src/upi/              pipeline: build_base, features_v1/v2/v4/v5/v6, bands, clean_labels, run_oof, exp_typing, make_submission
                      experiments not used by the final file: exp_mlp, exp_stage1, exp_stage2, fit_submit
scripts/eda/          exploratory analysis, 00 to 07 (run_all.py uses 00 and 04)
scripts/explore/      one-off checks behind the findings
colab/                driver and benchmark for running the same pipeline and model variants on Colab
notebooks/            organisers' starter notebook and the Colab notebooks
docs/                 problem statement, plan and findings, reports from review agents
reports/eda/          EDA report, figures and tables
data/, models/, submissions/   git-ignored
```

## Benchmark and variants

```bash
python -I colab/bench.py . s2_typ_all s2 oof_clean        # pooled and nested leave-one-month-out macro-F1 of saved models
```

Variants we tried (XGBoost, CatBoost, a neural network, a per-family type specialist, blends) all score within 0.002 of the second stage locally. To build the neural-network blend (needs PyTorch):

```bash
F=features_v1.parquet,features_v2.parquet,features_v3.parquet,features_v4.parquet,features_v5.parquet,features_v6.parquet
python -I src/upi/exp_mlp.py . mlp --features $F --labels models/oof_raw/y_clean.npy --device cuda --test
python -I src/upi/make_submission.py . s2 --blend s2,mlp --labels models/oof_raw/y_clean.npy --name blend_s2_mlp
```

Run Python as `python -I` so that nothing in the data folders can shadow a module.
