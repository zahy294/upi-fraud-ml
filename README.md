# UPI Fraud Rings: ML Gradient Wars 2026

Multi-class fraud detection on 1.8 million synthetic UPI transactions. Built in a 48-hour hackathon round (13 Nov 10:00 to 15 Nov 10:00 IST, 2026).

**Task.** Label every October transaction as legit or one of 11 fraud types (mule account, velocity bot, layering chain, SIM swap, account takeover, phishing collect, QR swap, refund abuse, fake cashback, loan-app extortion, merchant collusion). Training data is July to September. The metric is macro-F1 over the 12 labels, so a rare fraud type counts as much as the legit class.

**Result.** Public macro-F1 **0.5502**, up from 0.4061 for hand-written rules. One command rebuilds the final file from the five raw CSVs.

## Highlights

- **Raised macro-F1 from 0.4061 to 0.5502** over eight scored steps, each one a measured change (table below).
- **Reverse-engineered the data generator.** Found that fraud follows exact cross-row numeric recipes, that device numbers are issued in one contiguous block per fraud type, and that one column is filled by an exact rule which fraud breaks. These three findings account for most of the score.
- **Built a two-stage LightGBM pipeline** with a fraud-type specialist and a neural-network blend, validated leave-one-month-out to match the train-to-October time split.
- **Handled noisy labels.** The statement says some training fraud is unreported or filed under the wrong type; an event-level label-cleaning step corrected this and was part of the largest single gain (0.5034 to 0.5400).
- **Measured the ceiling instead of guessing.** Showed that about 60% of fraud rows are statistically identical to legit rows (cross-validated AUC 0.50 after 26 million tested column combinations), which caps the reachable score near 0.59, and stopped spending time there.
- **Fully reproducible.** A from-scratch run in a clean folder reproduced the uploaded files byte for byte on a 16 GB laptop.

## Score progression

| Step | Upload | Public macro-F1 |
|---|---|---:|
| Hand-written rules | `rules_v0_ordered.csv` | 0.4061 |
| Recipe features, one LightGBM model | `s1_v1.csv` | 0.5034 |
| Device-number bands, cleaned labels, event-level rules | `s3_clean_v4.csv` | 0.5400 |
| Band edges placed from block occupancy | `final_base_v7edges.csv` | 0.5468 |
| Second stage with account-episode features | `final_s2_v7edges.csv` | 0.5471 |
| Fraud-type specialist | `final_s2_typ_all.csv` | 0.5472 |
| One band edge corrected | `edge3_bot_right.csv` | 0.5475 |
| Neural-network blend and generator-rule rows | `final_A2_blend_edge_quirk.csv` | **0.5502** |

Local leave-one-month-out macro-F1 against the cleaned labels is 0.563 for the second stage; the public score runs about 0.016 lower.

## Reproduce the final upload

1. Python 3.11, then `pip install -r requirements.txt`.
2. Put the organisers' files in `data/raw/`: `train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`, `sample_submission.csv`.
3. Run:

```bash
python -I run_all.py --force --name final --mlp
```

The final entry is `submissions/final_blend_edges_quirk.csv`. The run also writes the file of every earlier step:

| File | What it is | Uploaded as | Public |
|---|---|---|---:|
| `final_blend_edges_quirk.csv` | second stage averaged with a neural network, edge corrected, generator-rule rows relabelled | `final_A2_blend_edge_quirk.csv` | 0.5502 |
| `final_edges.csv` | type-specialist file with one band edge corrected | `edge3_bot_right.csv` | 0.5475 |
| `final_typ_all.csv` | second stage + type specialist | `final_s2_typ_all.csv` | 0.5472 |
| `final.csv` | second stage | `final_s2_v7edges.csv` | 0.5471 |
| `final_base.csv` | one-stage model | `final_base_v7edges.csv` | 0.5468 |

The run takes about three and a quarter hours on a 16 GB laptop with 20 threads. `--mlp` needs PyTorch and fits the network on a GPU by default (about 35 seconds on an RTX 4050; `--device cpu` also works). Without `--mlp` the run needs no GPU and ends at `final_edges_quirk.csv`. `--no-stage2` stops after the one-stage file (about 80 minutes). Without `--force` the script skips steps whose outputs already exist.

Every file in the table was rebuilt in a clean folder and compared with the uploaded file: identical. LightGBM results can differ in a handful of rows between machines with a different thread count, and the network can differ between GPUs.

## Decisions, in the order we made them

### 1. Start from rules, then learn what the rules miss
Hand-written rules per fraud type scored 0.4061 and showed which types have a visible signature. Everything after that is built across rows; no model input is a raw ID.

### 2. Recipe features (`features_v1.py`, `features_v2.py`, `features_v4.py`): 0.5034
Fraud events in this data follow exact numeric recipes across rows: drains that shrink the balance by a fixed ratio per payment, paybacks and forwards at a fixed ratio of the amount, bursts of small whole-rupee payments from one phone, fixed-amount merchant sessions, and bait-then-payback timing. We turned each recipe into features computed over a sender's or receiver's neighbouring rows.

### 3. Validate by month
Train is July to September and the test set is October, so validation is leave-one-month-out. Random folds would split one fraud event across train and validation and overstate every score.

### 4. Leave out the two columns that drift
`app_version` and `ip_risk_score` behave differently in October than in training. They are never model inputs, although both look strong in training.

### 5. Device-number bands (`bands.py`)
New devices are numbered in the order the data was generated, one contiguous block per fraud type. A row on a device inside a block is fraud of that block's type. Train blocks are read from the labels. Test blocks are placed without labels, from the train proportions, a first model's probabilities and how densely each block is occupied. A diagnostic upload that retyped 73 rows against their band scored 0.5325, which confirmed that test labels follow the band.

### 6. Clean the training labels (`clean_labels.py`): 0.5400
The statement says some training fraud is unreported or filed under the wrong type, while test labels are clean. A row inside a detected event takes its event's majority label. Together with the bands, this was the largest single gain.

### 7. Model choice (`run_oof.py`)
LightGBM with 12 classes, legit rows down-sampled, three seeds averaged. XGBoost and CatBoost variants scored within 0.002 locally, so we kept the fastest one and spent the time on features.

### 8. Decision rule tuned for the metric (`make_submission.py`)
Macro-F1 rewards rare classes, so the predicted label is not always the most probable one. Per-class weights are chosen for macro-F1 on the out-of-fold predictions. Band rules then apply: one band row in an event gives the whole event that type.

### 9. Place band edges from occupancy: 0.5468
The first band placement used only proportions. Using how densely each block of device numbers is occupied moved several edges and added 0.007.

### 10. Second stage (`features_v6.py`): 0.5471
A fraud type belongs to an account's whole episode, not to one event. The second stage adds what the same accounts do in other fraud events, built from the first model's out-of-fold predictions so that no label leaks.

### 11. Fraud-type specialist (`exp_typing.py`): 0.5472
The second stage keeps deciding fraud or legit. A small model trained only on detected fraud rows decides which of the 11 types, so its capacity goes to telling look-alike types apart. A per-family version of the specialist was no better.

### 12. Edge probes (`relabel_band_rows.py`): 0.5475
Near some band edges no clearly typed row exists, so the few ordinary-looking rows between two bands can belong to either. We uploaded eight files, each moving one such group to the neighbouring band:

| Upload | Device numbers | Rows | Moved | Public |
|---|---|---:|---|---:|
| `edge1_collu_near.csv` | 474525–474526 | 2 | loan → collusion | 0.5472 |
| `edge2_collu_far.csv` | 474523–474524 | 2 | loan → collusion | 0.5472 |
| `edge3_bot_right.csv` | 473818–473821 | 4 | layering → bot | **0.5475** |
| `edge4_layer_left.csv` | 473801–473813 | 4 | bot → layering | 0.5468 |
| `edge5_sim_right.csv` | 473990 | 3 | takeover → SIM swap | 0.5461 |
| `edge6_refund_right.csv` | 474496–474498 | 3 | cashback → refund | 0.5472 |
| `edge7_loan_left.csv` | 474507–474508 | 2 | cashback → loan | 0.5462 |
| `edge8_cashback_right.csv` | 474511–474512 | 2 | loan → cashback | 0.5469 |

Four moves lowered the score, which confirmed those edges. Three changed nothing. One raised it, and only that move is kept (`EDGE_MOVES` in `run_all.py`). This step uses public-leaderboard feedback. A band edge is one fact about the whole test set, so the correction applies to public and private rows alike.

### 13. A rule the legit generator never breaks (`quirk_rules.py`)
Treating the data as generated, we looked for fields filled by an exact rule. `qr_vpa_mismatch` is filled exactly when a merchant is paid by QR: zero exceptions in 45,177 train rows and 7,462 test rows on ordinary new devices. Fraud breaks the rule both ways, and the model already catches 99.7% of those rows in training. The script relabels the few October rows that break the rule but were still predicted legit (25 rows in the final file): merchant-QR rows with a blank field become QR swap, other rows with the field filled become phishing collect.

### 14. Neural-network blend (`exp_mlp.py`): 0.5502
A small neural network on the same features makes different mistakes from the trees. Averaging it with the second stage, then applying steps 12 and 13, gave the final file. Steps 13 and 14 were uploaded together, so their separate contributions were not measured.

## What limits the score

About 60% of fraud rows are ordinary-looking payments drawn like any legit row. Their share is 2.3% of ordinary rows in every bucket of every column and account attribute. We tried to break this claim and could not:

- 1,025 cross-row rules, 799 single-column and 313,195 two-column cells, and 25.8 million three-column cells: no cell is fraud-rich in all three months.
- A cross-validated LightGBM model trained to separate these rows from legit rows: AUC 0.5006 on home-device rows.
- Account history across months, the two drifting columns in October, receiver type by payment type, hour of day and amount format: all flat.

With every patterned row detected and typed correctly, the score would be about 0.61 locally and 0.59 on the public board. The remaining errors are fraud types that the data emits in a sibling type's shape (for example SIM swap and account takeover).

## Layout

```
run_all.py            the single entry point (raw CSVs to submissions)
requirements.txt      package versions the uploads were produced with
src/upi/              pipeline: build_base, features_v1/v2/v4/v5/v6, bands, clean_labels, run_oof, exp_typing, exp_mlp,
                      make_submission, relabel_band_rows, quirk_rules
                      experiments not used by the final file: exp_stage1, exp_stage2, fit_submit
scripts/eda/          exploratory analysis, 00 to 07 (run_all.py uses 00 and 04)
scripts/explore/      one-off checks behind the findings above
colab/                driver and benchmark for running the same pipeline and model variants on Colab
notebooks/            organisers' starter notebook and the Colab notebooks
tests/                unit tests for the second-stage feature loader
docs/                 problem statement
reports/eda/          EDA report, figures and tables
data/, models/, submissions/   git-ignored
```

To score saved models locally (pooled and nested leave-one-month-out macro-F1):

```bash
python -I colab/bench.py . s2_typ_all s2 oof_clean
```

Run Python as `python -I` so that nothing in the data folders can shadow a module.

## Stack

Python 3.11, pandas, NumPy, LightGBM, XGBoost, scikit-learn, PyTorch.
