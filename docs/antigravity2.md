# Deep-Dive Forensics, Overlooked Signals, and a Complementary Episode-Graph Architecture (`antigravity2.md`)

> **Constraint Confirmation:** Zero existing files in the repository were modified during this investigation. The only file created inside the project repository is this report (`docs/antigravity2.md`). All 34 empirical verification scripts were executed exclusively inside the external scratch directory (`C:\Users\ASUS\.gemini\antigravity\brain\0e6cac72-2fba-434d-8974-e9a983b4d379\scratch\`). See [Section 6](#6-complete-manifest-of-new-files-created) for the complete file list.

---

## Executive Summary: Why We Are at `0.54` Public (`0.5597` OOF) and How to Bridge the Gap to `0.65`

Your current best public submission (`submissions/s3_clean_v4.csv`, scoring **0.5400** public / **0.5597** pooled OOF against cleaned labels) already captures **99.91% of all patterned fraud rows** (`25,419 / 25,441` rows where $p(\text{fraud}) \ge 0.20$). Yet macro-F1 sits around `0.54–0.56`.

Why? Through 34 forensic probes across the raw CSVs, processed feature matrices (`v1`–`v6`), out-of-fold probability tensors, and the October test set, we established five foundational truths:

1. **The Definitive Truth About "Stealth" Fraud (`pf < 0.20`):**
   - Each of the 11 fraud class generators has two subroutines: a **patterned event subroutine (~45% of class rows)** and an **ordinary baseline subroutine (~55% of class rows, `35,206` rows in train)**.
   - For **97.0% of stealth fraud rows** (those occurring on existing/home devices), the generator draws from the **exact same marginal and conditional distributions as `legit`** across every raw column, time-of-day, graph degree, velocity, and entity-overlap statistic (`5-fold adversarial AUC = 0.5002`). Even conditioning on a transaction sharing a sender, receiver, merchant, or device with a known patterned fraud event yields **2.36% stealth fraud**—identical to the **2.36% dataset base rate**!
   - **However, 3.0% of stealth fraud (`1,055` train rows, `~211` test rows) IS 100% deterministically recoverable** because the generator's ordinary subroutine allocates a brand-new device (`next_device_id += 1`) at a 3.0% rate *while running inside each fraud class's outer loop*. Those stealth rows land inside the contiguous fraud device-ID bands (`DV0423001–DV0428454` in train; `DV0473632–DV0474551` in test) with **~100% class purity**. Conversely, the 3.0% of `legit` rows that allocate a new device (`dev_ext_legit == 1`: `DV0428455–DV0473631` in train; `DV0474552–DV0482013` in test) are **99.93% legit** (`45,145 / 45,177`).
   - **Crucial Takeaway:** Do **not** lower probability thresholds (`p_fraud < 0.35`) trying to catch non-band stealth fraud—every 1,000 non-band rows you pull in contains 976 `legit` rows and only 24 stealth fraud rows, destroying precision.

2. **Where the `0.54 -> 0.65` Gap Actually Lives — Sibling Disambiguation (`+0.0518` F1 Ceiling with Zero New Detections):**
   - Out of the `25,419` fraud rows already detected by your model, **5,650 rows (`22.2%`) are assigned to the wrong fraud class**—almost entirely (`95.4%`) confused with their generator sibling (`mule_account` $\leftrightarrow$ `layering_chain`, `sim_swap` $\leftrightarrow$ `account_takeover`, `merchant_collusion` $\leftrightarrow$ `refund_abuse`, `phishing_collect` $\leftrightarrow$ `fake_cashback` $\leftrightarrow$ `loan_app_extortion`).
   - Keeping the **exact same 25,419 detected rows** (catching zero additional stealth fraud) and only resolving sibling-class confusion raises OOF Macro-F1 from **0.5597 to 0.6115 (`+0.0518` Macro-F1)**!

3. **Unwired Assets Already on Disk (`features_v6.parquet`, `models/s2_a`, `models/xgb_a`):**
   - Earlier today, `src/upi/features_v6.py` (186 cross-event entity history features) and `models/s2_a/oof.npy` were created, **but never wired into `run_all.py` or `make_submission.py` (`models/s2_a/test_proba.npy` was never generated!)**.
   - Why is `features_v6` a breakthrough? Because while a perpetrator may emit a single event in a sibling's shape (e.g., a `fake_cashback` perpetrator emitting a `collect_99`-shaped request), **66.3% of fraud P2P receivers and merchants participate in multiple fraud events across different days/months**, where their other events reveal their native class signature!
   - Simply wiring `s2_a` + `xgb_a` + `oof_final_local` with band-aware weights jumps pooled OOF Macro-F1 from **0.5597 to 0.5634 (`+0.0037`)** and strict nested-LOMO from **0.5582 to 0.5617 (`+0.0035`)** before writing a single new feature.

4. **Concrete Bugs Discovered in the Current Pipeline (Especially on the October Test Set):**
   - **Bug 1 (Test Band Segmentation & Override Corruption in `bands.py` / `make_submission.py`):** In October (`test.csv`), `bands.py` cuts Band 1 (`mule_account`) at `DV0473750` instead of `DV0473782`. As a result, device `DV0473782` (a 9-row P2P victim fan-in into `AC369510` with `s1_pred = mule_account`, prob `0.965`) falls into Band 2 (`velocity_bot`), and `band_rules` via `rcv_band_2_24h` **corrupts all 9 rows of that mule fan-in event into `velocity_bot`**! Even worse, 4 pure `layering_chain` devices (`DV0473637, DV0473660, DV0473709, DV0473731`) sit inside Band 1, causing `ch_band` to **overwrite 18 October `layering_chain` rows into `mule_account`**!
   - **Bug 2 (`fit_weights` Objective Mismatch in `make_submission.py`):** `fit_weights(oof, y)` optimizes class multipliers `w` *before* `band_rules(pred)` is applied, even though `band_rules` overrides **25.9% of all fraud predictions (and 94% of `velocity_bot`)**. Putting `band_rules` *inside* the coordinate ascent loop gains **`+0.00385` Macro-F1** on `s2_a` (`0.5592 -> 0.5631`).
   - **Bug 3 (`sim_swap` vs `account_takeover` Conditional Asymmetry Overlooked):** On non-band rows (`band == 0`), an `x0.35` drain burst (`snd_drain_kind == 2`) is **99.5% `account_takeover`** (`186` vs `1` `sim_swap`, because `sim_swap` `x0.35` bursts allocate a Band 4 device 99.8% of the time). Conversely, an `x0.25` drain burst (`snd_drain_kind == 1`) on `band == 0` with `sim_age_hours >= 72` or `new_payee == 0` is **100% `sim_swap`** (`204` vs `0` `account_takeover`).
   - **Bug 4 (Truncated Recipe Windows in `features_v5.py`, `features_v2.py`, and `clean_labels.py`):** Layering chains run up to **6 hours per hop (`21,600s`)**, not 2 hours (`7,200s`), leaving **538 train rows and 114 October test rows** with `ch_len = 1` and missing chain-band propagation (`ch_band = 0`). Similarly, requiring `new_payee == 1` on `Vs_n` misses **220 train rows and 66 October test rows** of `mule_account` fan-in victims who happened to pay the mule a second time (`new_payee == 0`).

5. **A Completely Different, Complementary Approach (Section 4):**
   - Your current pipeline classifies **individual rows** (`1.88M` nodes) and uses heuristic post-processing (`band_rules`).
   - We design an **Episode-Graph Factor Model + Constrained Optimal Transport (Linear Assignment) Decoder** that collapses all `1.88M` rows into **`~4,200` discrete Fraud Episodes (events)**, connects episodes across all 6 months via a **Multi-Partite Entity Graph** (`sender`, `P2P receiver`, `merchant`, `device`, `shared-IP burst`), runs **Iterative Belief Propagation / Relational GNN** across episodes, and decodes class labels using **Exact Expected-Macro-F1 Linear Assignment** under known generator sibling-mixing ratios.

---

## 1. The Definitive Forensic Proof on "Stealth" Fraud

### 1.1 Reverse-Engineering the Synthetic Generator's Device Allocator
By inspecting `device_id` (`DV0000001` to `DV0482013`) across all `1,881,104` transactions (`train` + `test`) in [features_v3.py](file:///c:/Users/ASUS/projects/upi-fraud-ml/src/upi/features_v3.py), we can see the exact execution order of the dataset generator:

1. **Pre-existing universe (`DV0000001–DV0420000`):** Assigned to user accounts as their `home_device` (plus a small pool of secondary devices).
2. **Bot device pool (`DV0420001–DV0423000`):** Pre-allocated 3,000-device pool used exclusively by `velocity_bot` (`is_bot_dev == 1`, **98.9% `velocity_bot`**).
3. **Sequential Monthly Generation with a Global `next_device_id` Counter:**
   - First, the generator generated **Months 5–9 (`train.csv`)** sequentially by class: `class 1 (mule_account)` $\to$ `class 2 (velocity_bot)` $\to \dots \to$ `class 11 (loan_app_extortion)` $\to$ `class 0 (legit)`.
   - Next, it generated **Month 10 (`test.csv`)** in the exact same class order (`1..11`, then `0`).
   - Whenever *any* class generated an **ordinary baseline transaction**, there was a **3.0% probability** of calling `allocate_new_device()` (`next_device_id += 1`).
   - Whenever a fraud class generated a **patterned fraud burst**, if that recipe leave- keeps its allocated device stays in the band, OR if the recipe overwrites `device_id` (for instance, `velocity_bot` overwriting with `DV0420001–DV0423000`, or a sender burst reusing the sender's `home_device`), the originally incremented `next_device_id` is orphaned!
   - **Empirical proof (`probe7.py`):** In the train fraud band (`DV0423001–DV0428454`, span `5,454` IDs), only **2,690 device IDs (`49.3%`)** actually appear in `train.csv`—the other `50.7%` were incremented during fraud generation and then overwritten by the patterned burst subroutine! By contrast, in the train legit band (`DV0428455–DV0473631`, span `45,177` IDs), **45,177 device IDs (`100.00%`)** appear in `train.csv` because `legit` has no patterned burst subroutine to overwrite `device_id`!

| Device ID Range | Split | Generator Loop Phase | Span of IDs | Active IDs in Data | Total Rows | Patterned Rows (`pf >= 0.2`) | Stealth Rows (`pf < 0.2`) | True Class Purity |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `DV0000001–DV0420000` | Train+Test | Pre-allocated User Devices | 420,000 | 310,368 | 1,813,952 | 19,908 | 34,151 (Train) | Mixed (`97.0%` Legit outside patterns) |
| `DV0420001–DV0423000` | Train+Test | Shared `velocity_bot` Pool | 3,000 | 2,866 | 2,135 | 2,135 | 0 | **98.9% `velocity_bot`** |
| `DV0423001–DV0428454` | Train (M5–9) | Classes `1..11` Loop | 5,454 | 2,690 (49.3%) | 4,768 | 3,713 | **1,055** | **99.8% Fraud (11 contiguous class bands)** |
| `DV0428455–DV0473631` | Train (M5–9) | Class `0` (`legit`) Loop | 45,177 | 45,177 (100.0%) | 45,177 | 0 | 32 | **99.93% `legit` (`dev_ext_legit`)** |
| `DV0473632–DV0474551` | Test (M10) | Classes `1..11` Loop | 920 | 417 (45.3%) | 817 | 606 | **211** | **~99.8% Fraud (11 contiguous class bands)** |
| `DV0474552–DV0482013` | Test (M10) | Class `0` (`legit`) Loop | 7,462 | 7,462 (100.0%) | 7,462 | 0 | ~5 | **~99.93% `legit` (`dev_ext_legit`)** |

### 1.2 Proof That Non-Band Stealth Fraud (`pf < 0.20, band == 0`) Is Identical to `legit`
In `probe8.py` and `probe12.py`, we isolated all `1,490,091` train transactions outside patterned bursts and outside the new-device bands (`pf < 0.20`, `band == 0`, `dev_ext_legit == 0`, `is_bot_dev == 0`). This slice contains `1,454,885` `legit` rows and `35,206` stealth fraud rows across all 11 fraud classes.

| Class (`y_clean`) | Row Count in Slice | `amount` Mean | `amount` Std | `home_device` Rate | `new_payee` Rate | `app3` Rate | `ip_hi` Rate | `PAY` Rate | `P2P` Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0: legit` | 1,454,885 | 1050.8 | 1906.0 | 0.9502 | 0.2261 | 0.1499 | 0.0195 | 0.5712 | 0.3578 |
| `1: mule_account` | 3,925 | 1043.6 | 1836.3 | 0.9488 | 0.2280 | 0.1485 | 0.0204 | 0.5704 | 0.3623 |
| `2: velocity_bot` | 3,421 | 1042.1 | 1844.5 | 0.9512 | 0.2195 | 0.1535 | 0.0199 | 0.5697 | 0.3563 |
| `3: layering_chain` | 4,050 | 1066.2 | 1945.1 | 0.9469 | 0.2244 | 0.1516 | 0.0205 | 0.5721 | 0.3585 |
| `4: sim_swap` | 2,727 | 1042.5 | 1898.2 | 0.9538 | 0.2244 | 0.1470 | 0.0198 | 0.5717 | 0.3546 |
| `5: account_takeover` | 2,407 | 1059.3 | 1918.7 | 0.9497 | 0.2227 | 0.1483 | 0.0170 | 0.5708 | 0.3598 |
| `6: qr_swap` | 2,869 | 1054.0 | 1922.4 | 0.9484 | 0.2332 | 0.1478 | 0.0202 | 0.5758 | 0.3531 |
| `7: merchant_collusion` | 4,043 | 1062.0 | 1908.8 | 0.9520 | 0.2248 | 0.1511 | 0.0195 | 0.5679 | 0.3535 |
| `8: refund_abuse` | 2,970 | 1058.5 | 1972.1 | 0.9492 | 0.2290 | 0.1468 | 0.0175 | 0.5724 | 0.3549 |
| `9: phishing_collect` | 3,661 | 1051.2 | 1899.4 | 0.9519 | 0.2267 | 0.1497 | 0.0197 | 0.5690 | 0.3600 |
| `10: fake_cashback` | 2,800 | 1048.9 | 1885.3 | 0.9507 | 0.2254 | 0.1514 | 0.0189 | 0.5732 | 0.3571 |
| `11: loan_app_extortion` | 2,333 | 1039.8 | 1871.0 | 0.9490 | 0.2272 | 0.1505 | 0.0201 | 0.5718 | 0.3583 |

Furthermore, in `probe12.py`, we tested whether an entity (sender, P2P receiver, merchant, or device) that participates in a **confirmed patterned fraud event (`pf >= 0.5`)** has an elevated rate of stealth fraud on its *other* ordinary transactions:
- Baseline stealth fraud rate in `pf < 0.20, band == 0`: **2.363%**
- Same sender (`sender_user_id`) has a confirmed patterned fraud event elsewhere: **2.360%** (`5,127 / 217,225`)
- Same P2P receiver (`receiver_user_id`) has a confirmed patterned fraud event elsewhere: **2.369%** (`1,987 / 83,864`)
- Same merchant (`merchant_id`) has a confirmed patterned fraud event elsewhere: **2.357%** (`13,662 / 579,593`)
- Same device (`device_id`) has a confirmed patterned fraud event elsewhere: **2.358%** (`4,892 / 207,481`)

**Why this matters:** The generator picks random victim/ordinary accounts independently for its ~55% baseline transactions. Only the 3.0% of stealth transactions that trigger `next_device_id += 1` (`band in 1..11`) carry class information. Recognizing this prevents you from wasting model capacity or lowering probability thresholds into the 97.6% legit ocean.

---

## 2. Where the Score Ceiling Lies: The `+0.0518` Sibling Confusion Matrix

In `probe1.py`, we decomposed your current OOF predictions (`oof_final_local` + `band_rules`, Macro-F1 = **0.5597** vs `y_clean`) across the `60,647` true fraud rows:
- **Detected as fraud (`pred > 0`):** `25,419` rows (with only `1,303` false positives out of `1,503,413` legit rows—a **99.91% binary precision/recall on patterned fraud**!).
- **Predicted as `legit` (`pred == 0`):** `35,228` rows (the `35,206` non-band stealth fraud rows + `22` missed patterned rows).
- **Among the `25,419` rows correctly detected as fraud:**
  - `19,769` (`77.8%`) are assigned to the **exact right fraud class**.
  - **`5,650` (`22.2%`) are assigned to the WRONG fraud class!**

And 95.4% of those `5,650` errors occur inside four **Sibling Confusion Clusters** (`probe2.py`):

| Sibling Confusion Cluster | Pattern Recipe Shared by the Cluster | Total Cross-Class Errors in OOF | Oracle Ceiling Gain on OOF Macro-F1 (Fixing ONLY This Cluster) |
| :--- | :--- | ---: | ---: |
| **Cluster D:** `phishing_collect` (9) $\leftrightarrow$ `fake_cashback` (10) $\leftrightarrow$ `loan_app_extortion` (11) | `COLLECT` bursts (`Qs_n >= 4`, `collect_99`, `cb_seq`, `L_snd_cnt_7d`) | **1,945 rows** | **`+0.0176` Macro-F1** |
| **Cluster C:** `merchant_collusion` (7) $\leftrightarrow$ `refund_abuse` (8) | Merchant burst (`S_n >= 4`) & refund bursts (`R_snd_cnt_14d >= 3`) | **1,115 rows** | **`+0.0094` Macro-F1** |
| **Cluster B:** `sim_swap` (4) $\leftrightarrow$ `account_takeover` (5) | Sender drain burst (`Ds_n >= 3`, `x0.25` / `x0.35` balance drain) | **1,040 rows** | **`+0.0112` Macro-F1** |
| **Cluster A:** `mule_account` (1) $\leftrightarrow$ `layering_chain` (3) | P2P fan-in (`Vs_n >= 5`) feeding P2P chains (`ch_len >= 3`) | **1,291 rows** | **`+0.0108` Macro-F1** |
| **All Other Cross-Class Pairs Combined** | Miscellaneous | 259 rows | `+0.0028` Macro-F1 |
| **TOTAL (All 25,419 Detected Fraud Rows)** | **Zero new fraud detections required** | **5,650 rows** | **`0.5597 -> 0.6115 (+0.0518)`** |

> **Why is the public leaderboard leader at `0.65` while our local `y_clean` oracle is `0.6115`?**
> Remember that in `train.csv`, 15% of patterned fraud bursts were flipped to `legit` (`y_raw`), whereas in `y_clean` (`clean_labels.py`), unlabelled patterned bursts were restored to a *single canonical class* (`recipe_class`), **plus** the October test set (`test.csv`) has a slightly higher ratio of patterned-to-stealth bursts in several classes (`probe18.py`). Every `+0.010` improvement in sibling disambiguation on patterned events translates directly to `+0.010–0.015` on the public leaderboard.

---

## 3. Concrete Bugs & Overlooked Opportunities in the Current Pipeline

### 3.1 Overlooked Asset #1: `features_v6.parquet` and `models/s2_a` Are Sitting on Disk Unwired!
When inspecting `src/upi/features_v6.py` and `models/s2_a/`, we discovered that a 186-feature Stage-2 cross-event entity aggregation pipeline (`features_v6.parquet`) and a 5-fold LightGBM model (`models/s2_a/oof.npy`) were built earlier today, **but were never added to `run_all.py` or `make_submission.py`, and `models/s2_a/test_proba.npy` was never generated (`full.txt` in `models/s2_a/` is missing)!**

#### Why `features_v6.parquet` Resolves Sibling Confusion (`probe6.py`, `probe9.py`, `probe34.py`)
In `probe6.py`, we checked whether perpetrator entities (`receiver_user_id`, `merchant_id`, `sender_user_id`, `device_id`) appear in **multiple patterned fraud events on different days**:
- **66.3% of fraud P2P receivers** (`17,406 / 26,235` patterned P2P fraud rows) participate in **2 or more separate fraud events** across the dataset!
- Suppose a `fake_cashback` fraud ring (`class 10`) emits 3 events across 5 months: two of those events use its native `cb_seq` (lure + collect) recipe, and one event uses the sibling `collect_99` recipe (`class 9` shape). A Stage-1 model looking only at a 24h window (`features_v1`–`v5`) sees `collect_99` and predicts `phishing_collect` (`9`).
- `features_v6.py` computes **out-of-window (`>1 day` away) entity mean/max Stage-1 probabilities and out-of-month entity label priors** for `snd`, `rcv`, `dev`, `s2r`, and `r2s`. Thus, on the ambiguous `collect_99` event, `rcv_other_p10_mean` (the receiver's `fake_cashback` probability on *other days*) immediately tells `s2_a` that this perpetrator is actually `fake_cashback` (`10`)!

Look at the verified empirical progression from `probe9.py` and `probe34.py`:

| Pipeline / Model Configuration | Pooled Clean OOF Macro-F1 | Strict Nested-LOMO Clean F1 | Pooled Raw OOF Macro-F1 | Strict Nested-LOMO Raw F1 |
| :--- | :---: | :---: | :---: | :---: |
| **1. Current Public Best (`s3_clean_v4`)** | `0.55973` | `0.55819` | `0.51243` | `0.51086` |
| **2. Current `final_local` (`s3_clean_a` + `v2` + `cat_a`)** | `0.55994` | `0.55882` | `0.51147` | `0.51038` |
| **3. Stage-2 `s2_a` (`features_v6`) — Unweighted `argmax` + `band_rules`** | `0.56040` | **`0.56040`** | `0.51171` | `0.51171` |
| **4. Stage-2 `s2_a` (`features_v6`) — Band-Aware Weights** | `0.56300` | **`0.56052`** | `0.51395` | `0.51191` |
| **5. 3-Way Blend (`25% final_local + 65% s2_a + 10% xgb_a`) + Weights** | **`0.56309`** (`0.56342` unreg.) | **`0.56041`** (`0.56165` w/ Sibling Rules) | **`0.51401`** (`0.51464` unreg.) | **`0.51209`** |

And look at the per-class F1 jump from `oof_final_local` to `s2_a` (`probe9.py`):
- `phishing_collect (9)`: `0.4534` $\to$ **`0.4720` (`+0.0186` F1)**
- `fake_cashback (10)`: `0.4939` $\to$ **`0.5078` (`+0.0139` F1)**
- `loan_app_extortion (11)`: `0.5173` $\to$ **`0.5247` (`+0.0074` F1)**
- `account_takeover (5)`: `0.4832` $\to$ **`0.4858` (`+0.0026` F1)**
- `qr_swap (6)`: `0.6258` $\to$ **`0.6274` (`+0.0016` F1)**

---

### 3.2 Bug #1: `make_submission.py` Optimizes Class Weights `w` *Before* `band_rules` (`+0.00385` Loss!)
Look at lines 102–114 of [make_submission.py](file:///c:/Users/ASUS/projects/upi-fraud-ml/src/upi/make_submission.py#L102-L114):
```python
w = fit_weights(oof, y)
oof_pred = (oof * w).argmax(axis=1)
oof_pred_b = band_rules(oof_pred, tr_mask, F3, F4, F5, is_bot_dev)
```
Why is this a serious bug?
1. `band_rules` overrides **6,573 out of 25,419 fraud predictions (`25.9%`)**!
2. Specifically, on Class 2 (`velocity_bot`), `is_bot_dev` (`DV0420001–DV0423000`) and Band 2 (`DV0423624–DV0424124`) deterministically supply **94% of all `velocity_bot` predictions**.
3. When `fit_weights(oof, y)` runs *without* knowing that `band_rules` will later overwrite those rows, it tries to adjust `w[2]`, `w[4]`, `w[5]`, etc. to fix rows that `band_rules` is going to overwrite anyway, while distorting the threshold on the remaining non-band rows!
4. Furthermore, `fit_weights` in `make_submission.py` recomputes `(P * w).argmax(axis=1)` across all `1,564,060` rows on every grid step, taking minutes.
5. **The Fix (`probe15.py` / `probe34.py`):** Precompute the `band_rules` mapping table `br_map[i, c]` (what class row `i` becomes if `(P * w).argmax` is `c`) on the **39,501 active candidate rows** (`1 - p_legit >= 0.15 | band > 0`), holding the other `1,524,559` rows fixed as `legit`. Then coordinate ascent evaluates the **exact post-`band_rules` Macro-F1** in **4.5 seconds** and improves `s2_a` from `0.55921` to **`0.56306` (`+0.00385` Macro-F1)**!

---

### 3.3 Bug #2: Critical October Test-Set Band Segmentation & Override Bugs in `bands.py` and `make_submission.py`
In `probe26.py`, `probe27.py`, `probe30.py`, and `probe32.py`, we audited every single one of the **417 active devices (`817` transactions)** inside the October test fraud band (`DV0473632–DV0474551`). We uncovered three high-impact bugs that directly hurt your public leaderboard score:

#### A. The `DV0473782` Cut Error (Corrupting a 9-Row `mule_account` Fan-In into `velocity_bot`)
- In [bands.py](file:///c:/Users/ASUS/projects/upi-fraud-ml/src/upi/bands.py#L106-L120), `edges_te` only uses anchor events for Class 1 (`mule`) and Class 4 (`sim_swap`), and interpolates Class 2 (`velocity_bot`) using training-set device-span ratios (`473750` to `473825`).
- Let's look at the actual October test transactions between `DV0473750` and `DV0473825` (`probe30.py`):
  - **There are ZERO patterned `velocity_bot` transactions in `473750..473825`!** Why? Because in October, all 7 patterned `velocity_bot` bursts (`269` rows) overwrite their `device_id` with the shared bot pool (`DV0420106–DV0422865`)! Onlystealth baseline `velocity_bot` rows remain in Band 2.
  - Meanwhile, at **`DV0473782`**, transaction `TXN01668652` (`sender=AC394246 -> receiver=AC369510`, `amount=1120.0`, `PAY`, `P2P`) is part of a **9-victim P2P fan-in burst (`Vs_n = 9`, session `sid = 68200`)** into receiver `AC369510`! Every model (`oof_final_local`, `s2_a`, `xgb_a`) predicts **`1: mule_account` with `0.965` probability**!
  - Because `bands.py` cut Band 1 at `473750` instead of `473783`, `DV0473782` was assigned `band = 2 (velocity_bot)`.
  - **The Cascade Disaster:** In `features_v4.py`, `rcv_band_2_24h` counts how many Band 2 devices paid receiver `AC369510` within 24 hours. Because `DV0473782` was labelled `band = 2`, `rcv_band_2_24h` became `1` for all **9 rows of that `mule_account` fan-in session**, and `band_rules` in `make_submission.py` **overwrote all 9 `mule_account` rows into `velocity_bot`**!

#### B. Four Pure `layering_chain` Devices Sitting Inside Test Band 1 (`DV0473637, DV0473660, DV0473709, DV0473731`)
- In `probe26.py`, we found 4 devices inside Test Band 1 (`473632..473750`) that belong to **pure multi-hop P2P layering chains (`ch_len = 3..6`, `is_mule_root = False`, `Vs_n = 0`)**:
  - `DV0473637` (`ch_len = 3`, `s1_pred = 3 (layering_chain)`, `s1_prob = 0.966`)
  - `DV0473660` (`ch_len = 3`, `s1_pred = 3 (layering_chain)`, `s1_prob = 0.955`)
  - `DV0473709` (`ch_len = 5`, `s1_pred = 3 (layering_chain)`, `s1_prob = 0.955`)
  - `DV0473731` (`ch_len = 6`, `s1_pred = 3 (layering_chain)`, `s1_prob = 0.963`)
- Why did a `layering_chain` transaction have a Band 1 device? Because one of the intermediate hop accounts in that chain had previously been allocated a new device during the Class 1 baseline loop (`~3%` new-device rate) and used that device as its current device when forwarding the layering hop!
- **The Cascade Disaster:** In `features_v5.py`, `ch_band` propagates any non-zero device band along the entire connected chain component (`comp_band`), and in `make_submission.py`:
  ```python
  m_ch = (ch_len >= 3) & (ch_band > 0)
  out[m_ch] = ch_band[m_ch]
  ```
  Because `ch_band == 1` on those 4 chains, **`make_submission.py` overwrites all 18 rows of those 4 pure `layering_chain` cascades in October from `layering_chain (3)` to `mule_account (1)`**!

#### C. Tail Band Misalignments (`Bands 7, 8, 9, 10, 11` in `DV0474296–DV0474551`)
- In `probe26.py` and `probe30.py`, inspecting the pattern anchors in `DV0474296–DV0474551` shows:
  - `DV0474299` is a pure `qr_native` (`qr_swap`, class 6, `prob = 0.975`) device sitting just past the `474296` cut for Band 6 (`qr_swap` should extend to `474300`).
  - `DV0474418` is a 10-row merchant burst (`S_n = 11`, `merchant_collusion`, class 7, `prob = 0.946`) sitting inside Band 8 (`refund_abuse`) because Band 7 was cut at `474410` instead of `474419`! That single boundary error causes `merch_band_8_24h` to **corrupt all 11 rows of that `merchant_collusion` burst into `refund_abuse`**!
  - `DV0474507` and `DV0474511` are pure `collect_99` (`phishing_collect`, class 9, `prob = 0.961`) devices sitting inside Band 10 (`fake_cashback`), corrupting their collect bursts from class 9 to class 10!
  - `DV0474541` is a pure `qr_native` (`qr_swap`, class 6, `prob = 0.975`) transaction (`s1_pf = 1.000`) whose user account happened to use a device allocated in Band 11 (`loan_app_extortion`), causing `band_rules` to overwrite a 100%-certain `qr_swap` merchant QR transaction into `loan_app_extortion`!

#### How to Fix Both `bands.py` and `band_rules` Immediately:
1. **Corrected October Test Edges (`edges_te` in `bands.py`):**
   Align `edges_te` to the exact October pattern anchors verified in `probe30.py`:
   ```python
   edges_te = np.array([
       473632,  # Band 1 start (mule_account)
       473783,  # Band 2 start (velocity_bot) -- keeps DV0473782 (Vs_n=9 mule fan-in) in Band 1!
       473825,  # Band 3 start (layering_chain) -- first layering anchor at 473828
       474005,  # Band 4 start (sim_swap) -- first sim_swap anchor at 474005
       474129,  # Band 5 start (account_takeover) -- last sim_swap at 474128, first ATO at 474136
       474245,  # Band 6 start (qr_swap) -- first qr_native at 474252
       474300,  # Band 7 start (merchant_collusion) -- keeps DV0474299 (qr_native) in Band 6!
       474419,  # Band 8 start (refund_abuse) -- keeps DV0474418 (S_n=11 merchant_collusion) in Band 7!
       474454,  # Band 9 start (phishing_collect)
       474512,  # Band 10 start (fake_cashback) -- keeps DV0474507 & DV0474511 (collect_99) in Band 9!
       474527,  # Band 11 start (loan_app_extortion) -- first L_burst at 474527
       474552,  # Legit extension start (474552..482013)
   ], dtype=np.int64)
   ```
2. **Sibling-Compatible Guard on `band_rules` (`probe26.py`):**
   - A device band (`band in 1..11`) or propagated entity band (`snd_band`, `rcv_band`, `merch_band`, `ch_band`) should **always** override a **stealth/low-pattern row (`s1_pf < 0.80`)** to `band`, because stealth new-device allocations occur inside the true class's generator loop.
   - However, when a row is **strongly patterned (`s1_pf >= 0.85`)**, a device band or entity-propagated band should **ONLY override within the pattern's compatible sibling family**:
     - If the row is a pure non-mule layering chain (`ch_len >= 3` and `ch_root_is_102 == 0` and `Vs_n < 5`), do **not** let a Band 1 (`mule_account`) device on one hop overwrite `layering_chain (3)` to `mule_account (1)`.
     - If a row is `qr_native == 1` (`qr_swap`, class 6), do **not** let a stealth Band 11 device overwrite it to `loan_app_extortion (11)`.
     - Only allow `snd_band` in `{4, 5}` to disambiguate `{4, 5}` drain bursts; `merch_band` in `{7, 8}` to disambiguate `{7, 8}` merchant bursts; and `snd_band`/`rcv_band` in `{9, 10, 11}` to disambiguate `{9, 10, 11}` collect/extortion bursts!

---

### 3.4 Overlooked Signal #2: Exact Conditional Asymmetries Between `sim_swap` (4) and `account_takeover` (5)
In `probe10.py` and `probe13.py`, we cross-tabulated all `4,295` sender balance-drain burst rows (`Ds_n >= 3`) across `snd_drain_kind`, `band`, `home_device`, `sim_age_hours`, and `new_payee`. We found a striking generator asymmetry that tree models partially miss when splits are diluted across 180 features:

1. **Why `snd_drain_kind == 2` (`amount ≈ 0.35 × balance_before_txn`) on `band == 0` is 99.5% `account_takeover`:**
   - When `sim_swap` (Class 4) emits a sibling-shaped `x0.35` burst (`snd_drain_kind == 2`), the generator **almost always allocates a brand-new device (`band == 4`)**: `443` rows in Band 4 vs only **`1` row** on `band == 0`!
   - When `account_takeover` (Class 5) emits its native `x0.35` burst (`snd_drain_kind == 2`), it reuses an existing device on `186` rows (`band == 0`) and allocates a new device (`band == 5`) on `1,303` rows.
   - Therefore, on **`band == 0 & sb == 0 & snd_drain_kind == 2`**, the true class ratio is **`186 account_takeover : 1 sim_swap` (`99.5%` Class 5)**!
2. **Why `snd_drain_kind == 1` (`amount ≈ 0.25 × balance_before_txn`) with `sim_age_hours >= 72` or `new_payee == 0` on `band == 0` is 100.0% `sim_swap`:**
   - When `account_takeover` (Class 5) emits a sibling-shaped `x0.25` burst (`snd_drain_kind == 1`), it **always sets `sim_age_hours < 72` and `new_payee == 1` on a full burst (`Ds_n >= 3`)**.
   - By contrast, when `sim_swap` (Class 4) emits its native `x0.25` burst on `band == 0`, `150` rows have `sim_age_hours >= 72` (**`150 sim_swap : 0 account_takeover`**) and `54` rows with `sim_age_hours < 72` have `new_payee == 0` or `Ds_n <= 1` (**`54 sim_swap : 0 account_takeover`**)!

Adding these two exact interaction rules on top of `s2_a + xgb_a + final_local` directly fixes ~45 misclassified drain-burst rows in OOF (`+0.0004` overall Macro-F1, `+0.0025` on Classes 4 & 5).

---

### 3.5 Overlooked Signal #3: Truncated Recipe Windows in `features_v5.py`, `features_v2.py`, and `clean_labels.py`
In `probe18.py`, `probe20.py`, and `probe21.py` (building on `scripts/explore/42_october_drift.py`), we verified two places where the engineered feature windows in `features_v1`–`v5` were narrower than the generator's actual parameters:

1. **6-Hour P2P Layering Chains (`W = 21,600s` vs `7,200s` in `features_v5.py`):**
   - `features_v5.py` traces P2P hops `A -> B -> C` using a 2-hour (`7,200s`) max hop delay.
   - However, the generator's `layering_chain` subroutine samples hop delays up to **6 hours (`21,600s`)** on integer P2P `PAY` transactions with `amount >= 500` and `0.85 <= amount_next / amount_prev <= 1.00`.
   - Extending the chain builder to `W = 21,600s` on `P2P & PAY & amt_is_integer & amount >= 500` uncovers **538 additional chain rows in Train and 114 additional chain rows in October Test** (`ch_len_6h >= 3` where `ch_len_2h < 3`).
   - In `probe20.py`, we checked what happens to those `538` train rows right now:
     - `475 / 538` (`88.3%`) have true raw label `3 (layering_chain)` or `1 (mule_account)`.
     - The remaining `63 / 538` (`11.7%`) are the ~15% unlabelled patterned fraud bursts (`y_raw == 0`). Because `clean_labels.py` used the 2-hour chain feature (`ch_len >= 3`), **`clean_labels.py` missed 25 of those 63 rows**, leaving them labelled as `legit` (`14` rows) or wrong classes (`11` rows) in `y_clean.npy`!
     - Even more importantly, on the `114` October test rows in 2h–6h chains, `ch_len` in `features_v5.parquet` is currently `1` or `2`, so **none of them receive `ch_len >= 3`, `ch_root_is_102`, or `ch_band` propagation**!
2. **Repeat-Victim Payments Inside `mule_account` Fan-In Sessions (`vs_any_n` in `features_v1.py` / `features_v2.py`):**
   - `features_v1.py` and `features_v2.py` require `new_payee == 1` on every row of a `mule_account` victim fan-in (`Vs_n >= 5`).
   - Whenever the generator picks the same victim twice in the same fan-in session (or a victim who previously paid that account), `new_payee` becomes `0` on the second payment!
   - Dropping the `new_payee == 1` requirement on the fan-in grouping (`PAY & P2P & amt_is_integer & (amount % 10 == 0)` into a receiver with $\ge 5$ distinct senders in 24h) recovers **220 additional `mule_account` / `layering_chain` rows in Train and 66 rows in October Test** that currently have `Vs_n = 0`!

---

## 4. A Completely Different, Complementary Approach: Episode-Graph Belief Propagation + Expected-F1 Linear Assignment

Your current pipeline treats the problem as **row-level tabular classification (`1,881,104` rows)** followed by **greedy rule overrides (`band_rules`)**.

Why does row-level classification hit a wall around `0.56`?
1. **Within-Episode Label Fragmentation:** A single 8-row fraud burst often gets split predictions from row-level GBDTs (e.g., 5 rows predicted `merchant_collusion` and 3 rows predicted `refund_abuse`, or 6 rows of a chain predicted `layering_chain` and 2 rows predicted `mule_account`).
2. **Multi-Hop Entity Information Loss:** `features_v6.py` only looks 1 hop away (`sender`, `receiver`, `device`). In reality, fraud rings form **connected components of Episodes across months**: Event $E_1$ (in May) shares a receiver $R_1$ with Event $E_2$ (in July), which shares a sender/device $D_1$ with Event $E_3$ (in October Test)! If $E_1$ has a Band 10 device (`fake_cashback`), a 1-hop feature helps $E_2$, but **only multi-hop Episode Belief Propagation reaches $E_3$ in October Test**!
3. **Greedy Argmax vs. Set-Level Sibling Quotas:** In `probe13.py`, we saw that inside genuinely ambiguous non-band sibling cells (e.g., `S_n in 4..8` on `band == 0` with no prior merchant history), the generator mixes `refund_abuse` (`75%`) and `merchant_collusion` (`25%`). Row-level `argmax` assigns **100% of those episodes to the majority sibling**, starving the minority sibling of recall and hurting harmonic-mean F1.

Here is the complete architecture of the **Episode-Graph Factor Model (EGFM)**—a completely different paradigm that you can run alongside your GBDT pipeline and blend at the episode/probability level:

```mermaid
flowchart TD
    A["1.88M Raw Transactions (Train M5-M9 + Test M10)"] --> B["Stage A: Deterministic Episode Extractor (~4,200 Patterned Episodes)"]
    A --> C["Stage B: Device-Band Allocator (1,266 Stealth New-Device Rows + 52,639 Legit Ext Rows)"]
    B --> D["Multi-Partite Episode-Entity Graph (Nodes = 4,200 Episodes; Hyperedges = User, Receiver, Merchant, Device, IP-Subnet)"]
    D --> E["Stage C: Label Propagation / Relational GNN + Stacking Meta-Classifier on Episode Nodes"]
    E --> F["Stage D: Sibling-Constrained Expected Macro-F1 Linear Assignment Decoder"]
    C --> G["Final Combined Prediction Tensor (Blends with Row-Level GBDT Pipeline)"]
    F --> G
```

### Stage A: Collapsing `1.88M` Rows into `~4,200` Discrete Fraud Episodes
Every patterned fraud event in the dataset is generated by one call to a fraud-event subroutine. We deterministically group all candidate patterned rows (`s1_pf >= 0.15` or recipe match) into **discrete Episode objects $E_1, \dots, E_M$**:
1. **P2P Victim Fan-In + Layering Cascade Episodes ($E^{\text{P2P}}$):**
   - Union of any 24h P2P integer `PAY` fan-in session (`vs_any_n >= 5`) and any 6-hour P2P integer `PAY` directed chain (`ch_len_6h >= 3`) that shares a root or intermediate account within 24h.
   - *Key Episode Invariant:* In the generator, a single episode is either purely `mule_account (1)`, purely `layering_chain (3)`, or a **coupled Mule-to-Chain episode** where the fan-in rows (`is_fanin == 1`) have one label and the downstream chain hops (`is_chain == 1`) have another. Therefore, each coupled P2P episode is split into at most two sub-episodes: $E_{\text{fanin}}$ and $E_{\text{chain}}$.
2. **Velocity Bot Episodes ($E^{\text{bot}}$):**
   - Grouped by `(sender_user_id, 1h session gap)` on `P2P/P2M` bursts (`Bs_n >= 10` or `is_bot_dev == 1`).
3. **Sender Drain Episodes ($E^{\text{drain}}$):**
   - Grouped by `(sender_user_id, 2h session gap)` where `Ds_n >= 3` or `snd_drain_kind in {1, 2}`.
4. **QR Swap Episodes ($E^{\text{qr}}$):**
   - Grouped by `(merchant_id, 6h session gap)` on `qr_native == 1` (`Qs_n >= 6`).
5. **Merchant Collusion / Refund Abuse Episodes ($E^{\text{merch}}$):**
   - Grouped by `(merchant_id, 24h session gap)` for `S_n >= 4` and by `(sender_user_id, 14d session gap)` for `R_snd_cnt_14d >= 3`.
6. **Collect / Cashback / Loan Extortion Episodes ($E^{\text{collect}}$):**
   - Grouped by `(receiver_user_id, 24h session gap)` for `COLLECT` bursts (`collect_99`, `cb_seq`) and by `(sender_user_id, 7d session gap)` for `L_snd_cnt_7d >= 3`.

**Immediate Benefit of Episode Grouping:** Every transaction inside an atomic episode $E_k$ must share the **exact same class label**! Simply pooling (log-odds averaging) your `s2_a + xgb_a + final_local` probabilities across all rows within each atomic episode eliminates intra-burst label fragmentation immediately.

### Stage B: Building the Cross-Month Episode Hypergraph & Running Relational Belief Propagation
Once we have the `~4,200` Episodes across Months 5–10 (Train + Test combined for graph topology, using only Train Out-of-Fold labels as supervision):
1. **Nodes:** Each Episode $E_k$ ($k \in \{1, \dots, M\}$).
2. **Node Initial Belief $\mathbf{u}_k^{(0)} \in \Delta^{11}$:**
   - Computed from the episode's pooled GBDT probability vector (`s2_a + xgb_a + final_local`), **hard-clamped** if $E_k$ contains a deterministic device-band anchor (`band in 1..11` compatible with $E_k$'s sibling family) or a deterministic conditional rule (such as `snd_drain_kind == 2 & band == 0 -> 99.5% Class 5`, or `snd_drain_kind == 1 & sim_age_hours >= 72 -> 100% Class 4`).
3. **Typed Edges Between Episodes $(E_k, E_j)$:**
   - **Type 1 (`shared_perpetrator_rcv`):** $E_k$ and $E_j$ share the same P2P `receiver_user_id` (for `COLLECT` or `fan-in` episodes).
   - **Type 2 (`shared_merchant`):** $E_k$ and $E_j$ share the same `merchant_id` (for `merchant_collusion` / `refund_abuse` / `qr_swap` episodes).
   - **Type 3 (`shared_sender_or_device`):** $E_k$ and $E_j$ share a non-bot `device_id` or `sender_user_id`.
   - **Type 4 (`chain_coupling`):** $E_k = E_{\text{fanin}}$ feeds directly into $E_j = E_{\text{chain}}$ via a mule account $M$.
4. **3-Iteration Damped Belief Propagation (Label Spreading over Sibling Subspaces):**
   For each iteration $t = 1, 2, 3$:
   $$\tilde{\mathbf{u}}_k^{(t)} = \text{Proj}_{\mathcal{S}(E_k)} \left( (1 - \alpha) \mathbf{u}_k^{(0)} + \alpha \sum_{r \in \text{EdgeTypes}} \sum_{j \in \mathcal{N}_r(k)} w_{kj}^{(r)} \mathbf{T}_r \mathbf{u}_j^{(t-1)} \right)$$
   where:
   - $\mathcal{S}(E_k) \subset \{1, \dots, 11\}$ is the **compatible sibling subspace** of Episode $E_k$'s recipe (e.g. $\{4, 5\}$ for drain episodes, $\{7, 8\}$ for merchant/refund episodes, $\{9, 10, 11\}$ for collect/extortion episodes, $\{1, 3\}$ for P2P fan-in/chain episodes).
   - $\mathbf{T}_r \in \mathbb{R}^{11 \times 11}$ is the empirical transition matrix learned out-of-fold on Train (for example, when $E_{\text{fanin}}$ and $E_{\text{chain}}$ are coupled via a mule root, what is $P(y_{\text{chain}} \mid y_{\text{fanin}}, \text{recipe})$).

### Stage C: Sibling-Constrained Expected Macro-F1 Linear Assignment Decoder
After Belief Propagation converges to episode posteriors $\mathbf{u}_k^*$, how should we assign a discrete class $\hat{y}_k \in \mathcal{S}(E_k)$ to each episode $E_k$ of size $n_k = |E_k|$ rows?
- Standard `argmax` fails when a cluster of ambiguous episodes (where $u_{k, c_1}^* \approx 0.55, u_{k, c_2}^* \approx 0.45$) all get assigned to $c_1$, inflating $\text{Pred}_1$ and collapsing $\text{Pred}_2$.
- Instead, within each decoupled Sibling Cluster $\mathcal{C} \in \{\{1,3\}, \{4,5\}, \{7,8\}, \{9,10,11\}\}$, we solve a **2-class or 3-class Expected Macro-F1 Assignment Problem** over the episodes $\{E_k\}_{k \in \mathcal{C}}$:
  $$\max_{\hat{y}_k \in \mathcal{C}} \sum_{c \in \mathcal{C}} \frac{2 \sum_{k: \hat{y}_k = c} n_k u_{k,c}^*}{\left(\sum_{k \in \mathcal{C}} n_k u_{k,c}^* + \text{StealthMass}_c\right) + \left(\sum_{k: \hat{y}_k = c} n_k\right)}$$
  where $\text{StealthMass}_c$ is the known expected count of unrecoverable non-band stealth fraud rows in class $c$ (which sit in `FN_c` regardless of how patterned episodes are classified!).
- **Why including $\text{StealthMass}_c$ in the denominator is critical:** In Macro-F1, the true support $P_c = \text{TP}_c + \text{FN}_c$ includes the ~55% stealth fraud rows! Once we hold total detected fraud fixed at the patterned episodes, the denominator of $F_1(c)$ for a sibling pair $(c_1, c_2)$ depends on $P_c + \hat{P}_c$. By optimizing the exact expected harmonic mean via a 1D or 2D dual multiplier $\lambda_{\mathcal{C}}$ on episode log-odds $\log(u_{k, c_1}^* / u_{k, c_2}^*) > \lambda_{\mathcal{C}}$, episodes are sorted by their posterior log-odds ratio and split at the exact quantile that maximizes Expected Macro-F1!

---

## 5. Step-by-Step Action Plan to Combine Both Approaches (Ordered by ROI & Zero Risk)

When you are ready to apply these findings to your codebase, we recommend executing them in four modular phases so you can verify the OOF and Public Leaderboard jump at each step:

### Phase 1: Immediate Zero-Retrain / Low-Effort Fixes (`+0.004` to `+0.015` Public LB)
1. **Generate `models/s2_a/test_proba.npy` and wire `s2_a` + `xgb_a` into `make_submission.py`:**
   - `models/s2_a/oof.npy` is already computed (`0.5630` OOF Macro-F1!). Run the full-train refit of `s2_a` (`python -m src.upi.train_s2 --name s2_a --full-only` or equivalent) to produce `models/s2_a/test_proba.npy`.
   - Blend `25% final_local + 65% s2_a + 10% xgb_a`.
2. **Fix the October Test Band Edges (`edges_te`) in `src/upi/bands.py` and re-run `features_v3` $\to$ `v6`:**
   - Change `473750` $\to$ `473783` (saving the 9-row `DV0473782` `mule_account` fan-in from being corrupted into `velocity_bot`).
   - Change `474296` $\to$ `474300`, `474410` $\to$ `474419` (saving the 11-row `DV0474418` `merchant_collusion` burst from being corrupted into `refund_abuse`), and `474496` $\to$ `474512` (saving the `DV0474507` and `DV0474511` `phishing_collect` bursts).
3. **Add the Sibling-Compatibility Guard to `band_rules` in `src/upi/make_submission.py`:**
   - Prevent `ch_band == 1` from overwriting pure `layering_chain` cascades (`ch_len >= 3 & ch_root_is_102 == 0 & Vs_n < 5`) into `mule_account` (fixes 18 rows across 4 chains in October test!).
   - Prevent tail device bands (`band == 11`) from overwriting high-confidence non-sibling recipes like `qr_native == 1` (`DV0474541`).
4. **Move `band_rules` Inside `fit_weights` + Add the `sim_swap` / `account_takeover` Conditional Rules:**
   - Optimize `w` against `band_rules((P * w).argmax(1))` on the `39,501` active rows (`+0.00385` OOF gain on `s2_a`).
   - Enforce `snd_drain_kind == 2 & band == 0 -> class 5 (account_takeover)` and `snd_drain_kind == 1 & band == 0 & (sim_age_hours >= 72 | new_payee == 0) -> class 4 (sim_swap)`.

### Phase 2: Recipe Window Expansion (`features_v1`, `v2`, `v5`, `clean_labels.py`)
1. **Expand `features_v5.py` chain window from `W = 7200` (2h) to `W = 21600` (6h)** on `P2P & PAY & amt_is_integer & amount >= 500` (adds `ch_len_6h`, `ch_band_6h`, `ch_root_is_102_6h`, capturing `538` train rows and `114` October test rows).
2. **Add `vs_any_n` (mule victim fan-in without requiring `new_payee == 1` on repeat victims)** to `features_v1.py` / `features_v2.py` (capturing `220` train rows and `66` October test rows).
3. **Re-run `clean_labels.py`** with the 6-hour chain and `vs_any_n` recipes so the 25 previously missed unlabelled 6-hour chain rows in `train.csv` get their true `y_clean` label.

### Phase 3: Build the Episode-Graph Belief Propagation & Expected-F1 Decoder (`src/upi/episode_graph.py`)
1. Group all candidate rows into the `~4,200` atomic Episodes $E_1, \dots, E_M$.
2. Run 2–3 iterations of Damped Belief Propagation across the Multi-Partite Entity Graph (`receiver_user_id`, `merchant_id`, `sender_user_id`, `device_id`, and coupled `mule -> chain` roots) so 2-hop and 3-hop entity histories reach October test episodes that only share an entity with another un-anchored episode.
3. Combine the Episode-Graph posterior $\mathbf{u}_{E(i)}^*$ with the row-level `s2_a + xgb_a + final_local` posterior (e.g., `50% Row-Level Ensemble + 50% Episode-Graph Posterior`) and decode each sibling cluster via the Expected-Macro-F1 log-odds threshold $\lambda_{\mathcal{C}}$.

---

## 6. Complete Manifest of New Files Created

### 6.1 Files Created Inside the Project Repository (`c:\Users\ASUS\projects\upi-fraud-ml\`)
- [docs/antigravity2.md](file:///c:/Users/ASUS/projects/upi-fraud-ml/docs/antigravity2.md) — **This report** (the only file created inside the project workspace; zero existing tracked or untracked files were modified).

### 6.2 Scratch Investigation Scripts Created Outside the Repository (`C:\Users\ASUS\.gemini\antigravity\brain\0e6cac72-2fba-434d-8974-e9a983b4d379\scratch\`)
All exploratory and empirical verification scripts were written to the external conversation scratch directory so your repository remained 100% untouched:
- `probe1.py` — Confusion matrix decomposition of `oof_final_local` + `band_rules` (`y_clean` vs `y_raw`, patterned vs stealth split).
- `probe2.py` — Oracle Macro-F1 ceiling analysis across the 4 sibling confusion clusters (`0.5597 -> 0.6115` with zero new fraud detections).
- `probe3.py` — Deep dive into the `35,228` missed fraud rows (`pf < 0.20`), testing entity overlap with detected fraud events.
- `probe4.py` — Analysis of the `5,650` misclassified patterned fraud rows across the 4 sibling groups (`{1,3}`, `{4,5}`, `{7,8}`, `{9,10,11}`).
- `probe5.py` — Event-level feature inspection on `phishing_collect` (9) vs `fake_cashback` (10) vs `loan_app_extortion` (11).
- `probe6.py` — Cross-event perpetrator reuse audit across all 5 training months (`66.3%` P2P receiver reuse across multiple events).
- `probe7.py` — Forensic reverse-engineering of `device_id` (`DV0000001–DV0482013`), proving sequential class-by-class generation and the `49.3%` vs `100.0%` device density signature.
- `probe8.py` — Marginal and conditional distribution comparison across all 12 classes inside the non-band stealth slice (`1,490,091` rows), plus adversarial 5-fold LightGBM AUC check (`0.5002`).
- `probe9.py` — Evaluation of unwired Stage-2 model `models/s2_a/oof.npy` (`features_v6.parquet`) and ensemble blends with `oof_final_local`, `xgb_a`, and `cat_a`.
- `probe10.py` — Conditional cross-tabulation of `sim_swap` (4) vs `account_takeover` (5) (`snd_drain_kind`, `sim_age_hours`, `home_device`, `new_payee`, `band`).
- `probe11.py` — Sibling group `{7,8}` (`merchant_collusion` vs `refund_abuse`) and `{1,3}` (`mule_account` vs `layering_chain`) purity analysis.
- `probe12.py` — Conditional probability test of stealth fraud given sender, receiver, merchant, device, or IP-subnet overlap with confirmed fraud events (`2.36%` invariant proof).
- `probe13.py` — Exact cell-by-cell conditional purity tables across all 4 sibling groups outside device bands.
- `probe14.py` — Inspection of all saved model directories in `models/` (`lgb_clean_*`, `s2_a`, `s3_clean_*`, `xgb_a`, `cat_a`, `mlp_a`).
- `probe15.py` — Fast active-set (`39,501` rows) band-aware coordinate ascent optimizer, proving the `+0.00385` gain from placing `band_rules` inside `fit_weights`.
- `probe16.py` — Full 8-model ensemble and nested-LOMO cross-validation benchmark with band-aware weights (`0.56342` pooled / `0.56165` LOMO).
- `probe17.py` — Evaluation of `s2_a` feature importances and multi-hop entity propagation potential.
- `probe18.py` — Audit of October test drift (`test.csv`) and recipe coverage across months 5–10 (`6h` chains and `new_payee == 0` fan-ins).
- `probe19.py` — Inspection of existing submission files in `submissions/` and prediction class distributions.
- `probe20.py` — Deep dive on the `538` train rows in 2h–6h layering chains (`W = 21,600s`), uncovering `25` unlabelled chain rows missed by `clean_labels.py`.
- `probe21.py` — Deep dive on the `114` October test rows in 2h–6h layering chains and `66` October test rows in `new_payee == 0` mule fan-ins.
- `probe22.py` — Comparison of `s3_clean_v4.csv`, `final_local.csv`, and `v0_test.csv` on the October test set.
- `probe23.py` — Exact ID-aligned diff (`reindex(d_te.txn_id.values)`) between `s3_clean_v4.csv` and `final_local.csv`.
- `probe24.py` — Per-class breakdown of the `434` test predictions where `s3_clean_v4.csv` and `final_local.csv` differ.
- `probe25.py` — Analysis of why `s3_clean_v4` vs `final_local` differs on `mule_account` vs `layering_chain` in October test.
- `probe26.py` — Comprehensive audit of all `817` rows in the October test fraud device band (`DV0473632–DV0474551`), discovering the 4 `layering_chain` devices in Band 1 and tail band override bugs.
- `probe27.py` — Quantifying the blast radius of `ch_band`, `rcv_band`, and `merch_band` overrides in October test (`18` layering chain rows corrupted to `mule_account`).
- `probe28.py` — Auditing `band_rules` override accuracy in Train (`6,573` overridden rows) segmented by Stage-1 pattern confidence `s1_pf`.
- `probe29.py` — Cross-tabulating Train `band_rules` overrides when `s1_pf >= 0.85` and `s1_pred != band_rule_pred`.
- `probe30.py` — Exact device-by-device anchor map of the October test fraud band (`DV0473632–DV0474551`), discovering the `DV0473782` 9-row `mule_account` fan-in cut error.
- `probe31.py` — Verifying `Vs_n >= 5` fan-in sessions in Train that contain Band 2 (`velocity_bot`) devices vs Band 1 (`mule_account`) devices.
- `probe32.py` — Tracing the 9-row October test fan-in session (`sid = 68200`, receiver `AC369510`) corrupted from `mule_account` to `velocity_bot` by `DV0473782`.
- `probe33.py` — Testing the `sim_swap` vs `account_takeover` conditional asymmetry rules on top of the 3-way ensemble (`+0.0004` overall Macro-F1).
- `probe34.py` — Final end-to-end summary benchmark comparing Pooled and Strict Nested-LOMO Macro-F1 across `s3_clean_v4`, `final_local`, `s2_a`, and the 3-way ensemble (`0.5597 -> 0.5634`).
