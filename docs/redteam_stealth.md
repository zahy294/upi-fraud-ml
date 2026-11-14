# Red-Team Audit of Stealth / Ordinary-Looking Fraud (`data/raw/train.csv`)

Reproduced by running `python -I scripts/explore/50_redteam_stealth.py` (threads capped at `4`, one 3-fold month-grouped cross-validated LightGBM with `200` trees, zero existing files modified).

## Verdict: Plain Negative Result

**The claim holds.** No rule built from columns available in `test.csv`, `accounts.csv`, and `merchants.csv` (excluding `app_version` and `ip_risk_score`) selects at least `300` ordinary rows in each of July, August, and September with a fraud-label share of at least `10%` in every month:

- **Cross-row count rules (`1,025` rules across `46` sender, receiver, device, and pair count features)**: **`0` rules** reach $\ge 10\%$ monthly fraud share with $\ge 300$ rows/month (best min-month fraud share is **`2.66%`** on `snd_cnt_month == 32`).
- **1-way binned scan (`116` columns, `799` cells, `732` with `min_n >= 300`)**: **`0` cells** reach $\ge 10\%$ (best min-month fraud share is **`5.75%`** on `qr_vpa_mismatch = 1.0`).
- **2-way pairwise interaction scan (`6,670` column pairs, `313,195` total cells, `180,659` with `min_n >= 300`)**: **`0` cells** reach $\ge 10\%$ (best min-month fraud share is **`6.02%`** on `os = android & qr_vpa_mismatch = 1.0`).
- **3-way interaction scan (`47,905` column triples across `67` core test/account/merchant/cross-row count columns, `25,819,312` total cells, `5,191,231` with `min_n >= 300`)**: **`0` cells** reach $\ge 10\%$ (best min-month fraud share is **`6.65%`** on `amt_is_integer = 1 & qr_vpa_mismatch = 0.0 & rcv_cnt_24h = 1`).
- **One LightGBM (`200` trees, `n_jobs = 4`) with 3-fold month-grouped cross-validation**:
  - **Pooled out-of-fold ROC AUC**: **`0.515027`** across all `1,538,598` ordinary rows (`0.516697` in July, `0.514756` in August, `0.513644` in September).
  - **Within `home_device == 1` (`1,418,862` rows, `92.22%` of ordinary rows)**: out-of-fold ROC AUC is **`0.500565`** (pure chance; the `0.5150` pooled lift comes solely from separating `home_device == 1` at `2.39%` fraud from `dev_ext_legit` new devices `> 428454` at `0.00%` fraud).
  - **Top-300 OOF predictions per month**: July `30 / 300` (`10.00%`), August `29 / 300` (`9.67%`), September `21 / 300` (`7.00%`) — min-month share **`7.00%`**, failing the $\ge 10\%$ every-month threshold.

---

## 1. Population Definition ("Ordinary" Train Rows)

Following the exact specification:
1. Load `P = models/s2_a/oof.npy` (`1,564,060 x 12`, aligned with `data/processed/base_v0.parquet` where `split == 'train'`).
2. Fit class weights `w = fit_weights_fast(P, y_clean)` from `colab/bench.py` against `models/oof_raw/y_clean.npy`:
   - `legit: 1.0000`, `mule_account: 0.3622`, `velocity_bot: 0.1970`, `layering_chain: 0.5721`, `sim_swap: 1.2252`, `account_takeover: 0.2294`, `phishing_collect: 0.9034`, `qr_swap: 0.4912`, `refund_abuse: 0.4912`, `fake_cashback: 1.9349`, `loan_app_extortion: 1.6616`, `merchant_collusion: 0.4218`.
3. Take rows with `(P * w).argmax(1) == 0` (`1,538,598` rows) and exclude device numbers `423001–428454` (`3,231` train rows total, `0` of which had `pred == 0`).

### Monthly & Class Breakdown of the `1,538,598` Ordinary Rows

| Month | Ordinary Rows | Fraud-Labelled Rows (`label != 'legit'`) | Fraud Share (%) |
|---|---:|---:|---:|
| `2026-07` (July) | `518,322` | `11,894` | `2.2947%` |
| `2026-08` (August) | `518,422` | `11,969` | `2.3087%` |
| `2026-09` (September) | `501,854` | `11,566` | `2.3047%` |
| **Total (Pooled)** | **`1,538,598`** | **`35,429`** | **`2.3027%`** |

- Note: `y_clean > 0` on ordinary rows has `35,432` positives (`2.3029%`), differing on only `3` rows out of `1,538,598`, and `0` ordinary rows have raw `month > '2026-09'`.
- By raw fraud class among the `1,538,598` ordinary rows: `mule_account: 6,828` (`0.4438%`), `velocity_bot: 4,754` (`0.3090%`), `qr_swap: 4,473` (`0.2907%`), `phishing_collect: 4,008` (`0.2605%`), `layering_chain: 3,847` (`0.2500%`), `account_takeover: 2,446` (`0.1590%`), `sim_swap: 2,439` (`0.1585%`), `refund_abuse: 1,931` (`0.1255%`), `merchant_collusion: 1,792` (`0.1165%`), `fake_cashback: 1,598` (`0.1039%`), `loan_app_extortion: 1,313` (`0.0853%`).

---

## 2. Best Cell Found Across All Binned & Cross-Row Scans (`n >= 300` per Month)

The single best cell found across all 1-way (`799` cells), 2-way (`313,195` cells), and 3-way (`25,819,312` cells) binned interactions and `1,025` cross-row count rules is a **3-way cell**:

- **Rule / Cell**: `amt_is_integer == 1 & qr_vpa_mismatch == 0.0 & rcv_cnt_24h == 1` (whole-rupee merchant QR transactions where `qr_vpa_mismatch == 0.0` at a merchant with only `1` transaction in $\pm 24\text{h}$)
- **July (`2026-07`)**: `337` ordinary rows, `32` fraud-labelled, **`9.4955%`** fraud share
- **August (`2026-08`)**: `339` ordinary rows, `27` fraud-labelled, **`7.9646%`** fraud share
- **September (`2026-09`)**: `361` ordinary rows, `24` fraud-labelled, **`6.6482%`** fraud share
- **Pooled (`Jul–Sep`)**: `1,037` ordinary rows, `83` fraud-labelled, **`8.0039%`** fraud share
- **Minimum monthly size / share**: `min_n = 337` ($\ge 300$), **`min_share = 6.6482%`** (**below** the `10.0%` requirement in every month).

Why this cell (and `qr_vpa_mismatch == 1.0`, below) sits around `6%–8%` rather than `2.3%`: in `s2_a`, `qr_swap` has decision weight `w[7] = 0.4912`, so `s2_a` only predicts `qr_swap` when `0.4912 * P(qr_swap) > P(legit)` (`P(qr_swap) > ~0.67`). Isolated merchant QR transactions with `qr_vpa_mismatch == 1.0` or whole-rupee QR amounts at a quiet merchant have a `~6%–9%` empirical `qr_swap` rate — well below the macro-F1 decision boundary, so `s2_a` leaves them predicted `legit`, and even on those small sub-threshold fringes the fraud share never reaches `10%` across all three months.

### Summary of Top Cells by Search Family (`min_n >= 300` in Jul, Aug, Sep)

| Search Family | Cells / Rules Tested (`min_n >= 300`) | Passing `>= 10%` in All 3 Months | Best Rule / Cell | Jul `f / n` (`%`) | Aug `f / n` (`%`) | Sep `f / n` (`%`) | Pooled (`%`) | Min-Month Share (`%`) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **Cross-row counts (1D)** | `1,025` (`1,025`) | **`0`** | `snd_cnt_month == 32` | `13 / 410` (`3.17%`) | `18 / 632` (`2.85%`) | `10 / 376` (`2.66%`) | `2.89%` | **`2.66%`** |
| **Cross-row counts (1D)** | — | **`0`** | `pair_cnt_7d >= 3` | `36 / 1,102` (`3.27%`) | `34 / 1,140` (`2.98%`) | `29 / 1,101` (`2.63%`) | `2.96%` | **`2.63%`** |
| **Cross-row counts (1D)** | — | **`0`** | `dev_uniq_acct_total == 3` | `30 / 1,036` (`2.90%`) | `28 / 1,052` (`2.66%`) | `28 / 1,072` (`2.61%`) | `2.72%` | **`2.61%`** |
| **Cross-row counts (1D)** | — | **`0`** | `pair_cnt_month >= 5` | `19 / 462` (`4.11%`) | `11 / 302` (`3.64%`) | `10 / 383` (`2.61%`) | `3.49%` | **`2.61%`** |
| **1-way binned (`116` cols)** | `799` (`732`) | **`0`** | `qr_vpa_mismatch = 1.0` | `36 / 586` (`6.14%`) | `31 / 539` (`5.75%`) | `33 / 574` (`5.75%`) | `5.89%` | **`5.75%`** |
| **1-way binned (`116` cols)** | — | **`0`** | `amt_tiny_le10 = 1` | `22 / 767` (`2.87%`) | `21 / 718` (`2.92%`) | `22 / 739` (`2.98%`) | `2.92%` | **`2.87%`** |
| **2-way binned (`6,670` pairs)** | `313,195` (`180,659`) | **`0`** | `os = android & qr_vpa_mismatch = 1.0` | `33 / 511` (`6.46%`) | `30 / 453` (`6.62%`) | `30 / 498` (`6.02%`) | `6.36%` | **`6.02%`** |
| **2-way binned (`6,670` pairs)** | — | **`0`** | `away_from_home = 0 & qr_vpa_mismatch = 1.0` | `33 / 552` (`5.98%`) | `31 / 513` (`6.04%`) | `32 / 542` (`5.90%`) | `5.97%` | **`5.90%`** |
| **3-way binned (`47,905` triples)** | `25,819,312` (`5,191,231`) | **`0`** | `amt_is_integer = 1 & qr_vpa_mismatch = 0.0 & rcv_cnt_24h = 1` | `32 / 337` (`9.50%`) | `27 / 339` (`7.96%`) | `24 / 361` (`6.65%`) | `8.00%` | **`6.65%`** |
| **3-way binned (`47,905` triples)** | — | **`0`** | `qr_vpa_mismatch = 1.0 & home_device = 1 & os = android` | `31 / 476` (`6.51%`) | `27 / 424` (`6.37%`) | `29 / 460` (`6.30%`) | `6.40%` | **`6.30%`** |
| **3-way binned (`47,905` triples)** | — | **`0`** | `qr_vpa_mismatch = 1.0 & os = android & pair_cnt_total = 1` | `31 / 457` (`6.78%`) | `27 / 400` (`6.75%`) | `28 / 454` (`6.17%`) | `6.56%` | **`6.17%`** |
| **3-way binned (`47,905` triples)** | — | **`0`** | `txn_type = QR & amt_is_integer = 1 & rcv_cnt_24h = 1` | `35 / 340` (`10.29%`) | `24 / 336` (`7.14%`) | `22 / 358` (`6.15%`) | `7.83%` | **`6.15%`** |

---

## 3. Month-Grouped Cross-Validated LightGBM (`200` Trees)

We trained a single `LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=4)` using 3-fold month-grouped cross-validation across all `120` test-available features (`15` categorical, `105` numeric, covering raw `test.csv` columns, `accounts.csv`, `merchants.csv`, and all cross-row counts and profile deviations per sender, receiver, device, and pair).

### Out-of-Fold ROC AUC

| Evaluation Slice | Rows | Fraud Rows | Out-of-Fold ROC AUC |
|---|---:|---:|---:|
| **Fold 1 — July (`2026-07` held out)** | `518,322` | `11,894` | `0.516697` |
| **Fold 2 — August (`2026-08` held out)** | `518,422` | `11,969` | `0.514756` |
| **Fold 3 — September (`2026-09` held out)** | `501,854` | `11,566` | `0.513644` |
| **Pooled All Ordinary Rows (`Jul–Sep`)** | **`1,538,598`** | **`35,429`** | **`0.515027`** |
| **Within `home_device == 1` (`DV0` + sender digits)** | **`1,418,862`** | **`33,962`** | **`0.500565`** |

### Per-Fraud-Class Out-of-Fold ROC AUC (Ordinary Fraud Class vs Ordinary Legit)

| Fraud Class | Ordinary Fraud Rows | OOF AUC (All Ordinary) | OOF AUC (`home_device == 1`) |
|---|---:|---:|---:|
| `mule_account` | `6,828` | `0.512681` | `0.498656` |
| `velocity_bot` | `4,754` | `0.513980` | `0.500750` |
| `layering_chain` | `3,847` | `0.506906` | `0.492025` |
| `sim_swap` | `2,439` | `0.514635` | `0.500847` |
| `account_takeover` | `2,446` | `0.508674` | `0.496435` |
| `phishing_collect` | `4,008` | `0.515840` | `0.500752` |
| `qr_swap` | `4,473` | `0.527770` | `0.511815` |
| `refund_abuse` | `1,931` | `0.517490` | `0.501219` |
| `fake_cashback` | `1,598` | `0.520218` | `0.507687` |
| `loan_app_extortion` | `1,313` | `0.512571` | `0.495760` |
| `merchant_collusion` | `1,792` | `0.514281` | `0.498894` |

### Top-K Per Month & Threshold Selection on LightGBM Out-of-Fold Scores

| Selection Rule on `oof_lgb` | July `f / n` (`%`) | August `f / n` (`%`) | September `f / n` (`%`) | Min-Month Share (`%`) |
|---|---:|---:|---:|---:|
| Top `300` rows per month | `30 / 300` (`10.00%`) | `29 / 300` (`9.67%`) | `21 / 300` (`7.00%`) | **`7.00%`** |
| Top `500` rows per month | `36 / 500` (`7.20%`) | `36 / 500` (`7.20%`) | `25 / 500` (`5.00%`) | **`5.00%`** |
| Top `1,000` rows per month | `50 / 1,000` (`5.00%`) | `46 / 1,000` (`4.60%`) | `42 / 1,000` (`4.20%`) | **`4.20%`** |
| Top `2,000` rows per month | `72 / 2,000` (`3.60%`) | `79 / 2,000` (`3.95%`) | `64 / 2,000` (`3.20%`) | **`3.20%`** |
| Top `5,000` rows per month | `151 / 5,000` (`3.02%`) | `152 / 5,000` (`3.04%`) | `130 / 5,000` (`2.60%`) | **`2.60%`** |
| Best global cutoff (`oof_lgb >= 0.04814`, `min_n >= 300`) | `54 / 1,222` (`4.42%`) | `36 / 572` (`6.29%`) | `25 / 516` (`4.84%`) | **`4.42%`** |
