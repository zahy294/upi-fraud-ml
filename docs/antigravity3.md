# Antigravity3 — Empirical Verification of Searches 1, 2, and 3

All numbers in this report are printed directly by `python -I scripts/explore/43_antigravity3.py --search 0` (thread count capped at `4`). No existing repository files were modified, and nothing was written under `data/`, `models/`, or `submissions/`.

## Executive Summary (Plain Negative Results)

1. **Search 1 (Is ordinary-looking fraud different from legit in October?) — Negative.**
   - In Test (`Oct`), comparing the `211` ordinary-looking fraud-band rows (`473632–474551`, `s1_pf < 0.2`) against the `7,452` `dev_ext_legit` rows (`s1_pf < 0.2`) reproduces the raw hint (`amount` median `707.04` vs `540.00`, geometric mean ratio `x1.1856`, raw Mann-Whitney `p = 0.0419` [13 marginal comparisons], raw KS `p = 0.0289` [13 marginal comparisons]; whole-rupee share `0.3128` vs `0.2570`, Fisher `p = 0.0790` [13 marginal comparisons]), while in Train (`Jul–Sep`, `1,150` vs `45,104` rows) the ratio is `x1.0269` (median `507.62` vs `455.12`, MW `p = 0.3261` [13 comparisons], KS `p = 0.1107` [13 comparisons]; whole-rupee share `0.2617` vs `0.2569`, Fisher `p = 0.7071` [13 comparisons]).
   - Under permutation testing across all `13` features and `78` pairs (`91` hypotheses; `1,492` scalar cells in Test, `1,519` in Train), the global family-wise error rate is **`p = 0.4477` in Test** (`max |z| = 4.207` across `1,492` cells) and **`p = 0.9468` in Train** (`max |z| = 3.080` across `1,519` cells). Neither `amount`, `amt_is_integer`, nor any marginal or pair survives correction.
   - The `147` Test rows that are whole-rupee (`66`) or `amount > 700` (`109`) have `0.000` same-device neighbors and `0.000` high-`s1_pf` P2P neighbors within $\pm 3$ days, forming zero uncaptured event shapes. Across `17,343` conjunction rules (`17,343` comparisons), **`0` rules** reach precision $\ge 50\%$ at recall $\ge 10\%$ (best Test precision at recall $\ge 10\%$ is `6.55%`, Train precision `3.71%`).
2. **Search 2 (Does the October shift cost detections?) — Negative.**
   - Across `248` input columns (`242` numeric $\times 12$ predicted classes = `2,904` KS comparisons), shifts among `legit`-predicted rows reflect calendar aging (`mer_age_days` median `660.0 -> 717.0`), `0.50x` arrival density (`rcv_sum_7d` median `8589.1 -> 5940.0`, `snd_out_cnt_24h` mean `1.2729 -> 1.1360`), and `x1.165` legit amount scaling (`459.20 -> 534.95`). Among flagged rows (`pred in 1..11`), `68` of `2,662` `(class, column)` pairs have `KS_p_bonf < 0.05` (`2,904` comparisons), driven by lower background receiver volume (`rcv_cnt_7d`, `pair_share_rcv_7d`) and smaller sampled October `velocity_bot` burst sizes (`Qs_n` median `37.0` vs `44.0`).
   - Near-miss 24h groups (`size >= 3`) and partially flagged events (`2b-ii`) both **decrease** in October per 100k rows (`0.39x` and `0.53x` max train month on `0.05 <= pf <= 0.50`; `0.13x` and `0.36x` on `0.15 <= pf <= 0.50`). Only one background-merchant-linked `velocity_bot` group (`6` rows at fuel merchant `MR009024`) exceeds `2x` the max train month, and those `6` rows are ordinary fuel transactions.
3. **Search 3 (A tell for sibling-shaped events using band devices as truth) — Negative.**
   - Testing `44` non-recipe features across all `5` sibling-shape families (`2,642` threshold rules across `7` pair tests, `3,052` including the multi-row control), **`0` rules** achieve $\ge 85\%$ monthly balanced accuracy on band events in Jul, Aug, and Sep while agreeing with non-band labels.

---

## Search 1: Is Ordinary-Looking Fraud Different from Legit in October?

### 1a. Permutation Tests on 13 Features and All 78 Pairs (`91` Hypotheses)

Populations (`models/s1/{oof,test_proba}.npy`, `s1_pf < 0.2` on new devices):
- **Test (`Oct`)**: `211` fraud-band rows (`473632–474551`) vs `7,452` `dev_ext_legit == 1` rows (`10,000` permutations; `1,492` scalar cells across `91` blocks).
- **Train (`Jul–Sep`)**: `1,150` fraud-band rows (`423001–428454`) vs `45,104` `dev_ext_legit == 1` rows (`4,000` permutations; `1,519` scalar cells across `91` blocks).
- **Global omnibus test**: Test `max |z| = 4.207`, **FWER `p = 0.4477`** (`1,492` scalar comparisons); Train `max |z| = 3.080`, **FWER `p = 0.9468`** (`1,519` scalar comparisons).

#### All 13 Marginal Tests (`13` marginal comparisons; `91` combined comparisons)

| Feature | `n_cols_te` | Test `max_z` | Test `p_perm_raw` (13 comp) | Test `p_bonf_13` (13 comp) | Test `p_wy_13` (13 comp) | Test `p_wy_91` (91 comp) | Train `max_z` | Train `p_perm_raw` (13 comp) | Train `p_wy_91` (91 comp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `amount` | 7 | 2.1445 | 0.1477 | 1.0000 | 0.8554 | 0.9990 | 1.7577 | 0.2997 | 1.0000 |
| `amt_is_integer` | 2 | 1.8261 | 0.0780 | 1.0000 | 0.6303 | 0.9850 | 0.3712 | 0.7166 | 1.0000 |
| `receiver_type` | 2 | 1.3175 | 0.1936 | 1.0000 | 0.9245 | 0.9995 | 1.2321 | 0.2062 | 1.0000 |
| `txn_type` | 5 | 1.2418 | 0.6003 | 1.0000 | 0.9999 | 1.0000 | 1.8101 | 0.2639 | 1.0000 |
| `merchant_category` | 13 | 1.3175 | 0.9107 | 1.0000 | 1.0000 | 1.0000 | 2.0322 | 0.4209 | 1.0000 |
| `hour` | 6 | 0.9707 | 0.7648 | 1.0000 | 1.0000 | 1.0000 | 1.4604 | 0.4384 | 1.0000 |
| `new_payee` | 2 | 0.3694 | 0.7376 | 1.0000 | 1.0000 | 1.0000 | 1.1747 | 0.2459 | 1.0000 |
| `pin_retries` | 5 | 0.8121 | 0.6749 | 1.0000 | 1.0000 | 1.0000 | 1.6928 | 0.2234 | 1.0000 |
| `login_to_txn_sec` | 6 | 1.6077 | 0.3762 | 1.0000 | 0.9975 | 1.0000 | 1.7863 | 0.2864 | 1.0000 |
| `sim_age_hours` | 7 | 1.0064 | 0.7676 | 1.0000 | 1.0000 | 1.0000 | 0.6560 | 0.9300 | 1.0000 |
| `device_age_days` | 7 | 3.2463 | 0.0089 | 0.1157 | 0.1063 | 0.4395 | 0.3535 | 0.9945 | 1.0000 |
| `balance_before` | 7 | 1.3335 | 0.5609 | 1.0000 | 0.9999 | 1.0000 | 0.7232 | 0.9295 | 1.0000 |
| `amt_over_bal` | 7 | 1.6927 | 0.3141 | 1.0000 | 0.9905 | 1.0000 | 1.4819 | 0.4491 | 1.0000 |

#### Top 10 Pair Tests in Test (`78` pair comparisons; `91` total hypotheses)

| Pair (`78` pairs, `91` total hypotheses) | `n_cols_te` | Test `max_z` | Test `p_perm_raw` (91 comp) | Test `p_bonf_91` (91 comp) | Test `p_wy_91` (91 comp) | Train `max_z` | Train `p_perm_raw` (91 comp) | Train `p_wy_91` (91 comp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `amount x amt_is_integer` | 11 | 3.7677 | 0.0021 | 0.1911 | 0.1416 | 2.3111 | 0.1632 | 1.0000 |
| `amount x receiver_type` | 11 | 3.6958 | 0.0025 | 0.2275 | 0.1648 | 2.6013 | 0.0800 | 0.9890 |
| `hour x device_age_days` | 17 | 4.2074 | 0.0039 | 0.3549 | 0.2365 | 2.1001 | 0.4299 | 1.0000 |
| `amt_is_integer x merchant_category` | 20 | 3.5711 | 0.0448 | 1.0000 | 0.9192 | 2.4123 | 0.3989 | 1.0000 |
| `hour x balance_before` | 17 | 3.1157 | 0.0500 | 1.0000 | 0.9353 | 1.3281 | 0.9548 | 1.0000 |
| `amount x new_payee` | 11 | 2.7688 | 0.0536 | 1.0000 | 0.9452 | 1.7422 | 0.5036 | 1.0000 |
| `amt_is_integer x new_payee` | 5 | 2.4353 | 0.0560 | 1.0000 | 0.9496 | 1.4787 | 0.4071 | 1.0000 |
| `amount x login_to_txn_sec` | 21 | 2.9751 | 0.0606 | 1.0000 | 0.9609 | 2.7278 | 0.1137 | 0.9988 |
| `login_to_txn_sec x balance_before` | 17 | 2.9443 | 0.0658 | 1.0000 | 0.9704 | 1.9264 | 0.5884 | 1.0000 |
| `amt_is_integer x pin_retries` | 7 | 2.4373 | 0.0849 | 1.0000 | 0.9893 | 1.8110 | 0.3449 | 1.0000 |

### 1b. Subset Decomposition & 3-Day Entity-Link Inspection of the `147` Test Rows

1. **Why raw whole-rupee and amount looked higher in Test**: In background traffic, `P2P` transactions are `70.1%` whole-rupee while `MERCHANT` transactions are `0.9%` whole-rupee. The `211` Test fraud-band rows had slightly more `P2P` (`85 / 211 = 40.28%` vs `2,673 / 7,452 = 35.87%`, Fisher `p = 0.2132` [13 comparisons]; Train `422 / 1,150 = 36.70%` vs `16,133 / 45,104 = 35.77%`, `p = 0.5312`). Within `P2P`, Test whole-rupee share is `65 / 85` (`0.7647`) vs `1,897 / 2,673` (`0.7097`, Fisher `p = 0.3292` [2 strata]; Train `298 / 422 = 0.7062` vs `11,447 / 16,133 = 0.7095`, `p = 0.8699`). The single pair cell with `|z| > 3` on `amount x amt_is_integer` is the upper tail `amount > 1500 & amt_is_integer == 1` (`19 / 211 = 9.00%` vs `290 / 7,452 = 3.89%`, `z = 3.7677`), which is spread across `9` bands (`band 1: 4, band 2: 1, band 3: 3, band 4: 1, band 6: 3, band 7: 3, band 8: 2, band 9: 1, band 11: 1`) and absent in Train (`49 / 1,150 = 4.26%` vs `1,445 / 45,104 = 3.20%`, `z = 2.0085`).
2. **$\pm 3$-day neighbor check (`scripts/explore/43_antigravity3.py` prints all `147` rows with `amt_is_integer == 1` [`66`] or `amount > 700` [`109`])**:
   - **Test (`147` fraud-band subset vs `3,897` `dev_ext_legit` subset)**: mean `n_snd_out = 0.354` vs `0.426`, `n_snd_in = 0.075` vs `0.070`, `n_dev = 0.000` vs `0.000` (`0 / 147` have any same-device neighbor in $\pm 3$d), `n_p2p_nbrs = 0.524` vs `0.624`, `n_nbr_int = 0.184` vs `0.214`, `n_nbr_hi_pf (s1_pf >= 0.2) = 0.000` vs `0.012` (`0 / 147` link to any suspicious P2P neighbor).
   - **Train (`652` fraud-band subset vs `24,992` `dev_ext_legit` subset)**: mean `n_snd_out = 0.799` vs `0.858`, `n_snd_in = 0.156` vs `0.157`, `n_dev = 0.000` vs `0.000`, `n_p2p_nbrs = 1.078` vs `1.196`, `n_nbr_int = 0.365` vs `0.388`, `n_nbr_hi_pf = 0.017` vs `0.019`.
   - **Conclusion**: They form **zero** uncaptured event shapes.

### 1c. Rule Search (`Precision >= 50%` at `Recall >= 10%` [`>= 22 / 211` rows])

Across `17,343` 1-, 2-, and 3-literal conjunction rules (`17,343` comparisons), **`0` rules** reach precision $\ge 50\%$ at recall $\ge 10\%$:

| Rule (`17,343` rules tested) | Test `tp / 211` (Rec) | Test `fp / 7,452` | Test Prec | Train `tp / 1,150` (Rec) | Train `fp / 45,104` | Train Prec | Test Existing-Dev Rows (`s1_pf < 0.2`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `amt>1500 & rcv==P2P & dev_age<30` | `23` (`10.90%`) | `328` | `6.55%` | `71` (`6.17%`) | `1,844` | `3.71%` | `787` |
| `amt>1500 & amt_int==1 & dev_age<30` | `16` (`7.58%`) | `223` | `6.69%` | `46` (`4.00%`) | `1,297` | `3.42%` | `549` |
| `amt>1200 & amt_int==1 & dev_age<20` | `22` (`10.43%`) | `317` | `6.49%` | `58` (`5.04%`) | `1,636` | `3.42%` | `647` |
| `amt>1500 & amt_int==1 & login_sec<5` | `9` (`4.27%`) | `79` | `10.23%` | `21` (`1.83%`) | `460` | `4.37%` | `7,448` |

---

## Search 2: Does the October Shift Cost Detections?

### 2a. Input Column Distribution Shifts (`242` numeric $\times 12$ classes = `2,904` KS comparisons)

- **Rows predicted `legit` (`pred == 0`, `1,538,641` Train vs `256,613` Oct)**: Top shifts (`2,904` comparisons) are calendar progression (`mer_age_days` median `660.0 -> 717.0`, `KS = 0.1664`, `p_bonf = 0.0000`; `rcv_acct_age_days` median `592.0 -> 649.0`, `KS = 0.0747`, `p_bonf = 0.0000`), `0.50x` arrival density (`rcv_sum_7d` median `8589.1 -> 5940.0`, `KS = 0.0903`; `r_rcv_dt_prev` median `281.0 -> 411.0`, `KS = 0.0881`; `snd_out_cnt_24h` mean `1.2729 -> 1.1360`, `KS = 0.0854`; `rcv_uniq_snd_day` mean `144.29 -> 73.24`, `KS = 0.0743`; `rcv_cnt_24h` mean `287.52 -> 145.85`, `KS = 0.0739`; all `p_bonf = 0.0000`), and `x1.165` legit amount scaling (`amount` median `459.20 -> 534.95`, `KS = 0.0492`, `p_bonf = 0.0000`).
- **Rows flagged by the model (`pred in 1..11`, `25,419` Train vs `3,969` Oct before band overrides)**: Out of `2,662` `(class, column)` pairs (`242 * 11`), `68` have `KS_p_bonf < 0.05` (`2,904` comparisons):

| Predicted Class (`pred_raw`) | `n_tr` | `n_oct` | Top Shifted Columns (`KS_stat`, `tr_med -> oct_med` or `tr_mean -> oct_mean`, `KS_p_bonf` across `2,904` tests) |
|---|---:|---:|---|
| `1 mule_account` | 3,896 | 798 | None significant (`mer_age_days` `KS = 0.2549`, `p_bonf = 1.0000`; `prev_in_amt_ratio_30m` `KS = 0.2406`, `p_bonf = 1.0000`) |
| `2 velocity_bot` | 3,533 | 291 | `Qs_n` / `Q_dev_cnt_1h` (`44.0 -> 37.0`, `KS = 0.3535`, `p_bonf = 0.0000`); `hour` (mean `13.69 -> 11.84`, `KS = 0.3138`, `p_bonf = 0.0000`) |
| `3 layering_chain` | 3,772 | 574 | None significant (`mer_age_days` `KS = 0.2873`, `p_bonf = 1.0000`; `r_rcv_dt_prev` `KS = 0.1607`, `p_bonf = 1.0000`) |
| `4 sim_swap` | 2,253 | 335 | `rcv_cnt_7d` (mean `125.15 -> 7.77`, `KS = 0.1690`, `p_bonf = 0.0003`); `pair_share_rcv_7d` (mean `0.8290 -> 0.9280`, `KS = 0.1677`, `p_bonf = 0.0004`) |
| `5 account_takeover` | 1,389 | 265 | None significant (`amt_over_snd_in_48h` `KS = 0.5707`, `p_bonf = 1.0000`; `snd_band_5_3h` `KS = 0.1249`, `p_bonf = 1.0000`) |
| `6 phishing_collect` | 3,110 | 510 | None significant (`amt_over_snd_in_48h` `KS = 0.3036`, `p_bonf = 1.0000`; `mer_age_days` `KS = 0.2761`, `p_bonf = 1.0000`) |
| `7 qr_swap` | 3,265 | 517 | `mer_age_days` (`777.0 -> 1203.0`, `KS = 0.2400`, `p_bonf = 0.0000`; calendar drift on merchants) |
| `8 refund_abuse` | 1,604 | 276 | `pair_share_rcv_7d` (`0.7143 -> 0.8750`, `KS = 0.1961`, `p_bonf = 0.0001`; lower Oct merchant volume) |
| `9 fake_cashback` | 1,153 | 208 | None significant (`prev_in_amt_ratio_30m` `KS = 0.4529`, `p_bonf = 1.0000`; `fwd_amt_ratio_30m` `KS = 0.4304`, `p_bonf = 1.0000`) |
| `10 loan_app_extortion` | 653 | 104 | `pair_share_rcv_7d` (`0.5000 -> 1.0000`, `KS = 0.3352`, `p_bonf = 0.0000`); `rcv_cnt_7d` (`2.0 -> 1.0`, `KS = 0.3352`, `p_bonf = 0.0000`) |
| `11 merchant_collusion` | 791 | 91 | `rcv_cnt_total` (`33.0 -> 116.0`, `KS = 0.6900`, `p_bonf = 0.0000`); `pair_share_of_rcv` (`0.1892 -> 0.0862`, `KS = 0.5144`); `S_n` (`18.0 -> 23.0`, `KS = 0.3656`, `p_bonf = 0.0000`) |

### 2b. Near-Miss 24h Groups (`size >= 3`) & Partially Flagged Events (`Jul`, `Aug`, `Sep`, `Oct`)

Because `oof_final_local` is trained on labels containing ~2.3% ordinary-looking fraud, ordinary rows have baseline `pf = 1 - P[:,0] ≈ 0.08`, so `95.4%` of all rows fall in `0.05 <= pf <= 0.50` (`Jul: 95,383.3/100k`, `Aug: 95,457.7/100k`, `Sep: 95,343.9/100k`, `Oct: 95,579.5/100k`). We therefore report both `0.05 <= pf <= 0.50` and `0.15 <= pf <= 0.50` (above the ordinary baseline):

| Class | `2b-i A` (`0.05<=pf<=0.50`, entity/recipe rcv) Jul / Aug / Sep / Oct (`/100k`, `Oct/MaxTr`) | `2b-i B` (`0.05<=pf<=0.50`, ALL rcv) Jul / Aug / Sep / Oct (`/100k`, `Oct/MaxTr`) | `2b-i C` (`0.15<=pf<=0.50`, ALL rcv) Jul / Aug / Sep / Oct (`/100k`, `Oct/MaxTr`) | `2b-ii A` (Mixed, companion `0.05<=pf<=0.50`) Jul / Aug / Sep / Oct (`/100k`, `Oct/MaxTr`) | `2b-ii B` (Mixed, companion `0.15<=pf<=0.50`) Jul / Aug / Sep / Oct (`/100k`, `Oct/MaxTr`) |
|---|---|---|---|---|---|
| `1 mule_account` | `428.0 / 378.7 / 336.0 / 111.7` (`0.26x`) | `292.6 / 413.4 / 298.5 / 421.8` (`1.02x`) | `59.20 / 1.90 / 1.18 / 0.00` (`0.00x`) | `91.45 / 79.15 / 90.56 / 100.93` (`1.10x`) | `3.23 / 3.42 / 3.14 / 1.92` (`0.56x`) |
| `2 velocity_bot` | `1.14 / 0.00 / 0.00 / 0.00` (`0.00x`) | `0.00 / 0.00 / 0.00 / 2.30` (**`>2x`**) | `0.00 / 0.00 / 0.00 / 0.00` (`0.00x`) | `17.08 / 16.13 / 15.29 / 3.07` (`0.18x`) | `2.85 / 1.90 / 1.57 / 0.38` (`0.13x`) |
| `3 layering_chain` | `116.1 / 326.7 / 203.1 / 44.9` (`0.14x`) | `68.7 / 227.4 / 111.0 / 129.3` (`0.57x`) | `0.00 / 0.00 / 0.00 / 0.00` (`0.00x`) | `45.16 / 43.66 / 49.01 / 35.31` (`0.72x`) | `5.88 / 5.88 / 6.66 / 2.30` (`0.35x`) |
| `4 sim_swap` | `1547.8 / 1408.4 / 1328.7 / 501.6` (`0.32x`) | `825.1 / 920.0 / 1050.5 / 930.6` (`0.89x`) | `3.60 / 0.19 / 2.35 / 1.92` (`0.53x`) | `24.85 / 24.67 / 24.89 / 6.52` (`0.26x`) | `1.71 / 1.52 / 1.57 / 0.00` (`0.00x`) |
| `5 account_takeover` | `3.98 / 0.00 / 1.18 / 0.00` (`0.00x`) | `1.14 / 0.57 / 0.59 / 0.00` (`0.00x`) | `0.00 / 0.00 / 0.00 / 0.00` (`0.00x`) | `19.54 / 9.49 / 12.74 / 7.68` (`0.39x`) | `1.14 / 0.95 / 0.78 / 1.15` (`1.01x`) |
| `6 phishing_collect` | `59.6 / 61.7 / 60.6 / 13.0` (`0.21x`) | `24.7 / 24.7 / 30.4 / 13.4` (`0.44x`) | `0.00 / 0.00 / 0.00 / 0.00` (`0.00x`) | `80.82 / 72.51 / 73.51 / 18.04` (`0.22x`) | `1.71 / 1.33 / 0.98 / 0.77` (`0.45x`) |
| `7 qr_swap` | `639.0 / 612.3 / 990.3 / 323.5` (`0.33x`) | `906.7 / 931.0 / 956.0 / 1101.8` (`1.15x`) | `0.76 / 0.00 / 0.59 / 0.00` (`0.00x`) | `151.59 / 171.02 / 156.43 / 69.84` (`0.41x`) | `0.38 / 1.71 / 0.39 / 0.00` (`0.00x`) |
| `8 refund_abuse` | `18.4 / 14.4 / 14.3 / 9.6` (`0.52x`) | `3.79 / 10.82 / 8.04 / 1.15` (`0.11x`) | `0.57 / 0.00 / 0.59 / 0.00` (`0.00x`) | `6.45 / 10.06 / 8.82 / 1.15` (`0.11x`) | `1.33 / 4.18 / 2.74 / 0.00` (`0.00x`) |
| `9 fake_cashback` | `111.8 / 261.0 / 668.8 / 127.0` (`0.19x`) | `191.8 / 162.3 / 212.1 / 154.7` (`0.73x`) | `0.00 / 0.00 / 0.59 / 0.00` (`0.00x`) | `14.42 / 18.22 / 22.93 / 4.61` (`0.20x`) | `0.57 / 0.95 / 0.78 / 0.38` (`0.40x`) |
| `10 loan_app_extortion` | `4.93 / 4.18 / 3.14 / 0.00` (`0.00x`) | `2.85 / 1.33 / 1.37 / 0.00` (`0.00x`) | `4.55 / 2.28 / 0.59 / 0.00` (`0.00x`) | `8.54 / 7.97 / 13.53 / 4.22` (`0.31x`) | `5.69 / 4.37 / 8.43 / 2.69` (`0.32x`) |
| `11 merchant_collusion` | `3343.6 / 3308.3 / 2783.7 / 1350.1` (`0.40x`) | `52022.6 / 51724.0 / 51556.3 / 45032.7` (`0.87x`) | `5.12 / 23.16 / 30.97 / 7.68` (`0.25x`) | `1.90 / 3.42 / 5.10 / 1.54` (`0.30x`) | `0.57 / 0.38 / 0.39 / 0.38` (`0.67x`) |
| **`TOTAL_ALL_TYPES`** | **`6274.4 / 6375.6 / 6389.8 / 2481.4` (`0.39x`)** | **`54340.0 / 54415.5 / 54224.8 / 47787.6` (`0.88x`)** | **`73.80 / 27.52 / 36.85 / 9.59` (`0.13x`)** | **`461.80 / 456.30 / 472.81 / 252.90` (`0.53x`)** | **`25.04 / 26.57 / 27.44 / 9.98` (`0.36x`)** |

- **Printed rows for the single cell exceeding `2x` max train month (`velocity_bot` under `2b-i B`: `6` Oct rows = `2.30 / 100k` vs `0.00` in Jul–Sep, `comp = 758150` at fuel merchant `MR009024`)**:
  - `TXaf36830a | 2026-10-17 13:11:13 | snd=AC060838 rcv=MR009024(MERCHANT:fuel) dev=DV0060838 | QR amt=1105.45 bal=3318.39 pf=0.088 best_cls=qr_swap`
  - `TX4cbbd223 | 2026-10-17 18:36:32 | snd=AC026475 rcv=MR009024(MERCHANT:fuel) dev=DV0026475 | QR amt=99.45 bal=1859.16 pf=0.094 best_cls=qr_swap`
  - `TXc8cc980f | 2026-10-18 15:02:46 | snd=AC107376 rcv=MR009024(MERCHANT:fuel) dev=DV0107376 | PAY amt=482.15 bal=7992.18 pf=0.405 best_cls=velocity_bot`
  - `TX83caa041 | 2026-10-18 17:12:46 | snd=AC218281 rcv=MR009024(MERCHANT:fuel) dev=DV0218281 | QR amt=846.54 bal=9704.86 pf=0.088 best_cls=qr_swap`
  - `TXa057998c | 2026-10-18 21:33:43 | snd=AC227532 rcv=MR009024(MERCHANT:fuel) dev=DV0227532 | QR amt=3247.42 bal=68349.11 pf=0.094 best_cls=sim_swap`
  - `TX65605efe | 2026-10-19 14:54:27 | snd=AC252857 rcv=MR009024(MERCHANT:fuel) dev=DV0252857 | QR amt=351.42 bal=3096.47 pf=0.092 best_cls=sim_swap`
  - These are `6` unrelated customer fuel purchases over 49 hours linked only because `MR009024` is a fuel station and `TXc8cc980f` (`pf = 0.405`) tilted the component's summed probability toward `velocity_bot`.

---

## Search 3: A Tell for Sibling-Shaped Events Using Band Devices as Truth

We extracted `44` non-recipe features across all `6` required families (`10` intra-event amount-order features, `10` second-level timestamp features, `15` sender/receiver `accounts.csv` features, `2` `merchants.csv` features, `3` `device_city` vs `home_city`/receiver-city features, and `4` position-in-month features) and tested `2,642` quantile/binary threshold rules (`3,052` including the `n_rows >= 2` control):

| Sibling Pair | Band Events (`Jul–Sep`) | Non-Band Labelled Events (`Jul–Sep`) | Rules Tested | Passing `>=85%` Monthly Balanced Acc + Non-Band Check | Best Rule on Band (`band_min_m_bal` / `band_min_m_acc` / `nonband_bal`) | Best Rule on Non-Band (`nonband_bal` / `band_min_m_bal`) |
|---|---|---|---:|---:|---|---|
| **Pair 1**: `x0.25` drains (`4 sim_swap` vs `5 account_takeover`) | `47` (`37` vs `10`) | `436` (`298` vs `138`) | `386` | **`0`** | `sec_even_share >= 0.5 -> sim_swap` (`0.600` / `0.556` / `0.475`) | `amt_rho >= -0.9396 -> sim_swap` (`0.593` / `0.444`) |
| **Pair 2**: Mule fan-ins (`1 mule_account` vs `2 velocity_bot`) | `77` (`70` vs `7`) | `118` (`106` vs `12`) | `576` | **`0`** | `amt_first_gt_last <= 1 -> mule_account` (`0.500` / `0.793` / `0.500`) | `dt_even_share <= 0.55 -> mule_account` (`0.705` / `0.283`) |
| **Pair 3**: Layering chains (`3 layering_chain` vs `1 mule_account`) | `60` (`51` vs `9`) | `501` (`414` vs `87`) | `394` | **`0`** | `min_mean <= 27.23 -> layering_chain` (`0.667` / `0.381` / `0.551`) | `home_eq_rcv <= 0 -> layering_chain` (`0.581` / `0.350`) |
| **Pair 4a**: Collect events (`6 phishing_collect` vs `9 fake_cashback`) | `56` (`51` vs `5`) | `1,751` (`1,519` vs `232`) | `250` | **`0`** | `snd_lt_rcv_id <= 0 -> phishing_collect` (`0.806` / `0.650` / `0.487`) | `snd_age_mean <= 551 -> phishing_collect` (`0.549` / `0.237`) |
| **Pair 4b**: Collect events (`6 phishing_collect` vs `10 loan_app_extortion`) | `51` (`51` vs `0`) | `1,589` (`1,519` vs `70`) | `248` | **`0`** | Constant `phishing_collect` (`0.500` / `1.000` / `0.500`) | `snd_lt_rcv_id <= 0 -> phishing_collect` (`0.604` / `0.306`) |
| **Pair 5a**: Fixed-amount merchant (`7 qr_swap` vs `8 refund_abuse`) | `41` (`31` vs `10`) | `484` (`416` vs `68`) | `340` | **`0`** | `sec_mod10_share <= 0 -> qr_swap` (`0.806` / `0.643` / `0.638`) | `amt_dec <= 0 -> qr_swap` (`0.758` / `0.611`) |
| **Pair 5b**: Fixed-amount merchant (`7 qr_swap` vs `11 merchant_collusion`) | `45` (`31` vs `14`) | `438` (`416` vs `22`) | `390` | **`0`** | `dev_eq_home >= 1 -> qr_swap` (`0.833` / `0.727` / `0.728`) | `dt_mod10_share <= 0 -> qr_swap` (`0.838` / `0.795`) |
| **Pair 5b (`n_rows >= 2`)**: Removing singleton length proxy | `39` (`25` vs `14`) | `291` (`269` vs `22`) | `410` | **`0`** | `sec_mod10_share <= 0 -> qr_swap` (`0.792` / `0.667` / `0.805`) | `dt_mod10_share <= 0 -> qr_swap` (`0.812` / `0.792`) |
| **Pair 5c**: Fixed-amount merchant (`8 refund_abuse` vs `11 merchant_collusion`) | `24` (`10` vs `14`) | `90` (`68` vs `22`) | `458` | **`0`** | `sec_std <= 16.07 -> refund_abuse` (`0.778` / `0.750` / `0.537`) | `dt_mod10_share <= 0 -> refund_abuse` (`0.717` / `0.333`) |

- **Why `dt_mod10_share <= 0 -> qr_swap` on Pair 5b is a burst-length proxy (`docs/codex_sibling_shapes.md`), not a timestamp tell**: `qr_swap` sessions in `S_flag == 1` have `1..3` rows (where `0` or `1–2` inter-arrival gaps rarely hit a multiple of 10s), whereas `merchant_collusion` sessions have `S_n >= 10` (where $\ge 9$ uniform gaps hit a multiple of 10s with probability $1 - 0.9^9 = 61\%$). Even with that length proxy included, its minimum monthly balanced accuracy on band events is `0.795` (`0.792` on `n_rows >= 2`), below the `0.85` threshold.
- **Conclusion**: When a generator emits a sibling-shaped event, it calls the exact same shared shape helper on the same entity pools and uniform second-level clock; **no tell separates the generators across Jul, Aug, and Sep.**
