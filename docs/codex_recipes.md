# Observable recipes for four fraud classes

13 November 2026. Source: `data/processed/base_v0.parquet`. Python was run in isolated mode (`python -I`), using `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe` because the Windows `python` alias was unavailable.

**Main finding.** There are both class-specific recipes and a shared merchant-payment recipe. The latter uses exactly **{1000, 2000, 2500, 5000, 10000} rupees**, PAY/QR transactions, and a balance floor of **1.1 × amount**. Its event size separates much of `qr_swap`, `refund_abuse`, and `merchant_collusion`. Treating every repeated round merchant payment as collusion confuses these classes. Loans instead repeat across **different payees**; native refunds repeat by **sender**, sometimes across merchants; native QR swaps are usually isolated payments to **P2P receivers with a shop category**.

These are reverse-engineered observations and exact detector definitions, not recovered generator source code. Fixed equalities are distinguished below from empirical ranges and imperfect event-typing rules. No model was fitted.

**September results.** Rules use transaction fields and relationships only. `app_major` and `ip_risk_score` are excluded from the rule-building function. The app-major-3 share is measured over **every selected September row**, including `legit` and other labels—not over the anchors alone.

| Assigned recipe | Selected rows | `app_major == "3"` rows | Share | Own label | `legit` | Other labels |
|---|---:|---:|---:|---:|---:|---:|
| `loan_app_extortion` | **195** | 118 | **0.6051** | 169 (86.67%) | 21 | 5 |
| `merchant_collusion` | **251** | 171 | **0.6813** | 206 (82.07%) | 30 | 15 |
| `qr_swap` | **974** | 580 | **0.5955** | 812 (83.37%) | 122 | 40 |
| `refund_abuse` | **506** | 319 | **0.6304** | 407 (80.43%) | 73 | 26 |

Using the supplied calibration, these shares are consistent with predominantly patterned populations. They are noisy proxies, not clean-label precision; 0.681 does not mean more than 100% patterned. The four rules do not overlap on this training file.

Full September label counts:

| Observed label | Loan rule | Collusion rule | QR rule | Refund rule |
|---|---:|---:|---:|---:|
| `legit` | 21 | 30 | 122 | 73 |
| `loan_app_extortion` | 169 | 1 | 1 | 0 |
| `merchant_collusion` | 0 | 206 | 14 | 1 |
| `qr_swap` | 1 | 5 | 812 | 8 |
| `refund_abuse` | 0 | 0 | 15 | 407 |
| `mule_account` | 1 | 2 | 0 | 1 |
| `velocity_bot` | 0 | 3 | 2 | 5 |
| `layering_chain` | 0 | 1 | 2 | 2 |
| `sim_swap` | 0 | 1 | 1 | 2 |
| `account_takeover` | 1 | 1 | 3 | 2 |
| `phishing_collect` | 0 | 1 | 1 | 4 |
| `fake_cashback` | 2 | 0 | 1 | 1 |

**Study population and procedure.** Only `split == "train"` was read: 1,564,060 rows, including 147 October-dated training tails. September reporting uses exactly `month == "2026-09"`: **509,996 rows**, excluding those tails. All training rows are available for label-free neighborhoods so that September events can extend into their October training tails. No test rows or test labels enter this investigation.

Anchors satisfy the requested label, `app_major == "3"`, and `ip_risk_score > 0.5`. For each class, eight July–August anchors were sampled with seed 17, deduplicating `(sender_id, receiver_id)`. Their **complete ±86,400-second contexts** were printed, using same sender, same receiver, or same device, plus the reverse sender/receiver links used by `14_event_examples.py`. There was no label-based filtering of context rows, no shortening to six hours, and no 28-row cap. Additional targeted examples and longer sender histories were examined when the ±1-day view could not expose a multi-day recipe. Representative full contexts are included below; the code at the end reproduces the initial 32 context prints.

All-train anchor coverage is a diagnostic, not a detector input:

| Anchor label | Trap-selected anchors | Covered by its final rule |
|---|---:|---:|
| `loan_app_extortion` | 167 | 129 |
| `merchant_collusion` | 200 | 145 |
| `qr_swap` | 684 | 597 |
| `refund_abuse` | 334 | 314 |

Trap-selected means patterned fraud, not necessarily the native recipe of the supplied label. For example, loan-labelled anchor `TX17befca7` is a P2P COLLECT for 1,950 with balance 2,145; collusion-labelled anchor `TX2c061c65` is inside a mule fan-in. Forcing those into loan/collusion rules would conflate recipes. Some remaining anchors also lack an identifiable event signature; coverage is not claimed to be complete.

**Exact common notation.** Amount equality and set membership are exact on the stored values; integer means `amount == floor(amount)`. All timestamps are the dataset's IST timestamps, without timezone conversion. A count includes the queried row and both time endpoints:

```text
C(E, key, w, i) = #{j: E(j), key(j) == key(i), |t[j] - t[i]| <= w}
```

Thus ±7 days is a 14-day-wide interval. These are bidirectional offline features, not causal streaming features. Neither labels nor trap columns determine membership.

Let `A = {1000, 2000, 2500, 5000, 10000}` and define the shared merchant pool:

```text
S(i) = receiver_type == "MERCHANT"
       AND txn_type in {"PAY", "QR"}
       AND amount in A
```

Sort only S rows by `(receiver_id, t)`. Start a new session when the receiver changes or the gap from the preceding **S row** is **greater than 86,400 seconds**. Other payments do not bridge sessions. Let `N_S(i)` be the number of S rows in its session; set it to zero off S. A session may span more than 24 hours through consecutive links. Its size is not a centered ±24-hour count.

The exact final rules are:

```text
L(i) = txn_type == "PAY"
       AND amount is integer AND 1500 <= amount < 15000
       AND new_payee == 1 AND hour(timestamp) in {20,21,22,23}
       AND (merchant_category == "lending" OR receiver_type == "P2P")

loan_app_extortion(i) = L(i) AND C(L, sender_id, 7*86400, i) >= 3

merchant_collusion(i) = S(i) AND N_S(i) >= 10

Q_native(i) = txn_type == "QR" AND receiver_type == "P2P"
              AND merchant_category in {"grocery", "food", "fuel", "pharmacy"}

qr_swap(i) = Q_native(i) OR (S(i) AND 1 <= N_S(i) <= 3)

R(i) = txn_type == "REFUND" AND receiver_type == "MERCHANT"
       AND merchant_category in {"ecommerce", "electronics"}
R_native(i) = R(i) AND C(R, sender_id, 14*86400, i) >= 3

refund_abuse(i) = R_native(i) OR (S(i) AND 4 <= N_S(i) <= 8)
```

The merchant size-nine case is deliberately unassigned; none occurs with this session definition in training. This is an empirical typing boundary, not proof that the generator prohibits nine rows. No app-major, IP-risk, balance-floor, mismatch-flag, or label condition is hidden in these rules.

**Loan recipe: repeat late-night payments to new, changing payees.** The selected training population contains **612 rows**. Its 139 sender sessions, split at gaps over seven days, have **3–6 payments**, and **every session has a different receiver for every payment**. There are 544 lending-merchant rows and 68 P2P rows. September alone has 173 merchant and 22 P2P payments. Requiring repeat payments to the same receiver would miss the central structure.

All 612 amounts are integer, empirically **1,509–14,967**; all are new-payee PAY transactions at 20:00–23:59. The typical session lasts days: duration p10/median/p90 is **2.005/4.114/6.097 days**. Consecutive qualifying sender gaps have median **23.96 hours**, with p10/p90 **0.69/70.87 hours**. Payments are not exactly daily or regularly spaced.

All 612 rows satisfy `balance_before >= 1.1 * amount - 0.011`; **356/612 (58.17%)** have `abs(balance_before - 1.1 * amount) <= 0.011`. This is consistent with a balance floor such as `max(sampled_balance, 1.1 * amount)`, not evidence that every balance is exactly 1.1 times amount. Across 473 eligible sender transitions, no next balance equals previous balance minus previous payment, and no amount is repeated. There is no fixed multiplicative balance sequence for this class. Two approximately ×0.35 links occur in one short three-row, ATO-labelled contaminating sequence; they are not a loan recurrence. The rule is useful but does not perfectly separate every overlapping recipe.

Full ±1-day context of loan anchor **TX75c1ce23**, 19 August 20:48:20, `AC139702`, device `DV0139702` (all four rows share that sender and device). All are PAY, category lending, new_payee=1, and labelled `loan_app_extortion`:

| Timestamp | Receiver | Amount | Balance before | Balance / amount | Anchor |
|---|---|---:|---:|---:|---|
| Aug 19 20:18:04 | MR005337 | 7,611 | 11,251.13 | 1.4783 | |
| Aug 19 20:48:20 | MR000647 | 9,658 | 10,623.80 | **1.1000** | * |
| Aug 19 23:57:09 | MR025924 | 5,478 | 6,025.80 | **1.1000** | |
| Aug 20 20:42:39 | MR001253 | 2,734 | 8,058.43 | 2.9475 | |

The P2P branch is real: anchor **TXbdc87834** pays 14,476 to `AC328694` at Aug 28 21:08:39 with balance 15,923.60 (=1.1×). Its ±1-day context contains only one other row: an unrelated 408.18 grocery QR earlier that day. The repeated-sender evidence is therefore necessarily wider than one day.

A lending-only row rule (same hour/amount/new-payee conditions, without repetition) selects 174 September rows, app-major share **0.6207**, with 151 loan labels. The final sender-repeat rule reaches **169 loan labels**. Broadening to all matching evening P2P rows without repetition would select 932 rows, app share only **0.3777**; many are other fraud recipes or ordinary rows.

**The shared fixed-amount merchant recipe.** S contains **2,011 training rows**. Every row satisfies `balance_before >= 1.1 * amount - 0.011`; **595** have equality within 0.011 rupee. The amounts are selected from the five-element set, so consecutive amount ratios lie in `{a/b: a,b in A}`. Ratios such as 0.5, 2, 5, and 0.2 reflect independent choices from that set. When both balances hit the 1.1 floor, their ratio automatically equals the amount ratio; it is not a separate balance-drain recipe.

| S session class | Sessions | Rows | Size range | Median distinct senders | Duration p10 / median / p90, hours | Exact 1.1-floor rows |
|---|---:|---:|---:|---:|---|---:|
| Small, assigned QR | 492 | 972 | 1–3 | 2 | 0 / 1.059 / 7.512 | 273 |
| Medium, assigned refund | 66 | 383 | 4–8 | 3 | 3.227 / 10.177 / 29.788 | 107 |
| Large, assigned collusion | 36 | 656 | 10–28 | 4 | 22.276 / 46.021 / 69.331 | 215 |

These are observations from the label-free session definitions, not claimed hard generator limits. The size classes were suggested by the July–August examples and their label composition. Categories vary widely, including lending; using category as the sole class identity would misclassify these merchant events.

**Merchant collusion: large sessions of fake sales, with repeated customers.** Large S sessions have a median of eight payments by their most frequent sender; observed distinct sender counts range **2–9**. There are **333 PAY and 323 QR** rows, **522 new_payee=0 and 134 new_payee=1** rows. Do not require new_payee=0 or one device. In 516 successive same-session, same-sender pairs, **103 repeat the amount**, but no balance follows the ledger equation `next_balance = previous_balance - previous_amount`. Only 215/656 balances hit the 1.1 floor, so an equality-only detector would miss most of the event.

Full ±1-day context of collusion anchor **TXf855edee**, Aug 26 08:15:50 (`AC159183`, `MR023286`, `DV0159183`). Every row is printed; the first is background associated through the sender. The merchant rows all have category ecommerce.

| Timestamp | Sender | Receiver | Device | Amount | Balance before | Type | Label | Anchor |
|---|---|---|---|---:|---:|---|---|---|
| Aug 25 15:44:25 | AC159183 | AC081187 | DV0159183 | 100 | 3,830.67 | PAY | legit | |
| Aug 25 22:21:38 | AC028006 | MR023286 | DV0028006 | 2,500 | 3,095.15 | PAY | merchant_collusion | |
| Aug 26 02:01:56 | AC411304 | MR023286 | DV0411304 | 2,000 | 6,321.84 | PAY | merchant_collusion | |
| Aug 26 03:38:50 | AC411304 | MR023286 | DV0411304 | 10,000 | 15,931.03 | PAY | legit | |
| Aug 26 08:15:50 | AC159183 | MR023286 | DV0159183 | 10,000 | **11,000** | QR | merchant_collusion | * |
| Aug 26 10:21:56 | AC159183 | MR023286 | DV0159183 | 2,500 | **2,750** | QR | merchant_collusion | |
| Aug 26 13:41:43 | AC028006 | MR023286 | DV0028006 | 10,000 | **11,000** | QR | merchant_collusion | |
| Aug 26 17:43:46 | AC159183 | MR023286 | DV0159183 | 2,000 | 4,110.75 | PAY | merchant_collusion | |

This two-day excerpt does not contain the whole multi-day merchant session. The complete session supplies `N_S`; a day boundary or pair-only threshold should not define the event.

**QR swap has two observable branches.** The native branch is **QR + P2P receiver + grocery/food/fuel/pharmacy category**. It selects **2,123 training rows**, all distinct sender–receiver pairs; there are no successive selected transactions from the same sender within one day. The relevant inconsistency is a shop category attached to a person-to-person receiver. Propagating the label to every nearby sender/device transaction is not justified.

Mismatch is 1 on **1,296** native rows and 0 on **827**: requiring it would miss 39% of this branch. Amounts can have paise; only 549/2,123 are whole rupees and 538 are multiples of ten. All balances are above 1.1×amount, but **none** equals the 1.1 floor within 0.011 rupee. No fixed cross-row numeric relation is supported.

Full ±1-day context of native anchor **TX437bab33** is just this one row:

| Timestamp | Sender | Receiver | Device | Amount | Balance before | Type/category | Mismatch | Label |
|---|---|---|---|---:|---:|---|---:|---|
| Jul 10 13:01:55 | AC204069 | AC148337 | DV0204069 | 680 | 2,349.38 | QR / food | **0** | qr_swap |

The other branch is small S sessions, including PAY rows without a QR mismatch. Full ±1-day context of trap-selected anchor **TX4bc018bd**, Jul 1 16:44:36:

| Timestamp | Sender | Receiver | Device | Amount | Balance before | Type/category | Label | Anchor |
|---|---|---|---|---:|---:|---|---|---|
| Jul 1 14:44:59 | AC296958 | AC403505 | DV0142359 | 80 | 1,532.98 | PAY / null | legit | |
| Jul 1 16:44:36 | AC310448 | MR002384 | DV0142359 | 5,000 | **5,500** | PAY / grocery | qr_swap | * |
| Jul 1 18:15:09 | AC062324 | MR002384 | DV0062324 | 2,500 | 4,559.46 | QR / grocery | qr_swap | |
| Jul 2 10:40:38 | AC296958 | MR007972 | DV0142359 | 508.06 | 3,831.18 | PAY / ecommerce | legit | |

The rule selects the two S rows, not the two background rows sharing the anchor's device. The matching merchant pair amounts are 5,000 and 2,500, but balance halves **do not** follow that amount ratio (4,559.46/5,500 is not 0.5).

The two QR branches contribute **663 + 311 = 974 September rows**, with app-major shares **0.6199 and 0.5434**, respectively. The small-S branch adds **252 QR labels**. It is less pure by the proxy, and its size-three boundary also admits 13 refund labels. That ambiguity should be retained as a feature, not described as perfect event typing.

A negative control is especially useful: merchant QR rows with `qr_vpa_mismatch == 1` select **584 September rows**, app-major share **0.1661**, comprising **543 legit, 24 qr_swap, and 17 other labels**. Mismatch alone is mostly background here. Adding it as an unconditional OR would dilute the recipe considerably.

**Refund abuse also has two branches.** Native refunds are repeated REFUND rows from one sender to ecommerce/electronics merchants. Using sender counts rather than pair counts permits merchant changes and device changes. The selected population has **1,103 training rows**: 848 ecommerce and 255 electronics; 1,086 fractional amounts and 17 integers. Sender sessions with a 14-day consecutive-gap limit are usually **3–8 rows**, typically one or two merchants. Two merged/reused-sender sessions have 10 and 11 rows and extend the observed maximum to 21.21 days, so do not impose an eight-row hard cap on the native branch. Median duration is **7.09 days**; median consecutive gap **24.73 hours**, p10/p90 **3.16/77.79 hours**.

There is **no exact amount/balance recurrence** to exploit in this branch. Across 912 eligible sender transitions, zero amounts repeat and zero next balances equal previous balance minus previous amount. No native refund has balance exactly 1.1×amount within 0.011 rupee; **268/1,103 are below that floor**. Amounts range 371.87–15,792.29, with median 2,282.55. There are **zero** preceding PAY/QR purchases to the same receiver from the same sender within 30 days for these 1,103 refunds. A matching-original-purchase or fixed refund-to-purchase-ratio requirement would therefore miss this recipe.

Full ±1-day context of refund anchor **TX6ef03551**, Aug 18 07:03:43, is the following two rows. Both use `AC068379`, `MR011091`, device `DV0352981`, type REFUND, category ecommerce, and label refund_abuse:

| Timestamp | Amount | Balance before | Balance / amount | Anchor |
|---|---:|---:|---:|---|
| Aug 17 19:31:48 | 2,146.91 | 10,965.76 | 5.1077 | |
| Aug 18 07:03:43 | 948.01 | 7,819.11 | 8.2479 | * |

The longer history is necessary for detection. For example, the qualifying refund sequence for `AC263754` is:

| Timestamp | Receiver | Amount | Balance before | Label |
|---|---|---:|---:|---|
| Jul 7 10:24:09 | MR003526 | 3,224.92 | 3,505.04 | legit |
| Jul 8 01:48:48 | MR003526 | 1,003.46 | 4,111.03 | refund_abuse |
| Jul 9 01:50:46 | MR003526 | 2,902.72 | 2,206.10 | legit |
| Jul 10 12:22:53 | MR003526 | 998.91 | 1,804.54 | refund_abuse |
| Jul 11 12:21:06 | MR006258 | 1,671.17 | 2,278.73 | refund_abuse |
| Jul 13 14:12:25 | MR006258 | 1,149.53 | 5,111.11 | refund_abuse |

This is the complete qualifying sequence, not the full unfiltered ±1-day context. It shows the second merchant and the legit-labelled event rows that the rule should still select.

The merchant-style refund branch has 4–8 S rows rather than REFUND transactions. It contributes **108 September rows**, including 81 refund labels, with app-major share **0.6019**. These are coherent groups, not merely isolated wrong-type labels. For trap-selected anchor **TX8f08f683**, the full ±1-day context has 15 rows:

| Timestamp | Sender | Receiver | Device | Amount | Balance before | Type/category | Label | Anchor |
|---|---|---|---|---:|---:|---|---|---|
| Jul 1 19:07:40 | AC278931 | MR017436 | DV0278931 | 130.44 | 2,335.64 | QR / lending | legit | |
| Jul 1 22:49:38 | AC248812 | MR017436 | DV0248812 | 1,169.72 | 12,435.08 | QR / lending | legit | |
| Jul 2 02:11:39 | AC066336 | MR016433 | DV0066336 | 1,000 | 1,749.24 | PAY / ecommerce | legit | |
| Jul 2 03:10:15 | AC066336 | MR016433 | DV0066336 | 2,500 | 3,118.72 | QR / ecommerce | legit | |
| Jul 2 08:26:28 | AC065737 | MR017436 | DV0172268 | 96 | 1,452.21 | PAY / lending | legit | |
| Jul 2 14:19:31 | AC015968 | MR017436 | DV0015968 | 5,000 | **5,500** | PAY / lending | refund_abuse | |
| Jul 2 15:16:13 | AC066336 | MR017436 | DV0066336 | 1,000 | 3,755.42 | QR / lending | refund_abuse | * |
| Jul 2 15:22:50 | AC397003 | MR017436 | DV0224472 | 2,500 | **2,750** | QR / lending | refund_abuse | |
| Jul 2 17:07:11 | AC397003 | MR017436 | DV0224472 | 2,500 | 4,007.29 | PAY / lending | refund_abuse | |
| Jul 2 17:14:27 | AC292066 | MR017436 | DV0292066 | 1,703.98 | 2,972.21 | QR / lending | legit | |
| Jul 2 17:48:09 | AC397003 | MR017436 | DV0224472 | 1,000 | 9,199.19 | PAY / lending | refund_abuse | |
| Jul 2 17:55:58 | AC333321 | MR017436 | DV0333321 | 2,500 | 3,179.14 | QR / lending | refund_abuse | |
| Jul 3 10:26:00 | AC001411 | MR017436 | DV0001411 | 463.89 | 2,212.10 | QR / lending | legit | |
| Jul 3 12:33:29 | AC330489 | MR017436 | DV0330489 | 187.86 | 2,446.99 | QR / lending | legit | |
| Jul 3 14:23:59 | AC009296 | MR017436 | DV0009296 | 351.12 | 8,658.72 | REFUND / lending | legit | |

The six fixed-amount rows to MR017436 form its medium S session. Neither the other two S rows to another merchant nor the small ordinary REFUND to the lending merchant is automatically part of that event. The native refund branch selects **398 September rows**, app-major share **0.6382**, including 326 refund labels; the union gives the reported 506 rows and 407 refund labels.

**Robustness and limits.** The equivalent July–August rule counts/app-major shares are loan **408/0.6397**, collusion **390/0.6410**, QR **2,121/0.5860**, and refund **893/0.6237**. These support the same qualitative recipes before September. This remains exploratory analysis, not a pristine untouched model-validation experiment.

For S sessions, using a 12-hour gap fragments large collusion events: the >=9-row family finds 217 September rows instead of 251. Increasing the gap from 24 to 48 hours leaves all September size-branch counts unchanged but merges one pair of differently typed July–August groups. A seven-day gap starts merging small QR groups and contaminating other size classes. The 24-hour rule is therefore explicit, conservative event reconstruction, not an alleged exact generator timer.

Size three overlaps the small-merchant refund and QR recipes. Large groups can also contain unrelated transactions, reused senders, and noisy labels. The report consequently does not claim to explain every trap-selected anchor or assign a clean class to every patterned row. Whole-rupee amounts, a balance ratio of 1.1, or app-major purity alone do not establish the fraud type.

Useful model inputs would be the individual eligibility flags, sender counts, S-session size/span/distinct-sender count, per-session sender repeat count, amount-set membership, `abs(balance_before - 1.1*amount)`, and the signed balance-floor residual. Their definitions above are label-free. Keep app/version/IP-risk columns out of the feature function. No implementation, model, feature cache, or prediction file was modified; this report is the only new repository file from this task.

**Runnable reproduction.** Save the following code to a temporary file outside the repository and run from the repository root with `python -I path/to/reproduce_recipes.py`. It independently builds the rules from a feature-only frame, verifies the reported counts, prints the full label mixtures, and prints the original 32 untruncated anchor contexts. It writes no files. pandas 2.3.3, NumPy and PyArrow were available in the analysis environment.

```python
import hashlib
import numpy as np
import pandas as pd

path = "data/processed/base_v0.parquet"
feature_cols = ["t", "timestamp", "sender_id", "receiver_id", "device_id",
                "receiver_type", "txn_type", "merchant_category", "amount",
                "balance_before", "new_payee", "qr_vpa_mismatch"]
cols = ["txn_id", "split", "month", "label", "app_major", "ip_risk_score"] + feature_cols
d = pd.read_parquet(path, columns=cols, filters=[("split", "==", "train")])
assert len(d) == 1564060 and d.txn_id.is_unique

def build_rules(f):
    assert {"label", "app_major", "app_version", "ip_risk_score"}.isdisjoint(f.columns)
    boolean = lambda s: s.fillna(False).to_numpy(dtype=bool)
    t = f.t.to_numpy(dtype=np.int64)
    scale = np.int64(2**26)
    assert t.min() >= 0 and t.max() + 14*86400 < scale
    assert f[["sender_id", "receiver_id", "device_id", "t"]].notna().all().all()
    key = pd.factorize(f.sender_id)[0].astype(np.int64) * scale + t

    def sender_count(pool, radius):
        times = np.sort(key[pool])
        return (np.searchsorted(times, key + radius, side="right")
                - np.searchsorted(times, key - radius, side="left"))

    integer = boolean(f.amount.eq(np.floor(f.amount)))
    L = (integer & boolean(f.txn_type.eq("PAY"))
         & boolean(f.amount.ge(1500) & f.amount.lt(15000))
         & boolean(f.new_payee.eq(1)) & boolean(f.timestamp.dt.hour.ge(20))
         & boolean(f.merchant_category.eq("lending") | f.receiver_type.eq("P2P")))
    loan = L & (sender_count(L, 7*86400) >= 3)

    native_qr = boolean(f.txn_type.eq("QR") & f.receiver_type.eq("P2P")
        & f.merchant_category.isin(["grocery", "food", "fuel", "pharmacy"]))
    R = boolean(f.txn_type.eq("REFUND") & f.receiver_type.eq("MERCHANT")
        & f.merchant_category.isin(["ecommerce", "electronics"]))
    native_refund = R & (sender_count(R, 14*86400) >= 3)

    S = boolean(f.receiver_type.eq("MERCHANT") & f.txn_type.isin(["PAY", "QR"])
        & f.amount.isin([1000, 2000, 2500, 5000, 10000]))
    x = f.loc[S].sort_values(["receiver_id", "t"], kind="stable").copy()
    x["session"] = (x.receiver_id.ne(x.receiver_id.shift())
                    | x.t.diff().gt(86400)).cumsum()
    session_n = pd.Series(0, index=f.index, dtype="int64")
    session_n.loc[x.index] = x.groupby("session").t.transform("size")
    n = session_n.to_numpy()
    rules = {"loan_app_extortion": loan,
             "merchant_collusion": S & (n >= 10),
             "qr_swap": native_qr | (S & (n >= 1) & (n <= 3)),
             "refund_abuse": native_refund | (S & (n >= 4) & (n <= 8))}
    branches = {"qr_native": native_qr, "qr_merchant_small": S & (n >= 1) & (n <= 3),
                "refund_native": native_refund,
                "refund_merchant_medium": S & (n >= 4) & (n <= 8)}
    return rules, branches

rules, branches = build_rules(d[feature_cols])
sep = d.month.eq("2026-09").to_numpy()
early = d.month.lt("2026-09").to_numpy()
old = d.app_major.eq("3").to_numpy()
assert int(sep.sum()) == 509996
expected = {"loan_app_extortion": (195, 118), "merchant_collusion": (251, 171),
            "qr_swap": (974, 580), "refund_abuse": (506, 319)}
for name, mask in {**rules, **branches}.items():
    selected = mask & sep
    n, old_n = int(selected.sum()), int((selected & old).sum())
    if name in expected:
        assert (n, old_n) == expected[name]
    print(name, "September", n, "app_major_3", old_n, "share", old_n/n)
    print(d.loc[selected, "label"].value_counts().to_dict())
    print("July-August", int((mask & early).sum()), "app_major_3_share", old[mask & early].mean())
assert np.stack(list(rules.values())).sum(axis=0).max() == 1

# Traps select examples and measure coverage only, after rule construction.
trap = d.app_major.eq("3") & d.ip_risk_score.gt(.5)
pd.set_option("display.width", 260)
for lab in rules:
    anchors = d[trap & d.label.eq(lab)]
    print("ANCHOR COVERAGE", lab, int(rules[lab][anchors.index].sum()), len(anchors))
    pool = anchors[anchors.month.lt("2026-09")].sample(frac=1, random_state=17)
    seen = set()
    for idx, a in pool.iterrows():
        pair = (a.sender_id, a.receiver_id)
        if pair in seen:
            continue
        seen.add(pair)
        linked = (d.sender_id.eq(a.sender_id) | d.receiver_id.eq(a.receiver_id)
                  | d.device_id.eq(a.device_id) | d.sender_id.eq(a.receiver_id)
                  | d.receiver_id.eq(a.sender_id))
        context = d.loc[linked & d.t.between(a.t-86400, a.t+86400)].sort_values("t").copy()
        context["anchor"] = np.where(context.index == idx, "*", "")
        context["balance_over_amount"] = context.balance_before / context.amount
        show = ["anchor", "txn_id", "timestamp", "sender_id", "receiver_id", "device_id",
                "amount", "balance_before", "balance_over_amount", "txn_type", "merchant_category",
                "new_payee", "qr_vpa_mismatch", "app_major", "ip_risk_score", "label"]
        print("\nANCHOR", lab, a.txn_id, a.timestamp, "CONTEXT ROWS", len(context))
        print(context[show].to_string(index=False))
        if len(seen) == 8:
            break
with open(path, "rb") as f:
    print("Input SHA256", hashlib.file_digest(f, "sha256").hexdigest())
```
