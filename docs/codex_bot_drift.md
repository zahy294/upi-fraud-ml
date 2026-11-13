# October bot drift: fewer observed events; no supported replacement shape

Analysis date: 13 November 2026. Input: `data/processed/base_v0.parquet`. Read the update box in `docs/IMPLEMENTATION_PLAN.md` and `scripts/explore/14_event_examples.py` before investigating. All Python executions used `python -I` (the installed Python 3.11 executable; the `python` Windows app alias was unavailable).

**Finding.** The single-device signature covers **1,066 September rows in 25 events**, versus **269 test rows in seven events**: **209.02 versus 103.23 rows per 100,000**, a rate ratio of **0.494**. The surviving October events retain September's amount, merchant, timing, and sender patterns. Searches for multiple devices, longer gaps, other categories, fractional amounts, larger amounts, and different transaction types did **not** identify a credible replacement population. **No tested burst definition restores September's rate while retaining evidence that the additional rows are bots.** The observable change is fewer qualifying events; the data do not establish what happened to the unobserved fraud activity.

There is a misleading numerical near-match: flagging every whole-rupee merchant PAY under Rs 500, with no burst requirement, selects **529 test rows**, close to the **544.67** expected from September's original burst rate. But that same broad definition selects **1,625 September rows**, so its own October/September rate ratio is only **0.637**. Its **559 additional September rows contain only two `velocity_bot` labels**. This is not evidence of recovering missing bots.

**Data boundaries and label isolation.** September means `split == "train"` and `month == "2026-09"` (509,996 rows). October means **all `split == "test"`**, as requested (260,582 rows): 260,536 are October-dated and 46 are November tails. The 147 October-dated training tails are excluded from both reporting denominators. Neighborhoods use all input rows, so boundaries do not truncate events. All 269 selected test burst rows are October-dated; using calendar October alone barely changes the reported rate (103.25 per 100,000).

The exploratory data reads explicitly excluded `label`, `app_version`, `app_major`, and `ip_risk_score`. Candidate definitions and test counts were computed without them. A separate parquet read with `filters=[("split", "==", "train")]` loaded **training labels only**, joined by `txn_id`, for the post-hoc September specificity checks below. No test labels, model probabilities, pseudo-labels, or leaderboard results were used. No existing repository files were edited, and no feature caches or predictions were rewritten.

Input SHA-256: `e88ee1fa76cc49613049c136534b6c7b16189c1859500caf07261f0ea2800cd9`.

**Exact definitions.** Let `t` be the existing seconds-valued timestamp. Define the eligible pools:

| Pool | Row condition |
|---|---|
| `M` | `receiver_type == "MERCHANT"` |
| `P` | `M & (txn_type == "PAY")` |
| `I` | `amount == floor(amount)`; exact equality, no rounding tolerance |
| `Q` | `P & I & (amount < 500)` |
| `J` | `P & I`, without an amount ceiling |
| `F500` | `P & (amount < 500)`, including fractional amounts |
| `F1000` | `P & (amount < 1000)`, including fractional amounts |
| `Qany` | `M & I & (amount < 500)`, any transaction type |

For pool `E`, entity key `k`, and radius `w` seconds, define:

```text
N(E,k,w,i) = {j : E(j), key_k(j) == key_k(i), |t_j - t_i| <= w}
C(E,k,w,i) = number of rows in N(E,k,w,i)
U(E,k,w,a,i) = number of distinct values of attribute a in N(E,k,w,i)
D(E,k,w,h,i) = E(i) AND C(E,k,w,i) >= h
```

Endpoints are inclusive, and counts include the current row if eligible. Distinct counts use full IDs. A radius of 600 seconds means **±10 minutes, a 20-minute interval**, not a fixed 10-minute bin. The method has no bin alignment or half-window offset. These are offline, bidirectional features; they use later transactions and are not causal streaming features. There are no missing sender/device/receiver IDs or `t` values in this input.

The baseline `B` is **`D(Q, device_id, 600, 3)`**. For event summaries only, sort Q rows by `(device_id, t)` and start a new session when the device changes or the gap from the preceding eligible row exceeds 600 seconds. Retain sessions with at least three rows. On this input, session membership equals `B` exactly. Session size is not generally interchangeable with a centered-window count: the equality here is empirical. Define session features as size, `max(t)-min(t)`, distinct sender and receiver counts, amount quantiles, and within-session consecutive timestamp gaps (omit the first gap; retain zero-second ties).

**The event distribution does not show a new October shape.** Counts below are deduplicated rows, not sums of overlapping neighborhoods.

| Period | All rows | Events | Burst rows | Burst rows / 100k | Mean event size |
|---|---:|---:|---:|---:|---:|
| July train | 527,065 | 33 | 1,307 | 247.98 | 39.61 |
| August train | 526,852 | 26 | 968 | 183.73 | 37.23 |
| September train | 509,996 | 25 | 1,066 | 209.02 | 42.64 |
| October test split | 260,582 | 7 | 269 | 103.23 | 38.43 |

| Measurement within selected bursts | September | October test |
|---|---:|---:|
| Event size: min / median / max | 20 / 44 / 60 | 29 / 37 / 58 |
| Event duration, seconds: p10 / median / p90 | 236.8 / 487 / 692.6 | 355.8 / 429 / 657.4 |
| Maximum event duration, seconds | 817 | 787 |
| Consecutive payment gap, seconds: p10 / median / p90 | 1 / 7 / 28 | 1 / 8 / 31.9 |
| Maximum consecutive gap, seconds | 156 | 102 |
| Amount, rupees: p10 / median / p90 | 52 / 241 / 434.5 | 66.8 / 253 / 452.2 |
| Amount min / max | 10 / 498 | 16 / 498 |
| Ecommerce / gaming rows | 864 / 202 | 210 / 59 |
| Merchant count per event: min / median / max | 1 / 2 / 3 | 1 / 2 / 3 |
| Events with exactly three sender accounts | 10 of 25 | 3 of 7 |
| Remaining events' distinct sender range | 26–60 | 32–58 |
| Median login-to-payment time, across rows | 4 seconds | 4 seconds |

All selected rows are whole-rupee merchant PAY transactions to gaming/ecommerce merchants; category was **not** part of `Q`. There are already September variants with **20 payments**, approximately **13.6-minute duration**, or **only three sender accounts**. Requiring at least 25 rows, at most ten minutes of total duration, or many distinct senders would unnecessarily discard established events. One October event has a median login time of 46 seconds; a short-login requirement would also be too restrictive.

**Candidate comparisons.** Both months use the same rule and their own all-row denominator. `Additional` means October rows outside `B`; rows lost from `B` are possible in definitions with higher thresholds. September precision is the observed fraction labelled `velocity_bot`, not an estimate of clean-label precision.

| Candidate | Sep rows | Test rows | Sep / 100k | Test / 100k | Test/Sep rate | Additional test rows | Sep bot precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B = D(Q,device,600,3)` | 1,066 | 269 | 209.02 | 103.23 | 0.494 | 0 | 84.62% |
| Require 25: `D(Q,device,600,25)` | 1,026 | 269 | 201.18 | 103.23 | 0.513 | 0 | 84.50% |
| Slow device: `D(Q,device,86400,3)` | 1,066 | 269 | 209.02 | 103.23 | 0.494 | 0 | 84.62% |
| Merchant grouping: `D(Q,receiver,600,3)` | 1,066 | 269 | 209.02 | 103.23 | 0.494 | 0 | 84.62% |
| Slow merchant: `D(Q,receiver,86400,3)` | 1,145 | 308 | 224.51 | 118.20 | 0.526 | 39 | 78.78% |
| All integer amounts: `D(J,receiver,86400,3)` | 1,589 | 469 | 311.57 | 179.98 | 0.578 | 200 | 56.95% |
| All amounts/categories: `D(P,device,600,5)` | 1,066 | 269 | 209.02 | 103.23 | 0.494 | 0 | 84.62% |
| All amounts, slow device: `D(P,device,86400,5)` | 1,098 | 284 | 215.30 | 108.99 | 0.506 | 15 | 82.15% |
| Allow fractional small amounts: `D(F500,receiver,600,10)` | 1,170 | 253 | 229.41 | 97.09 | 0.423 | 0 | 74.02% |
| All amounts, merchant: `D(P,receiver,600,10)` | 9,345 | 650 | 1,832.37 | 249.44 | 0.136 | 397 | 9.46% |
| Very broad merchant: `D(P,receiver,86400,3)` | 83,846 | 39,078 | 16,440.52 | 14,996.43 | 0.912 | 38,809 | 1.41% |
| No burst requirement: `Q` | 1,625 | 529 | 318.63 | 203.01 | 0.637 | 260 | 55.63% |
| No burst; gaming/ecommerce only: `Q & category in {gaming,ecommerce}` | 1,199 | 323 | 235.10 | 123.95 | 0.527 | 54 | 75.23% |
| No burst; any transaction type: `Qany` | 2,572 | 959 | 504.32 | 368.02 | 0.730 | 690 | 35.26% |

The pre-existing refined rule, `I & M & ((dev_cnt_5m >= 3) | (login_to_txn_sec <= 5)) & (home_device == 0)`, selects 1,067 September and 270 test rows (rate ratio 0.495). Thus the observed shortfall is not an artifact of the new baseline definition.

The rolling scan covered **200 definitions**: pools `{Q,J,F500,F1000,P}` × keys `{device_id,receiver_id}` × radii `{600,3600,21600,86400}` × thresholds `{3,5,10,20,25}`. The largest test/September rate ratio anywhere in that scan was **0.912**, the very broad, 1.41%-precision merchant rule in the table. None reached one. A separate **96-definition** session scan used pools `{Qany, M & I}` × keys `{device_id,receiver_id,sender_id}` × consecutive-gap limits `{600,3600,21600,86400}` × sizes `{3,5,10,20}`. It also failed to restore the rate; these long sessions can chain across days and are not fixed-width windows.

Removing **both** amount and transaction-type restrictions from 600-second device sessions, even allowing P2P receivers, still gives exactly 1,066 September and 269 test rows at session size >=10. This does not reveal bots moving to other amounts, transaction types, or receiver types on a shared device.

**Multiple-device and fully dispersed searches.** Define the explicit multiple-device flag:

```text
MD_w(i) = Q(i)
          AND C(Q,receiver_id,w,i) >= 3
          AND U(Q,receiver_id,w,device_id,i) >= 2
          AND U(Q,receiver_id,w,sender_id,i) >= 3
```

| Radius | September MD rows | Test MD rows |
|---|---:|---:|
| ±10 minutes | 0 | 0 |
| ±1 hour | 3 | 0 |
| ±6 hours | 20 | 3 |
| ±24 hours | 79 | 39 |

Every one of these rows is outside `B`. The 79 additional September rows at ±24 hours contain **zero** `velocity_bot` labels (78 `legit`, one other fraud label). The maximum eligible receiver-window size among the test multiple-device rows is only **six**, versus ten in September. The 39 test rows occur at three merchants and do not form a new 25–60-payment population.

For an even weaker linkage assumption, ignore **all** IDs and count `Q` rows globally within ±600 seconds. Requiring at least 20 selects **1,080 September / 270 test rows**, only **14 / one** outside `B`, respectively (rates 211.77 / 103.61 per 100,000; ratio 0.489). This search would include synchronized small whole-rupee payments even if they were spread across both devices and merchants. Using all merchant transaction types instead of PAY still adds only one test row at that threshold. Increasing the whole-rupee PAY ceiling to Rs 2,000 also adds only one. There is no missing dense population under those amount/time restrictions.

**Count distributions explain why threshold relaxation cannot help.** This table counts eligible `Q` rows by the size of their exact centered window. Empty bins are shown explicitly.

| Neighbor count | Device ±10m Sep | Device ±10m test | Receiver ±10m Sep | Receiver ±10m test | Receiver ±24h Sep | Receiver ±24h test |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 559 | 260 | 553 | 258 | 429 | 203 |
| 2 | 0 | 0 | 6 | 2 | 51 | 18 |
| 3–4 | 0 | 0 | 0 | 0 | 52 | 29 |
| 5–9 | 0 | 0 | 49 | 16 | 65 | 26 |
| 10–19 | 0 | 0 | 269 | 101 | 243 | 84 |
| 20–24 | 40 | 0 | 156 | 25 | 193 | 42 |
| 25–60 | 1,026 | 269 | 592 | 127 | 592 | 127 |
| 61+ | 0 | 0 | 0 | 0 | 0 | 0 |

There is no October mass at device counts 2–24 to recover with a lower threshold. All 260 non-burst `Q` test rows remain outside the device detector even at ±24 hours. Those residual rows occur at **99.78 per 100,000**, compared with **109.61** in September; they show no compensating increase. Even after dropping PAY, residual small whole-rupee merchant traffic falls from **295.30 to 264.79 per 100,000**.

**Specificity of additions, checked only on September training labels.** The baseline contains 902 bot-labelled, 130 legit-labelled, and 34 other-fraud-labelled rows. It captures 902 of 2,659 September `velocity_bot` labels (33.92% raw recall). This is consistent with substantial bot-labelled activity outside the recognizable recipe; it does not establish a label-free way to recover that activity in October.

| Broader definition | Additional Sep rows outside B | Bot labels in additions | Additional-row bot share |
|---|---:|---:|---:|
| Small PAY, receiver ±24h, >=3 | 79 | 0 | 0.00% |
| Any integer amount PAY, receiver ±24h, >=3 | 523 | 3 | 0.57% |
| Any amount PAY, device ±24h, >=5 | 32 | 0 | 0.00% |
| Fractional amounts allowed below 500, receiver ±10m, >=10 | 153 | 0 | 0.00% |
| Any amount PAY, receiver ±10m, >=10 | 8,328 | 18 | 0.22% |
| Small whole-rupee merchant PAY with no burst | 559 | 2 | 0.36% |
| Same no-burst pool, gaming/ecommerce only | 133 | 0 | 0.00% |

This is why matching 529 test rows against the **old** September target would be misleading. A same-definition, same-denominator comparison and inspection of the added training rows are both necessary. Broad merchant traffic can also appear more rate-stable while flagging tens of thousands of unrelated rows.

**Event frequency and uncertainty.** The observed event-rate ratio is **0.548** (seven versus 25, normalized by row exposure); October's mean event size is **0.901** of September's. Their product is the observed burst-row ratio **0.494**. Equal event frequency per row would predict **12.77 October events**, rather than seven. These are small event counts: under an independent Poisson-events-per-row model, an exact conditional binomial test gives a two-sided **p = 0.191** and a **95% event-rate-ratio interval of 0.200–1.303**. This model is a sensitivity check, not a claim about the dataset generator. Treating 1,335 burst rows as independent observations would overstate the evidence for drift. Neither unchanged prevalence nor a decrease in total bot fraud is established without test labels.

**Feature recommendation.** Retain the label-free `Q` flag and `C(Q,device,w)` at radii 600, 3,600, 21,600, and 86,400 seconds, along with `C(Q,receiver,w)` and the corresponding distinct-device/distinct-sender counts. The longer windows and `MD_w` are useful diagnostics, but this investigation gives **no evidence for promoting their additional rows to bots**. Session size, span, distinct senders, and distinct receivers describe the known recipe; do not require at least 25 payments or many senders, since valid training variants have 20 payments or three senders. The exact existing v1 feature `dev_intlt500_cnt_10m` counts all small integer transactions regardless of receiver/type; `C(Q,device,600)` deliberately counts only merchant PAY rows. Keep that distinction explicit when implementing.

Do not apply an October-specific rate-restoring threshold based on this analysis. A radically different pattern that changes several properties simultaneously remains possible, but the tested evidence does not identify one. The earlier statement that the bot drop itself demonstrates a shape change should be treated as a hypothesis, not a measured result. No feature implementation or prediction change is included here.

**Reproduction.** The following standalone code recreates the core counts, the full 200-definition rolling grid, event summaries, and training-label checks. Save it outside the repository and run it from the repository root with `python -I path/to/check_bot_drift.py`. On this machine the executable was `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe`. It reads data and prints results; it writes no repository files. Use pandas 2.3.3, NumPy, PyArrow, and SciPy.

```python
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

path = "data/processed/base_v0.parquet"
cols = ["txn_id", "t", "timestamp", "split", "month", "device_id",
        "sender_id", "receiver_id", "amount", "receiver_type", "txn_type",
        "merchant_category", "login_to_txn_sec"]
d = pd.read_parquet(path, columns=cols)  # no label or trap columns
assert d.txn_id.is_unique
sep = d.split.eq("train") & d.month.eq("2026-09")
october = d.split.eq("test")
assert (int(sep.sum()), int(october.sum())) == (509996, 260582)
M = d.receiver_type.eq("MERCHANT").to_numpy()
P = M & d.txn_type.eq("PAY").to_numpy()
I = d.amount.eq(np.floor(d.amount)).to_numpy()
Q = P & I & d.amount.lt(500).to_numpy()
pools = {"Q": Q, "J": P & I,
         "F500": P & d.amount.lt(500).to_numpy(),
         "F1000": P & d.amount.lt(1000).to_numpy(), "P": P}
t = d.t.to_numpy(dtype=np.int64)
SCALE = np.int64(2**26)
assert t.min() >= 0 and t.max() + 86400 < SCALE
assert d[["device_id", "receiver_id", "sender_id", "t"]].notna().all().all()
packed = {key: pd.factorize(d[key])[0].astype(np.int64) * SCALE + t
          for key in ("device_id", "receiver_id")}

def count(pool, key, radius):
    query = packed[key]
    events = np.sort(query[pool])
    return (np.searchsorted(events, query + radius, side="right")
            - np.searchsorted(events, query - radius, side="left"))

def flag(pool, key, radius, threshold):
    return pool & (count(pool, key, radius) >= threshold)

B = flag(Q, "device_id", 600, 3)
assert (int(B[sep].sum()), int(B[october].sum())) == (1066, 269)
assert np.array_equal(B, flag(Q, "device_id", 86400, 3))
assert np.array_equal(B, flag(Q, "receiver_id", 600, 3))

# This is the only label read: predicate-filtered training rows.
labels = pd.read_parquet(path, columns=["txn_id", "label"],
                         filters=[("split", "==", "train")])
labels = labels.set_index("txn_id").label
ysep = d.loc[sep, "txn_id"].map(labels)
assert ysep.notna().all()

def summary(name, mask):
    ns, nt = int(mask[sep].sum()), int(mask[october].sum())
    rs, rt = ns / sep.sum() * 1e5, nt / october.sum() * 1e5
    ys = ysep.loc[d.index[sep][mask[sep]]]
    added = mask & ~B
    ya = ysep.loc[d.index[sep][added[sep]]]
    return {"definition": name, "sep_n": ns, "test_n": nt,
            "sep_per100k": rs, "test_per100k": rt,
            "rate_ratio": rt / rs if rs else np.nan,
            "sep_bot": int(ys.eq("velocity_bot").sum()),
            "sep_extra": int(added[sep].sum()),
            "sep_extra_bot": int(ya.eq("velocity_bot").sum()),
            "test_extra": int(added[october].sum())}

results = []
for name, pool in pools.items():
    for key in packed:
        for radius in (600, 3600, 21600, 86400):
            c = count(pool, key, radius)
            for threshold in (3, 5, 10, 20, 25):
                results.append(summary(f"{name}/{key}/{radius}/{threshold}",
                                       pool & (c >= threshold)))
grid = pd.DataFrame(results)
print(grid.to_string(index=False))
print("Largest rate ratio:", grid.loc[grid.rate_ratio.idxmax()].to_dict())
print("No burst:", summary("Q", Q))

x = d.loc[Q].sort_values(["device_id", "t"], kind="stable").copy()
x["sid"] = (x.device_id.ne(x.device_id.shift()) | x.t.diff().gt(600)).cumsum()
s = x.groupby("sid").agg(n=("t", "size"), start=("t", "min"),
    end=("t", "max"), split=("split", "first"), month=("month", "first"),
    senders=("sender_id", "nunique"), merchants=("receiver_id", "nunique"))
s["span"] = s.end - s.start
members = np.zeros(len(d), dtype=bool)
members[x.index] = x.sid.isin(s.index[s.n >= 3])
assert np.array_equal(members, B)
s = s[s.n >= 3]
print(s.groupby(["split", "month"]).agg(events=("n", "size"), rows=("n", "sum")))
for name, m in [("Sep", s.split.eq("train") & s.month.eq("2026-09")),
                ("Test", s.split.eq("test"))]:
    print(name, s.loc[m, ["n", "span", "senders", "merchants"]]
          .quantile([0, .1, .5, .9, 1]))

# No entity key: checks simultaneous payments spread across all IDs.
qt = np.sort(t[Q])
global_count = (np.searchsorted(qt, t + 600, side="right")
                - np.searchsorted(qt, t - 600, side="left"))
print(summary("Q/global/600/20", Q & (global_count >= 20)))

# Exact distinct-key counts for the small Q pool; timestamps are sorted.
for radius in (600, 3600, 21600, 86400):
    md = np.zeros(len(d), dtype=bool)
    for _, g in d.loc[Q].sort_values("t").groupby("receiver_id"):
        tt = g.t.to_numpy()
        devices, senders = g.device_id.to_numpy(), g.sender_id.to_numpy()
        for pos, idx in enumerate(g.index):
            lo = np.searchsorted(tt, tt[pos] - radius, side="left")
            hi = np.searchsorted(tt, tt[pos] + radius, side="right")
            md[idx] = (hi - lo >= 3 and len(set(devices[lo:hi])) >= 2
                       and len(set(senders[lo:hi])) >= 3)
    print(summary(f"MD/{radius}", md))

exposure_ratio = october.sum() / sep.sum()
test = binomtest(7, 32, october.sum() / (sep.sum() + october.sum()))
ci = test.proportion_ci(confidence_level=.95, method="exact")
to_ratio = lambda p: p / (1 - p) / exposure_ratio
print("Event-rate p and CI:", test.pvalue, to_ratio(ci.low), to_ratio(ci.high))
with open(path, "rb") as f:
    print("SHA256:", hashlib.file_digest(f, "sha256").hexdigest())
```
