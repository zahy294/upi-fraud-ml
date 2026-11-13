"""Feature pack v1: exact-recipe ratios, burst counts and fan-in counts. Label-free, train+test together.

Reads data/processed/base_v0.parquet (row + relational_v0 features, sorted by t) and ratios_v0.parquet.
Usage: python -I src/upi/features_v1.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from events import Events, MIN, HOUR, DAY  # noqa: E402

ROOT = Path(sys.argv[1])
P = ROOT / "data/processed"
df = pd.read_parquet(P / "base_v0.parquet")
R = pd.read_parquet(P / "ratios_v0.parquet")
assert len(R) == len(df)
t = df.t.values.astype(np.int64)
amt = df.amount.values
bal = df.balance_before.values
snd = df.sender_id.str[2:].astype(np.int64).values
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
rcv_acct = np.where(is_p2p, rnum, -1)
dev = df.device_id.str[2:].astype(np.int64).values
pay = (df.txn_type == "PAY").values
newp = (df.new_payee == 1).values
F = pd.DataFrame(index=df.index)

# ---- A. ratios (prev/next in each entity sequence within 2 h) and distance to the exact recipe values ----
for c in R.columns:
    F["r_" + c.replace(":", "_").replace("/", "_")] = R[c].values
ba = bal / amt
F["bal_over_amt"] = ba
F["d_bal_amt_1p1"] = np.abs(ba - 1.1)
F["d_bal_amt_1p02"] = np.abs(ba - 1.02)
for k in ("snd", "dev"):
    pb, nb = R[f"{k}:bal/prevbal"].values, R[f"{k}:bal/nextbal"].values
    dp = np.fmin(np.abs(pb - 0.25), np.abs(pb - 0.35))
    dn = np.fmin(np.abs(nb - 4.0), np.abs(nb - 1 / 0.35))
    F[f"d_{k}_prevbal_q"] = dp
    F[f"d_{k}_nextbal_q"] = dn
    F[f"{k}_drain_link"] = ((dp < 0.0006) | (dn < 0.006)).astype(np.int8)
    F[f"{k}_drain_kind"] = np.where((np.abs(pb - 0.25) < 0.0006) | (np.abs(nb - 4) < 0.006), 1,
                                    np.where((np.abs(pb - 0.35) < 0.0006) | (np.abs(nb - 1 / 0.35) < 0.006), 2, 0)).astype(np.int8)

# ---- B. row flags ----
a_int = np.floor(amt) == amt
F["amt_mult10"] = (a_int & (amt % 10 == 0)).astype(np.int8)
F["amt_mult50"] = (a_int & (amt % 50 == 0)).astype(np.int8)
F["amt_mult100"] = (a_int & (amt % 100 == 0)).astype(np.int8)
F["amt_eq500"] = (amt == 500).astype(np.int8)
F["amt_int_lt500"] = (a_int & (amt < 500)).astype(np.int8)
F["bal_lt_amt"] = (bal < amt).astype(np.int8)
is11 = np.abs(ba - 1.1) < 0.0006
is102 = np.abs(ba - 1.02) < 0.0006
F["is_1p1"] = is11.astype(np.int8)
F["is_1p02"] = is102.astype(np.int8)

# ---- C. device bursts ----
dev_ev = Events(dev, t)
for nm, w in (("2m", 2 * MIN), ("10m", 10 * MIN), ("6h", 6 * HOUR)):
    F[f"dev_cnt_{nm}"] = dev_ev.count(dev, t, -w, w)
small = a_int & (amt < 500)
dev_small = Events(dev, t, mask=small)
F["dev_intlt500_cnt_10m"] = dev_small.count(dev, t, -10 * MIN, 10 * MIN)
F["dev_intlt500_cnt_1d"] = dev_small.count(dev, t, -DAY, DAY)
for nm, w in (("10m", 600), ("1h", 3600)):
    parts = {"snd": [], "rcv": []}
    for off in (0, w // 2):
        g = df.groupby([dev, (t + off) // w])
        parts["snd"].append(g.sender_id.transform("nunique").values)
        parts["rcv"].append(g.receiver_id.transform("nunique").values)
    F[f"dev_uniq_snd_{nm}"] = np.maximum(*parts["snd"])
    F[f"dev_uniq_rcv_{nm}"] = np.maximum(*parts["rcv"])
F["dev_dt_prev"] = R["dev:dt_prev"].values

# ---- D. receiver fan-in (victims paying one account) and what the receiver does afterwards ----
vict = pay & is_p2p & newp & a_int & (amt % 10 == 0)             # shape of a payment into a mule
in_all = Events(rcv, t, amt)
in_v = Events(rcv, t, amt, mask=vict)
for nm, w in (("6h", 6 * HOUR), ("24h", DAY), ("72h", 3 * DAY)):
    F[f"rcv_vict_cnt_{nm}"] = in_v.count(rcv, t, -w, w)
    F[f"rcv_vict_sum_{nm}"] = in_v.sum(rcv, t, -w, w)
F["rcv_in_cnt_6h"] = in_all.count(rcv, t, -6 * HOUR, 6 * HOUR)
F["rcv_vict_share_24h"] = F.rcv_vict_cnt_24h / np.maximum(in_all.count(rcv, t, -DAY, DAY), 1)
out_all = Events(snd, t, amt)
out_102 = Events(snd, t, amt, mask=is102)
F["rcv_fwd102_cnt_48h"] = out_102.count(rcv_acct, t, 1, 2 * DAY)       # receiver later sends ~98% of its balance
F["rcv_out_sum_48h"] = out_all.sum(rcv_acct, t, 1, 2 * DAY)
F["rcv_out_cnt_48h"] = out_all.count(rcv_acct, t, 1, 2 * DAY)
# onward transfer made by the mule itself: the sender collected victim-shaped payments before
F["snd_vict_in_cnt_48h"] = in_v.count(snd, t, -2 * DAY, -1)
F["snd_vict_in_sum_48h"] = in_v.sum(snd, t, -2 * DAY, -1)
snd_in_sum = in_all.sum(snd, t, -2 * DAY, -1)
F["snd_in_sum_48h"] = snd_in_sum
F["amt_over_snd_in_48h"] = amt / np.where(snd_in_sum > 0, snd_in_sum, np.nan)
F["snd_102_cnt_2d"] = out_102.count(snd, t, -2 * DAY, 2 * DAY)

# ---- E. sender drain bursts ----
drain_shape = pay & is_p2p & newp & a_int
out_d = Events(snd, t, amt, mask=drain_shape)
for nm, w in (("1h", HOUR), ("3h", 3 * HOUR)):
    F[f"snd_drainshape_cnt_{nm}"] = out_d.count(snd, t, -w, w)
    F[f"snd_out_cnt_{nm}"] = out_all.count(snd, t, -w, w)
dl = F.snd_drain_link.values == 1
F["snd_drainlink_cnt_3h"] = Events(snd, t, mask=dl).count(snd, t, -3 * HOUR, 3 * HOUR)
F["snd_eq500_cnt_3h"] = Events(snd, t, mask=amt == 500).count(snd, t, -3 * HOUR, 3 * HOUR)
F["snd_dt_prev"] = R["snd:dt_prev"].values

# ---- F. baits and paybacks (phishing / cashback) ----
pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
rev = np.where(is_p2p, rcv_acct * 2_000_000 + snd, -1)
rcode = pd.Index(puniq).get_indexer(rev)
tiny = amt <= 10
pair_ev = Events(pcode, t)
F["rev_tiny_before_7d"] = Events(pcode, t, mask=tiny).count(rcode, t, -7 * DAY, -1)
F["rev_11_after_7d"] = Events(pcode, t, mask=is11).count(rcode, t, 1, 7 * DAY)     # I sent a bait; the victim pays back in the 1.1 shape
F["rev_any_after_7d"] = pair_ev.count(rcode, t, 1, 7 * DAY)
F["rev_any_before_7d"] = pair_ev.count(rcode, t, -7 * DAY, -1)
F["snd_11_cnt_7d"] = Events(snd, t, mask=is11).count(snd, t, -7 * DAY, 7 * DAY)
rcv_11 = Events(rcv, t, mask=is11)
F["rcv_11_cnt_7d"] = rcv_11.count(rcv, t, -7 * DAY, 7 * DAY)
F["rcv_11_cnt_30d"] = rcv_11.count(rcv, t, -30 * DAY, 30 * DAY)
F["pair_11_cnt_30d"] = Events(pcode, t, mask=is11).count(pcode, t, -30 * DAY, 30 * DAY)
F["snd_tiny_out_cnt_1d"] = Events(snd, t, mask=tiny & is_p2p).count(snd, t, -DAY, DAY)

# ---- G. merchant / receiver context ----
rcv_cnt_7d = in_all.count(rcv, t, -7 * DAY, 7 * DAY)
F["rcv_cnt_7d"] = rcv_cnt_7d
F["rcv_sum_7d"] = in_all.sum(rcv, t, -7 * DAY, 7 * DAY)
F["pair_cnt_30d"] = pair_ev.count(pcode, t, -30 * DAY, 30 * DAY)
F["pair_share_rcv_7d"] = pair_ev.count(pcode, t, -7 * DAY, 7 * DAY) / np.maximum(rcv_cnt_7d, 1)
r500 = a_int & (amt % 500 == 0)
F["rcv_r500_share_7d"] = Events(rcv, t, mask=r500).count(rcv, t, -7 * DAY, 7 * DAY) / np.maximum(rcv_cnt_7d, 1)
F["snd_r500_cnt_7d"] = Events(snd, t, mask=r500).count(snd, t, -7 * DAY, 7 * DAY)
F["pair_r500_cnt_30d"] = Events(pcode, t, mask=r500).count(pcode, t, -30 * DAY, 30 * DAY)
is_ref = (df.txn_type == "REFUND").values
F["pair_refund_cnt_30d"] = Events(pcode, t, mask=is_ref).count(pcode, t, -30 * DAY, 30 * DAY)
F["snd_refund_cnt_7d"] = Events(snd, t, mask=is_ref).count(snd, t, -7 * DAY, 7 * DAY)
F["rcv_refund_cnt_7d"] = Events(rcv, t, mask=is_ref).count(rcv, t, -7 * DAY, 7 * DAY)
lend = (df.merchant_category.astype("object") == "lending").values
F["snd_lend_cnt_30d"] = Events(snd, t, mask=lend).count(snd, t, -30 * DAY, 30 * DAY)
F["pair_evening_cnt_30d"] = Events(pcode, t, mask=(df.is_late_evening.values == 1)).count(pcode, t, -30 * DAY, 30 * DAY)

F = F.astype("float32")
F.insert(0, "txn_id", df.txn_id.values)
F.to_parquet(P / "features_v1.parquet", index=False)
print("features_v1:", F.shape)
