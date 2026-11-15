"""Feature pack v2: event sessions for the fraud recipes (fixed-ratio drains, paybacks, forwards) and the bot bursts.

Label-free, train+test together. Usage: python -I src/upi/features_v2.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from events import Events, MIN, HOUR, DAY  # noqa: E402

ROOT = Path(sys.argv[1])
P = ROOT / "data/processed"
df = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "t", "sender_id", "receiver_id", "device_id", "receiver_type", "txn_type",
                                                      "merchant_category", "amount", "balance_before", "new_payee", "hour"])
t = df.t.values.astype(np.int64)
amt = df.amount.values
bal = df.balance_before.values
snd = df.sender_id.str[2:].astype(np.int64).values
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
dev = df.device_id.str[2:].astype(np.int64).values
cat = df.merchant_category.astype("object").fillna("NA").values
typ = df.txn_type.values
a_int = np.floor(amt) == amt
F = pd.DataFrame(index=df.index)


def sessions(mask, key, gap, prefix, extra_nunique=()):
    """Group rows in `mask` into sessions per `key`: a new session starts when the gap to the previous masked row exceeds `gap`."""
    x = df.loc[mask, ["t", "sender_id", "receiver_id", "device_id"]].copy()
    x["k"] = key[mask]
    x = x.sort_values(["k", "t"], kind="stable")
    x["sid"] = ((x.k != x.k.shift()) | (x.t.diff() > gap)).cumsum()
    g = x.groupby("sid")
    out = {"n": g.t.transform("size"), "span": g.t.transform("max") - g.t.transform("min"),
           "pos": g.cumcount() / np.maximum(g.t.transform("size") - 1, 1)}
    for col in extra_nunique:
        out["uniq_" + col[:3]] = g[col].transform("nunique")
    for nm, s in out.items():
        v = np.zeros(len(df), dtype="float32")
        v[x.index.values] = s.values
        F[f"{prefix}_{nm}"] = v
    return x


# ---- S: fixed-amount merchant payments (qr_swap small / refund_abuse medium / merchant_collusion large sessions) ----
S = (~is_p2p) & np.isin(typ, ["PAY", "QR"]) & np.isin(amt, [1000, 2000, 2500, 5000, 10000])
F["S_flag"] = S.astype("float32")
xs = sessions(S, rcv, DAY, "S", extra_nunique=("sender_id", "device_id"))
top = xs.groupby(["sid", "sender_id"]).t.transform("size")
v = np.zeros(len(df), dtype="float32")
v[xs.index.values] = top.values
F["S_sender_rows"] = v                                             # rows by this sender inside the session
F["S_floor_resid"] = np.where(S, bal - 1.1 * amt, np.nan)

# ---- L: late-evening new-payee payments (loan_app_extortion repeats to changing payees) ----
L = (typ == "PAY") & a_int & (amt >= 1500) & (amt < 15000) & (df.new_payee.values == 1) & (df.hour.values >= 20) & ((cat == "lending") | is_p2p)
F["L_flag"] = L.astype("float32")
Lev = Events(snd, t, mask=L)
F["L_snd_cnt_7d"] = Lev.count(snd, t, -7 * DAY, 7 * DAY)
F["L_snd_cnt_3d"] = Lev.count(snd, t, -3 * DAY, 3 * DAY)
Llend = Events(snd, t, mask=L & (cat == "lending"))
F["L_snd_lend_cnt_7d"] = Llend.count(snd, t, -7 * DAY, 7 * DAY)

# ---- native QR swap: a QR payment to a person whose row carries a shop category ----
F["qr_native"] = ((typ == "QR") & is_p2p & np.isin(cat, ["grocery", "food", "fuel", "pharmacy"])).astype("float32")
F["p2p_has_category"] = (is_p2p & (cat != "NA")).astype("float32")

# ---- native refund abuse: repeated refunds by one sender ----
Rm = (typ == "REFUND") & (~is_p2p) & np.isin(cat, ["ecommerce", "electronics"])
F["R_flag"] = Rm.astype("float32")
Rev = Events(snd, t, mask=Rm)
F["R_snd_cnt_14d"] = Rev.count(snd, t, -14 * DAY, 14 * DAY)
F["R_snd_cnt_7d"] = Rev.count(snd, t, -7 * DAY, 7 * DAY)

# ---- Q: small whole-rupee merchant PAY (velocity_bot bursts on one device) ----
Q = (~is_p2p) & (typ == "PAY") & a_int & (amt < 500)
F["Q_flag"] = Q.astype("float32")
Qdev = Events(dev, t, mask=Q)
Qrcv = Events(rcv, t, mask=Q)
for nm, w in (("10m", 10 * MIN), ("1h", HOUR), ("6h", 6 * HOUR), ("24h", DAY)):
    F[f"Q_dev_cnt_{nm}"] = Qdev.count(dev, t, -w, w)
    F[f"Q_rcv_cnt_{nm}"] = Qrcv.count(rcv, t, -w, w)
sessions(Q, dev, 10 * MIN, "Qs", extra_nunique=("sender_id", "receiver_id"))

# ---- drain bursts (sim_swap / account_takeover): session of whole-rupee PAY to new P2P payees by one sender ----
D = (typ == "PAY") & is_p2p & a_int & (df.new_payee.values == 1)
sessions(D, snd, 2 * HOUR, "Ds", extra_nunique=("receiver_id", "device_id"))

# ---- fan-in sessions (mule_account / layering_chain): victim-shaped payments into one P2P account ----
V = D & (amt % 10 == 0)
sessions(V, rcv, DAY, "Vs", extra_nunique=("sender_id",))

F = F.astype("float32")
F.insert(0, "txn_id", df.txn_id.values)
F.to_parquet(P / "features_v2.parquet", index=False)
print("features_v2:", F.shape)
