"""Feature pack v6: what the same accounts do in OTHER fraud events (cross-event context).

A fraud type's generator emits some events in a sibling type's shape (a loan-app or cashback episode containing a
plain collect request, a takeover in SIM-swap shape, ...). Inside such an event every row looks like the sibling type.
But the type is a property of the episode, and an account's separate events usually share it: for collect / bait
events, two events of the same sender more than a day apart carry the same label 95% of the time (53% by chance).
The earlier packs only look a few days around a row and never at account identity, so they cannot see this.

For every row and every link to other rows
    snd   same sender                         rcv   same receiver
    s2r   my sender is the receiver there     r2s   my receiver is the sender there (accounts only)
    dev   same device
this pack sums, over the OTHER rows that the first model flags as fraud (out-of-fold probability >= 0.5, or a
device band), the first model's class probabilities, split into "same event" (within one day) and "elsewhere"
(more than one day away), and counts the cleaned labels of those rows in the OTHER training months.

Leak safety: the probabilities are out-of-fold (test rows: the fit on all months); labels are only taken from
months other than the row's own, so a row never sees its own label or any label of its validation month.

Usage: python -I src/upi/features_v6.py <project_root> <stage1_tag> [--labels models/oof_raw/y_clean.npy]
Writes data/processed/features_v6.parquet.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS  # noqa: E402
from events import Events, DAY  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--labels", default="models/oof_raw/y_clean.npy")
ap.add_argument("--out", default="features_v6.parquet")
a = ap.parse_args()
ROOT = Path(a.root)
P = ROOT / "data/processed"
df = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "split", "month", "t", "sender_id", "receiver_id", "receiver_type", "device_id"])
n = len(df)
tr = (df.split == "train").values
K = len(LABELS)
prob = np.zeros((n, K))
prob[tr] = np.load(ROOT / "models" / a.tag / "oof.npy")
prob[~tr] = np.load(ROOT / "models" / a.tag / "test_proba.npy")
band = pd.read_parquet(P / "features_v3.parquet", columns=["band"]).band.values.astype(int)
flag = ((1 - prob[:, 0]) >= 0.5) | (band > 0)                       # rows the first model (or a device band) calls fraud
ev = prob.copy()                                                   # evidence a flagged row contributes: its class probabilities,
ev[band > 0] = np.eye(K)[band[band > 0]]                           # or the band type when the device says so
y = np.full(n, -1)
y[tr] = np.load(ROOT / a.labels)
month = df.month.values.copy()
month[tr & (month > "2026-09")] = "2026-09"                        # October tails of September events (same rule as common.load_xy)
mcode = pd.Series(month).map({"2026-07": 0, "2026-08": 1, "2026-09": 2}).fillna(3).astype(int).values   # 3 = test
t = df.t.values.astype(np.int64)

snd = df.sender_id.str[2:].astype(np.int64).values
is_acct = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_acct, rnum, rnum + 1_000_000)                    # merchants in their own number range
rcv_acct = np.where(is_acct, rnum, -1)
dev = df.device_id.str[2:].astype(np.int64).values
# link name -> (key of the query row, key of the rows being looked up, the row itself is among them)
LINKS = {"snd": (snd, snd, True), "rcv": (rcv, rcv, True), "dev": (dev, dev, True), "s2r": (snd, rcv_acct, False), "r2s": (rcv_acct, snd, False)}
BIG = np.int64(2 ** 24)                                            # about 194 days in seconds: wider than the data, inside the key range

F = {"txn_id": df.txn_id.values}
for name, (qkey, pkey, has_self) in LINKS.items():
    pool = flag & (pkey >= 0)
    me = (flag & (qkey == pkey) & (qkey >= 0)).astype(float) if not has_self else flag.astype(float)   # a self-transfer finds itself on the cross links too
    E_cnt = Events(pkey, t, np.ones(n), mask=pool)
    near = E_cnt.sum(qkey, t, -DAY, DAY) - me
    far = E_cnt.sum(qkey, t, -BIG, BIG) - me - near
    F[f"x_{name}_near_n"], F[f"x_{name}_far_n"] = near.astype("float32"), far.astype("float32")
    for k in range(1, K):
        E = Events(pkey, t, ev[:, k], mask=pool)
        own = ev[:, k] * me
        s_near = E.sum(qkey, t, -DAY, DAY) - own
        s_far = E.sum(qkey, t, -BIG, BIG) - own - s_near
        F[f"x_{name}_near_p{k}"] = s_near.astype("float32")
        F[f"x_{name}_far_p{k}"] = s_far.astype("float32")
    # cleaned labels of flagged rows on the same key in the OTHER training months
    lab_pool = pool & tr & (y > 0)
    tab = pd.DataFrame({"key": pkey[lab_pool], "km": pkey[lab_pool] * 4 + mcode[lab_pool], "y": y[lab_pool]})
    q_key, q_km = pd.Series(qkey), pd.Series(qkey * 4 + mcode)
    for k in range(1, K):
        tk = tab[tab.y == k]
        total = q_key.map(tk.groupby("key").size()).fillna(0).values
        own_m = q_km.map(tk.groupby("km").size()).fillna(0).values
        F[f"x_{name}_lab{k}"] = np.where(qkey >= 0, total - own_m, 0).astype("float32")
out = pd.DataFrame(F)                                              # every column is already float32: half the memory of float64
for name in LINKS:                                                 # shares: which type dominates the other events on this link
    far_p = out[[f"x_{name}_far_p{k}" for k in range(1, K)]].values
    lab = out[[f"x_{name}_lab{k}" for k in range(1, K)]].values
    out[f"x_{name}_far_top"] = np.where(far_p.sum(1) > 0, far_p.argmax(1) + 1, 0).astype("float32")
    out[f"x_{name}_far_topshare"] = (far_p.max(1) / np.maximum(far_p.sum(1), 1e-9)).astype("float32")
    out[f"x_{name}_lab_top"] = np.where(lab.sum(1) > 0, lab.argmax(1) + 1, 0).astype("float32")
    out[f"x_{name}_lab_topshare"] = (lab.max(1) / np.maximum(lab.sum(1), 1e-9)).astype("float32")
out.to_parquet(P / a.out, index=False)
print("features_v6:", out.shape, "| flagged rows:", int(flag.sum()),
      "| rows with a flagged row elsewhere on the same sender:", int((out.x_snd_far_n > 0).sum()))
