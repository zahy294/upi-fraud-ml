"""Feature pack v5: where a forwarding chain comes from.

A mule passes its collected money on, and the receiving account can pass it on again; those later hops look like a
layering chain but belong to the mule event. Follow each payment back through "the sender received 90-100% of this
amount within the previous two hours" to the first payment of the chain, and describe that root.

Label-free, train+test together. Usage: python -I src/upi/features_v5.py <project_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
P = ROOT / "data/processed"
df = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "t", "sender_id", "receiver_id", "receiver_type", "amount", "balance_before"])
f1 = pd.read_parquet(P / "features_v1.parquet", columns=["snd_vict_in_cnt_48h", "is_1p02", "rcv_vict_cnt_24h"])
n = len(df)
df["row"] = np.arange(n)
W, LO, HI = 7200, 0.90, 1.0

# parent: the last payment INTO my sender within W seconds before me
q = df[["row", "t", "sender_id", "amount"]].rename(columns={"sender_id": "k"}).sort_values("t")
inn = df[df.receiver_type == "P2P"][["row", "t", "receiver_id", "amount"]].rename(
    columns={"receiver_id": "k", "row": "p_row", "amount": "p_amt", "t": "p_t"}).sort_values("p_t")
m = pd.merge_asof(q, inn, left_on="t", right_on="p_t", by="k", direction="backward", allow_exact_matches=False, tolerance=W).set_index("row")
ratio = (m.amount / m.p_amt).reindex(df.row).values
par = np.where((ratio >= LO) & (ratio <= HI), m.p_row.reindex(df.row).fillna(-1).values, -1).astype(np.int64)

# child: the first payment my receiver makes within W seconds after me
q2 = df[df.receiver_type == "P2P"][["row", "t", "receiver_id", "amount"]].rename(columns={"receiver_id": "k"}).sort_values("t")
out = df[["row", "t", "sender_id", "amount"]].rename(columns={"sender_id": "k", "row": "c_row", "amount": "c_amt", "t": "c_t"}).sort_values("c_t")
m2 = pd.merge_asof(q2, out, left_on="t", right_on="c_t", by="k", direction="forward", allow_exact_matches=False, tolerance=W).set_index("row")
r2 = (m2.c_amt / m2.amount).reindex(df.row).values
chi = np.where((r2 >= LO) & (r2 <= HI), m2.c_row.reindex(df.row).fillna(-1).values, -1).astype(np.int64)


def walk(ptr, hops=8):
    depth = np.zeros(n, np.int16)
    end = np.arange(n)
    cur = ptr.copy()
    for _ in range(hops):
        has = cur >= 0
        depth += has
        end = np.where(has, cur, end)
        cur = np.where(has, ptr[np.maximum(cur, 0)], -1)
    return depth, end


d_back, root = walk(par)
d_fwd, leaf = walk(chi)
amt = df.amount.values
F = pd.DataFrame({"txn_id": df.txn_id.values})
F["ch_depth_back"] = d_back
F["ch_depth_fwd"] = d_fwd
F["ch_len"] = d_back + d_fwd + 1
F["ch_root_snd_vict_in"] = f1.snd_vict_in_cnt_48h.values[root]          # the first sender of the chain had collected victim payments
F["ch_root_is_102"] = f1.is_1p02.values[root]                           # the chain starts with a mule-style 98% payout
F["ch_root_amt_ratio"] = amt / amt[root]
F["ch_leaf_amt_ratio"] = amt[leaf] / amt
F["ch_root_rcv_vict"] = f1.rcv_vict_cnt_24h.values[root]
F["ch_is_root"] = ((d_back == 0) & (d_fwd > 0)).astype(np.int8)
# a device band on any row of the chain tells which generator made the whole chain (needs bands.py to have run)
band = pd.read_parquet(P / "features_v3.parquet", columns=["band"]).band.values.astype(int)
in_chain = (d_back + d_fwd) > 0
cb = pd.DataFrame({"root": root[in_chain], "band": band[in_chain]})
cb = cb[cb.band > 0].groupby("root").band.agg(["min", "max"])
one = cb[cb["min"] == cb["max"]]["max"]
F["ch_band"] = np.where(in_chain, pd.Series(root).map(one).fillna(0).values, 0)
num = [c for c in F.columns if c != "txn_id"]
F[num] = F[num].astype("float32")
F.to_parquet(P / "features_v5.parquet", index=False)
print("features_v5:", F.shape, "| rows in a chain of length >= 2:", int((F.ch_len >= 2).sum()), "| with a mule-style root:", int(((F.ch_len >= 2) & ((F.ch_root_is_102 == 1) | (F.ch_root_snd_vict_in > 0))).sum()))
