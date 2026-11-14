"""Build the two base tables every later step reads (rows sorted by time, train and test together):

  data/processed/base_v0.parquet    raw columns + row features + the 33 relational features
  data/processed/ratios_v0.parquet  ratios to the previous / next row of the same sender, receiver and device (within 2 h),
                                    and to the money the receiver sends on / the sender received just before

Needs data/interim/*.parquet (scripts/eda/00_load_and_overview.py) and data/processed/relational_v0.parquet
(scripts/eda/04_relational.py). Usage: python -I src/upi/build_base.py <project_root>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "scripts/eda"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import load, add_row_features  # noqa: E402

P = ROOT / "data/processed"
df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
R0 = pd.read_parquet(P / "relational_v0.parquet")
assert (R0.txn_id.values == df.txn_id.values).all()
df = pd.concat([df, R0.drop(columns=["txn_id", "split", "label", "month"])], axis=1)
df.to_parquet(P / "base_v0.parquet", index=False)
print("base_v0:", df.shape)

R = pd.DataFrame(index=df.index)
R["bal/amt"] = df.balance_before / df.amount


def seq(key, name, window=7200):
    o = df.sort_values([key, "t"], kind="stable")
    same_p = o[key].eq(o[key].shift())
    same_n = o[key].eq(o[key].shift(-1))
    dtp = (o.t - o.t.shift()).where(same_p)
    dtn = (o.t.shift(-1) - o.t).where(same_n)
    okp = same_p & (dtp <= window)
    okn = same_n & (dtn <= window)
    R[f"{name}:bal/prevbal"] = (o.balance_before / o.balance_before.shift()).where(okp).reindex(df.index)
    R[f"{name}:bal/nextbal"] = (o.balance_before / o.balance_before.shift(-1)).where(okn).reindex(df.index)
    R[f"{name}:amt/prevamt"] = (o.amount / o.amount.shift()).where(okp).reindex(df.index)
    R[f"{name}:amt/nextamt"] = (o.amount / o.amount.shift(-1)).where(okn).reindex(df.index)
    R[f"{name}:bal/prevamt"] = (o.balance_before / o.amount.shift()).where(okp).reindex(df.index)
    R[f"{name}:amt/prevbal"] = (o.amount / o.balance_before.shift()).where(okp).reindex(df.index)
    R[f"{name}:dt_prev"] = dtp.where(okp).reindex(df.index)


seq("sender_id", "snd")
seq("receiver_id", "rcv")
seq("device_id", "dev")
# money moving on: the next payment my receiver makes within 2 h, and the last payment my sender received within 2 h
out = df[["sender_id", "t", "amount", "balance_before"]].sort_values("t").rename(
    columns={"sender_id": "k", "amount": "o_amt", "balance_before": "o_bal", "t": "o_t"})
q = df[["receiver_id", "t", "amount"]].rename(columns={"receiver_id": "k"}).sort_values("t")
q["row"] = q.index
m1 = pd.merge_asof(q, out, left_on="t", right_on="o_t", by="k", direction="forward", allow_exact_matches=False, tolerance=7200).set_index("row")
R["fwd:o_amt/amt"] = (m1.o_amt / m1.amount).reindex(df.index)
R["fwd:o_bal/amt"] = (m1.o_bal / m1.amount).reindex(df.index)
inn = df[["receiver_id", "t", "amount", "balance_before"]].sort_values("t").rename(
    columns={"receiver_id": "k", "amount": "i_amt", "balance_before": "i_bal", "t": "i_t"})
q2 = df[["sender_id", "t", "amount", "balance_before"]].rename(columns={"sender_id": "k"}).sort_values("t")
q2["row"] = q2.index
m2 = pd.merge_asof(q2, inn, left_on="t", right_on="i_t", by="k", direction="backward", allow_exact_matches=False, tolerance=7200).set_index("row")
R["back:amt/i_amt"] = (m2.amount / m2.i_amt).reindex(df.index)
R["back:bal/i_amt"] = (m2.balance_before / m2.i_amt).reindex(df.index)
R = R.astype("float32")
R.insert(0, "txn_id", df.txn_id.values)                  # features_v1 checks this against the base table before using the ratios
R.to_parquet(P / "ratios_v0.parquet")
print("ratios_v0:", R.shape)
