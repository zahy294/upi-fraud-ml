"""Find exact-ratio 'recipes': values of simple cross-row ratios that are common among fraud rows and rare among legit rows."""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
df = pd.read_parquet("data/processed/base_v0.parquet", columns=["txn_id", "t", "sender_id", "receiver_id", "device_id", "receiver_type", "txn_type",
                     "amount", "balance_before", "label", "split", "app_major", "ip_risk_score", "new_payee", "login_to_txn_sec", "sim_age_hours", "device_age_days"])
R = pd.DataFrame(index=df.index)
R["bal/amt"] = df.balance_before / df.amount
def seq(key, name):
    o = df.sort_values([key, "t"], kind="stable")
    same_p = o[key].eq(o[key].shift()); same_n = o[key].eq(o[key].shift(-1))
    dtp = (o.t - o.t.shift()).where(same_p); dtn = (o.t.shift(-1) - o.t).where(same_n)
    okp = same_p & (dtp <= 7200); okn = same_n & (dtn <= 7200)
    R[f"{name}:bal/prevbal"] = (o.balance_before / o.balance_before.shift()).where(okp).reindex(df.index)
    R[f"{name}:bal/nextbal"] = (o.balance_before / o.balance_before.shift(-1)).where(okn).reindex(df.index)
    R[f"{name}:amt/prevamt"] = (o.amount / o.amount.shift()).where(okp).reindex(df.index)
    R[f"{name}:amt/nextamt"] = (o.amount / o.amount.shift(-1)).where(okn).reindex(df.index)
    R[f"{name}:bal/prevamt"] = (o.balance_before / o.amount.shift()).where(okp).reindex(df.index)
    R[f"{name}:amt/prevbal"] = (o.amount / o.balance_before.shift()).where(okp).reindex(df.index)
    R[f"{name}:dt_prev"] = dtp.where(okp).reindex(df.index)
seq("sender_id", "snd"); seq("receiver_id", "rcv"); seq("device_id", "dev")
# receiver as later sender (money moving on): next outgoing txn of my receiver within 2h
out = df[["sender_id", "t", "amount", "balance_before"]].sort_values("t").rename(columns={"sender_id": "k", "amount": "o_amt", "balance_before": "o_bal", "t": "o_t"})
q = df[["receiver_id", "t", "amount"]].rename(columns={"receiver_id": "k"}).sort_values("t")
q["row"] = q.index
mrg = pd.merge_asof(q, out, left_on="t", right_on="o_t", by="k", direction="forward", allow_exact_matches=False, tolerance=7200)
mrg = mrg.set_index("row")
R["fwd:o_amt/amt"] = (mrg.o_amt / mrg.amount).reindex(df.index)
R["fwd:o_bal/amt"] = (mrg.o_bal / mrg.amount).reindex(df.index)
inn = df[["receiver_id", "t", "amount", "balance_before"]].sort_values("t").rename(columns={"receiver_id": "k", "amount": "i_amt", "balance_before": "i_bal", "t": "i_t"})
q2 = df[["sender_id", "t", "amount", "balance_before"]].rename(columns={"sender_id": "k"}).sort_values("t"); q2["row"] = q2.index
m2 = pd.merge_asof(q2, inn, left_on="t", right_on="i_t", by="k", direction="backward", allow_exact_matches=False, tolerance=7200).set_index("row")
R["back:amt/i_amt"] = (m2.amount / m2.i_amt).reindex(df.index)
R["back:bal/i_amt"] = (m2.balance_before / m2.i_amt).reindex(df.index)
tr = (df.split == "train").values
fraud = tr & (df.label != "legit").values
pat = fraud & (df.app_major == "3").values & (df.ip_risk_score > 0.4).values
legit = tr & (df.label == "legit").values
print("patterned fraud rows:", pat.sum(), "legit:", legit.sum())
for c in R.columns:
    if c.endswith("dt_prev"):
        continue
    v = R[c].round(3)
    a = v[pat].value_counts(); b = v[legit].value_counts(); f = v[fraud].value_counts()
    t = pd.DataFrame({"pat_n": a, "fraud_n": f, "legit_n": b}).fillna(0)
    t["pat_share"] = t.pat_n / pat.sum(); t["legit_share"] = t.legit_n / legit.sum()
    t["lift"] = (t.pat_share + 1e-9) / (t.legit_share + 2e-6)
    t = t[(t.pat_n >= 25) & (t.lift >= 15)].sort_values("pat_n", ascending=False).head(8)
    cov = R[c].notna()[pat].mean()
    if len(t):
        print(f"\n== {c}  (non-null among patterned: {cov:.2f})")
        print(t[["pat_n", "fraud_n", "legit_n", "lift"]].round(1).to_string())
R.astype("float32").to_parquet("data/processed/ratios_v0.parquet")
