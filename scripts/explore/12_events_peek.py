"""Print a few raw fraud events (all rows of a sender within the event day) to see how labels sit inside events."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(sys.argv[1]); MODE = sys.argv[2]
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
df = pd.read_parquet(ROOT / "data/processed/base_v0.parquet")
df["old"] = (df.app_major == "3").astype(int)
df["day"] = df.t // 86400
tr = df[df.split == "train"]
cols = ["timestamp", "sender_id", "receiver_id", "device_id", "amount", "txn_type", "new_payee", "sim_age_hours", "device_age_days",
        "home_device", "away_from_home", "pin_retries", "login_to_txn_sec", "balance_before", "old", "ip_risk_score", "label"]
rng = np.random.RandomState(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
if MODE == "simato":
    f = tr[tr.label.isin(["sim_swap", "account_takeover"])]
    g = f.groupby(["sender_id", "day"]).size()
    keys = g[g >= 3].index
    print("sender-days with >=3 sim/ATO rows:", len(keys), "| label purity:",
          f.set_index(["sender_id", "day"]).loc[keys].groupby(level=[0, 1]).label.nunique().value_counts().to_dict())
    for k in [keys[i] for i in rng.choice(len(keys), 7, replace=False)]:
        d = tr[(tr.sender_id == k[0]) & (tr.day.between(k[1] - 1, k[1] + 1))]
        print(d[cols].drop(columns=["sender_id"]).to_string(index=False)); print()
elif MODE == "pair":
    a, b = sys.argv[4], sys.argv[5]
    f = tr[tr.label.isin([a, b])]
    for key in ["sender_id", "receiver_id"]:
        g = f.groupby([key, "day"]).label.agg(["size", "nunique"])
        g = g[g["size"] >= 2]
        print(f"{key}-days with >=2 rows of {a}/{b}: {len(g)}; share with both labels: {(g['nunique'] > 1).mean():.3f}")
    g = f.groupby(["receiver_id", "day"]).size(); keys = g[g >= 2].index
    for k in [keys[i] for i in rng.choice(len(keys), 6, replace=False)]:
        d = tr[((tr.receiver_id == k[0]) | (tr.sender_id == k[0])) & (tr.day.between(k[1], k[1]))]
        print(d[cols].to_string(index=False)); print()
