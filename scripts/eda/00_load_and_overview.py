"""Convert raw CSVs to parquet (data/interim) and print a structural overview.

Usage: python -I scripts/eda/00_load_and_overview.py <project_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"
INTERIM.mkdir(parents=True, exist_ok=True)

DT = {"merchant_category": "string", "qr_vpa_mismatch": "float32", "app_version": "string"}
frames = {}
for name in ["train", "test", "accounts", "merchants", "sample_submission"]:
    df = pd.read_csv(RAW / f"{name}.csv", dtype=DT if name in ("train", "test") else None)
    if name in ("train", "test"):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    frames[name] = df
    df.to_parquet(INTERIM / f"{name}.parquet", index=False)
    print(f"\n===== {name}: {df.shape[0]:,} rows x {df.shape[1]} cols, "
          f"{df.memory_usage(deep=True).sum()/2**20:,.0f} MiB in pandas")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(3),
        "n_unique": df.nunique(),
    })
    print(info.to_string())

tr, te, ac, me, ss = (frames[k] for k in ["train", "test", "accounts", "merchants", "sample_submission"])

print("\n===== integrity")
print("dup txn_id train:", tr.txn_id.duplicated().sum(), " test:", te.txn_id.duplicated().sum(),
      " overlap:", len(set(tr.txn_id) & set(te.txn_id)))
print("sample_submission ids == test ids (as sets):", set(ss.id) == set(te.txn_id),
      " same order:", (ss.id.values == te.txn_id.values).all())
print("sample_submission label values:", ss.label.value_counts().to_dict())
acc_ids, mer_ids = set(ac.account_id), set(me.merchant_id)
for nm, d in [("train", tr), ("test", te)]:
    print(f"[{nm}] sender in accounts: {d.sender_id.isin(acc_ids).mean():.5f}")
    p2p = d.receiver_type == "P2P"
    print(f"[{nm}] P2P receiver in accounts: {d.receiver_id[p2p].isin(acc_ids).mean():.5f}; "
          f"MERCHANT receiver in merchants: {d.receiver_id[~p2p].isin(mer_ids).mean():.5f}")
    print(f"[{nm}] receiver prefix vs type:\n", pd.crosstab(d.receiver_id.str[:2], d.receiver_type))
    print(f"[{nm}] time range: {d.timestamp.min()} -> {d.timestamp.max()}")
    print(f"[{nm}] device_id == 'DV0'+sender digits: "
          f"{(d.device_id == 'DV0' + d.sender_id.str[2:]).mean():.4f}")
    print(f"[{nm}] device_id prefix counts:", d.device_id.str[:3].value_counts().head(5).to_dict())

print("\n===== entity overlap train/test")
for col in ["sender_id", "receiver_id", "device_id"]:
    a, b = set(tr[col]), set(te[col])
    print(f"{col}: train {len(a):,}  test {len(b):,}  test seen in train {len(a & b)/len(b):.3f}")

print("\n===== accounts / merchants")
print(ac.describe(include="all").T.to_string())
print(me.describe(include="all").T.to_string())
print("accounts opened_on range:", ac.opened_on.min(), ac.opened_on.max())
print("merchants onboarded range:", me.onboarded_on.min(), me.onboarded_on.max())
