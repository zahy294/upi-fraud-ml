"""Monthly label mix, the Oct rows in train / Nov rows in test, and app_version by month.

Usage: python -I scripts/eda/01_time_probe.py <project_root>
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
tr = pd.read_parquet(ROOT / "data/interim/train.parquet")
te = pd.read_parquet(ROOT / "data/interim/test.parquet")
pd.set_option("display.width", 250)
for nm, d in [("train", tr), ("test", te)]:
    print(nm, "rows by month:", d.timestamp.dt.to_period("M").value_counts().sort_index().to_dict())
print("\ntrain label x month (counts):")
print(pd.crosstab(tr.label, tr.timestamp.dt.to_period("M"), margins=True))
late = tr[tr.timestamp >= "2026-10-01"]
print("\ntrain rows in Oct:", len(late), "max", late.timestamp.max())
print(late.label.value_counts())
print("\ntest rows in Nov:", (te.timestamp >= "2026-11-01").sum())
print("\napp_version train by month:")
print(pd.crosstab(tr.app_version, tr.timestamp.dt.to_period("M")))
print("\napp_version test:")
print(te.app_version.value_counts())
