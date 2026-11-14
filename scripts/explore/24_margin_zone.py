"""Margin zone: rows whose out-of-fold fraud probability is neither near 0 nor near 1.
For each bin: rows, share labelled fraud (raw / cleaned), and the trap-column estimate of how many are really
patterned fraud (app3 share: 0.15 = ordinary, 0.625 = patterned; ip>0.5 share: 0.02 vs 0.40). Measuring tool only."""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "oof_c3"
sys.path.insert(0, str(ROOT / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = ROOT / "data/processed"
base = pd.read_parquet(P / "base_v0.parquet", columns=["label", "split", "app_major", "ip_risk_score"])
tr = (base.split == "train").values
b = base[tr].reset_index(drop=True)
y_raw = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y = np.load(ROOT / "models/oof_raw/y_clean.npy")
oof = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
band = pd.read_parquet(P / "features_v3.parquet", columns=["band"]).band.values[tr].astype(int)
old = (b.app_major == "3").values
ip = (b.ip_risk_score > 0.5).values
pf = 1 - oof[:, 0]
edges = [0, 0.002, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.0001]
rows = []
nb = band == 0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = nb & (pf >= lo) & (pf < hi)
    ml = m & (y_raw == 0)
    rows.append({"pf": f"{lo:.3f}-{hi:.3f}", "rows": int(m.sum()), "raw_fraud": (y_raw[m] > 0).mean(), "clean_fraud": (y[m] > 0).mean(),
                 "app3_all": old[m].mean(), "est_patterned_all": (old[m].mean() - 0.1506) / (0.625 - 0.1506),
                 "raw_legit_rows": int(ml.sum()), "app3_rawlegit": old[ml].mean() if ml.any() else np.nan,
                 "est_patterned_rawlegit": (old[ml].mean() - 0.1506) / (0.625 - 0.1506) if ml.any() else np.nan,
                 "ip_rawlegit": ip[ml].mean() if ml.any() else np.nan,
                 "clean_legit_rows": int((m & (y == 0)).sum())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
# same by the top fraud class, for the wide middle zone
top = oof[:, 1:].argmax(1) + 1
print("\nrows with 0.03 <= pf < 0.85 by top class: rows, raw-legit rows, est. patterned share among raw-legit")
rows = []
for k in range(1, 12):
    m = nb & (pf >= 0.03) & (pf < 0.85) & (top == k)
    ml = m & (y_raw == 0)
    rows.append({"type": LABELS[k], "rows": int(m.sum()), "raw_legit": int(ml.sum()), "raw_own": int((m & (y_raw == k)).sum()),
                 "est_patterned_rawlegit": (old[ml].mean() - 0.1506) / (0.625 - 0.1506) if ml.any() else np.nan})
print(pd.DataFrame(rows).round(3).to_string(index=False))
