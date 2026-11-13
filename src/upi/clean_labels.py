"""Clean the training labels the way the test labels are clean: fraud that sits inside a recognisable event
but is labelled legit (unreported) or carries another type (misfiled) gets the event type.

Evidence used: leave-one-month-out probabilities from a model trained on the raw labels, and the device band.
The trap columns are NOT used to decide anything; they are only printed as an independent purity check
(patterned fraud has about 62% old app versions and 40% high ip scores, ordinary rows 15% and 2%).

Usage: python -I src/upi/clean_labels.py <project_root> <oof_tag> [--tau-unrep 0.5]
Writes models/<oof_tag>/y_clean.npy (train rows in base order).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, macro_f1, per_class, fit_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--tau-unrep", type=float, default=0.5)
ap.add_argument("--tau-type", type=float, default=0.6)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag

base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["txn_id", "label", "split", "month", "app_major", "ip_risk_score"])
tr = (base.split == "train").values
b = base[tr]
y = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
old = (b.app_major == "3").values
hi = (b.ip_risk_score > 0.5).values
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values[tr].astype(int)
P = np.load(OUT / "oof.npy").astype(float)
pf = 1 - P[:, 0]
top = P[:, 1:].argmax(1) + 1
ptop = P[np.arange(len(P)), top]
pown = P[np.arange(len(P)), y]


def purity(mask):
    return (old[mask].mean() - 0.1506) / (0.625 - 0.1506) if mask.any() else np.nan


print("legit-labelled rows by out-of-fold fraud probability: n, share old app, share high ip, implied patterned share")
for lo, hi_ in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    m = (y == 0) & (band == 0) & (pf >= lo) & (pf < hi_)
    print(f"  [{lo:.2f},{hi_:.2f})  n={m.sum():6d}  old={old[m].mean():.3f}  hi={hi[m].mean():.3f}  patterned~{purity(m):.2f}")

yc = y.copy()
r_band = (band > 0) & (y != band)
r_unrep = (band == 0) & (y == 0) & (pf >= a.tau_unrep)
r_type = (band == 0) & (y > 0) & (top != y) & (ptop >= a.tau_type) & (pown < 0.15)
yc[r_unrep | r_type] = top[r_unrep | r_type]
yc[r_band] = band[r_band]
print(f"\nrelabelled: band rule {r_band.sum()}, unreported {r_unrep.sum()} (patterned~{purity(r_unrep):.2f}), "
      f"misfiled {r_type.sum()} (patterned~{purity(r_type):.2f})")
ct = pd.DataFrame({"raw": np.bincount(y, minlength=12), "clean": np.bincount(yc, minlength=12)}, index=LABELS)
ct["change_%"] = ((ct.clean / ct.raw - 1) * 100).round(1)
print(ct.to_string())
np.save(OUT / "y_clean.npy", yc)

# how the same predictions score against raw and cleaned labels (weights fitted on each)
for name, target in (("raw labels", y), ("cleaned labels", yc)):
    w = fit_weights(P, target)
    pred = (P * w).argmax(1)
    print(f"\npooled leave-one-month-out macro-F1 vs {name}: {macro_f1(target, pred):.4f}  (and vs the other set: "
          f"{macro_f1(yc if target is y else y, pred):.4f})")
    for mth in ("2026-07", "2026-08", "2026-09"):
        mm = (b.month.values == mth) | ((mth == "2026-09") & (b.month.values > "2026-09"))
        print(f"   {mth}: {macro_f1(target[mm], pred[mm]):.4f}", end="")
    print()
    if target is yc:
        print(per_class(target, pred).round(3).to_string())
        np.save(OUT / "weights_clean.npy", w)
