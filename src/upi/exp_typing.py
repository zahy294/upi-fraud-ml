"""A different style: detect first, then type with a specialist model trained only on detected fraud rows.

The 12-class model spends almost all of its capacity on legit against fraud. Here an existing model (the detector)
keeps deciding whether a row is fraud, and a second, small model that only ever sees detected fraud rows decides
which type. Two modes:
  all      one 11-class model over every detected fraud row
  family   one model per look-alike family (mule/bot/layering, sim_swap/takeover, phishing/cashback/loan,
           qr/refund/collusion); the detector keeps the family, the specialist splits it

Output is a 12-column probability table like run_oof.py writes, so the benchmark and make_submission.py can use it:
  P[legit] = detector's;  P[type] = detector's fraud mass (or family mass) x specialist's type share.

Usage: python -I src/upi/exp_typing.py <project_root> <tag> --detector <tag> --features a.parquet,b.parquet
           [--labels models/x/y_clean.npy] [--mode all|family] [--trees 300] [--leaves 15] [--seeds 3] [--test]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, load_xy  # noqa: E402

FAMILIES = [[1, 2, 3], [4, 5], [6, 9, 10], [7, 8, 11]]
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--detector", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--labels", default="raw")
ap.add_argument("--mode", default="all", choices=["all", "family"])
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--min-child", type=int, default=10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--test", action="store_true")
a = ap.parse_args()
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag
OUT.mkdir(parents=True, exist_ok=True)
K = len(LABELS)

X, y_raw, meta = load_xy(ROOT, tuple(a.features.split(",")))
tr = (meta.split == "train").values
y = y_raw.copy()
if a.labels != "raw":
    y[tr] = np.load(ROOT / a.labels)
month = meta.month.values
n = len(y)
det = np.zeros((n, K))
det[tr] = np.load(ROOT / "models" / a.detector / "oof.npy")
have_test = (ROOT / "models" / a.detector / "test_proba.npy").exists()
if have_test:
    det[~tr] = np.load(ROOT / "models" / a.detector / "test_proba.npy")
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values.astype(int)
detected = (((1 - det[:, 0]) >= 0.5) | (band > 0)) & (tr | have_test)
print(f"detected rows: train {int((detected & tr).sum())}, of which labelled fraud {int((detected & tr & (y > 0)).sum())}; test {int((detected & ~tr).sum())}")


def type_share(fit_mask, pred_mask, classes, seed):
    """P(type | one of `classes`) for the rows in pred_mask, from a model fitted on detected fraud rows of fit_mask."""
    m = fit_mask & detected & np.isin(y, classes)
    present = [c for c in classes if (y[m] == c).any()]
    out = np.zeros((int(pred_mask.sum()), len(classes)))
    if len(present) == 1:
        out[:, classes.index(present[0])] = 1.0
        return out
    yy = np.searchsorted(present, y[m])
    mdl = lgb.LGBMClassifier(objective="multiclass" if len(present) > 2 else "binary", n_estimators=a.trees, learning_rate=a.lr, num_leaves=a.leaves,
                             min_child_samples=a.min_child, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1,
                             n_jobs=a.jobs, random_state=seed)
    mdl.fit(X[m], yy)
    p = mdl.predict_proba(X[pred_mask])
    for j, c in enumerate(present):
        out[:, classes.index(c)] = p[:, j]
    return out


def combine(pred_mask, fit_mask):
    d = det[pred_mask]
    out = np.zeros_like(d)
    out[:, 0] = d[:, 0]
    groups = [list(range(1, K))] if a.mode == "all" else FAMILIES
    for classes in groups:
        share = sum(type_share(fit_mask, pred_mask, classes, s) for s in range(a.seeds)) / a.seeds
        mass = d[:, classes].sum(1, keepdims=True)
        out[:, classes] = mass * share
    return out


oof = np.zeros((int(tr.sum()), K), dtype="float32")
tr_idx = np.where(tr)[0]
for mth in ("2026-07", "2026-08", "2026-09"):
    hold = tr & (month == mth)
    oof[np.searchsorted(tr_idx, np.where(hold)[0])] = combine(hold, tr & ~hold)
    print(f"fold {mth} done", flush=True)
np.save(OUT / "oof.npy", oof)
if a.test:
    assert have_test, "the detector tag has no test probabilities"
    np.save(OUT / "test_proba.npy", combine(~tr, tr).astype("float32"))
    print("test done")
