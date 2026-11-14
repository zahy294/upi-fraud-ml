"""Leave-one-month-out predictions for every train row, and (optionally) test predictions from a fit on all months.

Usage: python -I src/upi/run_oof.py <project_root> <tag> --features a.parquet,b.parquet [--labels models/x/y_clean.npy]
           [--trees 150] [--seeds 1] [--test]
Writes models/<tag>/oof.npy (train rows in base order) and, with --test, models/<tag>/test_proba.npy.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from common import LABELS, load_xy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--features", required=True)
ap.add_argument("--labels", default="raw")
ap.add_argument("--trees", type=int, default=150)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--legit-frac", type=float, default=0.25)
ap.add_argument("--test", action="store_true")
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--model", default="lgb", choices=["lgb", "xgb", "cat"])
ap.add_argument("--device", default="cpu", help="xgboost and catboost: cpu or cuda")
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--min-child", type=int, default=20)
ap.add_argument("--max-bin", type=int, default=255)
ap.add_argument("--depth", type=int, default=8, help="xgboost and catboost")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag
OUT.mkdir(parents=True, exist_ok=True)

X, y_raw, meta = load_xy(ROOT, tuple(a.features.split(",")))
tr = (meta.split == "train").values
y = y_raw.copy()
if a.labels != "raw":
    y[tr] = np.load(ROOT / a.labels)                     # training labels after cleaning, train rows in base order
month = meta.month.values
CAT_COLS = [c for c in X.columns if str(X[c].dtype) == "category"]
if a.model == "cat":
    for c in CAT_COLS:                                   # catboost wants plain strings for its categorical columns
        X[c] = X[c].astype(str)


def fit_predict(fit_mask, pred_mask, seed):
    rng = np.random.RandomState(seed)
    keep = fit_mask & ((y != 0) | (rng.rand(len(y)) < a.legit_frac))
    if a.model == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(loss_function="MultiClass", iterations=a.trees, learning_rate=a.lr, depth=a.depth, l2_leaf_reg=5.0,
                               bootstrap_type="Bernoulli", subsample=0.8, border_count=max(a.max_bin, 128) if a.device == "cuda" else 254,
                               task_type="GPU" if a.device == "cuda" else "CPU", thread_count=a.jobs, random_seed=seed, verbose=0,
                               cat_features=CAT_COLS, allow_writing_files=False)
    elif a.model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=a.trees, learning_rate=a.lr, max_depth=a.depth, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, max_bin=max(a.max_bin, 256), tree_method="hist", device=a.device,
                              enable_categorical=True, n_jobs=a.jobs, random_state=seed)
    else:
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=a.trees, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.min_child,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, max_bin=a.max_bin, verbose=-1, n_jobs=a.jobs,
                               random_state=seed)
    m.fit(X[keep], y[keep])
    return m.predict_proba(X[pred_mask])


t0 = time.time()
oof = np.zeros((tr.sum(), len(LABELS)), dtype="float32")
tr_idx = np.where(tr)[0]
for mth in ("2026-07", "2026-08", "2026-09"):
    hold = tr & (month == mth)
    p = sum(fit_predict(tr & ~hold, hold, s) for s in range(a.seeds)) / a.seeds
    oof[np.searchsorted(tr_idx, np.where(hold)[0])] = p
    print(f"fold {mth} done ({time.time()-t0:.0f}s)")
np.save(OUT / "oof.npy", oof)
if a.test:
    p = sum(fit_predict(tr, ~tr, s) for s in range(max(a.seeds, 3))) / max(a.seeds, 3)
    np.save(OUT / "test_proba.npy", p.astype("float32"))
    print(f"test done ({time.time()-t0:.0f}s)")
