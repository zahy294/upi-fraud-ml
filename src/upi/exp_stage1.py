"""Stage-1 experiment: fit on Jul-Aug, score September. Prints macro-F1 with tuned class weights and
trap-oracle diagnostics (how much of the patterned fraud each class finds, how pure the predictions are).

Usage: python -I src/upi/exp_stage1.py <project_root> <tag> [--traps] [--trees N] [--legit-frac F]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, load_xy, macro_f1, per_class, fit_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--traps", action="store_true")
ap.add_argument("--trees", type=int, default=500)
ap.add_argument("--legit-frac", type=float, default=0.25)
ap.add_argument("--features", default="features_v1.parquet")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
X, y, meta = load_xy(ROOT, tuple(f for f in a.features.split(",") if f), with_traps=a.traps)
tr = (meta.split == "train").values
fit = tr & (meta.month < "2026-09").values
val = tr & (meta.month == "2026-09").values
rng = np.random.RandomState(0)
keep = fit & ((y != 0) | (rng.rand(len(y)) < a.legit_frac))
print(f"features={X.shape[1]} fit rows={keep.sum():,} (of {fit.sum():,}) val rows={val.sum():,} load {time.time()-t0:.0f}s")

m = lgb.LGBMClassifier(objective="multiclass", n_estimators=a.trees, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, n_jobs=12, random_state=0)
t0 = time.time()
m.fit(X[keep], y[keep])
print(f"fit {time.time()-t0:.0f}s")
yv = y[val]
best = None
for n in sorted({a.trees // 4, a.trees // 2, a.trees}):
    P = m.predict_proba(X[val], num_iteration=n)
    w = fit_weights(P, yv)
    f = macro_f1(yv, (P * w).argmax(1))
    print(f"trees={n:4d}  argmax F1={macro_f1(yv, P.argmax(1)):.4f}  weighted F1={f:.4f}")
    if best is None or f > best[0]:
        best = (f, n, P, w)
f, n, P, w = best
pred = (P * w).argmax(1)
T = per_class(yv, pred)
# trap-oracle diagnostics on September
mv = meta[val]
pat = (mv.old.values & (mv.ip.values > 0.5))
T["rec_patterned"] = [np.nan] + [float((pred[(yv == c) & pat] == c).mean()) for c in range(1, len(LABELS))]
old_rate = np.array([mv.old.values[pred == c].mean() if (pred == c).any() else np.nan for c in range(len(LABELS))])
T["purity_est"] = np.clip((old_rate - 0.1506) / (0.625 - 0.1506), 0, 1.2)
T["w"] = w
print(f"\n== {a.tag}: best trees={n}  September macro-F1 = {f:.4f}")
print(T.round(3).to_string())
np.save(OUT / "sep_proba.npy", P.astype("float32"))
np.save(OUT / "weights.npy", w)
imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("\ntop features by gain %:", (imp / imp.sum() * 100).round(1).head(30).to_dict())
m.booster_.save_model(str(OUT / "lgb.txt"))
