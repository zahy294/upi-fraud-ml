"""Refit the stage-1 model on Jul-Sep and write a submission, using the tree count and class weights found on September.

Usage: python -I src/upi/fit_submit.py <project_root> <tag> --trees N [--seeds 3] [--features a.parquet,b.parquet]
Reads models/<tag>/weights.npy (from exp_stage1.py). Writes submissions/<tag>.csv and models/<tag>/test_proba.npy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, load_xy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--trees", type=int, required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--legit-frac", type=float, default=0.25)
ap.add_argument("--features", default="features_v1.parquet")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag

X, y, meta = load_xy(ROOT, tuple(f for f in a.features.split(",") if f))
tr = (meta.split == "train").values
te = ~tr
w = np.load(OUT / "weights.npy")
P = np.zeros((te.sum(), len(LABELS)))
for seed in range(a.seeds):
    rng = np.random.RandomState(seed)
    keep = tr & ((y != 0) | (rng.rand(len(y)) < a.legit_frac))
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=a.trees, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, n_jobs=12, random_state=seed)
    m.fit(X[keep], y[keep])
    P += m.predict_proba(X[te]) / a.seeds
    print("seed", seed, "done")
np.save(OUT / "test_proba.npy", P.astype("float32"))
pred = (P * w).argmax(1)
sub = pd.DataFrame({"id": meta.txn_id.values[te], "label": np.array(LABELS)[pred]})
sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
assert len(sub) == len(sample) and sub.label.notna().all() and (sub.id.values == sample.id.values).all()
assert set(sub.label) == set(LABELS), "every label must be predicted at least once"
path = ROOT / "submissions" / f"{a.tag}.csv"
sub.to_csv(path, index=False)
vc = sub.label.value_counts()
print("wrote", path, "| predicted share of rows per class (%):")
print((vc / len(sub) * 100).round(3).to_dict())
