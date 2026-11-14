from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
from common import CATS, LABELS, load_xy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="oof_clean_cat")
    ap.add_argument(
        "--features",
        default="features_v1.parquet,features_v2.parquet,features_v3.parquet,features_v4.parquet,features_v5.parquet",
    )
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    feats = tuple(a.features.split(","))
    X, y_raw, meta = load_xy(root, feats)
    for c in CATS:
        X[c] = X[c].astype(str)

    tr = (meta.split == "train").values
    y = y_raw.copy()
    y[tr] = np.load(root / "models/oof_raw/y_clean.npy")
    month = meta.month.values

    def fit_predict(fit_mask, pred_mask, seed):
        rng = np.random.RandomState(seed)
        keep = fit_mask & ((y != 0) | (rng.rand(len(y)) < 0.25))
        m = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=6.0,
            loss_function="MultiClass",
            task_type="GPU",
            devices="0",
            cat_features=CATS,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )
        m.fit(X.loc[keep], y[keep])
        return m.predict_proba(X.loc[pred_mask])

    t0 = time.time()
    seeds = a.seeds
    oof = np.zeros((tr.sum(), len(LABELS)), dtype="float32")
    tr_idx = np.where(tr)[0]
    for mth in ("2026-07", "2026-08", "2026-09"):
        hold = tr & (month == mth)
        p = sum(fit_predict(tr & ~hold, hold, s) for s in range(seeds)) / seeds
        oof[np.searchsorted(tr_idx, np.where(hold)[0])] = p
        print(f"fold {mth} done ({time.time() - t0:.0f}s)", flush=True)
    np.save(out / "oof.npy", oof)

    p_te = sum(fit_predict(tr, ~tr, s) for s in range(seeds)) / seeds
    np.save(out / "test_proba.npy", p_te.astype("float32"))
    print(f"test done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
