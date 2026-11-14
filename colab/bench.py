"""The fixed local benchmark, importable: pooled and nested leave-one-month-out macro-F1 of saved probabilities.

Pooled: class weights fitted on all three months. Nested: weights fitted on two months, applied to the third.
Band rules (as in src/upi/make_submission.py) are applied before scoring. Labels are the cleaned benchmark labels:
either models/oof_raw/y_clean.npy (train rows in base order) or a parquet with columns txn_id, y.

Command line: python -I colab/bench.py <project_root> spec [spec ...]     spec = tag[:weight][+tag[:weight]...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/upi"))
from common import LABELS, macro_f1, per_class  # noqa: E402


def fit_weights_fast(P, y, sweeps=3, grid=np.geomspace(0.05, 400, 60), w0=None):
    """Exact, fast version of common.fit_weights: one pass per class gives every row's flip point on the grid."""
    n, k = P.shape
    w = np.ones(k) if w0 is None else w0.copy()
    G = len(grid)
    rows = np.arange(n)
    for _ in range(sweeps):
        for c in range(1, k):
            Q = P * w
            Q[:, c] = -np.inf
            a = Q.argmax(1)
            m = Q[rows, a]
            with np.errstate(divide="ignore", invalid="ignore"):
                thr = np.where(P[:, c] > 0, m / P[:, c], np.inf)
            j = np.searchsorted(grid, thr, side="right")
            cm0 = np.bincount(y * k + a, minlength=k * k).reshape(k, k).astype(np.int64)
            flip = j < G
            rem = np.bincount(j[flip] * k * k + y[flip] * k + a[flip], minlength=G * k * k).reshape(G, k, k).cumsum(0)
            add = np.bincount(j[flip] * k * k + y[flip] * k + c, minlength=G * k * k).reshape(G, k, k).cumsum(0)
            cm = cm0[None] - rem + add
            tp = np.einsum("gii->gi", cm)
            s = (2 * tp / np.maximum(cm.sum(1) + cm.sum(2), 1)).mean(1)
            plateau = np.where(s >= s.max() - 0.0003)[0]
            w[c] = grid[plateau[len(plateau) // 2]]
    return w


class Bench:
    def __init__(self, root, labels=None):
        self.root = Path(root)
        P = self.root / "data/processed"
        base = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "label", "split", "month", "receiver_type"])
        tr = (base.split == "train").values
        self.tr = tr
        self.y_raw = base.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
        lab = Path(labels) if labels else self.root / "models/oof_raw/y_clean.npy"
        if lab.suffix == ".parquet":                                # keyed by txn_id, so the row order of this run does not matter
            t = pd.read_parquet(lab).set_index("txn_id").y
            y = t.reindex(base.txn_id.values[tr])
            assert y.notna().all(), "benchmark labels do not cover every train row"
            self.y = y.values.astype(int)
        else:
            self.y = np.load(lab).astype(int)
        assert len(self.y) == tr.sum()
        month = base.month.values[tr].copy()
        month[month > "2026-09"] = "2026-09"
        self.month = month
        F3 = pd.read_parquet(P / "features_v3.parquet")
        self.band = F3.band.values.astype(int)[tr]
        S = np.stack([F3[f"snd_band_{k}_3h"].values[tr] for k in range(1, 12)], 1)
        R = np.stack([F3[f"rcv_band_{k}_24h"].values[tr] for k in range(1, 12)], 1)
        self.sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)
        self.rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
        self.cb = pd.read_parquet(P / "features_v5.parquet", columns=["ch_band"]).ch_band.values[tr].astype(int)
        self.p2p = (base.receiver_type.values == "P2P")[tr]

    def rules(self, pred, m=slice(None)):
        b, s, r, c, p = self.band[m], self.sb[m], self.rb[m], self.cb[m], self.p2p[m]
        out = np.where(b > 0, b, pred)
        free = (b == 0) & (pred > 0)
        out = np.where(free & np.isin(pred, [4, 5]) & np.isin(s, [4, 5]), s, out)
        out = np.where(free & ((np.isin(pred, [1, 2, 3]) & p & np.isin(r, [1, 2, 3])) | (np.isin(pred, [7, 8, 11]) & ~p & np.isin(r, [7, 8, 11]))), r, out)
        out = np.where(free & (c > 0), c, out)
        return out

    def load(self, spec):
        parts = [p.split(":") for p in spec.split("+")]
        ws = np.array([float(p[1]) if len(p) > 1 else 1.0 for p in parts])
        ws = ws / ws.sum()
        return sum(w * np.load(self.root / "models" / p[0] / "oof.npy").astype(float) for w, p in zip(ws, parts))

    def score(self, P, per_class_table=False):
        y = self.y
        w = fit_weights_fast(P, y)
        pred = self.rules((P * w).argmax(1))
        nested = np.zeros(len(y), dtype=int)
        for mth in ("2026-07", "2026-08", "2026-09"):
            h = self.month == mth
            wm = fit_weights_fast(P[~h], y[~h])
            nested[h] = self.rules((P[h] * wm).argmax(1), h)
        out = {"pooled": macro_f1(y, pred), "pooled_raw": macro_f1(self.y_raw, pred), "nested": macro_f1(y, nested), "nested_raw": macro_f1(self.y_raw, nested)}
        if per_class_table:
            out["per_class"] = per_class(y, pred)
        return out

    def score_spec(self, spec, **kw):
        return self.score(self.load(spec), **kw)


if __name__ == "__main__":
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    labels = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--labels=")), None)
    B = Bench(sys.argv[1], labels)
    print(f"{'spec':<44} {'pooled':>8} {'raw':>8} {'nested':>8} {'raw':>8}")
    for spec in args:
        r = B.score_spec(spec, per_class_table="--per-class" in sys.argv)
        print(f"{spec:<44} {r['pooled']:>8.5f} {r['pooled_raw']:>8.5f} {r['nested']:>8.5f} {r['nested_raw']:>8.5f}", flush=True)
        if "per_class" in r:
            print(r["per_class"].round(3).to_string())
