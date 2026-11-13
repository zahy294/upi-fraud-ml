"""Independent, low-memory evidence audits and paired day-block intervals.

This is research evidence, not a submission pipeline. Revealed-label tests use
true positive seeds to study topology, and never claim deployable accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eda"))
from common import LABELS


def mf(cm):
    return float((2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


def main():
    result = {}
    df = pd.read_parquet(ROOT / "data/interim/train.parquet", columns=["timestamp", "sender_id", "receiver_id", "receiver_type", "device_id", "label"])
    t = ((df.timestamp - pd.Timestamp("2026-07-01")).dt.total_seconds()).to_numpy(dtype="int64")
    y = df.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy(dtype="int32")
    mo = df.timestamp.dt.month.to_numpy()
    trials = []
    for c, col, window in [(1, "receiver_id", 3600), (2, "device_id", 300)]:
        g = pd.factorize(df[col])[0].astype("int64")
        q = g * 2**26 + t
        for month in [8, 9]:
            population = mo == month
            if c == 1:
                population &= (df.receiver_type == "P2P").to_numpy()
            for seed in [137, 281, 419]:
                reveal = np.random.default_rng(seed).random(len(y)) < .2
                anchors = population & reveal & (y == c)
                k = np.sort(q[anchors])
                near = (np.searchsorted(k, q + window, "right") - np.searchsorted(k, q - window, "left")) > 0
                added = population & ~reveal & near
                tp = int((added & (y == c)).sum())
                n = int(added.sum())
                trials.append({"label": LABELS[c], "month": month, "seed": seed,
                    "revealed_true_seeds": int(anchors.sum()), "hidden_neighbours": n,
                    "correct": tp, "wrong": n - tp,
                    "precision": tp / n if n else None,
                    "hidden_class_recall": tp / int((population & ~reveal & (y == c)).sum())})
    result["revealed_label_topology_diagnostic"] = trials
    del df
    paths = {name: ROOT / f"prob_9_{name}.npz" for name in ["base", "interactions", "history_interactions"]}
    if all(p.exists() for p in paths.values()):
        arrays = {name: np.load(path) for name, path in paths.items()}
        base = arrays["base"]
        y, days = base["y"], base["days"]
        cms = {}
        for name, a in arrays.items():
            assert np.array_equal(y, a["y"]) and np.array_equal(days, a["days"])
            pred = (a["P"] * a["weights"]).argmax(1)
            cms[name] = np.array([np.bincount(y[days == d] * 12 + pred[days == d], minlength=144).reshape(12, 12) for d in np.unique(days)])
        indices = np.random.default_rng(137).integers(0, len(cms["base"]), size=(1000, len(cms["base"])))
        scores = {name: np.array([mf(m[ix].sum(0)) for ix in indices]) for name, m in cms.items()}
        result["paired_day_bootstrap"] = {}
        for a, b in [("interactions", "base"), ("history_interactions", "base"), ("history_interactions", "interactions")]:
            delta = scores[a] - scores[b]
            result["paired_day_bootstrap"][f"{a}_minus_{b}"] = {"mean": float(delta.mean()), "interval_95": np.quantile(delta, [.025, .975]).tolist(), "positive_fraction": float((delta > 0).mean())}
        result["september_halves"] = {}
        for name, a in arrays.items():
            pred = (a["P"] * a["weights"]).argmax(1)
            result["september_halves"][name] = [mf(np.bincount(y[m] * 12 + pred[m], minlength=144).reshape(12, 12)) for m in [days < 76, days >= 76]]
    (ROOT / "evidence_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
