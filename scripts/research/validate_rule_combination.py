"""Fit rule/model decisions on August; evaluate frozen decisions on September.

Usage: python validate_rule_combination.py AUG_ROOT SEP_ROOT [base|context]
Uses only saved probabilities and the pre-existing EDA rules. Writes JSON to
SEP_ROOT; never creates a competition submission.
"""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eda"))
from common import LABELS


def matrix(y, pred):
    return np.bincount(y * 12 + pred, minlength=144).reshape(12, 12)


def score(cm):
    return float((2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


def stats(cm):
    tp = cm.diagonal()
    return {"macro_f1": score(cm), "per_class": {
        name: {"precision": float(tp[c] / max(cm[:, c].sum(), 1)),
               "recall": float(tp[c] / max(cm[c].sum(), 1)),
               "f1": float(2 * tp[c] / max(cm[:, c].sum() + cm[c].sum(), 1)),
               "predicted": int(cm[:, c].sum()), "support": int(cm[c].sum())}
        for c, name in enumerate(LABELS)}}


def apply(a, thresholds):
    P, r = a["P"], a["rules"]
    pred = (P * a["weights"]).argmax(1)
    hit = (r > 0) & (P[np.arange(len(r)), r] >= thresholds[r])
    pred[hit] = r[hit]
    return pred


def main():
    aug_root, sep_root = map(Path, sys.argv[1:3])
    name = sys.argv[3] if len(sys.argv) > 3 else "base"
    a = np.load(aug_root / f"prob_8_{name}.npz")
    b = np.load(sep_root / f"prob_9_{name}.npz")
    thresholds = np.full(12, np.inf)
    base_aug = (a["P"] * a["weights"]).argmax(1)
    pred = base_aug.copy()
    for _ in range(2):
        for c in range(1, 12):
            hit = a["rules"] == c
            best = (score(matrix(a["y"], pred)), thresholds[c])
            for value in [0, .02, .05, .1, .2, .4, .6]:
                trial = pred.copy()
                trial[hit] = np.where(a["P"][hit, c] >= value, c, base_aug[hit])
                s = score(matrix(a["y"], trial))
                if s > best[0] + .0002:
                    best = s, value
            thresholds[c] = best[1]
            pred = apply(a, thresholds)
    base = (b["P"] * b["weights"]).argmax(1)
    combined = apply(b, thresholds)
    y, days = b["y"], b["days"]
    daily_base = np.array([matrix(y[days == d], base[days == d]) for d in np.unique(days)])
    daily_comb = np.array([matrix(y[days == d], combined[days == d]) for d in np.unique(days)])
    samples = np.random.default_rng(137).integers(0, len(daily_base), size=(1000, len(daily_base)))
    deltas = np.array([score(daily_comb[ix].sum(0)) - score(daily_base[ix].sum(0)) for ix in samples])
    result = {"family": name, "thresholds": dict(zip(LABELS, [None if np.isinf(v) else float(v) for v in thresholds])),
        "August": {"base": stats(matrix(a["y"], base_aug)), "combined": stats(matrix(a["y"], apply(a, thresholds)))},
        "September": {"base": stats(matrix(y, base)), "combined": stats(matrix(y, combined))},
        "paired_day_delta": {"mean": float(deltas.mean()), "interval_95": np.quantile(deltas, [.025, .975]).tolist(), "positive_fraction": float((deltas > 0).mean())},
        "September_halves": [{"base": score(matrix(y[m], base[m])), "combined": score(matrix(y[m], combined[m]))} for m in [days < 76, days >= 76]],
        "changes": {"corrected": int(((base != y) & (combined == y)).sum()), "broken": int(((base == y) & (combined != y)).sum())}}
    (sep_root / f"combination_{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"family": name, "aug_base": result["August"]["base"]["macro_f1"], "aug_combined": result["August"]["combined"]["macro_f1"], "sep_base": result["September"]["base"]["macro_f1"], "sep_combined": result["September"]["combined"]["macro_f1"], "bootstrap": result["paired_day_delta"], "halves": result["September_halves"]}, indent=2))


if __name__ == "__main__":
    main()
