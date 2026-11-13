"""Fit decisions on August only; apply the saved parameters unchanged."""
import numpy as np

from .constants import LABELS


def confusion(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected equal nonempty one-dimensional label arrays")
    if not np.isin(y, np.arange(12)).all() or not np.isin(prediction, np.arange(12)).all():
        raise ValueError("Labels must be integer class codes 0..11")
    return np.bincount(y.astype(int) * 12 + prediction.astype(int), minlength=144).reshape(12, 12)


def macro_f1(cm):
    return float((2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


def evaluate(y, prediction):
    cm = confusion(y, prediction)
    tp = cm.diagonal()
    return {"macro_f1": macro_f1(cm), "confusion_matrix": cm.tolist(), "per_class": {
        name: {"precision": float(tp[c] / max(cm[:, c].sum(), 1)),
               "recall": float(tp[c] / max(cm[c].sum(), 1)),
               "f1": float(2 * tp[c] / max(cm[:, c].sum() + cm[c].sum(), 1)),
               "support": int(cm[c].sum()), "predicted": int(cm[:, c].sum())}
        for c, name in enumerate(LABELS)}}


def validate_probabilities(P):
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[1] != 12 or not np.isfinite(P).all():
        raise ValueError("Expected finite probabilities with twelve ordered columns")
    if (P < 0).any() or (P > 1).any() or not np.allclose(P.sum(1), 1, atol=2e-5):
        raise ValueError("Invalid probability distribution")


def tune_weights(P, y, sweeps=2):
    validate_probabilities(P)
    best = (-1, None)
    for k in [1, 2, 3, 5, 8, 12, 20]:
        w = np.array([1] + [k] * 11, dtype=float)
        score = macro_f1(confusion(y, (P * w).argmax(1)))
        if score > best[0]:
            best = score, w
    shared = best[1].copy()
    weights = shared.copy()
    for _ in range(sweeps):
        for c in range(1, 12):
            scores = P * weights
            scores[:, c] = -1
            other = scores.argmax(1)
            mx = scores[np.arange(len(y)), other]
            grid = np.unique(np.r_[np.geomspace(.5, 30, 15), weights[c]])
            values = [macro_f1(confusion(y, np.where(P[:, c] * g > mx, c, other))) for g in grid]
            candidates = np.flatnonzero(np.asarray(values) >= max(values) - .0002)
            chosen = min(candidates, key=lambda i: abs(np.log(grid[i] / weights[c])))
            weights[c] = grid[chosen]
    return shared, weights


def apply_combination(P, rules, weights, thresholds):
    prediction = (P * np.asarray(weights)).argmax(1)
    rules = np.asarray(rules, dtype=int)
    threshold = np.array([np.inf if t is None else t for t in thresholds])
    hit = (rules > 0) & (P[np.arange(len(P)), rules] >= threshold[rules])
    prediction[hit] = rules[hit]
    return prediction


def fit_decisions(P, y, rules, config):
    shared, joint = tune_weights(P, y, config["decision_sweeps"])
    chosen = joint if config["decision_kind"] == "joint" else shared
    thresholds = [None] * 12
    base = (P * chosen).argmax(1)
    prediction = base.copy()
    for _ in range(config["decision_sweeps"]):
        for c in range(1, 12):
            hit = rules == c
            best = (macro_f1(confusion(y, prediction)), thresholds[c])
            for value in config["rule_threshold_grid"]:
                trial = prediction.copy()
                trial[hit] = np.where(P[hit, c] >= value, c, base[hit])
                score = macro_f1(confusion(y, trial))
                if score > best[0] + config["rule_min_improvement"]:
                    best = score, value
            thresholds[c] = best[1]
            prediction = apply_combination(P, rules, chosen, thresholds)
    return {"label_order": list(LABELS), "fitted_on": "August only",
            "decision_kind": config["decision_kind"], "shared_weights": shared.tolist(),
            "joint_weights": joint.tolist(), "weights": chosen.tolist(),
            "rule_thresholds": thresholds}


def predictions(P, rules, decisions):
    validate_probabilities(P)
    if decisions["label_order"] != list(LABELS):
        raise ValueError("Decision class order differs from model probability order")
    return {
        "raw": P.argmax(1),
        "shared": (P * decisions["shared_weights"]).argmax(1),
        "joint": (P * decisions["joint_weights"]).argmax(1),
        "model": (P * decisions["weights"]).argmax(1),
        "rules": rules,
        "combined": apply_combination(P, rules, decisions["weights"], decisions["rule_thresholds"]),
    }


def paired_day_bootstrap(y, baseline, combined, days, draws, seed):
    unique_days = np.unique(days)
    a = np.array([confusion(y[days == d], baseline[days == d]) for d in unique_days])
    b = np.array([confusion(y[days == d], combined[days == d]) for d in unique_days])
    samples = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    delta = np.array([macro_f1(b[ix].sum(0)) - macro_f1(a[ix].sum(0)) for ix in samples])
    return {"draws": draws, "mean_delta": float(delta.mean()),
            "delta_interval_95": np.quantile(delta, [.025, .975]).tolist(),
            "positive_fraction": float((delta > 0).mean()),
            "limitation": "Exploratory paired day resampling; not a public-score interval"}
