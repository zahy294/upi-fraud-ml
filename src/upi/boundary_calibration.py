"""Rule-aware class calibration from saved probabilities; no estimator fitting.

Search decision intervals, rather than a fixed grid of log offsets. Selection
on a later calibration block is still required: sharper fitting can overfit.
"""
import numpy as np
import pandas as pd
from .common import LABELS
from .make_submission import _proba, _labels, _apply_rules, _score


def rule_map(context, n):
    """The existing, label-free rules map each possible raw prediction."""
    return np.column_stack([_apply_rules(np.full(n, k), context) for k in range(12)])


def coordinate(logp, offsets, k, y, mapping, bound, penalty):
    """Find the best interior decision interval for one class, keeping others fixed.

    Equal margins move together; never split a tied group using its label.
    Actual predictions are rescored, including interval endpoints and the old
    offset, so floating point boundary effects cannot worsen the objective.
    """
    n = len(y)
    scores = logp + offsets
    scores[:, k] = -np.inf
    other = scores.argmax(axis=1)
    margin = scores[np.arange(n), other] - logp[:, k]
    before, after = mapping[np.arange(n), other], mapping[:, k]
    order = np.argsort(margin, kind="stable")
    sorted_margin = margin[order]
    edges = np.unique(np.r_[-bound, sorted_margin[(sorted_margin > -bound) & (sorted_margin < bound)], bound])
    values = (edges[:-1] + edges[1:]) / 2
    # Every value is between distinct margins. searchsorted counts all switches
    # strictly below it. Extremely close floating-point ties are rescored below.
    indices = np.searchsorted(sorted_margin, values, side="left")
    support = np.bincount(y, minlength=12)
    curve = np.zeros(len(values))
    for label in range(12):
        old, new = before == label, after == label
        delta = new.astype(np.int32) - old.astype(np.int32)
        count = old.sum() + np.r_[0, np.cumsum(delta[order], dtype=np.int64)][indices]
        tp_delta = delta * (y == label)
        tp = (old & (y == label)).sum() + np.r_[0, np.cumsum(tp_delta[order], dtype=np.int64)][indices]
        curve += (2 * tp / np.maximum(support[label] + count, 1)) / 12
    curve -= penalty * (np.sum(offsets[1:] ** 2) - offsets[k] ** 2 + values ** 2) / 11
    # Stable preference for smaller changes / offsets when several intervals tie.
    winner = np.lexsort((values, np.abs(values), np.abs(values - offsets[k]), -curve))[0]
    candidates = [offsets[k], float(values[winner]), -bound, bound, 0.]
    best_value, best_score = offsets[k], -np.inf
    for value in candidates:
        trial = offsets.copy()
        trial[k] = value
        raw = (logp + trial).argmax(axis=1)
        pred = mapping[np.arange(n), raw]
        objective = _score(y, pred) - penalty * np.mean(trial[1:] ** 2)
        if objective > best_score + 1e-12:
            best_value, best_score = value, objective
    return best_value, float(best_score)


def fit_boundary_offsets(proba, y, context, *, penalty=.002, bound=4., passes=3):
    p, y = _proba(proba), _labels(y, "calibration labels")
    dates = pd.DatetimeIndex(pd.to_datetime(context.get("dates", []), utc=True))
    if (not len(y) or len(p) != len(y) or len(dates) != len(y) or dates.isna().any()
            or len(dates.year.unique()) != 1
            or not ((dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)).all()):
        raise ValueError("boundary calibration requires aligned September16-23 rows only")
    if (not np.isfinite(penalty) or penalty < 0 or not np.isfinite(bound)
            or not 0 < bound <= 6 or not isinstance(passes, int) or not 1 <= passes <= 5):
        raise ValueError("invalid bounded calibration settings")
    logp, mapping = np.log(np.maximum(p, 1e-9)), rule_map(context, len(y))
    offsets, trace = np.zeros(12), []
    for sweep in range(passes):
        old = offsets.copy()
        for k in range(1, 12):
            offsets[k], objective = coordinate(logp, offsets, k, y, mapping, bound, penalty)
        trace.append({"pass": sweep, "objective": objective, "global": offsets.tolist()})
        if np.array_equal(old, offsets):
            break
    return {"global": offsets.tolist(), "conditional": {}, "class_order": list(LABELS),
            "method": "decision_intervals", "penalty": penalty, "bound": bound,
            "calibration_rows": len(y), "trace": trace}
