"""Label-free neighbour sums and means, with explicit month exclusions.

All intervals are closed, symmetric, and exclude only the query row itself.
The structured (entity, time) index avoids Events' packed-key range assumption.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS = {"receiver_24h": 86400, "device_10m": 600,
           "sender_3h": 10800, "reverse_pair_7d": 604800}


def logical_folds(meta):
    """October training tails belong to September; every test row has fold ''."""
    if not meta["split"].isin(["train", "test"]).all():
        raise ValueError("Unknown split")
    train = meta["split"].eq("train").to_numpy()
    month = meta["month"].astype(str).to_numpy(copy=True)
    month[train & (month > "2026-09")] = "2026-09"
    month[~train] = ""
    if set(month[train]) != {"2026-07", "2026-08", "2026-09"}:
        raise ValueError("This experiment requires July, August and September training folds")
    return month


def fit_months_for(available, query_month, source_month):
    """Exclude both the target's and the probability source's labelled month."""
    return tuple(sorted(set(available) - {query_month, source_month, ""}))


def probability_context(folds, available, query_month, provider, n_classes):
    """Build source probabilities safe for this query fold and outer fold.

    provider(months) must fit ONLY those months and predict every input row in
    its original order. Empty training sets use a fixed, label-independent
    uniform prior. This occurs at cross-month links in nested three-month CV.
    """
    folds = np.asarray(folds)
    out = np.empty((len(folds), n_classes), dtype=np.float32)
    plan = []
    for source_month in sorted(set(folds)):
        rows = np.flatnonzero(folds == source_month)
        months = fit_months_for(available, query_month, source_month)
        if months:
            p = provider(months)
            if p.shape != out.shape:
                raise ValueError("Probability cache shape/order contract violated")
            out[rows] = p[rows]
        else:
            out[rows] = 1.0 / n_classes
        plan.append({"query_month": query_month, "source_month": source_month,
                     "fit_months": list(months), "source_rows": len(rows),
                     "uniform_prior": not bool(months)})
    return out, plan


def restore_population_prior(probabilities, legit_fraction):
    """Undo class-conditional sampling: p(c|x) proportional to q(c|x)/s_c.

    This corrects the ideal sampling-prior shift, not arbitrary model
    miscalibration. Class zero is legit. Apply once, before aggregation.
    """
    if not 0 < legit_fraction <= 1:
        raise ValueError("legit_fraction must be in (0, 1]")
    p = np.asarray(probabilities, dtype=np.float64).copy()
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probabilities")
    p[:, 0] /= legit_fraction
    normalizer = p.sum(axis=1, keepdims=True)
    if (normalizer <= 0).any():
        raise ValueError("Zero probability mass")
    return (p / normalizer).astype(np.float32)


def _keys(group, time):
    result = np.empty(len(group), dtype=[("group", "i8"), ("time", "i8")])
    result["group"], result["time"] = group, time
    return result


@dataclass
class WindowIndex:
    order: np.ndarray
    left: np.ndarray
    right: np.ndarray
    self_present: np.ndarray

    @classmethod
    def build(cls, source_group, query_group, time, radius):
        order = np.lexsort((time, source_group))
        keys = _keys(source_group[order], time[order])
        left = np.searchsorted(keys, _keys(query_group, time - radius), side="left")
        right = np.searchsorted(keys, _keys(query_group, time + radius), side="right")
        missing = query_group < 0
        left[missing], right[missing] = 0, 0
        self_present = (source_group == query_group) & ~missing
        return cls(order, left, right, self_present)

    def aggregate(self, probabilities, rows):
        lo, hi = self.left[rows], self.right[rows]
        subtract_self = self.self_present[rows]
        count = hi - lo - subtract_self.astype(np.int64)
        sums = np.empty((len(rows), probabilities.shape[1]), dtype=np.float32)
        for c in range(probabilities.shape[1]):
            prefix = np.empty(len(probabilities) + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(probabilities[self.order, c], dtype=np.float64, out=prefix[1:])
            values = prefix[hi] - prefix[lo]
            values -= probabilities[rows, c] * subtract_self
            # Cumulative subtraction can leave a few ulps below zero.
            sums[:, c] = np.maximum(values, 0)
        sums[count == 0] = 0
        means = sums / np.maximum(count[:, None], 1)
        return count, sums, means.astype(np.float32)


class NeighborFeatures:
    """One reusable graph index; no labels, split flags, or probabilities in it."""

    def __init__(self, entities):
        required = ["txn_id", "sender_id", "receiver_id", "device_id", "t"]
        if entities[required].isna().any().any() or not entities.txn_id.is_unique:
            raise ValueError("Unique transaction IDs and nonmissing entity/time fields required")
        raw_time = entities.t.to_numpy()
        if not np.issubdtype(raw_time.dtype, np.integer):
            raise ValueError("t must contain integer seconds")
        time = raw_time.astype(np.int64)
        radius = max(WINDOWS.values())
        limits = np.iinfo(np.int64)
        if time.min() < limits.min + radius or time.max() > limits.max - radius:
            raise ValueError("Window arithmetic would overflow int64")
        self.n = len(time)
        self.windows = {}
        for name, key in [("receiver_24h", "receiver_id"), ("device_10m", "device_id"),
                          ("sender_3h", "sender_id")]:
            group = pd.factorize(entities[key])[0]
            self.windows[name] = WindowIndex.build(group, group, time, WINDOWS[name])
        # Full IDs preserve the account/merchant namespaces, including self-transfers.
        pairs = pd.MultiIndex.from_frame(entities[["sender_id", "receiver_id"]])
        group, unique = pd.factorize(pairs)
        reverse = pd.MultiIndex.from_arrays([entities.receiver_id, entities.sender_id])
        query = unique.get_indexer(reverse)
        self.windows["reverse_pair_7d"] = WindowIndex.build(
            group, query, time, WINDOWS["reverse_pair_7d"])

    def transform(self, probabilities, classes, rows=None):
        p = np.asarray(probabilities)
        if p.shape != (self.n, len(classes)) or not np.isfinite(p).all():
            raise ValueError("Probabilities must cover every graph row in the fixed class order")
        if (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Probabilities must be nonnegative and normalized")
        rows = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= self.n).any():
            raise ValueError("Invalid or duplicate query rows")
        columns = {f"s1_self_{c}": p[rows, i] for i, c in enumerate(classes)}
        for relation, window in self.windows.items():
            count, sums, means = window.aggregate(p, rows)
            columns[f"s1_{relation}_count"] = count.astype(np.float32)
            for i, c in enumerate(classes):
                columns[f"s1_{relation}_sum_{c}"] = sums[:, i]
                columns[f"s1_{relation}_mean_{c}"] = means[:, i]
        return pd.DataFrame(columns, index=rows, dtype=np.float32)
