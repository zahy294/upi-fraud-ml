"""Vectorized grouped event queries without composite-key overflow."""
from __future__ import annotations
import numpy as np

MIN, HOUR, DAY = 60, 3600, 86400
_KEY = np.dtype([("group", "i8"), ("time", "i8")])


def _integers(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if a.size and a.dtype.kind == "u" and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return a.astype(np.int64)


def _keys(group, time):
    result = np.empty(len(group), dtype=_KEY)
    result["group"], result["time"] = group, time
    return result


class Events:
    """Closed windows; strict previous/next. Negative groups never match."""
    def __init__(self, group, tt, vals=None, mask=None, *, values=None):
        if vals is not None and values is not None:
            raise ValueError("supply vals or values, not both")
        vals = values if vals is None else vals
        group, tt = _integers(group, "group"), _integers(tt, "tt")
        if len(group) != len(tt):
            raise ValueError("group/time lengths differ")
        vals = None if vals is None else np.asarray(vals, dtype=float)
        if vals is not None and (vals.ndim != 1 or len(vals) != len(tt) or not np.isfinite(vals).all()):
            raise ValueError("values must be finite and match time shape")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype.kind != "b" or mask.shape != tt.shape:
                raise ValueError("mask must be a matching boolean array")
            group, tt = group[mask], tt[mask]
            vals = None if vals is None else vals[mask]
        order = np.lexsort((tt, group))
        self.group, self.time = group[order], tt[order]
        self.values = None if vals is None else vals[order]
        self.keys, self.n = _keys(self.group, self.time), len(order)
        self.cs = None if vals is None else np.r_[0., np.cumsum(self.values)]

    def _query(self, qg, qt):
        qg, qt = _integers(qg, "qg"), _integers(qt, "qt")
        if qg.shape != qt.shape:
            raise ValueError("query shapes differ")
        return qg, qt

    @staticmethod
    def _shift(time, delta):
        if not isinstance(delta, (int, np.integer)):
            raise ValueError("bounds must be integer seconds")
        limits = np.iinfo(np.int64)
        if len(time) and (int(time.min()) + int(delta) < limits.min or int(time.max()) + int(delta) > limits.max):
            raise ValueError("query time bound exceeds int64")
        return time + int(delta)

    def _lr(self, qg, qt, lo, hi):
        qg, qt = self._query(qg, qt)
        if lo > hi:
            raise ValueError("lo must not exceed hi")
        left = np.searchsorted(self.keys, _keys(qg, self._shift(qt, lo)), side="left")
        right = np.searchsorted(self.keys, _keys(qg, self._shift(qt, hi)), side="right")
        return qg, left, right

    def count(self, qg, qt, lo, hi):
        qg, left, right = self._lr(qg, qt, lo, hi)
        return np.where(qg >= 0, right - left, 0)

    def sum(self, qg, qt, lo, hi):
        if self.cs is None:
            raise ValueError("values are required for sum queries")
        qg, left, right = self._lr(qg, qt, lo, hi)
        return np.where(qg >= 0, self.cs[right] - self.cs[left], 0.)

    def _nearest(self, qg, qt, forward, within=None, value=False):
        qg, qt = self._query(qg, qt)
        if value and self.values is None:
            raise ValueError("values are required for value queries")
        if within is not None and (not np.isfinite(within) or within < 0):
            raise ValueError("within must be finite and nonnegative")
        if not self.n:
            return np.full(len(qg), np.nan)
        pos = np.searchsorted(self.keys, _keys(qg, qt), side="right" if forward else "left")
        if not forward:
            pos -= 1
        safe = np.clip(pos, 0, self.n - 1)
        valid = (pos >= 0) & (pos < self.n) & (qg >= 0) & (self.group[safe] == qg)
        event = self.time[safe]
        if len(qt) and (int(max(event.max(), qt.max())) - int(min(event.min(), qt.min())) > np.iinfo(np.int64).max):
            raise ValueError("timestamp difference exceeds int64")
        gap = (event - qt) if forward else (qt - event)
        valid &= gap > 0
        if within is not None:
            valid &= gap <= within
        return np.where(valid, self.values[safe] if value else gap.astype(float), np.nan)

    def prev_dt(self, qg, qt, within):
        return self._nearest(qg, qt, False, within)

    def next_dt(self, qg, qt, within):
        return self._nearest(qg, qt, True, within)

    def previous_gap(self, query_group, query_time):
        return self._nearest(query_group, query_time, False)

    def next_gap(self, query_group, query_time):
        return self._nearest(query_group, query_time, True)

    def previous_value(self, query_group, query_time, within):
        return self._nearest(query_group, query_time, False, within, True)

    def next_value(self, query_group, query_time, within):
        return self._nearest(query_group, query_time, True, within, True)
