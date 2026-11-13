"""Vectorised time-window lookups over events grouped by an entity (sender, receiver, device, pair)."""
import numpy as np

SCALE, OFF = np.int64(2 ** 26), np.int64(2 ** 25)
MIN, HOUR, DAY = 60, 3600, 86400


class Events:
    """Events sorted by (group, time). Queries are O(log n) per row via searchsorted."""

    def __init__(self, group, tt, vals=None, mask=None):
        if mask is not None:
            group, tt = group[mask], tt[mask]
            vals = vals[mask] if vals is not None else None
        o = np.lexsort((tt, group))
        self.K = group[o].astype(np.int64) * SCALE + tt[o] + OFF
        self.v = vals[o].astype(np.float64) if vals is not None else None
        self.cs = np.r_[0.0, np.cumsum(self.v)] if vals is not None else None
        self.n = len(self.K)

    def _lr(self, qg, qt, lo, hi):
        b = qg.astype(np.int64) * SCALE + qt + OFF
        return np.searchsorted(self.K, b + lo, "left"), np.searchsorted(self.K, b + hi, "right")

    def count(self, qg, qt, lo, hi):
        """Number of events of group qg with time in [qt+lo, qt+hi]; 0 where qg < 0."""
        lo_i, hi_i = self._lr(qg, qt, lo, hi)
        return np.where(qg >= 0, hi_i - lo_i, 0)

    def sum(self, qg, qt, lo, hi):
        """Sum of values over the same window."""
        lo_i, hi_i = self._lr(qg, qt, lo, hi)
        return np.where(qg >= 0, self.cs[hi_i] - self.cs[lo_i], 0.0)
