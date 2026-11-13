from __future__ import annotations

import numpy as np

MIN, HOUR, DAY = 60, 3600, 86400


class Events:
    """Vectorized window queries over events sorted by integer group and time."""

    def __init__(self, group: np.ndarray, time_s: np.ndarray, values: np.ndarray | None = None,
                 mask: np.ndarray | None = None, *, vals: np.ndarray | None = None):
        if vals is not None:
            if values is not None:
                raise ValueError("supply values or vals, not both")
            values = vals
        group = np.asarray(group, dtype=np.int64)
        time_s = np.asarray(time_s, dtype=np.int64)
        values = None if values is None else np.asarray(values)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            group, time_s = group[mask], time_s[mask]
            values = None if values is None else values[mask]
        order = np.lexsort((time_s, group))
        self.group = group[order]
        self.time = time_s[order]
        self.values = None if values is None else values[order]
        self.n = len(order)
        self.scale = np.int64(max(2**32, (int(time_s.max() - time_s.min()) + 2) if len(time_s) else 2))
        self.offset = np.int64(-int(time_s.min()) + 1 if len(time_s) else 1)
        self.keys = self.group * self.scale + self.time + self.offset

    def _base(self, query_group: np.ndarray, query_time: np.ndarray) -> np.ndarray:
        return np.asarray(query_group, dtype=np.int64) * self.scale + np.asarray(query_time, dtype=np.int64) + self.offset

    def count(self, query_group: np.ndarray, query_time: np.ndarray, lo: int, hi: int) -> np.ndarray:
        base = self._base(query_group, query_time)
        left = np.searchsorted(self.keys, base + lo, side="left")
        right = np.searchsorted(self.keys, base + hi, side="right")
        return np.where(np.asarray(query_group) >= 0, right - left, 0).astype(np.int32)

    def sum(self, query_group: np.ndarray, query_time: np.ndarray, lo: int, hi: int) -> np.ndarray:
        if self.values is None:
            raise ValueError("values are required for sum queries")
        base = self._base(query_group, query_time)
        left = np.searchsorted(self.keys, base + lo, side="left")
        right = np.searchsorted(self.keys, base + hi, side="right")
        prefix = np.r_[0.0, np.cumsum(self.values, dtype=np.float64)]
        return np.where(np.asarray(query_group) >= 0, prefix[right] - prefix[left], 0.0)

    def next_value(self, query_group: np.ndarray, query_time: np.ndarray, within: int) -> np.ndarray:
        if self.values is None:
            raise ValueError("values are required for value queries")
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="right")
        safe = np.minimum(pos, max(self.n - 1, 0))
        valid = (pos < self.n) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.group[safe] == np.asarray(query_group)
            valid &= self.keys[safe] <= base + within
            return np.where(valid, self.values[safe], np.nan)
        return np.full(len(base), np.nan)

    def previous_value(self, query_group: np.ndarray, query_time: np.ndarray, within: int) -> np.ndarray:
        if self.values is None:
            raise ValueError("values are required for value queries")
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="left") - 1
        safe = np.maximum(pos, 0)
        valid = (pos >= 0) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.group[safe] == np.asarray(query_group)
            valid &= self.keys[safe] >= base - within
            return np.where(valid, self.values[safe], np.nan)
        return np.full(len(base), np.nan)

    def previous_gap(self, query_group: np.ndarray, query_time: np.ndarray) -> np.ndarray:
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="left") - 1
        safe = np.maximum(pos, 0)
        valid = (pos >= 0) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.group[safe] == np.asarray(query_group)
        gap = np.asarray(query_time, dtype=np.int64) - (self.time[safe] if self.n else 0)
        return np.where(valid & (gap > 0), gap, np.nan)

    def next_gap(self, query_group: np.ndarray, query_time: np.ndarray) -> np.ndarray:
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="right")
        safe = np.minimum(pos, max(self.n - 1, 0))
        valid = (pos < self.n) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.group[safe] == np.asarray(query_group)
        gap = (self.time[safe] if self.n else 0) - np.asarray(query_time, dtype=np.int64)
        return np.where(valid & (gap > 0), gap, np.nan)

    def prev_dt(self, qg: np.ndarray, qt: np.ndarray, within: int) -> np.ndarray:
        gap = self.previous_gap(qg, qt)
        return np.where(gap <= within, gap, np.nan)

    def next_dt(self, qg: np.ndarray, qt: np.ndarray, within: int) -> np.ndarray:
        gap = self.next_gap(qg, qt)
        return np.where(gap <= within, gap, np.nan)

