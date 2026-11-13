from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import apply_weights, fit_class_weights, macro_f1_ids

MINUTE, HOUR, DAY = 60, 3600, 86400


class MultiValEvents:
    """Vectorized 2D multi-class window queries over events sorted by integer group and time."""

    def __init__(self, group: np.ndarray, time_s: np.ndarray, values_2d: np.ndarray):
        group = np.asarray(group, dtype=np.int64)
        time_s = np.asarray(time_s, dtype=np.int64)
        values_2d = np.asarray(values_2d, dtype=np.float64)
        order = np.lexsort((time_s, group))
        self.group = group[order]
        self.time = time_s[order]
        self.values = values_2d[order]
        self.n, self.k = self.values.shape
        self.scale = np.int64(max(2**32, (int(time_s.max() - time_s.min()) + 2) if len(time_s) else 2))
        self.offset = np.int64(-int(time_s.min()) + 1 if len(time_s) else 1)
        self.keys = self.group * self.scale + self.time + self.offset
        self.prefix = np.vstack([np.zeros((1, self.k), dtype=np.float64), np.cumsum(self.values, axis=0)])

    def _base(self, query_group: np.ndarray, query_time: np.ndarray) -> np.ndarray:
        return np.asarray(query_group, dtype=np.int64) * self.scale + np.asarray(query_time, dtype=np.int64) + self.offset

    def mean_excluding_self(self, query_group: np.ndarray, query_time: np.ndarray,
                            self_values: np.ndarray, lo: int, hi: int,
                            exclude_self: bool = True) -> np.ndarray:
        base = self._base(query_group, query_time)
        left = np.searchsorted(self.keys, base + lo, side="left")
        right = np.searchsorted(self.keys, base + hi, side="right")
        valid_group = np.asarray(query_group) >= 0
        cnt = np.where(valid_group, right - left, 0)
        total = self.prefix[right] - self.prefix[left]
        if exclude_self:
            cnt = np.maximum(cnt - 1, 0)
            total = np.maximum(total - self_values, 0.0)
        out = np.zeros_like(self_values, dtype=np.float32)
        mask = cnt > 0
        if mask.any():
            out[mask] = (total[mask] / cnt[mask, None]).astype(np.float32)
        return out

    def next_values(self, query_group: np.ndarray, query_time: np.ndarray, within: int) -> tuple[np.ndarray, np.ndarray]:
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="right")
        safe = np.minimum(pos, max(self.n - 1, 0))
        valid = (pos < self.n) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.keys[safe] <= base + within
        out = np.zeros((len(base), self.k), dtype=np.float32)
        if valid.any():
            out[valid] = self.values[safe[valid]].astype(np.float32)
        return out, valid

    def previous_values(self, query_group: np.ndarray, query_time: np.ndarray, within: int) -> tuple[np.ndarray, np.ndarray]:
        base = self._base(query_group, query_time)
        pos = np.searchsorted(self.keys, base, side="left") - 1
        safe = np.maximum(pos, 0)
        valid = (pos >= 0) & (np.asarray(query_group) >= 0)
        if self.n:
            valid &= self.keys[safe] >= base - within
        out = np.zeros((len(base), self.k), dtype=np.float32)
        if valid.any():
            out[valid] = self.values[safe[valid]].astype(np.float32)
        return out, valid


CHANNEL_NAMES = (
    "p2p_rcv_1h",
    "dev_5m",
    "dev_1h",
    "snd_24h",
    "pair_15d",
    "rev_24h",
    "flow_30m",
)


def compute_neighbour_channels(frame: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    """Return array of shape (N, n_channels, n_classes) with leave-one-out neighbour probabilities."""
    n, k = probabilities.shape
    t = pd.to_datetime(frame.timestamp).to_numpy(dtype="datetime64[s]").astype(np.int64)
    p2p = frame.receiver_type.eq("P2P").to_numpy()

    sender = pd.factorize(frame.sender_id, sort=False)[0].astype(np.int64)
    receiver = pd.factorize(frame.receiver_id, sort=False)[0].astype(np.int64)
    p2p_receiver = np.where(p2p, receiver, -1).astype(np.int64)
    device = pd.factorize(frame.device_id, sort=False)[0].astype(np.int64)

    pair_index = pd.MultiIndex.from_arrays([frame.sender_id, frame.receiver_id])
    pair, unique_pairs = pd.factorize(pair_index, sort=False)
    pair = pair.astype(np.int64)
    reverse_index = pd.MultiIndex.from_arrays([frame.receiver_id, frame.sender_id])
    reverse_code = unique_pairs.get_indexer(reverse_index).astype(np.int64)

    account_index = pd.Index(pd.unique(pd.concat([frame.sender_id, frame.receiver_id[p2p]], ignore_index=True)))
    sender_account = account_index.get_indexer(frame.sender_id).astype(np.int64)
    receiver_account = np.where(p2p, account_index.get_indexer(frame.receiver_id), -1).astype(np.int64)

    p2p_rcv_ev = MultiValEvents(np.where(p2p, receiver, np.arange(n) + len(receiver) + 10), t, probabilities)
    dev_ev = MultiValEvents(device, t, probabilities)
    snd_ev = MultiValEvents(sender, t, probabilities)
    pair_ev = MultiValEvents(pair, t, probabilities)
    out_ev = MultiValEvents(sender_account, t, probabilities)
    in_ev = MultiValEvents(np.where(p2p, receiver_account, np.arange(n) + len(account_index) + 10), t, probabilities)

    ch_p2p_rcv_1h = p2p_rcv_ev.mean_excluding_self(p2p_receiver, t, probabilities, -HOUR, HOUR, exclude_self=True)
    ch_dev_5m = dev_ev.mean_excluding_self(device, t, probabilities, -5 * MINUTE, 5 * MINUTE, exclude_self=True)
    ch_dev_1h = dev_ev.mean_excluding_self(device, t, probabilities, -HOUR, HOUR, exclude_self=True)
    ch_snd_24h = snd_ev.mean_excluding_self(sender, t, probabilities, -DAY, DAY, exclude_self=True)
    ch_pair_15d = pair_ev.mean_excluding_self(pair, t, probabilities, -15 * DAY, 15 * DAY, exclude_self=True)
    ch_rev_24h = pair_ev.mean_excluding_self(reverse_code, t, probabilities, -DAY, DAY, exclude_self=False)

    next_out, valid_out = out_ev.next_values(receiver_account, t, 30 * MINUTE)
    prev_in, valid_in = in_ev.previous_values(sender_account, t, 30 * MINUTE)
    denom = valid_out.astype(np.float32) + valid_in.astype(np.float32)
    ch_flow_30m = np.where(denom[:, None] > 0, (next_out + prev_in) / np.maximum(denom[:, None], 1.0), 0.0).astype(np.float32)

    return np.stack([
        ch_p2p_rcv_1h,
        ch_dev_5m,
        ch_dev_1h,
        ch_snd_24h,
        ch_pair_15d,
        ch_rev_24h,
        ch_flow_30m,
    ], axis=1)


def apply_propagation(probabilities: np.ndarray, channels: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Combine base probabilities (N, C) with neighbour channels (N, M, C) using alpha (M, C)."""
    boosted = probabilities.astype(np.float64) + np.einsum("nmc,mc->nc", channels, alpha, optimize=True)
    return boosted.astype(np.float32)


def fit_propagation_and_weights(probabilities: np.ndarray, channels: np.ndarray, y_true: np.ndarray,
                                sweeps: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate ascent over per-class neighbour channel alphas and class decision weights."""
    _, n_channels, n_classes = channels.shape
    alpha = np.zeros((n_channels, n_classes), dtype=np.float64)
    weights = fit_class_weights(probabilities, y_true, sweeps=3)
    best_score = macro_f1_ids(y_true, apply_weights(probabilities, weights), n_classes)
    alpha_grid = np.array([0.0, 0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5])

    # Domain-relevant channels per fraud class to avoid spurious combinations and speed up search
    # 1:mule, 2:bot, 3:layer, 4:sim, 5:ato, 6:phish, 7:qr, 8:refund, 9:cashback, 10:loan, 11:collusion
    allowed_channels = {
        1: [0, 6],          # mule_account: p2p_rcv_1h, flow_30m
        2: [1, 2],          # velocity_bot: dev_5m, dev_1h
        3: [6, 0],          # layering_chain: flow_30m, p2p_rcv_1h
        4: [3, 2],          # sim_swap: snd_24h, dev_1h
        5: [3, 2],          # account_takeover: snd_24h, dev_1h
        6: [0, 5],          # phishing_collect: p2p_rcv_1h, rev_24h
        7: [0, 4],          # qr_swap: p2p_rcv_1h, pair_15d
        8: [3, 4],          # refund_abuse: snd_24h, pair_15d
        9: [5, 4],          # fake_cashback: rev_24h, pair_15d
        10: [4, 3],         # loan_app_extortion: pair_15d, snd_24h
        11: [4],            # merchant_collusion: pair_15d
    }

    for _ in range(sweeps):
        for class_id in range(1, n_classes):
            for ch_idx in allowed_channels.get(class_id, []):
                best_a = alpha[ch_idx, class_id]
                for cand_a in alpha_grid:
                    if cand_a == alpha[ch_idx, class_id]:
                        continue
                    trial_alpha = alpha.copy()
                    trial_alpha[ch_idx, class_id] = cand_a
                    prop = apply_propagation(probabilities, channels, trial_alpha)
                    score = macro_f1_ids(y_true, apply_weights(prop, weights), n_classes)
                    if score > best_score + 1e-4:
                        best_score = score
                        best_a = cand_a
                alpha[ch_idx, class_id] = best_a
        prop = apply_propagation(probabilities, channels, alpha)
        weights = fit_class_weights(prop, y_true, sweeps=2)
        best_score = macro_f1_ids(y_true, apply_weights(prop, weights), n_classes)

    return alpha, weights
