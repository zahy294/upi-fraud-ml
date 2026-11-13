"""Planning-only temporal-flow probe; no model training or test-label inference.

Run: python scripts/research/probe_flow_matches.py
Fixed definitions, evaluated separately within each labelled calendar month.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
          'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
          'fake_cashback', 'loan_app_extortion', 'merchant_collusion']


def summary(mask, labels, target):
    n = int(mask.sum())
    tp = int((mask & (labels == target)).sum())
    support = int((labels == target).sum())
    return {'flagged': n, 'true_positive': tp, 'target_support': support,
            'precision': tp / n if n else 0,
            'recall': tp / support if support else 0,
            'label_counts': {str(k): int(v) for k, v in
                             pd.Series(labels[mask]).value_counts().items()}}


def run_month(d):
    d = d.sort_values('timestamp', kind='stable').reset_index(drop=True)
    # Normalize explicitly: pandas versions can use different datetime units.
    t = pd.to_datetime(d.timestamp).to_numpy(dtype='datetime64[s]').astype('int64')
    t -= t.min()
    snd = d.sender_id.str[2:].astype('int64').to_numpy()
    p2p = d.receiver_type.eq('P2P').to_numpy()
    rcv = d.receiver_id.str[2:].astype('int64').to_numpy()
    amount = d.amount.to_numpy()
    labels = d.label.to_numpy()
    integer = np.isclose(amount, np.round(amount), rtol=0, atol=1e-8)
    masks = {}
    # SCALE exceeds the monthly time span plus every window; group identities
    # are checked explicitly too. ID values are lookup keys, never predictors.
    scale = np.int64(2**32)
    forward = np.zeros((8, len(d)), dtype=bool)
    backward = np.zeros_like(forward)
    for direction in ('forward', 'backward'):
        if direction == 'forward':
            source = np.arange(len(d))
            group, query = snd, rcv
        else:
            source = np.flatnonzero(p2p)
            group, query = rcv[source], snd
        order = source[np.lexsort((t[source], group))]
        groups = snd[order] if direction == 'forward' else rcv[order]
        keys = groups * scale + t[order]
        q = query * scale + t
        base = np.searchsorted(keys, q, side='right' if direction == 'forward' else 'left')
        for j in range(8):
            ix = base + j if direction == 'forward' else base - j - 1
            valid = (ix >= 0) & (ix < len(order))
            safe = np.clip(ix, 0, len(order)-1)
            candidate = order[safe]
            gap = t[candidate] - t if direction == 'forward' else t - t[candidate]
            ratio = amount[candidate] / np.maximum(amount, 1e-9) if direction == 'forward' else amount / np.maximum(amount[candidate], 1e-9)
            valid &= (groups[safe] == query) & (gap > 0) & (gap <= 1800)
            valid &= (ratio >= .8) & (ratio <= .9999)
            if direction == 'forward':
                forward[j] = valid & p2p
            else:
                backward[j] = valid
    eligible = p2p & integer & (amount >= 5000)
    for k in (1, 3, 8):
        masks[f'layer_first_{k}'] = eligible & (forward[:k].any(axis=0) | backward[:k].any(axis=0))
    masks['layer_added_by_8'] = masks['layer_first_8'] & ~masks['layer_first_1']

    # An outgoing mule leg can combine multiple inbound payments. Claim rows
    # are excluded from the input flow because REFUND means refund claim.
    source = np.flatnonzero(p2p & d.txn_type.ne('REFUND').to_numpy())
    order = source[np.lexsort((t[source], rcv[source]))]
    keys = rcv[order] * scale + t[order]
    q = snd * scale + t
    left = np.searchsorted(keys, q-1800, side='left')
    right = np.searchsorted(keys, q, side='left')
    sums = np.r_[0., amount[order].cumsum()]
    incoming_sum = sums[right] - sums[left]
    count = right - left
    ratio = amount / np.maximum(incoming_sum, 1e-9)
    masks['mule_aggregate_outflow'] = eligible & (count >= 3) & (ratio >= .7) & (ratio <= 1.05)
    masks['mule_aggregate_without_nearest_ratio'] = masks['mule_aggregate_outflow'] & ~backward[0]
    results = {}
    for name, mask in masks.items():
        target = 'mule_account' if name.startswith('mule') else 'layering_chain'
        results[name] = summary(mask, labels, target)
    return {'rows': len(d), 'metrics': results}


def main():
    started = time.perf_counter()
    path = ROOT / 'upi-fraud-rings/train.csv'
    cols = ['txn_id','timestamp','sender_id','receiver_id','receiver_type','amount','txn_type','label']
    data = pd.read_csv(path, usecols=cols)
    months = data.timestamp.str[:7]
    result = {'purpose':'planning diagnostic, not trained-model or untouched holdout performance',
              'config':{'forward_seconds':1800,'candidate_limits':[1,3,8],
                        'layer_ratio':[.8,.9999],'minimum_amount':5000,
                        'aggregate_min_count':3,'aggregate_ratio':[.7,1.05],
                        'context':'same calendar month only; edges are censored'},
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
              'train_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
              'script_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),
              'excluded_train_tail_rows':int((months>'2026-09').sum()), 'months':{}}
    for month in ('2026-07','2026-08','2026-09'):
        result['months'][month] = run_month(data.loc[months == month].copy())
    result['elapsed_seconds'] = time.perf_counter()-started
    out = ROOT / 'reports/research/flow_signal_probe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

