"""One-pass, label-free evidence propagation between related transactions.

Unlike teacher features, this operates on the final model probabilities. Each
query excludes itself. Connections are restricted by transaction role, time,
calendar month and split. There is no recursive propagation of inferred labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .events import Events
    from .features_v6 import validate_ids, validate_probabilities
except ImportError:
    from events import Events
    from features_v6 import validate_ids, validate_probabilities


def event_evidence(frame, probabilities, *, seed_threshold=.5):
    """Return (N,11) leave-self-out seed mass per eligible peer, plus counts.

    Only confident class predictions contribute mass, but *all* eligible peers
    enter the denominator. This prevents one seed in a busy merchant account
    from assigning its class to every payment. No labels or trap columns read.
    """
    validate_ids(frame)
    p, available = validate_probabilities(probabilities, len(frame))
    if not available.all() or not .5 <= seed_threshold < 1:
        raise ValueError("complete probabilities and seed threshold in [0.5,1) required")
    dates = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if dates.isna().any() or not frame.split.isin(["train", "test"]).all():
        raise ValueError("valid timestamps and explicit train/test splits required")
    t = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 10**9
    month = dates.dt.strftime("%Y-%m").to_numpy()
    amount = frame.amount.to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("finite nonnegative amounts required")
    p2p = frame.receiver_type.eq("P2P").to_numpy()
    merchant = frame.receiver_type.eq("MERCHANT").to_numpy()
    pay = frame.txn_type.eq("PAY").to_numpy()
    new = frame.new_payee.eq(1).to_numpy()
    integer = amount == np.floor(amount)
    evidence = np.zeros((len(frame), 11), dtype=np.float32)
    coverage = {}

    def rolling(name, columns, eligible, classes, seconds):
        valid = eligible.copy()
        for col in columns:
            valid &= frame[col].notna().to_numpy()
        keys = pd.MultiIndex.from_arrays([frame.split.to_numpy(), month] +
                                        [frame[c].to_numpy() for c in columns])
        codes = pd.factorize(keys, sort=False)[0].astype(np.int64)
        codes[~valid] = -1
        total = Events(codes, t, mask=valid).count(codes, t, -seconds, seconds)
        count = np.maximum(total - valid.astype(int), 0)
        usable = valid & (count > 0)
        coverage[name] = int(usable.sum())
        for k in classes:
            seeds = valid & (p[:, k] >= seed_threshold) & (p.argmax(1) == k)
            mass = Events(codes, t, p[:, k], mask=seeds).sum(codes, t, -seconds, seconds)
            mass -= np.where(seeds, p[:, k], 0.)
            mean = np.where(usable, np.maximum(mass, 0.) / np.maximum(count, 1), 0.)
            evidence[:, k - 1] = np.maximum(evidence[:, k - 1], np.clip(mean, 0., 1.))

    rolling("fanin", ["receiver_type", "receiver_id"], p2p & pay, [1, 3], 3600)
    rolling("device_burst", ["device_id"], merchant & pay & integer & (amount < 1000), [2], 600)
    rolling("drain", ["sender_id"], p2p & pay & new, [4, 5], 7200)
    rolling("merchant_session", ["receiver_type", "receiver_id"],
            merchant & frame.txn_type.isin(["PAY", "QR"]).to_numpy()
            & np.isin(amount, [1000, 2000, 2500, 5000, 10000]), [7, 8, 11], 86400)
    rolling("refund", ["sender_id"], merchant & frame.txn_type.eq("REFUND").to_numpy(), [8], 14*86400)
    rolling("evening", ["sender_id"], pay & new & (dates.dt.hour.to_numpy() >= 20), [10], 7*86400)
    rolling("collect_pair", ["sender_id", "receiver_id"], p2p & frame.txn_type.eq("COLLECT").to_numpy(), [6, 9], 7*86400)
    return evidence, coverage


def propagate(probabilities, evidence, strength):
    """Move bounded probability mass from legit to peer-supported fraud types."""
    p, available = validate_probabilities(probabilities, len(probabilities))
    evidence = np.asarray(evidence, dtype=float)
    if (not available.all() or evidence.shape != (len(p), 11)
            or not np.isfinite(evidence).all() or (evidence < 0).any()
            or (evidence > 1).any() or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("invalid event evidence or strength")
    q = p.copy()
    q[:, 1:] = np.maximum(p[:, 1:], strength * evidence)
    added = (q[:, 1:] - p[:, 1:]).sum(1)
    q[:, 0] = np.maximum(p[:, 0] - added, 1e-9)
    return q / q.sum(1, keepdims=True)
