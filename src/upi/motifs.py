"""Label-free bounded sequence and entity-combination features."""
import numpy as np
import pandas as pd
try:
    from .features_v6 import validate_ids
except ImportError:
    from features_v6 import validate_ids


def build_motifs(base):
    validate_ids(base)
    base = base.reset_index(drop=True)
    dates = pd.to_datetime(base.timestamp, utc=True, errors="raise")
    if dates.isna().any() or not base.split.isin(["train", "test"]).all():
        raise ValueError("motifs require valid timestamps and train/test splits")
    t = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 10**9
    n = len(base)
    month = dates.dt.strftime("%Y-%m").to_numpy()
    amount = base.amount.to_numpy(dtype=float)
    balance = base.balance_before.to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("motifs require finite nonnegative amounts")
    result = {"txn_id": base.txn_id.to_numpy()}
    # Search up to four subsequent eligible transfers, not just the immediate
    # transaction. Missing sender IDs never form a shared anonymous account.
    keys = pd.MultiIndex.from_arrays([base.split, month, base.sender_id])
    code = pd.factorize(keys, sort=False)[0]
    eligible = (base.sender_id.notna().to_numpy() & base.receiver_type.eq("P2P").to_numpy()
                & base.txn_type.eq("PAY").to_numpy() & (amount > 0)
                & np.isfinite(balance) & (balance > 0))
    rows = np.flatnonzero(eligible)
    rows = rows[np.lexsort((base.txn_id.to_numpy()[rows], t[rows], code[rows]))]
    for factor, name in ((.25, "quarter"), (.35, "third")):
        nearest, skipped = np.full(n, np.inf), np.full(n, np.inf)
        count = np.zeros(n)
        for hop in range(1, 5):
            a, b = rows[:-hop], rows[hop:]
            good = (code[a] == code[b]) & (t[b] > t[a]) & (t[b] - t[a] <= 7200)
            a, b = a[good], b[good]
            delta = np.abs(np.log(balance[b] / balance[a]) - np.log(factor))
            target = nearest if hop == 1 else skipped
            np.minimum.at(target, a, delta)
            np.minimum.at(target, b, delta)
            match = delta <= .002
            np.add.at(count, a[match], 1)
            np.add.at(count, b[match], 1)
        result[f"motif_{name}_nearest_residual"] = np.where(np.isfinite(nearest), nearest, np.nan)
        result[f"motif_{name}_skip_residual"] = np.where(np.isfinite(skipped), skipped, np.nan)
        result[f"motif_{name}_matching_links"] = count
    # Short bounded device episodes: timing regularity, not just total volume.
    mask = (base.device_id.notna().to_numpy() & base.receiver_type.eq("MERCHANT").to_numpy()
            & base.txn_type.eq("PAY").to_numpy() & (amount == np.floor(amount)) & (amount < 1000))
    frame = pd.DataFrame({"split": base.split, "month": month, "device": base.device_id,
                          "cell": t // 1800, "t": t, "amount": amount})
    frame = frame.loc[mask].sort_values(["split", "month", "device", "cell", "t"], kind="stable")
    columns = ["split", "month", "device", "cell"]
    group = frame.groupby(columns, sort=False, observed=True)
    frame["gap"] = group.t.diff()
    stats = {"count": group.t.transform("size"),
             "span": group.t.transform("max") - group.t.transform("min"),
             "gap_cv": group.gap.transform("std") / group.gap.transform("mean").replace(0, np.nan),
             "amount_cv": group.amount.transform("std") / group.amount.transform("mean").replace(0, np.nan)}
    for name, values in stats.items():
        out = np.full(n, np.nan)
        out[frame.index] = values
        result[f"motif_device_{name}"] = out
    # Joint account/payee weekly amount context; raw identifiers are not features.
    frame = pd.DataFrame({"split": base.split, "month": month, "sender": base.sender_id,
                          "receiver": base.receiver_id, "type": base.receiver_type,
                          "week": t // (7*86400), "amount": amount})
    group = frame.groupby(["split", "month", "sender", "receiver", "type", "week"],
                          observed=True, sort=False)
    result["motif_pair_week_count"] = group.amount.transform("size").to_numpy()
    result["motif_pair_amount_relative"] = amount / group.amount.transform("mean").replace(0, np.nan).to_numpy()
    result["motif_pair_amount_cv"] = (group.amount.transform("std") /
                                     group.amount.transform("mean").replace(0, np.nan)).to_numpy()
    return pd.DataFrame({k: v if k == "txn_id" else np.asarray(v, dtype=np.float32)
                         for k, v in result.items()})
