"""Label-free composition of nearby transactions; no propagated predictions."""
import numpy as np
import pandas as pd
from .events import Events
from .features_v6 import validate_ids

COLUMNS = ["txn_id", "timestamp", "split", "sender_id", "receiver_id", "receiver_type",
           "device_id", "amount", "balance_before", "new_payee", "device_age_days", "sim_age_hours"]


def build_event_composition(base):
    validate_ids(base)
    base = base.reset_index(drop=True)
    dates = pd.to_datetime(base.timestamp, utc=True, errors="raise")
    if dates.isna().any() or not base.split.isin(["train", "test"]).all():
        raise ValueError("valid timestamps and explicit splits required")
    t = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 10**9
    month = dates.dt.strftime("%Y-%m")
    amount, balance = base.amount.to_numpy(float), base.balance_before.to_numpy(float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("finite nonnegative amounts required")
    # Missing measurements do not count as suspicious or as observed peers.
    signals = {
        "new_payee": np.where(base.new_payee.notna(), base.new_payee.eq(1).astype(float), np.nan),
        "fresh_sim": np.where(base.sim_age_hours.notna(), base.sim_age_hours.le(24).astype(float), np.nan),
        "new_device": np.where(base.device_age_days.notna(), base.device_age_days.le(7).astype(float), np.nan),
        "drain_fraction": np.where(np.isfinite(balance) & (balance > 0), amount / np.maximum(balance, 1e-9), np.nan),
        "log_amount": np.log1p(amount),
    }
    signals["drain_fraction"] = np.clip(signals["drain_fraction"], 0, 5)
    result = {"txn_id": base.txn_id.to_numpy()}
    for name, columns, seconds in (("sender2h", ["sender_id"], 7200),
                                  ("receiver1h", ["receiver_type", "receiver_id"], 3600),
                                  ("device10m", ["device_id"], 600)):
        valid = base[columns].notna().all(axis=1).to_numpy()
        keys = pd.MultiIndex.from_arrays([base.split, month] + [base[c] for c in columns])
        codes = pd.factorize(keys, sort=False)[0].astype(np.int64)
        codes[~valid] = -1
        count = Events(codes, t, mask=valid).count(codes, t, -seconds, seconds) - valid.astype(int)
        result[f"ec_{name}_peers"] = np.maximum(count, 0)
        for signal, values in signals.items():
            observed = valid & np.isfinite(values)
            ev = Events(codes, t, np.nan_to_num(values), mask=observed)
            mass = ev.sum(codes, t, -seconds, seconds) - np.where(observed, values, 0)
            support = ev.count(codes, t, -seconds, seconds) - observed.astype(int)
            mean = np.where(support > 0, np.maximum(mass, 0) / np.maximum(support, 1), np.nan)
            result[f"ec_{name}_{signal}_mean"] = mean
            result[f"ec_{name}_{signal}_support"] = np.maximum(support, 0)
            if signal in {"log_amount", "drain_fraction"}:
                result[f"ec_{name}_{signal}_deviation"] = values - mean
    return pd.DataFrame({k: v if k == "txn_id" else np.asarray(v, dtype="float32")
                         for k, v in result.items()})
