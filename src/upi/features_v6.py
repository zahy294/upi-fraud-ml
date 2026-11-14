"""Pure, ID-aligned feature fusion and temporally restricted target encoding."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd

try:
    from .common import LABELS, CATS, DROP, TRAPS
    from .events import Events
except ImportError:
    from common import LABELS, CATS, DROP, TRAPS
    from events import Events

EXPECTED_LABELS = [
    "legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap",
    "account_takeover", "phishing_collect", "qr_swap", "refund_abuse",
    "fake_cashback", "loan_app_extortion", "merchant_collusion",
]
if list(LABELS) != EXPECTED_LABELS:
    raise ValueError("common.LABELS differs from the fixed 12-class order")


def validate_ids(frame):
    if "txn_id" not in frame or not frame.columns.is_unique:
        raise ValueError("unique columns including txn_id are required")
    if frame.txn_id.isna().any() or frame.txn_id.duplicated().any():
        raise ValueError("txn_id must be non-null and unique")


def align_pack(base, pack):
    """Exact ID-set validation, one-to-one merge, restore base order."""
    validate_ids(base)
    validate_ids(pack)
    if len(pack) != len(base) or not pd.Index(base.txn_id).isin(pack.txn_id).all():
        raise ValueError("feature pack ID set differs from base")
    return base[["txn_id"]].merge(pack, on="txn_id", how="left",
                                 validate="one_to_one", sort=False).reset_index(drop=True)


def fit_mask(base, fit_ids):
    validate_ids(base)
    ids = pd.Index(fit_ids)
    if ids.has_duplicates or ids.isna().any() or not ids.isin(base.txn_id).all():
        raise ValueError("fit_ids must be unique, non-null known IDs")
    mask = base.txn_id.isin(ids).to_numpy()
    if "split" in base and (base.loc[mask, "split"] != "train").any():
        raise ValueError("fit_ids must contain training rows only")
    return mask


def validate_probabilities(proba, n):
    p = np.asarray(proba, dtype=float)
    if p.shape != (n, 12):
        raise ValueError("teacher_proba must have shape (len(base),12)")
    available = np.isfinite(p).all(axis=1)
    missing = np.isnan(p).all(axis=1)
    if not (available | missing).all():
        raise ValueError("teacher rows must be valid probabilities or all NaN")
    if ((p[available] < 0).any() or (p[available] > 1).any()
            or not np.allclose(p[available].sum(axis=1), 1, atol=1e-5)):
        raise ValueError("invalid teacher probabilities")
    return p, available


def entity_keys(frame, entity):
    if entity == "rcv":
        values = list(zip(frame.receiver_type, frame.receiver_id))
        missing = (frame.receiver_type.isna() | frame.receiver_id.isna()).to_numpy()
    else:
        column = {"snd": "sender_id", "dev": "device_id"}[entity]
        values = frame[column].tolist()
        missing = frame[column].isna().to_numpy()
    return pd.Series([None if bad else value for value, bad in zip(values, missing)],
                     dtype=object)


def entity_codes(frame, entity):
    return pd.factorize(entity_keys(frame, entity), sort=False)[0].astype(np.int64)


def _rolling_distinct(group, time, values, window):
    """Linear sliding counter after sorting; all equal-time rows share a result."""
    order = np.lexsort((time, group))
    out = np.zeros(len(time), dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and group[order[end]] == group[order[start]]:
            end += 1
        if group[order[start]] >= 0:
            counts = Counter()
            left, right = start, start
            while right < end:
                stop = right + 1
                now = time[order[right]]
                while stop < end and time[order[stop]] == now:
                    stop += 1
                while left < right and time[order[left]] < now - window:
                    value = values[order[left]]
                    if value >= 0:
                        counts[value] -= 1
                        if counts[value] == 0:
                            del counts[value]
                    left += 1
                out[order[right:stop]] = len(counts)
                for row in order[right:stop]:
                    value = values[row]
                    if value >= 0:
                        counts[value] += 1
                right = stop
        start = end
    return out


def build_label_free_v6(base: pd.DataFrame, packs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    validate_ids(base)
    time = base.t.to_numpy()
    amount = base.amount.to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("amounts must be finite and nonnegative")
    result = {"txn_id": base.txn_id.to_numpy()}
    codes = {entity: entity_codes(base, entity) for entity in ("snd", "rcv", "dev")}
    for entity, group in codes.items():
        ev = Events(group, time, amount)
        for name, window in (("10m", 600), ("1h", 3600), ("24h", 86400)):
            result[f"v6_{entity}_n_{name}"] = ev.count(group, time, -window, -1)
            result[f"v6_{entity}_amt_sum_{name}"] = ev.sum(group, time, -window, -1)
        if entity != "rcv":
            background = ev.count(group, time, -21600, -601)
            mu = background * 600 / 21000
            result[f"v6_{entity}_velocity_z"] = np.clip(
                (result[f"v6_{entity}_n_10m"] - mu) / np.sqrt(mu + 1), -20, 20)
    result["v6_dev_unique_senders_24h"] = _rolling_distinct(
        codes["dev"], time, codes["snd"], 86400)
    if "v4" not in packs:
        raise ValueError("packs['v4'] with bait timing features is required")
    f4 = align_pack(base, packs["v4"])
    bait = f4.bait_prev_dt_h.to_numpy(dtype=float)
    tiny = f4.pair_tiny_prev_dt_h.to_numpy(dtype=float)
    count = f4.bait_cnt_before_1d.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["v6_bait_delay_fraction"] = bait / (bait + tiny + 1 / 3600)
        result["v6_bait_recency_density"] = count / (1 + bait)
    result["v6_bait_prev_missing"] = np.isnan(bait)
    result["v6_pair_tiny_prev_missing"] = np.isnan(tiny)
    out = pd.DataFrame(result)
    columns = out.columns.drop("txn_id")
    out[columns] = out[columns].astype("float32")
    if np.isinf(out[columns].to_numpy()).any():
        raise ValueError("infinite v6 feature")
    return out


def _months(frame):
    if frame.month.isna().any():
        raise ValueError("month cannot be missing")
    return pd.PeriodIndex(frame.month.astype(str), freq="M")


def _times(frame):
    if "timestamp" in frame:
        t = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if t.isna().any():
            raise ValueError("timestamps cannot be missing")
        return t
    if "t" not in frame or not np.isfinite(frame.t.to_numpy(dtype=float)).all():
        raise ValueError("frozen TE requires timestamp or finite base-convention t")
    return frame.t


def build_temporal_te(query: pd.DataFrame, reference: pd.DataFrame, *,
                      alpha: float = 50.0, expanding_months: bool = True) -> pd.DataFrame:
    validate_ids(query)
    validate_ids(reference)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    qm, rm = _months(query), _months(reference)
    if not expanding_months and len(query) and len(reference):
        if _times(reference).max() >= _times(query).min() or rm.max() > qm.min():
            raise ValueError("frozen reference must strictly precede queries")
    out = {"txn_id": query.txn_id.to_numpy()}
    for entity in ("snd", "rcv", "dev"):
        for k in range(12):
            out[f"v6_te_{entity}_{k}"] = np.full(len(query), 1 / 12, dtype=np.float32)
        out[f"v6_te_{entity}_log_support"] = np.zeros(len(query), dtype=np.float32)
    for month in qm.unique():
        qmask = np.asarray(qm == month)
        rmask = np.asarray(rm < month) if expanding_months else np.ones(len(reference), bool)
        ref = reference.loc[rmask].reset_index(drop=True)
        # Only eligible labels are read; future/held-out labels are untouched.
        y = ref.label.map({label: k for k, label in enumerate(LABELS)})
        if y.isna().any():
            raise ValueError("eligible reference has invalid original labels")
        y = y.to_numpy(dtype=int)
        prior = (np.bincount(y, minlength=12) + 1) / (len(y) + 12)
        for entity in ("snd", "rcv", "dev"):
            rkeys = entity_keys(ref, entity)
            qkeys = entity_keys(query.loc[qmask], entity)
            counts = {}
            for key, label in zip(rkeys, y):
                if key is not None:
                    if key not in counts:
                        counts[key] = np.zeros(12)
                    counts[key][label] += 1
            matrix = np.array([counts.get(key, np.zeros(12)) for key in qkeys]).reshape(-1, 12)
            support = matrix.sum(axis=1)
            encoded = (matrix + alpha * prior) / (support[:, None] + alpha)
            for k in range(12):
                out[f"v6_te_{entity}_{k}"][qmask] = encoded[:, k]
            out[f"v6_te_{entity}_log_support"][qmask] = np.log1p(support)
    return pd.DataFrame(out)


def assemble_features(base: pd.DataFrame, packs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    validate_ids(base)
    combined = base.reset_index(drop=True).copy()
    for name, pack in packs.items():
        aligned = align_pack(base, pack)
        for col in aligned.columns.drop("txn_id"):
            if col in combined:
                if not combined[col].equals(aligned[col]):
                    # Numeric dtype differences alone do not make values conflict.
                    equal = (combined[col].eq(aligned[col]) | (combined[col].isna() & aligned[col].isna())).all()
                    if not equal:
                        raise ValueError(f"conflicting feature {col} from {name}")
            else:
                combined[col] = aligned[col]
    excluded = set(DROP) | set(TRAPS) | {"y_raw", "y_clean", "reason", "teacher_available"}
    columns = [c for c in combined if c not in excluded]
    mappings = {}
    X = pd.DataFrame(index=combined.index)
    for col in columns:
        if col in CATS:
            levels = sorted(combined[col].dropna().astype(str).unique().tolist())
            mappings[col] = levels
            X[col] = combined[col].astype("string").map({v: i for i, v in enumerate(levels)}).fillna(-1).astype("int32")
        else:
            if not pd.api.types.is_numeric_dtype(combined[col]):
                raise ValueError(f"unexpected nonnumeric predictor: {col}")
            X[col] = combined[col].astype("float32")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("infinite assembled feature")
    schema = {"columns": columns, "categorical_mappings": mappings,
              "unknown_category_code": -1, "class_order": list(LABELS),
              "row_ids": base.txn_id.tolist()}
    return X, schema


def apply_feature_schema(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Encode a raw predictor frame with a previously persisted mapping."""
    result = pd.DataFrame(index=frame.index)
    for col in schema["columns"]:
        if col in schema["categorical_mappings"]:
            levels = schema["categorical_mappings"][col]
            result[col] = frame[col].astype("string").map({v: i for i, v in enumerate(levels)}).fillna(-1).astype("int32")
        else:
            result[col] = frame[col].astype("float32")
    return result
