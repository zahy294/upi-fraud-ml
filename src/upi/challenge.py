"""Training-only robustness and fold-safe peer evidence for the challenge run."""
from copy import copy
import numpy as np
import pandas as pd
try:
    from .features_v6 import validate_ids, validate_probabilities, entity_keys
except ImportError:
    from features_v6 import validate_ids, validate_probabilities, entity_keys


def stopping_metric(name, classes):
    classes = np.asarray(classes, dtype=int)
    if name not in {"macro_f1", "balanced_logloss"}:
        raise ValueError("unknown custom stopping metric")
    def metric(y_encoded, probability):
        y = np.asarray(y_encoded, dtype=int)
        p = np.asarray(probability)
        if p.shape != (len(y), len(classes)):
            raise ValueError("custom metric expects 2D multiclass probabilities")
        if name == "balanced_logloss":
            losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.))
            counts = np.bincount(y, minlength=len(classes))
            totals = np.bincount(y, weights=losses, minlength=len(classes))
            return name, float(np.mean(totals[counts > 0] / counts[counts > 0])), False
        actual, pred = classes[y], classes[p.argmax(axis=1)]
        cm = np.bincount(actual * 12 + pred, minlength=144).reshape(12, 12)
        score = np.mean(2 * np.diag(cm) / np.maximum(cm.sum(axis=0) + cm.sum(axis=1), 1))
        return name, float(score), True
    return metric


def warmup_stopping(callback, minimum_rounds):
    """Initialize LightGBM early stopping at the minimum eligible round."""
    if not isinstance(minimum_rounds, int) or minimum_rounds < 1:
        raise ValueError("minimum_rounds must be a positive integer")
    def wrapped(env):
        start = env.begin_iteration + minimum_rounds - 1
        if env.iteration < start:
            return
        adjusted = copy(env)
        adjusted.begin_iteration = start
        callback(adjusted)
    wrapped.order = 30
    wrapped.before_iteration = False
    return wrapped


def reliability_weights(cleaned, *, suspected_weight=.35, promoted_weight=.6):
    """Downweight only eligible fitting rows; never alter held-out labels."""
    if not 0 < suspected_weight <= 1 or not 0 < promoted_weight <= 1:
        raise ValueError("reliability weights must be in (0,1]")
    pf = cleaned.teacher_fraud_probability.to_numpy(dtype=float)
    raw, target = cleaned.y_raw.to_numpy(), cleaned.y_clean.to_numpy()
    suspicious = ((raw > 0) & (target == raw) & cleaned.reason.eq("unchanged").to_numpy()
                  & (cleaned.direct_band.to_numpy() == 0) & np.isfinite(pf) & (pf < .1))
    weights = np.ones(len(cleaned))
    weights[suspicious] = suspected_weight
    weights[cleaned.reason.eq("unreported").to_numpy()] = promoted_weight
    return weights


def build_peer_features(base, teacher):
    """Leave-self-out teacher means within entity/time cells, never across months.

    Teacher predictions must be forward-fold for train rows. Grouping by split
    prevents test teachers trained on all labels from feeding training rows.
    Unavailable teacher queries (the July warmup) always have missing evidence.
    """
    validate_ids(base)
    p, available = validate_probabilities(teacher, len(base))
    dates = pd.to_datetime(base.timestamp, utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("peer features require valid timestamps")
    if not base.split.isin(["train", "test"]).all():
        raise ValueError("peer features require explicit train/test split")
    result = {"txn_id": base.txn_id.to_numpy()}
    epoch_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    months = dates.dt.strftime("%Y-%m")
    for entity, seconds in (("snd", 3600), ("rcv", 3600), ("dev", 600)):
        keys = entity_keys(base, entity)
        valid = available & keys.notna().to_numpy()
        # Epoch cells plus month/split isolate teacher provenance boundaries.
        time_cell = epoch_ns // (seconds * 1_000_000_000)
        index = pd.MultiIndex.from_arrays([base.split.to_numpy(), months, keys, time_cell])
        codes, unique = pd.factorize(index, sort=False)
        count = np.bincount(codes[valid], minlength=len(unique))
        peer_count = count[codes] - valid.astype(int)
        result[f"peer_{entity}_count"] = np.where(valid, peer_count, 0).astype("float32")
        for k in range(1, 12):
            sums = np.bincount(codes[valid], weights=p[valid, k], minlength=len(unique))
            own = np.where(valid, np.nan_to_num(p[:, k]), 0.)
            means = (sums[codes] - own) / np.maximum(peer_count, 1)
            result[f"peer_{entity}_class_{k}"] = np.where(valid & (peer_count > 0),
                                                         np.clip(means, 0., 1.), np.nan).astype("float32")
    return pd.DataFrame(result)
