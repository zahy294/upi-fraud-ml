"""Fold-restricted, event-preserving label corrections; no rows are removed."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .common import LABELS
    from .events import Events, MIN, DAY
    from .features_v6 import align_pack, fit_mask, validate_probabilities, entity_codes
except ImportError:
    from common import LABELS
    from events import Events, MIN, DAY
    from features_v6 import align_pack, fit_mask, validate_probabilities, entity_codes


def clean_training_labels(base: pd.DataFrame, teacher_proba: np.ndarray,
                          bands: pd.DataFrame, sessions: pd.DataFrame,
                          fit_ids: pd.Index, *, tau_unrep: float = 0.85) -> pd.DataFrame:
    mask = fit_mask(base, fit_ids)
    p, available = validate_probabilities(teacher_proba, len(base))
    if not 0 <= tau_unrep <= 1:
        raise ValueError("tau_unrep must be within [0,1]")
    band_all = align_pack(base, bands).band
    if band_all.isna().any() or not band_all.isin(range(12)).all():
        raise ValueError("band must be an integer class index 0..11")
    session_all = align_pack(base, sessions)
    b = base.loc[mask].reset_index(drop=True)
    band = band_all.to_numpy(dtype=int)[mask]
    s_flag = session_all.S_flag.to_numpy()[mask] == 1
    p, available = p[mask], available[mask]
    y = b.label.map({lab: i for i, lab in enumerate(LABELS)})
    if y.isna().any():
        raise ValueError("fit labels must belong to the fixed class order")
    y = y.to_numpy(dtype=int)
    if not len(b):
        return pd.DataFrame({"txn_id": b.txn_id, "y_raw": y, "y_clean": y.copy(),
                             "reason": pd.Series(dtype=object),
                             "teacher_available": available,
                             "teacher_fraud_probability": np.empty(0), "direct_band": np.empty(0, dtype=int)})
    pf = np.where(available, 1 - p[:, 0], np.nan)
    top = np.where(available, np.nan_to_num(p[:, 1:], nan=-1).argmax(axis=1) + 1, 0)
    detected = (pf >= 0.5) | (band > 0)
    t = b.t.to_numpy()
    snd, rcv, dev = (entity_codes(b, key) for key in ("snd", "rcv", "dev"))
    is_p2p = b.receiver_type.eq("P2P").to_numpy()
    rcv_evt = np.where(is_p2p | s_flag, rcv, -1)
    pair_keys = list(zip(b.sender_id, b.receiver_type, b.receiver_id))
    pairs, unique = pd.factorize(pd.MultiIndex.from_tuples(pair_keys), sort=False)
    lookup = {key: i for i, key in enumerate(unique)}
    reverse = np.array([lookup.get((r, "P2P", s), -1) if p2p else -1
                        for s, r, p2p in zip(b.sender_id, b.receiver_id, is_p2p)], dtype=np.int64)
    votes = np.zeros((len(b), 12))
    for k in range(1, 12):
        ind = ((y == k) & detected).astype(float)
        if not ind.any():
            continue
        v = (Events(snd, t, ind).sum(snd, t, -7 * DAY, 7 * DAY) - ind
             + Events(rcv_evt, t, ind, mask=rcv_evt >= 0).sum(rcv_evt, t, -DAY, DAY)
             - ind * (rcv_evt >= 0)
             + Events(dev, t, ind).sum(dev, t, -10 * MIN, 10 * MIN) - ind
             + Events(pairs, t, ind).sum(reverse, t, -7 * DAY, 7 * DAY))
        votes[:, k] = np.maximum(v, 0)
    total = votes.sum(axis=1)
    winner = votes.argmax(axis=1)
    event_ok = (votes.max(axis=1) >= 1) & (votes.max(axis=1) / np.maximum(total, 1e-9) >= 0.6)
    event = np.where(event_ok, winner, 0)
    clean = y.copy()
    reason = np.full(len(y), "unchanged", dtype=object)
    unreported = (band == 0) & (y == 0) & available & (pf >= tau_unrep)
    # A missing teacher disables teacher-dependent detection. A known band can
    # still detect a neighbour; direct band correction below remains available.
    changed_type = (band == 0) & (y > 0) & detected & (event > 0) & (event != y)
    band_change = (band > 0) & (band != y)
    clean[unreported] = np.where(event[unreported] > 0, event[unreported], top[unreported])
    reason[unreported] = "unreported"
    clean[changed_type], reason[changed_type] = event[changed_type], "event_type"
    clean[band > 0] = band[band > 0]
    reason[band_change] = "band"
    return pd.DataFrame({"txn_id": b.txn_id.to_numpy(), "y_raw": y, "y_clean": clean,
                         "reason": reason, "teacher_available": available,
                         "teacher_fraud_probability": pf, "direct_band": band})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--tau-unrep", type=float, default=0.85)
    a = ap.parse_args(argv)
    processed = a.root / "data/processed"
    base = pd.read_parquet(processed / "base_v0.parquet")
    train = base.split.eq("train").to_numpy()
    out = a.root / "models" / a.tag
    p = np.full((len(base), 12), np.nan)
    p[train] = np.load(out / "oof.npy")
    result = clean_training_labels(
        base, p, pd.read_parquet(processed / "features_v3.parquet"),
        pd.read_parquet(processed / "features_v2.parquet"), pd.Index(base.txn_id[train]),
        tau_unrep=a.tau_unrep)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_clean.npy", result.y_clean.to_numpy())
    result.to_parquet(out / "cleaning_rows.parquet", index=False)
    print(result.reason.value_counts().to_dict())
    print("Cleaning diagnostics only; validation must use original held-out labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
