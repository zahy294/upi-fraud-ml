"""Multiclass decision calibration, event-band rules and strict submission output."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .common import LABELS
    from .features_v6 import align_pack, validate_ids
except ImportError:
    from common import LABELS
    from features_v6 import align_pack, validate_ids


def _labels(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu" or ((a < 0) | (a >= 12)).any():
        raise ValueError(f"{name} must be integer labels in 0..11")
    return a.astype(int)


def _proba(values):
    p = np.asarray(values, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 12 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("probabilities must be finite normalized (N,12)")
    return p


def prepare_rule_context(meta, f3, f5):
    validate_ids(meta)
    f3, f5 = align_pack(meta, f3), align_pack(meta, f5)
    b = f3.band.to_numpy()
    cb = f5.ch_band.to_numpy()
    for a in (b, cb):
        if not np.isfinite(a).all() or not np.isin(a, np.arange(12)).all():
            raise ValueError("band labels must be 0..11")
    S = f3[[f"snd_band_{k}_3h" for k in range(1, 12)]].to_numpy(dtype=float)
    R = f3[[f"rcv_band_{k}_24h" for k in range(1, 12)]].to_numpy(dtype=float)
    if not np.isfinite(S).all() or not np.isfinite(R).all() or (S < 0).any() or (R < 0).any():
        raise ValueError("band counts must be finite and nonnegative")
    result = {
        "prepared": True, "txn_id": meta.txn_id.to_numpy(),
        "band": b.astype(int), "chain_band": cb.astype(int),
        "sender_band": np.where((S > 0).sum(axis=1) == 1, S.argmax(axis=1) + 1, 0),
        "receiver_band": np.where((R > 0).sum(axis=1) == 1, R.argmax(axis=1) + 1, 0),
        "p2p": meta.receiver_type.eq("P2P").to_numpy(),
    }
    if "timestamp" in meta:
        result["dates"] = pd.to_datetime(meta.timestamp, utc=True).to_numpy()
    return result


def _context(context):
    if context.get("prepared", False):
        return context
    return prepare_rule_context(context["meta"], context["f3"], context["f5"])


def slice_rule_context(context, mask):
    context = _context(context)
    mask = np.asarray(mask)
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in context.items()}


def _apply_rules(pred, context):
    b, cb = context["band"], context["chain_band"]
    if len(pred) != len(b):
        raise ValueError("prediction/context row counts differ")
    sb, rb, p2p = context["sender_band"], context["receiver_band"], context["p2p"]
    out = np.where(b > 0, b, pred)
    free = (b == 0) & (pred > 0)
    drain = free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    out = np.where(drain, sb, out)
    out = np.where(fanin | shop, rb, out)
    return np.where(free & (cb > 0), cb, out).astype(int)


def apply_band_rules(pred: np.ndarray, meta: pd.DataFrame,
                     f3: pd.DataFrame, f5: pd.DataFrame) -> np.ndarray:
    return _apply_rules(_labels(pred, "pred"), prepare_rule_context(meta, f3, f5))


def event_groups(meta, f3, f5, f2, f4):
    """Disjoint, deterministic priority: band > chain > shop > drain > bait."""
    f3, f5, f2, f4 = [align_pack(meta, f) for f in (f3, f5, f2, f4)]
    return np.select(
        [f3.band.to_numpy() > 0, f5.ch_len.to_numpy() > 1,
         f2.S_flag.to_numpy() == 1, f2.Ds_n.to_numpy() >= 2,
         (f4.is_tiny_p2p_pay.to_numpy() == 1) | np.isfinite(f4.bait_prev_dt_h.to_numpy())],
        ["direct_band", "chain", "merchant_session", "drain", "bait"], default="other")


def evaluate_predictions(y_true, y_pred) -> dict:
    y, pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if len(y) != len(pred):
        raise ValueError("label/prediction lengths differ")
    cm = np.bincount(y * 12 + pred, minlength=144).reshape(12, 12)
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    return {
        "macro_f1": float(f1.mean()), "n": len(y), "empty": len(y) == 0,
        "fraud_prediction_rate": float(np.mean(pred > 0)) if len(pred) else 0.,
        "class_order": list(LABELS), "confusion_matrix": cm.tolist(),
        "per_class": {lab: {"precision": float(precision[k]), "recall": float(recall[k]),
                            "f1": float(f1[k]), "support": int(support[k]),
                            "predicted": int(predicted[k])} for k, lab in enumerate(LABELS)},
    }


def _score(y, pred):
    cm = np.bincount(y * 12 + pred, minlength=144).reshape(12, 12)
    return float(np.mean(2 * np.diag(cm) / np.maximum(cm.sum(axis=0) + cm.sum(axis=1), 1)))


def _decision(logp, groups, offsets, context):
    a = np.asarray(offsets["global"], dtype=float)
    if a.shape != (12,) or not np.isfinite(a).all() or a[0] != 0:
        raise ValueError("invalid global offsets")
    scores = logp + a
    for group, values in offsets.get("conditional", {}).items():
        delta = np.asarray(values, dtype=float)
        if delta.shape != (12,) or not np.isfinite(delta).all() or delta[0] != 0:
            raise ValueError("invalid conditional offsets")
        if group == "direct_band" and np.any(delta != 0):
            raise ValueError("conditional offsets forbidden for direct bands")
        scores[groups == group] += delta
    return _apply_rules(scores.argmax(axis=1), context)


def predict_labels(proba: np.ndarray, groups: np.ndarray,
                   offsets: dict, rule_context: dict) -> np.ndarray:
    p, groups = _proba(proba), np.asarray(groups).astype(str)
    if groups.shape != (len(p),) or offsets.get("class_order") != list(LABELS):
        raise ValueError("group shape or offset class order mismatch")
    return _decision(np.log(np.maximum(p, 1e-9)), groups, offsets, _context(rule_context))


def _optimize(logp, y, groups, context, config, conditional):
    result = {"global": [0.] * 12, "conditional": {}, "class_order": list(LABELS)}
    trace = []

    def objective(offsets):
        pred = _decision(logp, groups, offsets, context)
        a = np.asarray(offsets["global"])
        penalty = 0.002 * np.mean(a[1:] ** 2)
        deltas = np.array(list(offsets["conditional"].values()), dtype=float)
        if deltas.size:
            penalty += 0.01 * np.mean(deltas ** 2)
        return _score(y, pred) - penalty

    best = objective(result)
    for step in config.get("steps", [0.4, 0.2, 0.1, 0.05]):
        for sweep in range(min(int(config.get("max_passes", 3)), 3)):
            changed = False
            for k in range(1, 12):
                old = result["global"][k]
                chosen = old
                for value in sorted({float(np.clip(old - step, -1.5, 1.5)), float(np.clip(old + step, -1.5, 1.5))}):
                    result["global"][k] = value
                    score = objective(result)
                    if score > best + 1e-12 or (abs(score - best) <= 1e-12 and abs(value) < abs(chosen)):
                        chosen, best = value, score
                result["global"][k] = chosen
                changed |= chosen != old
            trace.append({"phase": "global", "step": float(step), "pass": sweep, "objective": float(best)})
            if not changed:
                break
    if conditional:
        for group in sorted(set(groups) - {"direct_band"}):
            mask = groups == group
            n = int(mask.sum())
            if n < 200:
                continue
            result["conditional"][group] = [0.] * 12
            for k in range(1, 12):
                if int(((y == k) & mask).sum()) < 30:
                    continue
                chosen = 0.
                for raw in [-0.25, -0.125, 0., 0.125, 0.25]:
                    value = raw * n / (n + 2000)
                    result["conditional"][group][k] = value
                    score = objective(result)
                    if score > best + 1e-12 or (abs(score - best) <= 1e-12 and abs(value) < abs(chosen)):
                        chosen, best = value, score
                result["conditional"][group][k] = chosen
            trace.append({"phase": "conditional", "group": group, "n": n, "objective": float(best)})
    result["trace"] = trace
    return result


def fit_offsets(proba: np.ndarray, y_raw: np.ndarray, groups: np.ndarray,
                rule_context: dict, *, config: dict) -> dict:
    p, y, groups = _proba(proba), _labels(y_raw, "y_raw"), np.asarray(groups).astype(str)
    context = _context(rule_context)
    if not len(y) or len(p) != len(y) or groups.shape != y.shape:
        raise ValueError("nonempty aligned calibration rows are required")
    if "dates" not in context:
        raise ValueError("calibration requires original timestamps")
    dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], utc=True))
    if len(dates) != len(y) or dates.isna().any() or not ((dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)).all() or len(dates.year.unique()) != 1:
        raise ValueError("offset fitting is restricted to September16-23")
    logp = np.log(np.maximum(p, 1e-9))
    early, late = dates.day <= 19, dates.day >= 20
    gate = {"enabled": False, "global_only": None, "conditional": None}
    if early.any() and late.any() and config.get("conditional", True):
        early_context, late_context = slice_rule_context(context, early), slice_rule_context(context, late)
        global_inner = _optimize(logp[early], y[early], groups[early], early_context, config, False)
        conditional_inner = _optimize(logp[early], y[early], groups[early], early_context, config, True)
        sg = _score(y[late], _decision(logp[late], groups[late], global_inner, late_context))
        sc = _score(y[late], _decision(logp[late], groups[late], conditional_inner, late_context))
        gate = {"enabled": sc > sg + 1e-12, "global_only": sg, "conditional": sc}
    global_full = _optimize(logp, y, groups, context, config, False)
    selected = _optimize(logp, y, groups, context, config, True) if gate["enabled"] else global_full.copy()
    selected["global_only_offsets"] = global_full
    selected["conditional_gate"] = gate
    selected["calibration_rows"] = len(y)
    return selected


def write_submission(test_ids: np.ndarray, pred: np.ndarray,
                     sample: pd.DataFrame, path: Path) -> dict:
    ids = pd.Index(np.asarray(test_ids))
    pred = _labels(pred, "pred")
    if ids.has_duplicates or ids.isna().any() or len(ids) != len(pred):
        raise ValueError("test IDs must be unique, non-null and match predictions")
    if "id" not in sample or sample.id.isna().any() or sample.id.duplicated().any():
        raise ValueError("sample must contain unique non-null IDs")
    if len(ids) != len(sample) or not ids.isin(sample.id).all():
        raise ValueError("sample and prediction ID sets differ")
    frame = pd.DataFrame({"id": ids, "label": np.asarray(LABELS)[pred]})
    frame = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one", sort=False)
    if frame.label.isna().any() or not np.array_equal(frame.id, sample.id):
        raise ValueError("submission ordering or missing label failure")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    counts = {lab: int((frame.label == lab).sum()) for lab in LABELS}
    return {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "predicted_counts": counts, "absent_classes": [lab for lab, n in counts.items() if n == 0],
            "fraud_prediction_rate": float((frame.label != "legit").mean()) if len(frame) else 0.}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="profile directory containing validation/ and refit/")
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)
    # Artifact identity/class checks are shared with the training driver.
    try:
        from .fit_submit import load_probability_artifact
    except ImportError:
        from fit_submit import load_probability_artifact
    refit = a.run_dir / "refit"
    meta = pd.read_parquet(refit / "meta.parquet")
    p = load_probability_artifact(refit / "test_proba.npz", meta.txn_id.to_numpy())
    f3, f5 = pd.read_parquet(refit / "features_v3.parquet"), pd.read_parquet(refit / "features_v5.parquet")
    groups = np.load(refit / "groups.npy", allow_pickle=False)
    offsets = json.loads((a.run_dir / "validation/offsets.json").read_text(encoding="utf-8"))
    pred = predict_labels(p, groups, offsets, {"meta": meta, "f3": f3, "f5": f5})
    result = write_submission(meta.txn_id.to_numpy(), pred, pd.read_csv(a.sample),
                              a.output or refit / "submission.csv")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
