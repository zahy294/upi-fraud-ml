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
                            "missed_as_legit": int(cm[k, 0]) if k else 0,
                            "wrong_fraud_type": int(cm[k, 1:].sum() - tp[k]) if k else 0,
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
    bound = float(config.get("offset_bound", 1.5))
    if not np.isfinite(bound) or bound <= 0 or bound > 6:
        raise ValueError("offset_bound must be in (0,6]")
    grid = [float(v) for v in config.get("coarse_grid", [])]
    if any(not np.isfinite(v) or abs(v) > bound for v in grid):
        raise ValueError("coarse_grid must be finite and within offset_bound")

    # Macro-F1 is piecewise constant. Small local steps can fail to cross any
    # decision boundary, even when a larger jump recovers true fraud cases.
    # Joint fraud shifts also move the legit/fraud boundary without changing
    # the ordering among fraud classes.
    if grid:
        chosen = result["global"].copy()
        for value in sorted(set(grid + [0.]), key=lambda v: (abs(v), v)):
            result["global"] = [0.] + [value] * 11
            score = objective(result)
            if score > best + 1e-12:
                chosen, best = result["global"].copy(), score
        result["global"] = chosen
        trace.append({"phase": "joint_fraud", "objective": float(best)})
        for k in range(1, 12):
            chosen = result["global"][k]
            for value in sorted(set(grid + [chosen]), key=lambda v: (abs(v), v)):
                result["global"][k] = value
                score = objective(result)
                if score > best + 1e-12:
                    chosen, best = value, score
            result["global"][k] = chosen
        trace.append({"phase": "coarse_class", "objective": float(best)})
    for step in config.get("steps", [0.4, 0.2, 0.1, 0.05]):
        for sweep in range(min(int(config.get("max_passes", 3)), 3)):
            changed = False
            for k in range(1, 12):
                old = result["global"][k]
                chosen = old
                for value in sorted({float(np.clip(old - step, -bound, bound)), float(np.clip(old + step, -bound, bound))}):
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


def select_recall_policy(base_proba, recall_proba, y_raw, groups, rule_context, *,
                         baseline_config, config):
    """Select on Sep20-23 after fitting offsets on Sep16-19; never assessment.

    Candidate zero retains the previous optimizer and ensemble. Other
    candidates use broad offset search and optionally a fraud-weighted member.
    Only the selected policy is recalibrated on all Sep16-23 afterwards.
    """
    base, recall = _proba(base_proba), _proba(recall_proba)
    y, groups = _labels(y_raw, "y_raw"), np.asarray(groups).astype(str)
    ctx = _context(rule_context)
    dates = pd.DatetimeIndex(pd.to_datetime(ctx.get("dates", []), utc=True))
    if (base.shape != recall.shape or len(y) != len(base) or groups.shape != y.shape
            or len(dates) != len(y) or dates.isna().any() or len(dates.year.unique()) != 1
            or not ((dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)).all()):
        raise ValueError("recall policy selection requires aligned September16-23 rows")
    alphas = [float(a) for a in config.get("blend_weights", [0., .25, .5])]
    margin = float(config.get("min_gain", .001))
    if (not alphas or len(set(alphas)) != len(alphas)
            or any(not np.isfinite(a) or not 0 <= a <= 1 for a in alphas)
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("invalid recall blend weights or minimum gain")
    enhanced = {**baseline_config, **config.get("calibration", {})}
    early, late = dates.day <= 19, dates.day >= 20
    chosen_alpha, chosen_config, chosen_name = 0., baseline_config, "legacy"
    trials = []
    if early.any() and late.any():
        early_ctx, late_ctx = slice_rule_context(ctx, early), slice_rule_context(ctx, late)
        candidates = [("legacy", 0., baseline_config)] + [
            (f"recall_{a:g}", a, enhanced) for a in alphas]
        incumbent = None
        baseline_score = None
        for name, alpha, calibration_config in candidates:
            probs = (1 - alpha) * base + alpha * recall
            # The existing conditional optimizer cannot pass its own temporal
            # gate inside four days, so inner policies are global-only.
            inner = fit_offsets(probs[early], y[early], groups[early], early_ctx,
                                config=calibration_config)
            pred = predict_labels(probs[late], groups[late], inner, late_ctx)
            report = evaluate_predictions(y[late], pred)
            score = report["macro_f1"]
            trials.append({"name": name, "recall_weight": alpha, "metrics": report})
            if baseline_score is None:
                baseline_score = incumbent = score
            elif score > baseline_score + margin and score > incumbent + 1e-12:
                chosen_alpha, chosen_config, chosen_name = alpha, calibration_config, name
                incumbent = score
    mixed = (1 - chosen_alpha) * base + chosen_alpha * recall
    offsets = fit_offsets(mixed, y, groups, ctx, config=chosen_config)
    selection = {"selected": chosen_name, "recall_weight": chosen_alpha,
                 "min_gain": margin, "fit_days": "Sep16-19", "gate_days": "Sep20-23",
                 "trials": trials, "fallback_reason": None if trials else "missing_inner_partition"}
    return offsets, selection


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
