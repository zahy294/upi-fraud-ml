"""Reproduce the team's five-model v4 blend and audit bounded tuning policies.

No model training. Input root is the friend's complete artifact bundle, including
the original base row order. Never mix old positional arrays with a rebuilt base.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.upi.common import LABELS, fit_weights
from src.upi.features_v6 import validate_ids
from src.upi.fit_submit import file_hash
from src.upi.make_submission import (prepare_rule_context, slice_rule_context,
    _apply_rules, evaluate_predictions, write_submission)

MEMBERS = {"oof_clean": .35, "oof_v6_lgb": .25, "oof_clean_xgb": .15,
           "oof_v6_xgb": .10, "oof_clean_cat": .15}
CAVEAT = ("Saved leave-one-month-out predictions and shared band features are not "
          "certified forward/fold-safe validation. Cleaned labels are not ground truth. "
          "Public score 0.5411 is user-reported; no new public score is measured.")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def required_paths(root, baseline):
    files = [root / "data/processed" / f for f in
             ("base_v0.parquet", "features_v3.parquet", "features_v5.parquet")]
    files += [root / "data/raw/sample_submission.csv", root / "models/oof_raw/y_clean.npy", baseline]
    files += [root / "models" / tag / name for tag in MEMBERS for name in ("oof.npy", "test_proba.npy")]
    return files


def load_probabilities(path, n):
    p = np.load(path, mmap_mode="r", allow_pickle=False)
    if p.shape != (n, 12) or p.dtype.kind not in "fiu":
        raise ValueError(f"invalid probability shape/dtype: {path}")
    for start in range(0, n, 100000):
        block = p[start:start+100000]
        if (not np.isfinite(block).all() or (block < 0).any() or (block > 1).any()
                or not np.allclose(block.sum(1), 1., atol=1e-5)):
            raise ValueError(f"invalid probabilities: {path}")
    return p


def blend(probabilities, weights):
    if set(weights) != set(MEMBERS) or set(probabilities) != set(MEMBERS):
        raise ValueError("all five named members required")
    w = np.asarray(list(weights.values()), float)
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.):
        raise ValueError("blend weights must be finite, nonnegative and sum to one")
    result = np.zeros(next(iter(probabilities.values())).shape, dtype=np.float64)
    for tag in MEMBERS:
        # Float64 matches load_tag in the friend's master builder.
        result += float(weights[tag]) * np.asarray(probabilities[tag], dtype=np.float64)
    return result


def compare_submission(candidate, baseline):
    for name, frame in (("candidate", candidate), ("baseline", baseline)):
        if (list(frame.columns) != ["id", "label"] or frame.isna().any().any()
                or frame.id.duplicated().any() or not frame.label.isin(LABELS).all()):
            raise ValueError(f"invalid {name} submission")
    if len(candidate) != len(baseline) or set(candidate.id) != set(baseline.id):
        raise ValueError("baseline and candidate ID sets differ")
    old = baseline.set_index("id").loc[candidate.id, "label"].to_numpy()
    return int((candidate.label.to_numpy() != old).sum())


def load_inputs(root):
    base = pd.read_parquet(root / "data/processed/base_v0.parquet",
                           columns=["txn_id", "timestamp", "split", "label", "receiver_type"])
    validate_ids(base)
    if not base.split.isin(["train", "test"]).all():
        raise ValueError("unknown split")
    train = base.split.eq("train").to_numpy()
    if not train.any() or train.all():
        raise ValueError("both train and test required")
    raw = base.loc[train, "label"].map({label: k for k, label in enumerate(LABELS)})
    if raw.isna().any():
        raise ValueError("invalid original labels")
    clean = np.load(root / "models/oof_raw/y_clean.npy", allow_pickle=False)
    if clean.shape != (train.sum(),) or clean.dtype.kind not in "iu" or not np.isin(clean, np.arange(12)).all():
        raise ValueError("invalid positional cleaned labels")
    packs = []
    for name in ("features_v3.parquet", "features_v5.parquet"):
        frame = pd.read_parquet(root / "data/processed" / name)
        validate_ids(frame)
        # The source scripts used positional access, so verify that contract
        # before our helper performs any ID alignment.
        if not np.array_equal(frame.txn_id, base.txn_id):
            raise ValueError(f"original positional feature order differs: {name}")
        packs.append(frame)
    context = prepare_rule_context(base, *packs)
    oof = {tag: load_probabilities(root / "models" / tag / "oof.npy", int(train.sum())) for tag in MEMBERS}
    test = {tag: load_probabilities(root / "models" / tag / "test_proba.npy", int((~train).sum())) for tag in MEMBERS}
    return base, train, raw.to_numpy(dtype=int), clean, context, oof, test


def specifications():
    specs = [("fixed_clean", dict(MEMBERS), "clean"),
             ("fixed_raw", dict(MEMBERS), "raw"),
             ("equal_clean", {tag: .2 for tag in MEMBERS}, "clean")]
    for omitted in MEMBERS:
        weights = {tag: (weight / (1 - MEMBERS[omitted]) if tag != omitted else 0.)
                   for tag, weight in MEMBERS.items()}
        specs.append(("without_" + omitted, weights, "clean"))
    return specs


def tune(probabilities, raw, clean, dates, context, *, min_gain=.002):
    """Freeze selection before evaluating Sep24–30. No test predictions used."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if dates.isna().any() or len(dates.year.unique()) != 1 or not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("invalid tuning calendar or gain margin")
    september = dates.month == 9
    fit, gate, assess = (september & (dates.day <= 15),
                         september & (dates.day >= 16) & (dates.day <= 23),
                         september & (dates.day >= 24) & (dates.day <= 30))
    if not fit.any() or not gate.any() or not assess.any():
        raise ValueError("Sep1–15, Sep16–23 and Sep24–30 all required")
    trials, chosen, best, baseline = [], None, -1., None
    selected_spec = None
    for name, weights, target in specifications():
        p = blend({tag: values[fit | gate] for tag, values in probabilities.items()}, weights)
        fit_sub, gate_sub = fit[fit | gate], gate[fit | gate]
        y_fit = (clean if target == "clean" else raw)[fit]
        decision_weights = fit_weights(p[fit_sub], y_fit)
        prediction = _apply_rules((p[gate_sub] * decision_weights).argmax(1), slice_rule_context(context, gate))
        metrics = evaluate_predictions(raw[gate], prediction)
        score = metrics["macro_f1"]
        trial = {"name": name, "blend": weights, "weight_fit_labels": target,
                 "decision_weights": decision_weights.tolist(), "gate_metrics": metrics}
        trials.append(trial)
        print(f"{name}: Sep16–23 raw macro-F1 {score:.6f}", flush=True)
        if baseline is None or (score > baseline + min_gain and score > best):
            chosen, selected_spec, best = trial, (name, weights, target), score
        if baseline is None:
            baseline = score
    # No assessment label is read until the selection above has been fixed.
    name, weights, target = selected_spec
    calibration = fit | gate
    p = blend(probabilities, weights)
    decision_weights = fit_weights(p[calibration], (clean if target == "clean" else raw)[calibration])
    pred = _apply_rules((p[assess] * decision_weights).argmax(1), slice_rule_context(context, assess))
    baseline_p = blend(probabilities, MEMBERS)
    baseline_w = fit_weights(baseline_p[calibration], clean[calibration])
    baseline_pred = _apply_rules((baseline_p[assess] * baseline_w).argmax(1), slice_rule_context(context, assess))
    return {"caveat": CAVEAT, "selected": name, "selected_blend": weights,
            "weight_fit_labels": target, "decision_weights": decision_weights.tolist(),
            "min_gain": min_gain, "trials": trials,
            "assessment": evaluate_predictions(raw[assess], pred),
            "fixed_blend_assessment": evaluate_predictions(raw[assess], baseline_pred),
            "weight_fit_dates": "Sep1–15", "selection_dates": "Sep16–23",
            "final_weight_fit_dates": "Sep1–23", "assessment_dates": "Sep24–30"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, default=Path("."))
    ap.add_argument("--baseline-csv", type=Path, default=Path("submissions/v4_master_ensemble.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/v4_master_audit"))
    ap.add_argument("--stage", choices=["check", "reproduce", "tune"], default="check")
    a = ap.parse_args(argv)
    root = a.artifact_root.resolve()
    baseline = a.baseline_csv if a.baseline_csv.is_absolute() else root / a.baseline_csv
    files = required_paths(root, baseline)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print("Missing original v4 artifacts; no fitting or tuning performed:")
        print("\n".join(missing))
        return 2
    if a.stage == "check":
        print("All 16 required files present. Reproduction will validate contents and compare submitted labels.")
        return 0
    output = a.output_dir.resolve()
    if output.exists():
        raise FileExistsError("choose a fresh output directory; existing experiments are preserved")
    base, train, raw, clean, context, oof, test = load_inputs(root)
    sample = pd.read_csv(root / "data/raw/sample_submission.csv", dtype={"id": str})
    gold = pd.read_csv(baseline, dtype={"id": str})
    provenance = {"caveat": CAVEAT, "blend": MEMBERS, "class_order": list(LABELS),
                  "positional_arrays": "Original base order supplied by owner; NPY files have no IDs.",
                  "input_hashes": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): file_hash(p) for p in files},
                  "source_hashes": {name: file_hash(Path(__file__).parent / name) for name in
                    ("v4_master.py", "src/upi/common.py", "src/upi/make_submission.py", "src/upi/features_v6.py")}}
    weights = fit_weights(blend(oof, MEMBERS), clean)
    pred = _apply_rules((blend(test, MEMBERS) * weights).argmax(1), slice_rule_context(context, ~train))
    output.mkdir(parents=True)
    submission = write_submission(base.txn_id.to_numpy()[~train], pred, sample, output / "reproduced_v4.csv")
    candidate = pd.read_csv(output / "reproduced_v4.csv", dtype={"id": str})
    differences = compare_submission(candidate, gold)
    provenance.update(decision_weights=weights.tolist(), submission=submission,
                      label_differences_vs_submitted= differences,
                      byte_identical=file_hash(output / "reproduced_v4.csv") == file_hash(baseline))
    write_json(output / "reproduction.json", provenance)
    print(f"ID-aligned differences versus submitted v4: {differences}")
    if differences:
        print("Reproduction mismatch: tuning stopped. Resolve artifact order, settings or rule differences first.")
        return 2
    if a.stage == "tune":
        result = tune(oof, raw, clean, base.loc[train, "timestamp"], slice_rule_context(context, train))
        write_json(output / "tuning.json", result)
        report = result["assessment"]
        lines = ["# Saved v4 probability diagnostic", "", CAVEAT, "",
                 f"Verified submitted labels: zero differences. Selected: {result['selected']}",
                 f"Assessment raw macro-F1: {report['macro_f1']:.6f}",
                 f"Fixed blend assessment: {result['fixed_blend_assessment']['macro_f1']:.6f}", ""]
        lines += [f"- {lab}: precision {v['precision']:.4f}; recall {v['recall']:.4f}; F1 {v['f1']:.4f}; support {v['support']}"
                  for lab, v in report["per_class"].items()]
        lines += ["", "No tuned submission generated. Review upstream fold safety before using an upload."]
        (output / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
