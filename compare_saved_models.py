"""Compare saved A/B/C probabilities; tune decisions without training models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from postprocess_saved import LABELS, load_proba, digest, align_ids
from make_submission import (fit_offsets, predict_labels, evaluate_predictions,
                             prepare_rule_context, slice_rule_context, write_submission)

WEIGHTS = {"original": [.5, .3, .2], "A": [1., 0., 0.], "B": [0., 1., 0.],
           "C": [0., 0., 1.], "equal": [1/3] * 3,
           "AB": [.5, .5, 0.], "AC": [.5, 0., .5], "BC": [0., .5, .5]}
BROAD = {"coarse_grid": [-1., 0., .5, 1., 1.5, 2., 2.5, 3.],
         "offset_bound": 3., "conditional": False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_labels(path, ids):
    """Read only ID/label columns, retaining requested validation IDs only."""
    columns = pd.read_csv(path, nrows=0).columns
    id_col = "txn_id" if "txn_id" in columns else "id"
    if id_col not in columns or "label" not in columns:
        raise ValueError("labels CSV requires txn_id (or id) and original label")
    wanted = pd.Index(np.asarray(ids).astype(str))
    parts = []
    for frame in pd.read_csv(path, usecols=[id_col, "label"], dtype={id_col: str, "label": str}, chunksize=250000):
        parts.append(frame.loc[frame[id_col].isin(wanted), [id_col, "label"]])
    found = pd.concat(parts, ignore_index=True)
    if found.isna().any().any() or not found.label.isin(LABELS).all():
        raise ValueError("missing/invalid original validation labels")
    found = found.iloc[align_ids(found[id_col], wanted)]
    return found.label.map({k: i for i, k in enumerate(LABELS)}).to_numpy(dtype=int)


def load_stage(folder, stage, expected_signature=None):
    manifest = read_json(folder / "manifest.json")
    if (manifest.get("stage") != stage or manifest.get("profile") != "ensemble"
            or manifest.get("class_order") != list(LABELS)):
        raise ValueError("expected an ensemble stage manifest with all 12 classes")
    if expected_signature is not None and manifest["signature"] != expected_signature:
        raise ValueError("validation and refit signatures differ")
    meta = pd.read_parquet(folder / "meta.parquet")
    if meta.txn_id.isna().any() or meta.txn_id.duplicated().any():
        raise ValueError("invalid stage IDs")
    probabilities = {}
    artifact_names = {k: f"{k}_proba.npz" for k in "ABC"}
    artifact_names["original"] = "validation_proba.npz" if stage == "validation" else "test_proba.npz"
    for member, filename in artifact_names.items():
        path = folder / filename
        with np.load(path, allow_pickle=False) as saved:
            provenance = json.loads(str(saved["provenance"]))
        for key in ("signature", "stage", "profile", "fit_ids_hash", "sampled_fit_ids_hash"):
            if provenance.get(key) != manifest.get(key):
                raise ValueError(f"{member} probability provenance mismatch: {key}")
        probabilities[member] = load_proba(path, meta.txn_id)
    groups = np.load(folder / "groups.npy", allow_pickle=False)
    if groups.shape != (len(meta),):
        raise ValueError("group/meta row counts differ")
    context = prepare_rule_context(meta, pd.read_parquet(folder / "features_v3.parquet"),
                                   pd.read_parquet(folder / "features_v5.parquet"))
    return manifest, meta, probabilities, groups, context


def mix(probabilities, weights):
    return sum(float(w) * probabilities[k] for k, w in zip("ABC", weights))


def select_policy(probabilities, y, groups, context, config, min_gain=.001, fraud_gate_search=False):
    """Only calibration rows are accepted; assessment labels cannot enter."""
    dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], utc=True))
    if (not len(y) or len(dates) != len(y) or dates.isna().any()
            or len(dates.year.unique()) != 1
            or not ((dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)).all()):
        raise ValueError("selection accepts September16-23 calibration rows only")
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("minimum gain must be finite and nonnegative")
    early, late = dates.day <= 19, dates.day >= 20
    if not early.any() or not late.any():
        raise ValueError("both Sep16-19 and Sep20-23 are required")
    early_ctx, late_ctx = slice_rule_context(context, early), slice_rule_context(context, late)
    specs = [("legacy", WEIGHTS["original"], config)] + [
        (name + "_broad", w, {**config, **BROAD}) for name, w in WEIGHTS.items()]
    if fraud_gate_search:
        for name in ("original", "equal"):
            for threshold in (-.5, 0., .5, 1.):
                specs.append((f"{name}_fraud_gate_{threshold:g}", WEIGHTS[name],
                              {**config, **BROAD, "fraud_gate_log_odds": threshold}))
    trials = []
    selected, best, baseline = None, -1., None
    for number, (name, weights, settings) in enumerate(specs, 1):
        print(f"[{number}/{len(specs)}] Calibrating {name} on Sep16-19...", flush=True)
        p = probabilities.get("original") if name == "legacy" else None
        if p is None:
            p = mix(probabilities, weights)
        offsets = fit_offsets(p[early], y[early], groups[early], early_ctx, config=settings)
        metrics = evaluate_predictions(y[late], predict_labels(p[late], groups[late], offsets, late_ctx))
        score = metrics["macro_f1"]
        trial = {"name": name, "weights": weights, "config": settings,
                 "inner_offsets": offsets, "gate_metrics": metrics}
        trials.append(trial)
        print(f"  Sep20-23 macro-F1: {score:.6f}", flush=True)
        if baseline is None:
            selected, best, baseline = trial, score, score
        elif score > baseline + min_gain and score > best + 1e-12:
            selected, best = trial, score
    return selected, trials


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-dir", type=Path, default=Path("runs/local_podium/ensemble"))
    ap.add_argument("--labels-csv", type=Path, default=Path("../upi-fraud-rings/train.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/saved_model_comparison"))
    ap.add_argument("--min-gain", type=float, default=.001)
    ap.add_argument("--fraud-gate-search", action="store_true",
                    help="also test total-fraud-mass gates on original and equal blends")
    a = ap.parse_args(argv)
    if a.output_dir.exists():
        raise FileExistsError("output folder already exists; choose a new --output-dir")
    run_manifest = read_json(a.profile_dir.parent / "manifest.json")
    expected_hash = run_manifest["input_hashes"]["train.csv"]
    print("Checking original label-file hash...", flush=True)
    actual_hash = digest(a.labels_csv)
    if expected_hash != actual_hash:
        raise ValueError("labels CSV differs from the original run's train.csv")
    val = a.profile_dir / "validation"
    manifest, meta, probabilities, groups, context = load_stage(val, "validation")
    # The comparator is deliberately the old A/B/C run, not a previously
    # selected recall-model blend masquerading as that baseline.
    if manifest["config"].get("recall", {}).get("enabled", False):
        raise ValueError("use the original local_podium A/B/C run, not recall_v7")
    members = manifest["config"]["members"]
    if ([m["name"] for m in members] != list("ABC")
            or not np.allclose([m["weight"] for m in members], WEIGHTS["original"])):
        raise ValueError("expected original A/B/C weights 0.5/0.3/0.2")
    dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], utc=True))
    if dates.isna().any() or len(dates.year.unique()) != 1:
        raise ValueError("invalid saved validation dates")
    cal = (dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)
    assess = (dates.month == 9) & (dates.day >= 24) & (dates.day <= 30)
    if not cal.any() or not assess.any():
        raise ValueError("missing calibration or assessment rows")
    y = read_labels(a.labels_csv, meta.txn_id)
    config = manifest["config"].get("calibration", {})
    selected, trials = select_policy({k: p[cal] for k, p in probabilities.items()}, y[cal], groups[cal],
                                    slice_rule_context(context, cal), config, a.min_gain, a.fraud_gate_search)
    # Freeze the decision before examining any assessment metrics.
    print(f"Selected {selected['name']}; recalibrating on Sep16-23...", flush=True)
    selected_p = probabilities["original"] if selected["name"] == "legacy" else mix(probabilities, selected["weights"])
    offsets = fit_offsets(selected_p[cal], y[cal], groups[cal], slice_rule_context(context, cal),
                          config=selected["config"])
    legacy_p = probabilities["original"]
    legacy_offsets = read_json(val / "offsets.json")
    assessment_context = slice_rule_context(context, assess)
    legacy_metrics = evaluate_predictions(y[assess], predict_labels(
        legacy_p[assess], groups[assess], legacy_offsets, assessment_context))
    # Retain exact saved legacy decisions when the gate rejects all candidates.
    if selected["name"] == "legacy":
        offsets = legacy_offsets
    selected_metrics = evaluate_predictions(y[assess], predict_labels(
        selected_p[assess], groups[assess], offsets, assessment_context))
    report = {"selected": selected["name"], "weights": selected["weights"],
              "selection_dates": {"fit": "Sep16-19", "gate": "Sep20-23"},
              "final_offset_fit_dates": "Sep16-23", "assessment_dates": "Sep24-30",
              "min_gain": a.min_gain, "original_labels": True,
              "assessment_reused_project_data": True, "legacy_assessment": legacy_metrics,
              "selected_assessment": selected_metrics, "trials": trials}
    assessment_rows = []
    for trial in trials:
        p = probabilities["original"] if trial["name"] == "legacy" else mix(probabilities, trial["weights"])
        metrics = evaluate_predictions(y[assess], predict_labels(
            p[assess], groups[assess], trial["inner_offsets"], assessment_context))
        trial["assessment_using_inner_offsets"] = metrics
        assessment_rows.append({"candidate": trial["name"],
                                "gate_macro_f1": trial["gate_metrics"]["macro_f1"],
                                "assessment_with_inner_offsets": metrics["macro_f1"]})
    refit = a.profile_dir / "refit"
    _, test_meta, test_probs, test_groups, test_context = load_stage(refit, "refit", manifest["signature"])
    test_p = test_probs["original"] if selected["name"] == "legacy" else mix(test_probs, selected["weights"])
    pred = predict_labels(test_p, test_groups, offsets, test_context)
    sample = pd.read_csv(refit / "submission.csv", dtype={"id": str})[["id"]]
    test_ids = test_meta.txn_id.to_numpy(dtype=str)
    # Validate all input relationships before creating output files.
    align_ids(test_ids, sample.id)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    result = write_submission(test_ids, pred, sample, a.output_dir / "submission.csv")
    np.savez_compressed(a.output_dir / "test_proba.npz", ids=test_ids, proba=test_p.astype("float32"),
                        classes=np.asarray(LABELS))
    report["submission"] = result
    report["same_submission_as_original"] = result["sha256"] == digest(refit / "submission.csv")
    paths = [a.profile_dir.parent / "manifest.json", val / "offsets.json",
             val / "validation_proba.npz", refit / "test_proba.npz", refit / "submission.csv"]
    for folder in (val, refit):
        paths += [folder / name for name in ("manifest.json", "meta.parquet", "features_v3.parquet",
                                             "features_v5.parquet", "groups.npy", "A_proba.npz",
                                             "B_proba.npz", "C_proba.npz")]
    report["input_hashes"] = {str(path): digest(path) for path in paths}
    report["input_hashes"][str(a.labels_csv)] = actual_hash
    report["source_hashes"] = {str(path): digest(path) for path in
        (Path(__file__), Path(__file__).parent / "postprocess_saved.py",
         Path(__file__).parent / "src/upi/make_submission.py")}
    write_json(a.output_dir / "comparison.json", report)
    write_json(a.output_dir / "offsets.json", offsets)
    pd.DataFrame(assessment_rows).to_csv(a.output_dir / "candidates.csv", index=False)
    per_class = [{"policy": name, "label": lab, **values} for name, metrics in
                 (("legacy", legacy_metrics), ("selected", selected_metrics))
                 for lab, values in metrics["per_class"].items()]
    pd.DataFrame(per_class).to_csv(a.output_dir / "per_class_metrics.csv", index=False)
    lines = ["Saved-model comparison (no model training)",
             f"Selected on Sep20-23: {selected['name']}; A/B/C weights: {selected['weights']}",
             f"Original ensemble assessment macro-F1: {legacy_metrics['macro_f1']:.6f}",
             f"Selected policy assessment macro-F1: {selected_metrics['macro_f1']:.6f}",
             f"Change: {selected_metrics['macro_f1'] - legacy_metrics['macro_f1']:+.6f}",
             f"Submission identical to original: {report['same_submission_as_original']}",
             "Original-label assessment: Sep24-30; already used in this project.",
             "Candidate assessment columns use Sep16-19 offsets; selected headline uses Sep16-23 offsets.",
             "Public score is unknown. A negative assessment change is evidence against submitting this candidate.",
             "", "Selected assessment precision / recall / F1 / support:"]
    for label, values in selected_metrics["per_class"].items():
        lines.append(f"{label}: {values['precision']:.4f} / {values['recall']:.4f} / {values['f1']:.4f} / {values['support']}")
    (a.output_dir / "COMPARISON.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Saved report and candidate: {a.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
