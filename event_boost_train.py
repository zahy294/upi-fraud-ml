"""One-model LightGBM ablation of the 39 cached event-composition features.

check is read-only. validate fits one new model; refit requires corroborating
validation evidence. Existing hierarchy/forest runs are immutable inputs.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from hierarchical_train import read, write, check_inputs, prepare, verify_complete
from src.upi.common import LABELS
from src.upi.fit_submit import (file_hash, ids_hash, train_member, model_proba,
    load_probability_artifact, save_probability_artifact)
from src.upi.make_submission import (prepare_rule_context, slice_rule_context, fit_offsets,
    predict_labels, evaluate_predictions, write_submission)

MIN_GAIN = .001


def preflight(root, source, features, teacher, reference):
    sm = read(source / "manifest.json")
    verify_complete(source / "validation", sm["signature"])
    ready = read(source / "validation/cache/ready.json")
    for name, sha in ready["hashes"].items():
        if file_hash(source / "validation/cache" / name) != sha:
            raise ValueError(f"hierarchy cache changed: {name}")
    fm = read(features / "manifest.json")
    feature_hash = read(features / "features.json")
    if (fm["source_manifest"] != file_hash(source / "manifest.json") or fm["source_cache"] != ready
            or file_hash(features / "event_composition.parquet") != feature_hash["sha256"]
            or file_hash(root / "src/upi/event_composition.py") != fm["code"]["src/upi/event_composition.py"]):
        raise ValueError("event features were built from different source data/code")
    inputs, current = check_inputs(root, teacher, sm["config"])
    if inputs["signature"] != sm["source_signature"] or current["source_artifacts"] != sm["source_artifacts"]:
        raise ValueError("cached teachers or fold contexts differ")
    ref = read(reference / "comparison.json")
    reference_f1 = ref["selected_assessment"]["macro_f1"]
    if ref["selected"] != "equal_fraud_gate_-0.5" or abs(reference_f1 - .520807) > .000001:
        raise ValueError("expected the recorded strongest local reference")
    manifest = {"source_manifest": file_hash(source / "manifest.json"), "cache": ready,
        "features": feature_hash, "teacher_signature": inputs["signature"],
        "reference_hash": file_hash(reference / "comparison.json"), "reference_f1": reference_f1,
        "config": sm["config"], "min_gain": MIN_GAIN,
        "environment": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "lightgbm")},
        "python": platform.python_version(), "source_hashes": {name: file_hash(root / name) for name in
            ("event_boost_train.py", "hierarchical_train.py", "src/upi/fit_submit.py",
             "src/upi/make_submission.py", "src/upi/event_composition.py")}}
    manifest["signature"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return sm, inputs, manifest


def append_features(X, rows, features):
    if len(X) != len(rows) or rows.txn_id.isna().any() or rows.txn_id.duplicated().any():
        raise ValueError("invalid cached rows/matrix alignment")
    if features.txn_id.isna().any() or features.txn_id.duplicated().any():
        raise ValueError("event feature IDs must be unique and nonmissing")
    extra_columns = [name for name in features.columns if name != "txn_id"]
    if len(extra_columns) != 39 or set(extra_columns) & set(X.columns):
        raise ValueError("expected 39 new event columns without duplicate feature names")
    if not rows.txn_id.isin(features.txn_id).all():
        raise ValueError("missing event feature IDs")
    extra = features.set_index("txn_id").loc[rows.txn_id].reset_index(drop=True)
    return pd.concat([X.reset_index(drop=True), extra], axis=1)


def compare_candidate(p, baseline, y, rows, groups, context, settings, baseline_offsets, reference_f1):
    """Choose on Sep20-23; evaluate on Sep24-30 only after freezing selection."""
    dates = pd.DatetimeIndex(pd.to_datetime(rows.timestamp, utc=True))
    early = (dates.month == 9) & (dates.day >= 16) & (dates.day <= 19)
    gate = (dates.month == 9) & (dates.day >= 20) & (dates.day <= 23)
    cal, assess = early | gate, (dates.month == 9) & (dates.day >= 24) & (dates.day <= 30)
    if not all(mask.any() for mask in (early, gate, assess)):
        raise ValueError("missing chronological calibration, gate or assessment block")
    candidates = {}
    for name, probabilities in (("baseline", baseline), ("event_features", p)):
        inner = fit_offsets(probabilities[early], y[early], groups[early],
                            slice_rule_context(context, early), config=settings)
        candidates[name] = {"gate": evaluate_predictions(y[gate], predict_labels(probabilities[gate],
            groups[gate], inner, slice_rule_context(context, gate))), "inner_offsets": inner}
    chosen = ("event_features" if candidates["event_features"]["gate"]["macro_f1"] >
              candidates["baseline"]["gate"]["macro_f1"] + MIN_GAIN else "baseline")
    offsets = fit_offsets(p[cal], y[cal], groups[cal], slice_rule_context(context, cal), config=settings)
    for name, probabilities, policy in (("baseline", baseline, baseline_offsets), ("event_features", p, offsets)):
        candidates[name]["assessment"] = evaluate_predictions(y[assess], predict_labels(
            probabilities[assess], groups[assess], policy, slice_rule_context(context, assess)))
    score = candidates["event_features"]["assessment"]["macro_f1"]
    baseline_score = candidates["baseline"]["assessment"]["macro_f1"]
    eligible = chosen == "event_features" and score > max(reference_f1, baseline_score) + MIN_GAIN
    return offsets, {"selected_on_gate": chosen, "candidates": candidates, "reference_f1": reference_f1,
        "eligible_for_refit": eligible, "original_labels": True, "assessment_reused_project_data": True,
        "public_score": None, "min_gain": MIN_GAIN,
        "dates": {"early_stop": "Sep1-15", "calibration": "Sep16-19", "gate": "Sep20-23", "assessment": "Sep24-30"}}


def train_stage(root, source, feature_run, out, manifest, stage):
    signature = manifest["signature"]
    folder = out / stage
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        verify_complete(folder, signature)
        print(f"{stage} already complete; outputs verified", flush=True)
        return
    validation = stage == "validation"
    cache = source / "validation/cache" if validation else folder / "cache"
    rows = pd.read_parquet(cache / "rows.parquet")
    X = append_features(pd.read_parquet(cache / "X.parquet"), rows,
                        pd.read_parquet(feature_run / "event_composition.parquet"))
    fit, query = rows["fit"].to_numpy(), rows["query"].to_numpy()
    stop = rows.early_stop.to_numpy() if validation else np.zeros(len(rows), dtype=bool)
    if (fit & query).any() or not fit.any() or not query.any() or (validation and not stop.any()):
        raise ValueError("invalid fitting/query/stopping masks")
    if validation:
        dates = pd.DatetimeIndex(pd.to_datetime(rows.timestamp, utc=True))
        if not np.isin(dates.month[fit], [7, 8]).all() or not (
                (dates.month[stop] == 9) & (dates.day[stop] <= 15)).all():
            raise ValueError("fitting/stopping masks are not chronological")
    schema = read(cache / "schema.json")
    config = manifest["config"]
    spec = {**config["model_defaults"], **config["flat"],
            "categorical_columns": list(schema["categorical_mappings"])}
    rounds = None if validation else read(out / "validation/model.json")["rounds"]
    y = rows.y.to_numpy(dtype=int)
    print(f"Fitting one event-feature LightGBM ({stage}): {fit.sum():,} rows, {X.shape[1]} features", flush=True)
    model, best = train_member(X.loc[fit], y[fit], rows.weight.to_numpy()[fit],
        X.loc[stop] if validation else None, y[stop] if validation else None,
        spec=spec, device="cpu", fixed_rounds=rounds)
    model.booster_.save_model(str(folder / "model.txt"))
    write(folder / "model.json", {"rounds": best, "spec": spec, "columns": X.columns.tolist(),
                                  "classes": model.podium_classes_.tolist()})
    p = model_proba(model, X.loc[query])
    del X, model
    gc.collect()
    qrows = rows.loc[query].reset_index(drop=True)
    groups = np.load(cache / "groups.npy", allow_pickle=False)[query]
    ctx = prepare_rule_context(qrows[["txn_id", "timestamp", "receiver_type"]],
        pd.read_parquet(cache / "v3.parquet").loc[query].reset_index(drop=True),
        pd.read_parquet(cache / "v5.parquet").loc[query].reset_index(drop=True))
    save_probability_artifact(folder / "probabilities.npz", qrows.txn_id, p,
        provenance={"signature": signature, "stage": stage, "fit_ids_hash": ids_hash(rows.loc[fit, "txn_id"])})
    if validation:
        source_signature = read(source / "manifest.json")["signature"]
        baseline = load_probability_artifact(source / "validation/flat_proba.npz", qrows.txn_id,
            expected_provenance={"signature": source_signature, "stage": "validation", "class_order": list(LABELS),
                                 "fit_ids_hash": ids_hash(rows.loc[fit, "txn_id"])})
        offsets, report = compare_candidate(p, baseline, qrows.y.to_numpy(dtype=int), qrows, groups, ctx,
            config["calibration"], read(source / "validation/baseline_offsets.json"), manifest["reference_f1"])
        source_metrics = read(source / "validation/metrics.json")["partitions"]["assessment"]["flat_baseline"]
        if abs(report["candidates"]["baseline"]["assessment"]["macro_f1"] - source_metrics["macro_f1"]) > 1e-12:
            raise ValueError("same-feature source baseline failed to reproduce")
        write(folder / "offsets.json", offsets)
        write(folder / "comparison.json", report)
        score = report["candidates"]["event_features"]["assessment"]
        base_score = report["candidates"]["baseline"]["assessment"]["macro_f1"]
        lines = ["Event-feature LightGBM ablation", f"Selected on gate: {report['selected_on_gate']}",
            f"Unchanged-feature LightGBM assessment: {base_score:.6f}",
            f"Event-feature LightGBM assessment: {score['macro_f1']:.6f}",
            f"Best saved local reference: {manifest['reference_f1']:.6f}",
            f"Eligible for refit: {report['eligible_for_refit']}",
            "Original noisy labels; assessment is reused project data. Public score unknown.", "",
            "Class: precision / recall / F1 / support"]
        lines += [f"{label}: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f} / {v['support']}"
                  for label, v in score["per_class"].items()]
        (out / "RUN_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
    else:
        offsets = read(out / "validation/offsets.json")
        pred = predict_labels(p, groups, offsets, ctx)
        sample = pd.read_csv(root / "data/raw/sample_submission.csv", dtype={"id": str})
        result = write_submission(qrows.txn_id.to_numpy(dtype=str), pred, sample, folder / "submission.csv")
        restored = pd.read_csv(folder / "submission.csv", dtype={"id": str})
        expected = pd.Series(np.asarray(LABELS)[pred], index=qrows.txn_id.astype(str)).loc[sample.id].to_numpy()
        if restored.id.tolist() != sample.id.tolist() or not np.array_equal(restored.label, expected):
            raise ValueError("submission CSV read-back mismatch")
        write(folder / "submission_report.json", result)
        print(f"Submission saved: {folder / 'submission.csv'}", flush=True)
    write(folder / "complete.json", {"signature": signature, "hashes": {
        p.name: file_hash(p) for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--source-run", type=Path, default=Path("runs/hierarchy_v9"))
    ap.add_argument("--feature-run", type=Path, default=Path("runs/forest_v10"))
    ap.add_argument("--teacher-run", type=Path, default=Path("runs/challenge_v8"))
    ap.add_argument("--reference-dir", type=Path, default=Path("runs/fraud_gate_comparison"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/event_boost_v11"))
    ap.add_argument("--stage", choices=["check", "validate", "refit"], default="check")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    source, features, teacher, reference, out = [p.resolve() if p.is_absolute() else (root/p).resolve()
        for p in (a.source_run, a.feature_run, a.teacher_run, a.reference_dir, a.output_dir)]
    if any(out == p or out in p.parents or p in out.parents for p in (source, features, teacher, reference)):
        raise ValueError("experiment output must be separate from source runs")
    sm, inputs, manifest = preflight(root, source, features, teacher, reference)
    if a.stage == "check":
        print("Preflight passed. Cached features, baseline, teachers and checksums verified. No model fitted.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read(out / "manifest.json")["signature"] != manifest["signature"]:
        raise ValueError("inputs/code/config/environment changed; choose a new output directory")
    write(out / "manifest.json", manifest)
    stage = "refit" if a.stage == "refit" else "validation"
    if stage == "refit":
        verify_complete(out / "validation", manifest["signature"])
        if not read(out / "validation/comparison.json")["eligible_for_refit"]:
            raise ValueError("validation did not corroborate an improvement; do not refit this candidate")
        schema_dir = out / "validation/cache"
        schema_dir.mkdir(exist_ok=True)
        (schema_dir / "schema.json").write_bytes((source / "validation/cache/schema.json").read_bytes())
        prepare(root, teacher, out, sm["config"], inputs, "refit")
    train_stage(root, source, features, out, manifest, stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
