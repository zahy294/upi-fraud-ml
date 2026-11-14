"""Bounded comparison of saved models and adaptive class decisions. No training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from compare_saved_models import (load_stage, read_labels, read_json, write_json, mix)
from postprocess_saved import digest, load_proba, align_ids
from src.upi.common import LABELS
from src.upi.fit_submit import ids_hash
from src.upi.boundary_calibration import fit_boundary_offsets
from src.upi.make_submission import (predict_labels, evaluate_predictions, slice_rule_context,
                                     write_submission, fit_offsets)

GATE_CONFIG = {"steps": [.4, .2, .1, .05], "max_passes": 3, "conditional": False,
               "coarse_grid": [-1., 0., .5, 1., 1.5, 2., 2.5, 3.], "offset_bound": 3.,
               "fraud_gate_log_odds": -.5}


def candidate_specs(mode="boundaries"):
    # Declared before looking at assessment outcomes. Do not expand this list
    # based on assessment results: the project has already reused that week.
    if mode not in ("boundaries", "gate"):
        raise ValueError("unknown comparison mode")
    specs = [{"name": "reference", "member": "reference", "share": 0., "penalty": None}]
    if mode == "boundaries":
        specs += [{"name": "reference_boundaries", "member": "reference", "share": 0., "penalty": .002},
                  {"name": "reference_unpenalized", "member": "reference", "share": 0., "penalty": 0.}]
    for member in ("forest", "hierarchy"):
        for share in (.1, .25, .5):
            specs.append({"name": f"{member}_{share:g}", "member": member,
                          "share": share, "penalty": .002})
    return [{**s, "calibration": mode} for s in specs]


def blend(probabilities, spec):
    share = spec["share"]
    return probabilities["reference"] if share == 0 else (
        (1 - share) * probabilities["reference"] + share * probabilities[spec["member"]])


def calibrate(p, y, groups, context, spec):
    if spec["calibration"] == "gate":
        return fit_offsets(p, y, groups, context, config=GATE_CONFIG)
    return fit_boundary_offsets(p, y, context, penalty=spec["penalty"])


def select(probabilities, y, groups, context, reference_inner, min_gain=.001, mode="boundaries"):
    """Accept calibration rows only; later assessment labels never enter selection."""
    dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], utc=True))
    if (not len(y) or len(dates) != len(y) or dates.isna().any()
            or len(dates.year.unique()) != 1
            or not ((dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)).all()):
        raise ValueError("selection requires September16-23 rows only")
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("minimum gain must be finite and nonnegative")
    early, gate = dates.day <= 19, dates.day >= 20
    if not early.any() or not gate.any():
        raise ValueError("both calibration and selection blocks are required")
    early_context, gate_context = slice_rule_context(context, early), slice_rule_context(context, gate)
    trials, selected, baseline = [], None, None
    for spec in candidate_specs(mode):
        p = blend(probabilities, spec)
        offsets = reference_inner if spec["name"] == "reference" else calibrate(
            p[early], y[early], groups[early], early_context, spec)
        metrics = evaluate_predictions(y[gate], predict_labels(p[gate], groups[gate], offsets, gate_context))
        trial = {**spec, "inner_offsets": offsets, "gate_metrics": metrics}
        trials.append(trial)
        score = metrics["macro_f1"]
        print(f"{spec['name']}: Sep20-23 macro-F1 {score:.6f}", flush=True)
        if baseline is None:
            baseline, selected = score, trial
        elif score > baseline + min_gain and score > selected["gate_metrics"]["macro_f1"] + 1e-12:
            selected = trial
    return selected, trials


def load_extra(run, filename, expected_ids, cache, expected_fit_hash):
    manifest, complete = read_json(run / "manifest.json"), read_json(run / "validation/complete.json")
    path = run / "validation" / filename
    if complete["signature"] != manifest["signature"] or digest(path) != complete["hashes"][filename]:
        raise ValueError(f"saved probability checksum/signature mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        provenance = json.loads(str(saved["provenance"]))
    if (provenance.get("signature") != manifest["signature"] or provenance.get("stage") != "validation"
            or provenance.get("fit_ids_hash") != expected_fit_hash):
        raise ValueError(f"saved probability provenance mismatch: {path}")
    ready = read_json(cache / "ready.json")
    if digest(cache / "rows.parquet") != ready["hashes"]["rows.parquet"]:
        raise ValueError("validation row cache checksum mismatch")
    return load_proba(path, expected_ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-dir", type=Path, default=Path("runs/local_podium/ensemble"))
    ap.add_argument("--reference-dir", type=Path, default=Path("runs/fraud_gate_comparison"))
    ap.add_argument("--hierarchy-dir", type=Path, default=Path("runs/hierarchy_v9"))
    ap.add_argument("--forest-dir", type=Path, default=Path("runs/forest_v10"))
    ap.add_argument("--labels-csv", type=Path, default=Path("../upi-fraud-rings/train.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/diversity_v11"))
    ap.add_argument("--min-gain", type=float, default=.001)
    ap.add_argument("--mode", choices=["boundaries", "gate"], default="gate")
    a = ap.parse_args(argv)
    if a.output_dir.exists():
        raise FileExistsError("choose a new output directory; existing experiments are immutable")
    source = read_json(a.profile_dir.parent / "manifest.json")
    if digest(a.labels_csv) != source["input_hashes"]["train.csv"]:
        raise ValueError("original training labels have changed")
    manifest, meta, old, groups, context = load_stage(a.profile_dir / "validation", "validation")
    reference = read_json(a.reference_dir / "comparison.json")
    if reference["selected"] != "equal_fraud_gate_-0.5" or not np.allclose(reference["weights"], [1/3]*3):
        raise ValueError("expected the frozen equal-model fraud-gate reference")
    reference_trial = next(t for t in reference["trials"] if t["name"] == reference["selected"])
    reference_offsets = read_json(a.reference_dir / "offsets.json")
    # The comparison's source hashes must match this run, not just its filename.
    for name in ("manifest.json", "meta.parquet", "A_proba.npz", "B_proba.npz", "C_proba.npz",
                 "features_v3.parquet", "features_v5.parquet", "groups.npy"):
        path = a.profile_dir / "validation" / name
        recorded = next((sha for p, sha in reference["input_hashes"].items()
                         if Path(p).resolve() == path.resolve()), None)
        if recorded != digest(path):
            raise ValueError(f"reference input mismatch: {path}")
    probabilities = {"reference": mix(old, [1/3]*3)}
    del old
    cache = a.hierarchy_dir / "validation/cache"
    rows = pd.read_parquet(cache / "rows.parquet")
    query_rows = rows.loc[rows["query"]].iloc[align_ids(rows.loc[rows["query"], "txn_id"], meta.txn_id)]
    if not np.array_equal(pd.to_datetime(query_rows.timestamp, utc=True), pd.to_datetime(meta.timestamp, utc=True)):
        raise ValueError("query timestamps differ between model runs")
    fit_hash = ids_hash(rows.loc[rows["fit"], "txn_id"])
    probabilities["hierarchy"] = load_extra(a.hierarchy_dir, "hierarchy_proba.npz", meta.txn_id, cache, fit_hash)
    forest_manifest = read_json(a.forest_dir / "manifest.json")
    if (forest_manifest["source_manifest"] != digest(a.hierarchy_dir / "manifest.json")
            or forest_manifest["source_cache"] != read_json(cache / "ready.json")):
        raise ValueError("forest uses a different hierarchy feature cache")
    probabilities["forest"] = load_extra(a.forest_dir, "probabilities.npz", meta.txn_id, cache, fit_hash)
    y = read_labels(a.labels_csv, meta.txn_id)
    if not np.array_equal(y, query_rows.y.to_numpy(dtype=int)):
        raise ValueError("query labels differ from original training labels")
    dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], utc=True))
    cal = (dates.month == 9) & (dates.day >= 16) & (dates.day <= 23)
    assess = (dates.month == 9) & (dates.day >= 24) & (dates.day <= 30)
    if not assess.any():
        raise ValueError("missing assessment week")
    selected, trials = select({k: p[cal] for k, p in probabilities.items()}, y[cal], groups[cal],
        slice_rule_context(context, cal), reference_trial["inner_offsets"], a.min_gain, a.mode)
    if not np.isclose(trials[0]["gate_metrics"]["macro_f1"], reference_trial["gate_metrics"]["macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("reference selection score failed to reproduce")
    # Selection is frozen before accessing assessment labels.
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir / "selection.json", {"selected": selected, "trials": trials,
                                               "candidate_specs": candidate_specs(a.mode), "min_gain": a.min_gain})
    p = blend(probabilities, selected)
    offsets = reference_offsets if selected["name"] == "reference" else calibrate(
        p[cal], y[cal], groups[cal], slice_rule_context(context, cal), selected)
    baseline = evaluate_predictions(y[assess], predict_labels(probabilities["reference"][assess], groups[assess],
        reference_offsets, slice_rule_context(context, assess)))
    if not np.isclose(baseline["macro_f1"], reference["selected_assessment"]["macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("reference assessment score failed to reproduce")
    metrics = evaluate_predictions(y[assess], predict_labels(p[assess], groups[assess], offsets,
                                                             slice_rule_context(context, assess)))
    report = {"selected": selected["name"], "selected_spec": selected, "reference_assessment": baseline,
              "selected_assessment": metrics, "assessment_reused_project_data": True, "original_labels": True,
              "public_score": None, "model_training": False, "submission": None,
              "selection_dates": {"calibration": "Sep16-19", "gate": "Sep20-23", "assessment": "Sep24-30"}}
    # Only generate a candidate when assessment corroborates selection and no
    # new member needs refitting. This does not upload or recommend a public score.
    if selected["share"] == 0 and metrics["macro_f1"] > baseline["macro_f1"] + a.min_gain:
        _, test_meta, test_probs, test_groups, test_context = load_stage(a.profile_dir / "refit", "refit", manifest["signature"])
        test_p = mix(test_probs, [1/3]*3)
        pred = predict_labels(test_p, test_groups, offsets, test_context)
        sample = pd.read_csv(a.profile_dir / "refit/submission.csv", dtype={"id": str})[["id"]]
        report["submission"] = write_submission(test_meta.txn_id.to_numpy(dtype=str), pred, sample, a.output_dir / "submission.csv")
    files = [a.labels_csv, a.profile_dir.parent / "manifest.json", a.reference_dir / "comparison.json",
             a.reference_dir / "offsets.json", cache / "rows.parquet", cache / "ready.json"]
    for run, filename in ((a.hierarchy_dir, "hierarchy_proba.npz"), (a.forest_dir, "probabilities.npz")):
        files += [run / "manifest.json", run / "validation/complete.json", run / "validation" / filename]
    report["input_hashes"] = {str(path): digest(path) for path in files}
    report["source_hashes"] = {name: digest(Path(__file__).parent / name) for name in
        ("compare_diverse_models.py", "compare_saved_models.py", "postprocess_saved.py",
         "src/upi/boundary_calibration.py", "src/upi/make_submission.py")}
    write_json(a.output_dir / "comparison.json", report)
    write_json(a.output_dir / "offsets.json", offsets)
    lines = ["Saved-model diversity comparison (no model training)", f"Selected: {selected['name']}",
             f"Reference assessment macro-F1: {baseline['macro_f1']:.6f}",
             f"Selected assessment macro-F1: {metrics['macro_f1']:.6f}",
             f"Change: {metrics['macro_f1'] - baseline['macro_f1']:+.6f}",
             "Original noisy labels; assessment week has already been reused. Public score unknown.",
             f"Submission generated: {report['submission'] is not None}", "",
             "Class: precision / recall / F1 / support"]
    lines += [f"{label}: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f} / {v['support']}"
              for label, v in metrics["per_class"].items()]
    (a.output_dir / "COMPARISON.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
