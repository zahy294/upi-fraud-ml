"""Extra Trees + event composition, reusing the completed hierarchy feature cache.

prepare never trains. validate fits one forest; refit requires completed
validation and reuses its decision offsets. No competition upload is made.
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
from src.upi.common import LABELS
from src.upi.event_composition import build_event_composition, COLUMNS
from src.upi.fit_submit import file_hash, align_proba, save_probability_artifact, ids_hash
from src.upi.make_submission import (prepare_rule_context, slice_rule_context,
    fit_offsets, predict_labels, evaluate_predictions, write_submission)
from hierarchical_train import read, write, prepare, check_inputs, verify_complete


def encode_categories(X, schema):
    """Fixed one-hot columns; avoid imposing numeric order on category codes."""
    cats = schema["categorical_mappings"]
    parts = [X.drop(columns=list(cats)).astype("float32")]
    for col, levels in cats.items():
        codes = X[col].to_numpy()
        for k in range(-1, len(levels)):
            parts.append(pd.DataFrame({f"forest_cat_{col}_{k}": (codes == k).astype("float32")}, index=X.index))
    return pd.concat(parts, axis=1)


def fit_forest(X, y, weights, config):
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import ExtraTreesClassifier
    # Only fitting rows determine imputation; held-out data never enters fit.
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    matrix = imputer.fit_transform(X)
    model = ExtraTreesClassifier(**config)
    model.fit(matrix, y, sample_weight=weights)
    return imputer, model


def probability(imputer, model, X):
    result = np.zeros((len(X), 12), dtype=np.float32)
    for start in range(0, len(X), 25000):
        result[start:start+25000] = align_proba(
            model.predict_proba(imputer.transform(X.iloc[start:start+25000])), model.classes_)
    return result


def run_stage(root, source, out, config, signature, stage):
    cache = (source / "validation/cache") if stage == "validation" else out / "refit/cache"
    folder = out / stage
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        verify_complete(folder, signature)
        print(f"{stage} already complete")
        return
    row = pd.read_parquet(cache / "rows.parquet")
    X = pd.read_parquet(cache / "X.parquet")
    schema = read(cache / "schema.json")
    features = pd.read_parquet(out / "event_composition.parquet").set_index("txn_id")
    extra = features.loc[row.txn_id].reset_index(drop=True)
    X = encode_categories(pd.concat([X, extra], axis=1), schema)
    del features, extra
    gc.collect()
    fit, query = row.fit.to_numpy(), row["query"].to_numpy()
    print(f"Training Extra Trees on {int(fit.sum()):,} rows, {len(X.columns)} input features", flush=True)
    imputer, model = fit_forest(X.loc[fit], row.y.to_numpy()[fit], row.weight.to_numpy()[fit], config)
    p = probability(imputer, model, X.loc[query])
    import joblib
    joblib.dump({"imputer": imputer, "model": model, "columns": X.columns.tolist(),
                 "class_order": LABELS, "signature": signature}, folder / "forest.joblib", compress=3)
    del X, model, imputer
    gc.collect()
    meta = row.loc[query, ["txn_id", "timestamp", "receiver_type"]].reset_index(drop=True)
    context = prepare_rule_context(meta,
        pd.read_parquet(cache / "v3.parquet").loc[query].reset_index(drop=True),
        pd.read_parquet(cache / "v5.parquet").loc[query].reset_index(drop=True))
    groups = np.load(cache / "groups.npy", allow_pickle=False)[query]
    save_probability_artifact(folder / "probabilities.npz", meta.txn_id, p,
        provenance={"signature": signature, "stage": stage, "fit_ids_hash": ids_hash(row.loc[fit, "txn_id"])})
    if stage == "validation":
        cal = row.calibration.to_numpy()[query]
        y = row.y.to_numpy(dtype=int)[query]
        settings = read(source / "manifest.json")["config"]["calibration"]
        offsets = fit_offsets(p[cal], y[cal], groups[cal], slice_rule_context(context, cal), config=settings)
        write(folder / "offsets.json", offsets)
    else:
        offsets = read(out / "validation/offsets.json")
    pred = predict_labels(p, groups, offsets, context)
    if stage == "validation":
        metrics = {"label_source": "original", "public_score": None, "partitions": {}}
        for part in ("early_stop", "calibration", "assessment", "full_september", "tail"):
            mask = row[part].to_numpy()[query]
            metrics["partitions"][part] = evaluate_predictions(y[mask], pred[mask])
        write(folder / "metrics.json", metrics)
        report = metrics["partitions"]["assessment"]
        lines = ["# Extra Trees event-composition experiment", "",
                 f"Assessment macro-F1: {report['macro_f1']:.6f}",
                 "Previous local flat: 0.519038; best saved policy: 0.520807.",
                 "Team public baseline 0.5411 is not a comparable local metric.",
                 "Architecture and feature block both changed; no isolated feature gain is claimed.",
                 "Assessment has been reused in this project. No public score measured.", ""]
        lines += [f"- {lab}: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f} / {v['support']}"
                  for lab, v in report["per_class"].items()]
        (out / "RUN_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(f"Assessment macro-F1: {report['macro_f1']:.6f}", flush=True)
    else:
        sample = pd.read_csv(root / "data/raw/sample_submission.csv")
        report = write_submission(meta.txn_id.to_numpy(), pred, sample, folder / "submission.csv")
        restored = pd.read_csv(folder / "submission.csv")
        expected = pd.Series(np.asarray(LABELS)[pred], index=meta.txn_id).loc[sample.id].to_numpy()
        if not np.array_equal(restored.label, expected):
            raise ValueError("final CSV read-back mismatch")
        write(folder / "submission_report.json", report)
    write(folder / "complete.json", {"signature": signature, "hashes": {
        p.name: file_hash(p) for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--source-run", type=Path, default=Path("runs/hierarchy_v9"))
    ap.add_argument("--teacher-run", type=Path, default=Path("runs/challenge_v8"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/forest_v10"))
    ap.add_argument("--config", type=Path, default=Path("configs/extra_trees.json"))
    ap.add_argument("--stage", choices=["prepare", "validate", "refit"], default="prepare")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    source, teacher, out, cp = [p.resolve() if p.is_absolute() else (root/p).resolve()
                               for p in (a.source_run, a.teacher_run, a.output_dir, a.config)]
    if any(out == p or p in out.parents or out in p.parents for p in (source, teacher)):
        raise ValueError("new experiment output must be separate from its sources")
    sm = read(source / "manifest.json")
    verify_complete(source / "validation", sm["signature"])
    ready = read(source / "validation/cache/ready.json")
    for name, sha in ready["hashes"].items():
        if file_hash(source / "validation/cache" / name) != sha:
            raise ValueError(f"source cache changed: {name}")
    inputs, current_source = check_inputs(root, teacher, sm["config"])
    if inputs["signature"] != sm["source_signature"]:
        raise ValueError("teacher source differs from cached validation source")
    if current_source["source_artifacts"] != sm["source_artifacts"]:
        raise ValueError("cached teachers or source contexts have changed")
    config = read(cp)
    manifest = {"config": config, "source_manifest": file_hash(source / "manifest.json"),
                "source_cache": ready, "teacher_signature": inputs["signature"],
                "source_artifacts": sm["source_artifacts"],
                "python": platform.python_version(),
                "environment": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "pandas", "pyarrow", "joblib")},
                "code": {name: file_hash(root/name) for name in ("forest_experiment.py", "src/upi/event_composition.py", "hierarchical_train.py", "src/upi/events.py", "src/upi/features_v6.py", "src/upi/make_submission.py", "src/upi/fit_submit.py", "src/upi/common.py")}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["signature"] = signature
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read(out / "manifest.json")["signature"] != signature:
        raise ValueError("inputs/code/config/environment changed; use a new output directory")
    write(out / "manifest.json", manifest)
    feature_file = out / "event_composition.parquet"
    feature_hash = out / "features.json"
    if feature_hash.exists():
        if file_hash(feature_file) != read(feature_hash)["sha256"]:
            raise ValueError("event feature checksum mismatch")
    else:
        print("Preparing label-free event composition; no model fitting", flush=True)
        base = pd.read_parquet(root / "data/processed/base_v0.parquet", columns=COLUMNS)
        features = build_event_composition(base)
        features.to_parquet(feature_file, index=False)
        write(feature_hash, {"sha256": file_hash(feature_file), "rows": len(features),
                             "features": len(features.columns)-1})
        print(f"Prepared {len(features):,} rows and {len(features.columns)-1} new features", flush=True)
        del base, features
        gc.collect()
    if a.stage == "prepare":
        return 0
    if a.stage == "refit":
        verify_complete(out / "validation", signature)
        # Only the old feature schema is reused. New fitting rows/cleaning/bands
        # are constructed in this experiment's output, never in the source run.
        schema_dir = out / "validation/cache"
        schema_dir.mkdir(exist_ok=True)
        (schema_dir / "schema.json").write_bytes((source / "validation/cache/schema.json").read_bytes())
        prepare(root, teacher, out, sm["config"], inputs, "refit")
    run_stage(root, source, out, config, signature, "refit" if a.stage == "refit" else "validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
