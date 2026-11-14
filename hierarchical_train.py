"""CPU hierarchy experiment reusing a completed run's teachers and preparation.

prepare/check never fit models. validate fits flat, binary and conditional-type
models. refit uses frozen selection/rounds and produces a checked submission.
Existing source runs are read-only. No competition upload is performed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.upi.common import LABELS
from src.upi.fit_submit import (file_hash, ids_hash, calendar_dates, split_masks,
    sample_training, _frozen_features, _raw_labels, train_member, model_proba,
    save_probability_artifact, load_probability_artifact, rebuild_chain_features)
from src.upi.features_v6 import validate_ids, align_pack
from src.upi.bands import build_bands
from src.upi.clean_labels import clean_training_labels
from src.upi.motifs import build_motifs
from src.upi.hierarchical import fit_detector, fit_types, hierarchy_proba
from src.upi.make_submission import (event_groups, prepare_rule_context, slice_rule_context,
    select_member_policy, fit_offsets, predict_labels, evaluate_predictions, write_submission)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_complete(folder, signature):
    value = read(folder / "complete.json")
    if value["signature"] != signature:
        raise ValueError("completed stage signature differs")
    for name, sha in value["hashes"].items():
        if file_hash(folder / name) != sha:
            raise ValueError(f"completed artifact checksum mismatch: {name}")


def check_inputs(root, source, config):
    inputs = read(source / "inputs.json")
    for name, expected in inputs["hashes"].items():
        if name == "source":
            for module, sha in expected.items():
                if file_hash(root / "src/upi" / module) != sha:
                    raise ValueError(f"source run implementation differs: {module}")
        elif file_hash(root / "data/processed" / name) != expected:
            raise ValueError(f"prepared input differs from source run: {name}")
    required = [source / "bootstrap" / n for n in
                ("manifest.json", "forward_teacher.npz", "refit_test_teacher.npz")]
    required += [source / "contexts/validation" / n for n in
                 ("features_v3.parquet", "features_v5.parquet", "cleaning_rows.parquet")]
    artifact_hashes = {str(p.relative_to(source)): file_hash(p) for p in required}
    if read(source / "bootstrap/manifest.json")["signature"] != inputs["signature"]:
        raise ValueError("source bootstrap signature mismatch")
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "scipy")}
    versions["python"] = platform.python_version()
    sources = {n: file_hash(root / n) for n in
               ("hierarchical_train.py", "configs/hierarchy.json", "src/upi/hierarchical.py",
                "src/upi/motifs.py", "src/upi/events.py")}
    payload = {"source_signature": inputs["signature"], "source_artifacts": artifact_hashes,
               "source_hashes": inputs["hashes"], "config": config,
               "code": sources, "environment": versions, "class_order": list(LABELS)}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return inputs, {**payload, "signature": signature}


def prepare(root, source, out, config, inputs, stage):
    folder = out / stage
    cache = folder / "cache"
    ready = cache / "ready.json"
    if ready.exists():
        manifest = read(ready)
        for name, sha in manifest["hashes"].items():
            if file_hash(cache / name) != sha:
                raise ValueError(f"cache checksum mismatch: {name}")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    print(f"Preparing {stage} features; no model fitting in this step", flush=True)
    base = pd.read_parquet(root / "data/processed/base_v0.parquet").reset_index(drop=True)
    validate_ids(base)
    source_config = inputs["config"]
    dates = calendar_dates(base, source_config)
    base["month"] = dates.dt.strftime("%Y-%m")
    masks, _ = split_masks(base, dates, source_config)
    fit = masks["fit"] if stage == "validation" else masks["train"]
    if stage == "validation":
        context_dir = source / "contexts/validation"
        cleaned = pd.read_parquet(context_dir / "cleaning_rows.parquet")
    else:
        # Reuse previously trained forward teachers; fit-specific label-derived
        # context is rebuilt using all training labels only for the final stage.
        context_dir = folder / "context"
        context_dir.mkdir(exist_ok=True)
        bootstrap = source / "bootstrap"
        bm = read(bootstrap / "manifest.json")
        teacher = load_probability_artifact(bootstrap / "forward_teacher.npz", base.txn_id,
            allow_missing=True, expected_provenance={"signature": inputs["signature"], "folds": bm["folds"]})
        teacher[masks["test"]] = load_probability_artifact(bootstrap / "refit_test_teacher.npz",
            base.txn_id[masks["test"]], expected_provenance={"signature": inputs["signature"],
                                                          "fit_ids_hash": ids_hash(base.txn_id[fit])})
        f1 = align_pack(base, pd.read_parquet(root / "data/processed/features_v1.parquet"))
        f2 = align_pack(base, pd.read_parquet(root / "data/processed/features_v2.parquet"))
        bands, diagnostic = build_bands(base, teacher, f1, pd.Index(base.txn_id[fit]))
        chain = rebuild_chain_features(base, f1, bands)
        cleaned = clean_training_labels(base, teacher, bands, f2, pd.Index(base.txn_id[fit]))
        bands.to_parquet(context_dir / "features_v3.parquet", index=False)
        chain.to_parquet(context_dir / "features_v5.parquet", index=False)
        cleaned.to_parquet(context_dir / "cleaning_rows.parquet", index=False)
        write(context_dir / "band_diagnostics.json", diagnostic)
        del teacher, f1, f2, bands, chain
        gc.collect()
    validate_ids(cleaned)
    if set(cleaned.txn_id) != set(base.txn_id[fit]):
        raise ValueError("cleaned fitting IDs differ from eligible chronological fit IDs")
    y = cleaned.set_index("txn_id").loc[base.txn_id[fit], "y_clean"].to_numpy(dtype=int)
    keep, weights = sample_training(base.txn_id[fit], y, rate=config["legit_fraction"])
    chosen_fit = np.flatnonzero(fit)[keep]
    query = ((masks["full_september"] | masks["tail"]) & ~fit
             if stage == "validation" else masks["test"])
    selected = query.copy()
    selected[chosen_fit] = True
    full_ids = pd.Index(base.txn_id)
    # Build motif context before sampling, never from the sampled negatives.
    motifs = build_motifs(base).loc[selected].reset_index(drop=True) if config["include_motifs"] else None
    selected_base = base.loc[selected].reset_index(drop=True)
    row = selected_base[["txn_id", "timestamp", "receiver_type"]].copy()
    row["fit"] = fit[selected]
    row["query"] = query[selected]
    row["y"] = -1
    row.loc[row.fit, "y"] = y[keep]
    row["weight"] = 1.
    row.loc[row.fit, "weight"] = weights
    for name in ("early_stop", "calibration", "assessment", "full_september", "tail"):
        row[name] = masks[name][selected]
    if stage == "validation":
        row.loc[row["query"], "y"] = _raw_labels(base, query)
    del base, cleaned, dates
    gc.collect()
    packs = {}
    for name in ("v1", "v2", "v3", "v4", "v5"):
        path = (context_dir if name in {"v3", "v5"} else root / "data/processed") / f"features_{name}.parquet"
        pack = pd.read_parquet(path)
        validate_ids(pack)
        if len(pack) != len(full_ids) or not full_ids.isin(pack.txn_id).all():
            raise ValueError(f"full feature ID set differs: {name}")
        packs[name] = pack.set_index("txn_id").loc[selected_base.txn_id].reset_index()
        del pack
    if motifs is not None:
        packs["motifs"] = motifs
    groups = event_groups(selected_base, packs["v3"], packs["v5"], packs["v2"], packs["v4"])
    schema = None if stage == "validation" else read(out / "validation/cache/schema.json")
    X, schema, unknown = _frozen_features(selected_base, packs, schema)
    X.to_parquet(cache / "X.parquet", index=False)
    row.to_parquet(cache / "rows.parquet", index=False)
    for name in ("v3", "v5"):
        packs[name].to_parquet(cache / f"{name}.parquet", index=False)
    np.save(cache / "groups.npy", groups.astype(str))
    write(cache / "schema.json", schema)
    write(cache / "unknown_categories.json", unknown)
    files = ["X.parquet", "rows.parquet", "v3.parquet", "v5.parquet", "groups.npy", "schema.json", "unknown_categories.json"]
    write(ready, {"rows": len(X), "features": len(X.columns),
                  "hashes": {f: file_hash(cache / f) for f in files}})
    print(f"Prepared {len(X):,} rows and {len(X.columns)} features", flush=True)
    return folder


def train_stage(root, out, config, signature, stage):
    folder, cache = out / stage, out / stage / "cache"
    complete = folder / "complete.json"
    if complete.exists():
        verify_complete(folder, signature)
        print(f"{stage} already complete", flush=True)
        return
    X, row = pd.read_parquet(cache / "X.parquet"), pd.read_parquet(cache / "rows.parquet")
    if len(X) != len(row):
        raise ValueError("cached matrix/row lengths differ")
    schema = read(cache / "schema.json")
    fit, stop, query = row.fit.to_numpy(), row.early_stop.to_numpy(), row["query"].to_numpy()
    y = row.y.to_numpy(dtype=int)
    validation = stage == "validation"
    prior = None if validation else read(out / "validation/models.json")
    specs = {name: {**config["model_defaults"], **config[name],
                    "categorical_columns": list(schema["categorical_mappings"])}
             for name in ("flat", "detector", "types")}
    X_fit, X_stop, X_query = X.loc[fit], X.loc[stop] if validation else None, X.loc[query]
    del X
    gc.collect()
    w = row.weight.to_numpy()[fit]
    models, info = {}, {}
    for name in ("flat", "detector", "types"):
        print(f"Fitting {name} ({stage})", flush=True)
        started = time.perf_counter()
        kwargs = {"spec": specs[name]}
        rounds = None if validation else prior[name]["rounds"]
        args = (X_fit, y[fit], w, X_stop, y[stop] if validation else None)
        if name == "flat":
            model, best = train_member(*args, device="cpu", fixed_rounds=rounds, **kwargs)
        elif name == "detector":
            model, best = fit_detector(*args, rounds=rounds, **kwargs)
        else:
            model, best = fit_types(*args, rounds=rounds, **kwargs)
        model.booster_.save_model(str(folder / f"{name}.txt"))
        info[name] = {"rounds": best, "spec": specs[name], "seconds": time.perf_counter() - started,
                      "classes": (model.classes_ if name == "detector" else model.podium_classes_).tolist()}
        models[name] = model
        print(f"Completed {name}: {best} rounds", flush=True)
    write(folder / "models.json", info)
    flat = model_proba(models["flat"], X_query)
    hierarchical = hierarchy_proba(models["detector"], models["types"], X_query)
    del models, X_fit, X_stop, X_query
    gc.collect()
    meta = row.loc[query, ["txn_id", "timestamp", "receiver_type"]].reset_index(drop=True)
    groups = np.load(cache / "groups.npy", allow_pickle=False)[query]
    f3 = pd.read_parquet(cache / "v3.parquet").loc[query].reset_index(drop=True)
    f5 = pd.read_parquet(cache / "v5.parquet").loc[query].reset_index(drop=True)
    context = prepare_rule_context(meta, f3, f5)
    provenance = {"signature": signature, "stage": stage, "class_order": list(LABELS),
                  "fit_ids_hash": ids_hash(row.loc[fit, "txn_id"])}
    for name, p in (("flat", flat), ("hierarchy", hierarchical)):
        save_probability_artifact(folder / f"{name}_proba.npz", meta.txn_id, p, provenance=provenance)
    if validation:
        cal = row.calibration.to_numpy()[query]
        offsets, selection = select_member_policy(
            {"A": flat[cal], "H": hierarchical[cal]}, {"A": 1., "H": 0.},
            y[query][cal], groups[cal], slice_rule_context(context, cal),
            config={"min_gain": config["min_gain"], "calibration": config["calibration"],
                    "baseline_calibration": config["calibration"]})
        write(folder / "selection.json", selection)
        write(folder / "offsets.json", offsets)
        baseline_offsets = fit_offsets(flat[cal], y[query][cal], groups[cal],
            slice_rule_context(context, cal), config=config["calibration"])
        write(folder / "baseline_offsets.json", baseline_offsets)
    else:
        selection = read(out / "validation/selection.json")
        offsets = read(out / "validation/offsets.json")
    weights = selection["member_weights"]
    p = weights["A"] * flat + weights["H"] * hierarchical
    pred = predict_labels(p, groups, offsets, context)
    save_probability_artifact(folder / "selected_proba.npz", meta.txn_id, p, provenance=provenance)
    if validation:
        metrics = {"selection": selection, "partitions": {}, "label_source": "original",
                   "assessment_reused_project_data": True, "public_score": None}
        baseline_pred = predict_labels(flat, groups, baseline_offsets, context)
        for part in ("early_stop", "calibration", "assessment", "full_september", "tail"):
            mask = row[part].to_numpy()[query]
            metrics["partitions"][part] = evaluate_predictions(y[query][mask], pred[mask])
            metrics["partitions"][part]["flat_baseline"] = evaluate_predictions(y[query][mask], baseline_pred[mask])
        write(folder / "metrics.json", metrics)
        report = metrics["partitions"]["assessment"]
        lines = ["# Hierarchy validation", "", f"Selected: {selection['selected']}; weights: {weights}",
                 f"Assessment macro-F1: {report['macro_f1']:.6f}",
                 f"Same-feature flat baseline: {report['flat_baseline']['macro_f1']:.6f}",
                 "Reference results: challenge_v8 0.517891; best saved comparison 0.520807.",
                 "Original noisy labels; this does not predict the public score.", "",
                 "Class: precision / recall / F1 / support", ""]
        lines += [f"- {label}: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f} / {v['support']}"
                  for label, v in report["per_class"].items()]
        lines += ["", "Validation only. Review this report before refitting or submitting."]
        (out / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Assessment macro-F1: {report['macro_f1']:.6f}", flush=True)
    else:
        sample = pd.read_csv(root / "data/raw/sample_submission.csv")
        submission = write_submission(meta.txn_id.to_numpy(), pred, sample, folder / "submission.csv")
        # Read back the exact file a teammate will upload, not just an array.
        restored = pd.read_csv(folder / "submission.csv")
        expected = pd.Series(np.asarray(LABELS)[pred], index=meta.txn_id).loc[sample.id].to_numpy()
        if list(restored.columns) != ["id", "label"] or not np.array_equal(restored.label, expected):
            raise ValueError("submission read-back mismatch")
        write(folder / "submission_report.json", submission)
    files = [p for p in folder.iterdir() if p.is_file() and p.name != "complete.json"]
    write(complete, {"signature": signature, "hashes": {p.name: file_hash(p) for p in files}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--source-run", type=Path, default=Path("runs/challenge_v8"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/hierarchy_v9"))
    ap.add_argument("--config", type=Path, default=Path("configs/hierarchy.json"))
    ap.add_argument("--stage", choices=["check", "prepare", "validate", "refit"], default="validate")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    source = args.source_run if args.source_run.is_absolute() else root / args.source_run
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("source and output run folders must be separate")
    config = read(args.config if args.config.is_absolute() else root / args.config)
    inputs, manifest = check_inputs(root, source, config)
    if args.stage == "check":
        print("Preflight passed: prepared data, source code, cached teachers and fold contexts verified. No training performed.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        if read(out / "manifest.json")["signature"] != manifest["signature"]:
            raise ValueError("inputs/code/config/environment changed; use a new output directory")
    else:
        write(out / "manifest.json", manifest)
    stage = "refit" if args.stage == "refit" else "validation"
    if stage == "refit" and not (out / "validation/complete.json").exists():
        raise ValueError("complete validation before requesting refit")
    if stage == "refit":
        verify_complete(out / "validation", manifest["signature"])
    prepare(root, source, out, config, inputs, stage)
    if args.stage != "prepare":
        train_stage(root, out, config, manifest["signature"], stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
