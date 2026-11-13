"""CLI: validate, refit, predict and check-submission. No network writes."""
import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .artifacts import (cached_features, data_hashes, digest, environment, git_revision,
                        save_probabilities, sha256, source_hashes, write_json)
from .constants import FEATURES, LABELS
from .data import align, check_submission, load_data
from .decisions import evaluate, fit_decisions, paired_day_bootstrap, predictions
from .model import fit_model, predict_proba

REPO = Path(__file__).resolve().parents[2]


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config["feature_pack"] != "research_baseline_v1":
        raise ValueError("Only the measured feature pack is implemented")
    if config["decision_kind"] not in ("joint", "shared"):
        raise ValueError("decision_kind must be joint or shared")
    if not config["august_start"] < config["september_start"] < config["october_start"]:
        raise ValueError("Temporal cutoffs must be chronological")
    return config


def identity(config, hashes):
    return {"config": config, "data_sha256": hashes, "source_sha256": source_hashes(),
            "git_revision": git_revision(), "environment": environment(), "label_order": list(LABELS)}


def verify_artifacts(manifest):
    for name, record in manifest["artifacts"].items():
        if not Path(record["path"]).exists() or sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Missing or changed artifact: {name}")


def records(paths):
    return {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)} for name, path in paths.items()}


def compare_reference(probabilities_path, research_path):
    """Compare by ID, never by position; no research artifact is required."""
    if not research_path.exists():
        return {"status": "Research probability artifact not present"}
    frame = pd.read_parquet(probabilities_path)
    with np.load(research_path, allow_pickle=True) as ref:
        expected = pd.DataFrame(ref["P"], columns=[f"prob_{c}" for c in LABELS])
        expected.insert(0, "txn_id", ref["ids"])
        expected = align(expected, frame.txn_id)
    delta = np.abs(frame[[f"prob_{c}" for c in LABELS]].to_numpy() - expected.drop(columns="txn_id").to_numpy())
    return {"research_path": str(research_path.resolve()), "rows": len(frame),
            "max_probability_difference": float(delta.max()), "exact_probability_match": bool((delta == 0).all())}


def write_comparison(path, metrics, halves, bootstrap):
    lines = ["# Local candidate validation", "",
        "Reference: reproduced research baseline. The generating run of the supplied 0.5034 public CSV remains unidentified.", "",
        "August selects decisions; September evaluates frozen decisions. Original labels and complete validation populations are used.", "",
        "| Variant | August macro-F1 | September macro-F1 |", "|---|---:|---:|"]
    for variant in ["raw", "shared", "joint", "rules", "combined"]:
        lines.append(f"| {variant} | {metrics['august'][variant]['macro_f1']:.6f} | {metrics['september'][variant]['macro_f1']:.6f} |")
    lines += ["", "| Class | Model F1 | Combined F1 | Delta |", "|---|---:|---:|---:|"]
    base = metrics["september"]["model"]["per_class"]
    combined = metrics["september"]["combined"]["per_class"]
    for label in LABELS:
        a, b = base[label]["f1"], combined[label]["f1"]
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += ["", "September halves (model → combination):"]
    for name, values in halves.items():
        lines.append(f"- {name}: {values['model']['macro_f1']:.6f} → {values['combined']['macro_f1']:.6f}")
    lines += ["", f"Paired day-resampled delta interval: {bootstrap['delta_interval_95']}. This is a sensitivity diagnostic, not an October score forecast.", "",
        "The EDA rules and experiment choices have already been inspected against September; it is a development set, not an untouched final holdout. Train labels are noisy and test labels are unavailable."]
    path.write_text("\n".join(lines) + "\n")


def validate(args, config):
    started = time.time()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = data_hashes(args.data_dir)
    run_identity = identity(config, hashes)
    manifest_path = out / "validation_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume or old["identity"] != run_identity:
            raise ValueError("Existing validation run: use --resume for identical inputs/code or choose a fresh output directory")
        verify_artifacts(old)
        print(f"Verified completed validation run: {manifest_path}")
        return
    transactions, accounts, merchants, _ = load_data(args.data_dir)
    targets = transactions[["txn_id", "label"]]
    all_metrics, details, paths = {}, {}, {}
    decisions = None
    for name, cutoff, end in [
        ("august", config["august_start"], config["september_start"]),
        ("september", config["september_start"], config["october_start"]),
    ]:
        print(f"Validating {name}: labels before {cutoff}, transaction context before {end}", flush=True)
        f, feature_info = cached_features(transactions, accounts, merchants, end, args.cache_dir, hashes)
        valid = ((f.split == "train") & (f.timestamp >= pd.Timestamp(cutoff)) &
                 (f.timestamp < pd.Timestamp(end))).to_numpy()
        query = f.loc[valid].reset_index(drop=True)
        booster, train_info = fit_model(f, targets, cutoff, config)
        P = predict_proba(booster, query)
        truth = align(targets, query.txn_id).label.map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype="int32")
        rules = query.rule_code.to_numpy(dtype="int32")
        if name == "august":
            decisions = fit_decisions(P, truth, rules, config)
            decisions["fit_ids_sha256"] = digest(query.txn_id.tolist())
            decisions["fit_max_timestamp"] = str(query.timestamp.max())
            write_json(out / "decisions.json", decisions)
        pred = predictions(P, rules, decisions)
        all_metrics[name] = {variant: evaluate(truth, code) for variant, code in pred.items()}
        fold_dir = out / name
        fold_dir.mkdir(exist_ok=True)
        booster.save_model(str(fold_dir / "model.txt"))
        write_json(fold_dir / "schema.json", {"features": list(FEATURES), "label_order": list(LABELS),
                   "category_schema": feature_info["category_schema"]})
        artifact = fold_dir / "probabilities.parquet"
        save_probabilities(artifact, query, P, rules, pred, truth)
        write_json(fold_dir / "metrics.json", all_metrics[name])
        details[name] = {**train_info, "feature_cache": feature_info, "validation_rows": len(query),
            "validation_min_timestamp": str(query.timestamp.min()), "validation_max_timestamp": str(query.timestamp.max()),
            "research_reproduction": compare_reference(artifact, REPO / f"data/processed/research_{'aug' if name == 'august' else 'sep'}/prob_{8 if name == 'august' else 9}_base.npz")}
        for file in ["model.txt", "schema.json", "probabilities.parquet", "probabilities.schema.json", "metrics.json"]:
            paths[f"{name}/{file}"] = fold_dir / file
        if name == "september":
            halves = {}
            for half, mask in [("1–15 September", query.timestamp.dt.day.to_numpy() <= 15),
                               ("16–30 September", query.timestamp.dt.day.to_numpy() > 15)]:
                halves[half] = {variant: evaluate(truth[mask], pred[variant][mask]) for variant in ["model", "combined"]}
            bootstrap = paired_day_bootstrap(truth, pred["model"], pred["combined"],
                query.timestamp.dt.normalize().to_numpy(), config["bootstrap_draws"], config["seed"])
        print(name, {k: round(v["macro_f1"], 6) for k, v in all_metrics[name].items()}, flush=True)
        del f, query, booster, P
        gc.collect()
    write_comparison(out / "comparison.md", all_metrics, halves, bootstrap)
    write_json(out / "validation_metrics.json", {"folds": all_metrics, "september_halves": halves, "paired_day_bootstrap": bootstrap})
    paths.update({file: out / file for file in ["decisions.json", "comparison.md", "validation_metrics.json"]})
    manifest = {"identity": run_identity, "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "research_baseline_v1; generating pipeline of s1_v1.csv unidentified",
        "folds": details, "artifact_format": "parquet keyed by txn_id, twelve named probability columns",
        "validation_labels_modified": False, "decision_fit": "August only; frozen for September and test",
        "history_features": False, "elapsed_seconds": time.time() - started,
        "artifacts": records(paths)}
    write_json(manifest_path, manifest)
    print(f"Saved {manifest_path}", flush=True)


def require_validation(args, config):
    out = args.output_dir.resolve()
    manifest = json.loads((out / "validation_manifest.json").read_text())
    if manifest["identity"] != identity(config, data_hashes(args.data_dir)):
        raise ValueError("Validation identity differs from current data/config/code/environment")
    verify_artifacts(manifest)
    decisions = json.loads((out / "decisions.json").read_text())
    return out, manifest, decisions


def refit(args, config):
    started = time.time()
    out, validation, _ = require_validation(args, config)
    path = out / "refit_manifest.json"
    if path.exists():
        if not args.resume:
            raise ValueError("Final model already exists; use --resume to verify it")
        verify_artifacts(json.loads(path.read_text()))
        print(f"Verified completed refit: {path}")
        return
    transactions, accounts, merchants, _ = load_data(args.data_dir)
    features, cache = cached_features(transactions, accounts, merchants, None, args.cache_dir,
                                      validation["identity"]["data_sha256"])
    booster, train_info = fit_model(features, transactions[["txn_id", "label"]], None, config)
    booster.save_model(str(out / "final_model.txt"))
    write_json(out / "final_schema.json", {"features": list(FEATURES), "label_order": list(LABELS),
                "category_schema": cache["category_schema"]})
    write_json(path, {"identity": validation["identity"], "train": train_info, "feature_cache": cache,
        "validation_manifest_sha256": sha256(out / "validation_manifest.json"),
        "includes_147_supplied_october_training_tails": True,
        "elapsed_seconds": time.time() - started,
        "artifacts": records({name: out / name for name in ["final_model.txt", "final_schema.json", "decisions.json"]})})
    print(f"Saved final model and {path}", flush=True)


def predict(args, config):
    started = time.time()
    out, validation, decisions = require_validation(args, config)
    refit_manifest = json.loads((out / "refit_manifest.json").read_text())
    if refit_manifest["identity"] != validation["identity"]:
        raise ValueError("Final model identity differs from validated identity")
    verify_artifacts(refit_manifest)
    schema = json.loads((out / "final_schema.json").read_text())
    if schema["features"] != list(FEATURES) or schema["label_order"] != list(LABELS):
        raise ValueError("Saved model schema mismatch")
    transactions, accounts, merchants, sample = load_data(args.data_dir)
    features, cache = cached_features(transactions, accounts, merchants, None, args.cache_dir,
        validation["identity"]["data_sha256"], schema=schema["category_schema"])
    # IDs determine output order; file position and timestamp order never do.
    query = align(features.loc[features.split == "test"], sample.id)
    booster = lgb.Booster(model_file=str(out / "final_model.txt"))
    P = predict_proba(booster, query)
    pred = predictions(P, query.rule_code.to_numpy(dtype="int32"), decisions)
    save_probabilities(out / "test_probabilities.parquet", query, P, query.rule_code.to_numpy(), pred)
    submission = pd.DataFrame({"id": sample.id, "label": np.asarray(LABELS)[pred["combined"]]})
    summary = check_submission(submission, sample)
    submission_path = args.submission.resolve() if args.submission else out / "candidate.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.exists():
        raise ValueError("Refusing to overwrite an existing submission; choose a fresh path")
    submission.to_csv(submission_path, index=False)
    if submission_path.stat().st_size > 15_000_000:
        raise ValueError("Submission exceeds 15 MB")
    summary["sha256"] = sha256(submission_path)
    summary["bytes"] = submission_path.stat().st_size
    comparison = None
    if args.reference_submission:
        reference = pd.read_csv(args.reference_submission)
        check_submission(reference, sample)
        joined = reference.merge(submission, on="id", suffixes=("_reference", "_candidate"), validate="one_to_one", sort=False)
        equal = joined.label_reference == joined.label_candidate
        fraud = (joined.label_reference != "legit") | (joined.label_candidate != "legit")
        comparison = {"path": str(args.reference_submission.resolve()), "sha256": sha256(args.reference_submission),
            "reported_public_score": 0.5034, "candidate_public_score": None,
            "different_rows": int((~equal).sum()), "overall_agreement": float(equal.mean()),
            "fraud_union_rows": int(fraud.sum()), "fraud_union_agreement": float(equal[fraud].mean()) if fraud.any() else None}
        joined.loc[~equal].to_csv(out / "differences_from_s1_v1.csv", index=False)
    month = query.timestamp.dt.strftime("%Y-%m")
    monthly = {str(m): pd.Series(np.asarray(LABELS)[pred["combined"]][month == m]).value_counts().to_dict() for m in month.unique()}
    train = transactions[transactions.split == "train"]
    seen = {}
    for role in ["sender_id", "receiver_id", "device_id"]:
        mask = query[role].isin(train[role]).to_numpy()
        seen[role] = {"seen_rows": int(mask.sum()), "unseen_rows": int((~mask).sum()),
            "seen_predicted_fraud": int(((pred["combined"] > 0) & mask).sum()),
            "unseen_predicted_fraud": int(((pred["combined"] > 0) & ~mask).sum())}
    write_json(out / "prediction_manifest.json", {"identity": validation["identity"],
        "submission": summary, "reference_comparison": comparison,
        "month_prediction_counts": monthly, "seen_unseen_diagnostics": seen,
        "feature_cache": cache, "elapsed_seconds": time.time() - started,
        "validation_manifest_sha256": sha256(out / "validation_manifest.json"),
        "refit_manifest_sha256": sha256(out / "refit_manifest.json"),
        "artifacts": records({"submission": submission_path, "model": out / "final_model.txt",
            "probabilities": out / "test_probabilities.parquet", "decisions": out / "decisions.json",
            "schema": out / "final_schema.json"})})
    print(json.dumps({"submission": str(submission_path), **summary, "reference_comparison": comparison}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "refit", "predict", "check-submission"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "models/measured_v1")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data/processed/measured_v1")
    parser.add_argument("--config", type=Path, default=REPO / "configs/measured.json")
    parser.add_argument("--reference-submission", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "check-submission":
        if not args.submission:
            parser.error("check-submission requires --submission")
        summary = check_submission(pd.read_csv(args.submission), pd.read_csv(args.data_dir / "sample_submission.csv"))
        if args.submission.stat().st_size > 15_000_000:
            raise ValueError("Submission exceeds 15 MB")
        print(json.dumps({**summary, "sha256": sha256(args.submission)}, indent=2))
        return
    config = read_config(args.config)
    {"validate": validate, "refit": refit, "predict": predict}[args.command](args, config)


if __name__ == "__main__":
    main()
