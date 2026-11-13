"""Turnkey Colab preparation, chronological training and artifact handoff.

Only standard-library imports at module load. --dry-run performs no file I/O.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap",
          "account_takeover", "phishing_collect", "qr_swap", "refund_abuse",
          "fake_cashback", "loan_app_extortion", "merchant_collusion"]
INPUTS = ("train.csv", "test.csv", "accounts.csv", "merchants.csv", "sample_submission.csv")
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow", "matplotlib")
SOURCE_FILES = (
    "colab_train_submit.py", "requirements-colab.txt", "configs/podium.json",
    "src/upi/fit_submit.py", "src/upi/make_submission.py", "src/upi/build_base.py",
    "src/upi/features_v1.py", "src/upi/features_v2.py", "src/upi/features_v4.py",
    "src/upi/features_v5.py", "src/upi/features_v6.py", "src/upi/bands.py",
    "src/upi/clean_labels.py", "src/upi/common.py", "run_all.py",
    "scripts/eda/common.py", "scripts/eda/00_load_and_overview.py",
    "scripts/eda/04_relational.py",
)
PREPARED = tuple(f"data/interim/{name.removesuffix('.csv')}.parquet" for name in INPUTS) + (
    "data/processed/relational_v0.parquet", "data/processed/base_v0.parquet",
    "data/processed/ratios_v0.parquet", "data/processed/features_v1.parquet",
    "data/processed/features_v2.parquet", "data/processed/features_v4.parquet",
)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/podium.json"))
    ap.add_argument("--tag", default="podium_v6")
    ap.add_argument("--stage", choices=["prepare", "bootstrap", "validate", "refit", "all"], default="all")
    ap.add_argument("--profiles", default="ensemble", help="comma-separated control,fusion,temporal,ensemble")
    ap.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--baseline-csv", type=Path)
    for flag in ("install-deps", "mount-drive", "resume", "dry-run"):
        ap.add_argument("--" + flag, action="store_true")
    ap.add_argument("--_action", choices=["environment", "backend", "ratios", "validate"], help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not a._action and (a.data_dir is None or a.output_dir is None):
        ap.error("--data-dir and --output-dir are required")
    if Path(a.tag).name != a.tag or a.tag in {".", ".."}:
        ap.error("--tag must be one directory name")
    profiles = a.profiles.split(",")
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in {"control", "fusion", "temporal", "ensemble"} for p in profiles):
        ap.error("invalid or duplicate --profiles")
    return a


def _stage_commands(args):
    root = args.root.resolve()
    run = root / "runs" / args.tag
    py = str(root / ".colab-venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")) if args.install_deps else sys.executable
    script = str(root / "colab_train_submit.py")
    environment = []
    # A resume must not recreate/reseed a working environment (which can
    # silently upgrade pip and violate the saved dependency fingerprint).
    existing_resume_env = args.resume and not args.dry_run and Path(py).is_file()
    if args.install_deps and not existing_resume_env:
        environment += [
            [sys.executable, "-m", "pip", "install", "virtualenv==20.31.2"],
            [sys.executable, "-m", "virtualenv", "--no-download", str(root / ".colab-venv")],
        ]
        environment.append([py, "-m", "pip", "install", "pip==25.1.1"])
        environment.append([py, "-m", "pip", "install", "--requirement", str(root / "requirements-colab.txt")])
    environment += [[py, "-m", "pip", "check"],
                    [py, script, "--root", str(root), "--tag", args.tag, "--_action", "environment"]]
    if args.device != "cpu":
        probe = [py, script, "--root", str(root), "--tag", args.tag,
                 "--_action", "backend", "--device", args.device]
        if args.allow_cpu_fallback:
            probe.append("--allow-cpu-fallback")
        environment.append(probe)
    commands = [
        [py, str(root / "scripts/eda/00_load_and_overview.py"), str(root)],
        [py, str(root / "scripts/eda/04_relational.py"), str(root)],
        [py, str(root / "src/upi/build_base.py"), str(root)],
        [py, script, "--root", str(root), "--tag", args.tag, "--_action", "ratios"],
    ]
    commands += [[py, str(root / f"src/upi/features_{v}.py"), str(root)] for v in ("v1", "v2", "v4")]
    stages = [("environment", environment), ("prepare", commands)]
    limit = {"prepare": 0, "bootstrap": 1, "validate": 2, "refit": 3, "all": 3}[args.stage]
    for name in ("bootstrap", "validate", "refit")[:limit]:
        commands = [[py, str(root / "src/upi/fit_submit.py"), str(root),
                     "--config", str(run / "effective_config.json"), "--stage", name]]
        if name in {"validate", "refit"}:
            commands.append([py, script, "--root", str(root), "--tag", args.tag,
                             "--_action", "validate"])
        stages.append((name, commands))
    # Every stage value is a list of argv lists.
    return [(name, value if not value or isinstance(value[0], list) else [value]) for name, value in stages]


def plan_commands(args) -> list[list[str]]:
    return [cmd for _, commands in _stage_commands(args) for cmd in commands]


def run_checked(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", subprocess.list2cmdline(command), flush=True)
    print("Log:", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                       check=True, text=True)
    print("Completed:", log_path.name, flush=True)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _capture(command, root):
    return subprocess.run(command, cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _copy_inputs(args):
    root, source = args.root.resolve(), args.data_dir.resolve()
    destination = root / "data/raw"
    hashes = {}
    # Validate all inputs and all existing destinations before copying any.
    for name in INPUTS:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
        hashes[name] = _hash(path)
        target = destination / name
        if target.exists() and _hash(target) != hashes[name]:
            raise ValueError(f"existing input differs: {target}; use a fresh checkout or resolve the data mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        source_file, target = source / name, destination / name
        if source_file.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source_file, target)
        if _hash(target) != hashes[name]:
            raise ValueError(f"copy hash mismatch: {target}")
    return hashes


def _copy_tree_changed(source, destination):
    """Copy artifact trees without rewriting equal-content Drive files."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_dir():
            _copy_tree_changed(item, destination / item.name)
        else:
            target = destination / item.name
            if not target.exists() or item.stat().st_size != target.stat().st_size or _hash(item) != _hash(target):
                shutil.copy2(item, target)


def _backup(args, manifest):
    root = args.root.resolve()
    run, target = root / "runs" / args.tag, args.output_dir.resolve() / args.tag
    if run == target:
        return
    if target.is_relative_to(run):
        raise ValueError("backup directory cannot be inside the local run directory")
    _copy_tree_changed(run, target)
    # Preparation caches allow a fresh Colab runtime to resume without silently
    # relying on missing local parquet files. Raw CSVs remain supplied separately.
    for relative, expected in manifest.get("stages", {}).get("prepare", {}).get("artifacts", {}).items():
        path = root / relative
        if path.is_file():
            dest = target / "prepared_cache" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or _hash(dest) != expected:
                shutil.copy2(path, dest)


def _restore(args):
    root, backup = args.root.resolve(), args.output_dir.resolve() / args.tag
    run = root / "runs" / args.tag
    if not (run / "manifest.json").exists() and (backup / "manifest.json").exists():
        if run.exists() and any(run.iterdir()):
            raise ValueError("nonempty local run without manifest; restore requires an empty destination")
        # Exclude the separate large preparation cache from the run tree.
        for item in backup.iterdir():
            if item.name == "prepared_cache":
                continue
            if item.is_dir():
                _copy_tree_changed(item, run / item.name)
            else:
                run.mkdir(parents=True, exist_ok=True)
                target = run / item.name
                if target.exists() and _hash(target) != _hash(item):
                    raise ValueError(f"restore would overwrite differing artifact: {target}")
                shutil.copy2(item, target)
    if (run / "manifest.json").exists():
        manifest = _read_json(run / "manifest.json")
        for relative, expected in manifest.get("stages", {}).get("prepare", {}).get("artifacts", {}).items():
            dest, cached = root / relative, backup / "prepared_cache" / relative
            if dest.exists():
                if _hash(dest) != expected:
                    raise ValueError(f"resume preparation artifact differs: {dest}")
            elif cached.is_file():
                if _hash(cached) != expected:
                    raise ValueError(f"Drive preparation cache hash mismatch: {cached}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, dest)


def _stage_artifacts(root, run, stage):
    if stage == "prepare":
        paths = [root / p for p in PREPARED]
    elif stage == "bootstrap":
        paths = [p for p in (run / "bootstrap").rglob("*") if p.is_file()]
    else:
        suffix = "validation" if stage == "validate" else "refit"
        paths = [p for p in (run / "contexts" / suffix).rglob("*") if p.is_file()]
        for profile in _read_json(run / "effective_config.json")["profiles"]:
            paths += [p for p in (run / profile / suffix).rglob("*") if p.is_file()]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError(f"stage {stage} did not produce all required outputs")
    if stage == "bootstrap":
        for name in ("manifest.json", "forward_teacher.npz", "validation_test_teacher.npz", "refit_test_teacher.npz"):
            if not (run / "bootstrap" / name).is_file():
                raise ValueError(f"missing bootstrap artifact {name}")
    if stage in {"validate", "refit"}:
        suffix = "validation" if stage == "validate" else "refit"
        for name in ("features_v3.parquet", "features_v5.parquet", "cleaning_rows.parquet",
                     "band_diagnostics.json", "cleaning_diagnostics.json"):
            if not (run / "contexts" / suffix / name).is_file():
                raise ValueError(f"missing context artifact {suffix}/{name}")
        required = ("manifest.json", "schema.json", "meta.parquet", "features_v3.parquet", "features_v5.parquet", "groups.npy")
        required += ("metrics.json", "per_class_metrics.csv", "offsets.json", "validation_proba.npz") if stage == "validate" else ("test_proba.npz", "submission.csv", "submission_metrics.json")
        for profile in _read_json(run / "effective_config.json")["profiles"]:
            for name in required:
                if not (run / profile / suffix / name).is_file():
                    raise ValueError(f"missing {profile}/{suffix}/{name}")
    return {str(path.relative_to(root)).replace("\\", "/"): _hash(path) for path in paths}


def _resume_stage(root, manifest, stage, dependencies):
    saved = manifest.get("stages", {}).get(stage)
    if saved is None:
        return False
    if saved.get("dependencies") != dependencies:
        raise ValueError(f"upstream stage dependencies differ: {stage}")
    for relative, expected in saved["artifacts"].items():
        path = root / relative
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"resume artifact missing/changed: {path}")
    return True


def _load_api(root):
    # Runtime-only import, never reached by planning/dry-run.
    sys.path.insert(0, str(root / "src/upi"))
    import fit_submit
    import make_submission
    return fit_submit, make_submission


def _baseline_comparison(submission, baseline, labels):
    import pandas as pd
    current, old = pd.read_csv(submission), pd.read_csv(baseline)
    for name, frame in (("submission", current), ("baseline", old)):
        if set(frame.columns) != {"id", "label"} or frame.id.isna().any() or frame.id.duplicated().any() or frame.label.isna().any() or not frame.label.isin(labels).all():
            raise ValueError(f"invalid {name} CSV")
    if len(old) != len(current) or not current.id.isin(old.id).all():
        raise ValueError("baseline/submission ID sets differ")
    old = current[["id"]].merge(old, on="id", validate="one_to_one", sort=False)
    changed = int(current.label.ne(old.label).sum())
    def distribution(frame):
        return {"fraud_rate": float(frame.label.ne("legit").mean()),
                "class_shares": {lab: float(frame.label.eq(lab).mean()) for lab in labels}}
    return {"available": True, "interpretation": "distribution comparison only; no true test labels",
            "changed_rows": changed, "disagreement_rate": changed / len(current) if len(current) else 0.,
            "current": distribution(current), "reported_05400_baseline": distribution(old)}


def validate_artifacts(root: Path, run_dir: Path) -> dict:
    """Strict external-run checks; reads only that run and supplied raw IDs."""
    import numpy as np
    import pandas as pd
    root, run = Path(root), Path(run_dir)
    fit, submit = _load_api(root)
    config = _read_json(run / "effective_config.json")
    report = {"class_order": list(LABELS), "profiles": {}}
    baseline = _read_json(run / "manifest.json").get("baseline_csv")
    for profile in config["profiles"]:
        directory = run / profile
        validation = directory / "validation"
        metrics = _read_json(validation / "metrics.json")
        if metrics["label_source"] != "original":
            raise ValueError("headline metrics must use original labels")
        meta = pd.read_parquet(validation / "meta.parquet")
        probability = fit.load_probability_artifact(validation / "validation_proba.npz", meta.txn_id)
        if probability.shape != (len(meta), 12):
            raise ValueError("validation probability shape mismatch")
        for folder in (validation, directory / "refit"):
            if not folder.is_dir():
                continue
            stage_meta = pd.read_parquet(folder / "meta.parquet")
            for artifact in folder.glob("*proba.npz"):
                with np.load(artifact, allow_pickle=False) as saved:
                    ids = saved["ids"]
                index = pd.Index(ids)
                if index.has_duplicates or index.isna().any() or not index.isin(stage_meta.txn_id).all():
                    raise ValueError(f"invalid probability artifact IDs: {artifact}")
                fit.load_probability_artifact(artifact, ids)
        for partition in ("early_stop", "calibration", "assessment", "full_september"):
            values = metrics["partitions"][partition]
            if values["class_order"] != LABELS or set(values["per_class"]) != set(LABELS):
                raise ValueError(f"missing 12-class report for {profile}/{partition}")
            if not np.isfinite(values["macro_f1"]) or not 0 <= values["macro_f1"] <= 1:
                raise ValueError("invalid macro-F1")
        pd.DataFrame(metrics["partitions"]["assessment"]["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
            validation / "assessment_confusion_matrix.csv", index_label="actual")
        profile_report = {"original_label_macro_f1": {
            name: metrics["partitions"][name]["macro_f1"]
            for name in ("early_stop", "calibration", "assessment", "full_september")},
            "assessment_per_class": metrics["partitions"]["assessment"]["per_class"],
            "recall_selection": metrics.get("recall_selection"),
            "assessment_legacy_baseline": metrics["partitions"]["assessment"].get("legacy_baseline"),
            "baseline_comparison": {"available": False, "reason": "no baseline CSV supplied"}}
        refit = directory / "refit"
        if refit.exists():
            meta = pd.read_parquet(refit / "meta.parquet")
            p = fit.load_probability_artifact(refit / "test_proba.npz", meta.txn_id)
            sample = pd.read_csv(root / "data/raw/sample_submission.csv")
            csv_path = refit / "submission.csv"
            result = pd.read_csv(csv_path)
            if result.columns.tolist() != ["id", "label"] or result.isna().any().any() or result.id.duplicated().any():
                raise ValueError("invalid submission columns, missing values or duplicate IDs")
            if not np.array_equal(result.id.to_numpy(), sample.id.to_numpy()) or not result.label.isin(LABELS).all():
                raise ValueError("submission/sample order, IDs or labels differ")
            if len(result) != len(p) or len(meta) != len(sample):
                raise ValueError("submission row counts differ")
            f3, f5 = pd.read_parquet(refit / "features_v3.parquet"), pd.read_parquet(refit / "features_v5.parquet")
            groups = np.load(refit / "groups.npy", allow_pickle=False)
            offsets = _read_json(validation / "offsets.json")
            expected = submit.predict_labels(p, groups, offsets, {"meta": meta, "f3": f3, "f5": f5})
            keyed = pd.DataFrame({"id": meta.txn_id, "label": np.asarray(LABELS)[expected]})
            expected_csv = sample[["id"]].merge(keyed, on="id", validate="one_to_one", sort=False)
            if not np.array_equal(result.label, expected_csv.label):
                raise ValueError("submission does not match saved probabilities/context/offsets")
            profile_report["submission"] = {
                "path": str(csv_path), "rows": len(result), "sha256": _hash(csv_path),
                "fraud_rate": float(result.label.ne("legit").mean()),
                "class_shares": {lab: float(result.label.eq(lab).mean()) for lab in LABELS}}
            if baseline:
                profile_report["baseline_comparison"] = _baseline_comparison(csv_path, baseline, LABELS)
            _json(refit / "prediction_distribution.json", profile_report["submission"])
        schema = _read_json(validation / "schema.json")
        report["profiles"][profile] = profile_report
        report.setdefault("feature_schema_hashes", {})[profile] = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()).hexdigest()
    _json(run / "validation_report.json", report)
    lines = ["# Colab run report", "", "Metrics use original training labels.",
             "Public leaderboard performance is unmeasured by this runner.", ""]
    for profile, values in report["profiles"].items():
        lines += [f"## {profile}", ""]
        lines += [f"- {name}: {score:.6f}" for name, score in values["original_label_macro_f1"].items()]
        selection = values.get("recall_selection")
        legacy = values.get("assessment_legacy_baseline")
        if selection:
            lines += ["", f"Selected policy: {selection['selected']}; fraud-weighted member share: {selection['recall_weight']:.2f}.",
                      "Selection uses Sep20-23, after threshold fitting on Sep16-19."]
        if legacy:
            score = values["original_label_macro_f1"]["assessment"]
            lines += [f"Legacy ensemble assessment: {legacy['macro_f1']:.6f}; change: {score - legacy['macro_f1']:+.6f}.",
                      "", "Assessment change versus legacy (precision / recall / F1):", ""]
            for lab, value in values["assessment_per_class"].items():
                old = legacy["per_class"][lab]
                lines.append(f"- {lab}: {value['precision'] - old['precision']:+.4f} / {value['recall'] - old['recall']:+.4f} / {value['f1'] - old['f1']:+.4f}")
        lines += ["", "Assessment per class (precision / recall / F1 / support):", ""]
        for lab, value in values["assessment_per_class"].items():
            lines.append(f"- {lab}: {value['precision']:.4f} / {value['recall']:.4f} / {value['f1']:.4f} / {value['support']}")
        lines += ["", f"Baseline distribution comparison: {values['baseline_comparison']}", ""]
        if "submission" in values:
            lines += [f"Submission: {values['submission']['path']}",
                      f"SHA256: {values['submission']['sha256']}", ""]
    (run / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _action(args):
    root, run = args.root.resolve(), args.root.resolve() / "runs" / args.tag
    if args._action == "environment":
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
        required = {}
        for line in (root / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.strip().startswith("#"):
                name, version = line.strip().split("==", 1)
                required[name] = version
        for name, expected in required.items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"dependency {name} must equal {expected}; use --install-deps")
        distributions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        _json(run / "environment.json", {"python": platform.python_version(),
              "platform": platform.platform(), "versions": versions,
              "installed_distributions": distributions})
        (run / "requirements-resolved.txt").write_text(
            "\n".join(f"{name}=={version}" for name, version in sorted(distributions.items())) + "\n",
            encoding="utf-8")
    elif args._action == "validate":
        validate_artifacts(root, run)
    elif args._action == "ratios":
        import numpy as np
        import pandas as pd
        base = pd.read_parquet(root / "data/processed/base_v0.parquet", columns=["txn_id"])
        ratios = pd.read_parquet(root / "data/processed/ratios_v0.parquet")
        if ratios.index.name != "txn_id" or not np.array_equal(ratios.index, base.txn_id):
            raise ValueError("ratio IDs do not equal base order before v1")
        print("Ratio/base ID alignment verified")
    elif args._action == "backend":
        import numpy as np
        import lightgbm as lgb
        X = np.random.default_rng(17).normal(size=(24, 4))
        y = np.arange(24) % 3
        actual, error = args.device, None
        try:
            model = lgb.LGBMClassifier(n_estimators=2, num_leaves=3, min_child_samples=1,
                                      objective="multiclass", n_jobs=1, verbosity=-1,
                                      device_type=args.device, random_state=17)
            model.fit(X, y)
        except lgb.basic.LightGBMError as exc:
            if not args.allow_cpu_fallback:
                raise RuntimeError(f"requested {args.device} backend probe failed: {exc}") from exc
            actual, error = "cpu", str(exc)
            cpu = lgb.LGBMClassifier(n_estimators=2, num_leaves=3, min_child_samples=1,
                                    objective="multiclass", n_jobs=1, verbosity=-1,
                                    device_type="cpu", random_state=17)
            cpu.fit(X, y)
        _json(run / "backend.json", {"requested": args.device, "actual": actual,
              "gpu_probe_passed": actual == args.device, "fallback_reason": error})
        print(f"Requested backend: {args.device}; selected backend: {actual}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps({"branch": "codex/target-070", "stage": args.stage,
                          "requested_device": args.device, "commands": plan_commands(args)},
                         indent=2))
        return 0
    if args._action:
        return _action(args)
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        print("Pinned Colab stack requires Python3.11-3.13; use a supported Colab runtime.", file=sys.stderr)
        return 2
    manifest, run, root = None, None, args.root.resolve()
    original_environment = None
    try:
        if args.mount_drive:
            from google.colab import drive
            drive.mount("/content/drive")
        if not root.is_dir():
            raise FileNotFoundError(f"project root missing: {root}")
        run = root / "runs" / args.tag
        backup = args.output_dir.resolve() / args.tag
        if backup.is_relative_to(run) and backup != run:
            raise ValueError("backup cannot be inside the local run")
        config_path = args.config if args.config.is_absolute() else root / args.config
        config = _read_json(config_path)
        config.update(tag=args.tag, profiles=args.profiles.split(","), device=args.device,
                      allow_cpu_fallback=args.allow_cpu_fallback, output_dir=str(run))
        input_hashes = _copy_inputs(args)
        commit = _capture(["git", "rev-parse", "HEAD"], root)
        dirty = _capture(["git", "status", "--porcelain"], root)
        source_hashes = {name: _hash(root / name) for name in SOURCE_FILES}
        identity = {"commit_sha": commit, "source_hashes": source_hashes,
                    "config": json.loads(json.dumps(config)), "input_hashes": input_hashes,
                    "baseline_sha256": _hash(args.baseline_csv) if args.baseline_csv else None}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if args.resume:
            restore_manifest = run / "manifest.json" if (run / "manifest.json").exists() else backup / "manifest.json"
            if restore_manifest.exists() and _read_json(restore_manifest).get("fingerprint") != fingerprint:
                raise ValueError("resume source/config/commit/input hashes differ")
            _restore(args)
        previous = _read_json(run / "manifest.json") if (run / "manifest.json").exists() else None
        if previous is not None:
            if not args.resume:
                raise ValueError("run already exists; use --resume with unchanged inputs/config/source or a fresh --tag")
            if previous["fingerprint"] != fingerprint:
                raise ValueError("resume source/config/commit/input hashes differ")
            manifest = previous
            original_environment = manifest.get("environment")
        else:
            if run.exists() and any(run.iterdir()):
                raise ValueError("nonempty run directory without a manifest; use a fresh --tag")
            manifest = {**identity, "fingerprint": fingerprint, "class_order": LABELS,
                        "dirty_state": dirty, "cli": vars(args).copy(), "stages": {},
                        "baseline_csv": str(args.baseline_csv.resolve()) if args.baseline_csv else None,
                        "started_utc": datetime.now(timezone.utc).isoformat()}
            manifest["cli"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["cli"].items()}
        run.mkdir(parents=True, exist_ok=True)
        manifest["baseline_csv"] = str(args.baseline_csv.resolve()) if args.baseline_csv else None
        for name in ("data/interim", "data/processed", "reports/eda/tables", "reports/eda/figures"):
            (root / name).mkdir(parents=True, exist_ok=True)
        _json(run / "manifest.json", manifest)
        dependencies = {}
        for stage, commands in _stage_commands(args):
            if stage != "environment" and args.resume and _resume_stage(root, manifest, stage, dependencies):
                print("Verified checkpoint; skipping:", stage, flush=True)
                dependencies[stage] = hashlib.sha256(json.dumps(manifest["stages"][stage], sort_keys=True).encode()).hexdigest()
                continue
            start = time.monotonic()
            manifest["active_stage"] = stage
            _json(run / "manifest.json", manifest)
            for i, command in enumerate(commands):
                run_checked(command, cwd=root, log_path=run / "logs" / f"{stage}_{i:02d}.log")
            if stage == "environment":
                environment = _read_json(run / "environment.json")
                if original_environment and (environment["versions"] != original_environment["versions"] or environment["python"] != original_environment["python"] or environment["installed_distributions"] != original_environment["installed_distributions"]):
                    raise ValueError("resume dependency versions differ")
                manifest["environment"] = environment
                backend = (_read_json(run / "backend.json") if args.device != "cpu"
                           else {"requested": "cpu", "actual": "cpu", "gpu_probe_passed": False})
                if manifest.get("backend") and any(manifest["backend"][key] != backend[key] for key in ("requested", "actual", "gpu_probe_passed")):
                    raise ValueError("resume backend changed")
                manifest["backend"] = backend
                config["device"] = backend["actual"]
                manifest["effective_config"] = json.loads(json.dumps(config))
                _json(run / "effective_config.json", config)
                # Keep Card2's input/source fingerprints and metadata in a
                # separate file: runner manifest is not overwritten by it.
            else:
                if stage == "bootstrap":
                    card2_inputs = _read_json(run / "inputs.json")
                    year = card2_inputs["dataset_year"]
                    manifest["dataset_year"] = year
                    manifest["resolved_split_dates"] = {
                        name: [f"{year}-{start}", f"{year}-{end}"]
                        for name, (start, end) in config["splits"].items()}
                if stage in {"validate", "refit"}:
                    report = _read_json(run / "validation_report.json")
                    manifest["feature_schema_hashes"] = report["feature_schema_hashes"]
                artifacts = _stage_artifacts(root, run, stage)
                manifest["stages"][stage] = {"dependencies": dict(dependencies),
                    "artifacts": artifacts, "duration_seconds": time.monotonic() - start}
                dependencies[stage] = hashlib.sha256(json.dumps(manifest["stages"][stage], sort_keys=True).encode()).hexdigest()
            manifest.setdefault("timings_seconds", {})[stage] = time.monotonic() - start
            _json(run / "manifest.json", manifest)
            _backup(args, manifest)
        manifest["last_completed_stage"] = args.stage
        manifest.pop("last_failure", None)
        manifest.pop("active_stage", None)
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _json(run / "manifest.json", manifest)
        _backup(args, manifest)
        print("Completed. Artifacts:", run)
        print("Persistent copy:", backup)
        if args.stage in {"validate", "refit", "all"}:
            print((run / "RUN_REPORT.md").read_text(encoding="utf-8"))
        return 0
    except Exception as exc:
        code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) and exc.returncode else 1
        print(f"Pipeline stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if manifest is not None and run is not None:
            try:
                manifest["last_failure"] = {"error": str(exc), "exit_code": code,
                                           "utc": datetime.now(timezone.utc).isoformat()}
                _json(run / "manifest.json", manifest)
                _backup(args, manifest)
            except Exception as backup_error:
                print(f"Failure artifact copy also failed: {backup_error}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
