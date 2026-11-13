from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import environment_manifest, git_revision, write_json, write_submission
from .config import resolve_path
from .constants import LABELS, LABEL_TO_ID, NON_FEATURE_COLUMNS
from .data import audit_inputs, read_inputs, sha256_file
from .events import Events
from .features import build_feature_table, feature_fingerprint
from .metrics import apply_weights, classification_report_dict, fit_class_weights, macro_f1_ids
from .model import best_iteration, create_model, fit_model, predict_aligned, save_model
from .stage2 import CHANNEL_NAMES, apply_propagation, compute_neighbour_channels, fit_propagation_and_weights

MINUTE, HOUR = 60, 3600


def _input_hashes(raw_dir: Path) -> dict[str, str]:
    return {name: sha256_file(raw_dir / name) for name in
            ("train.csv", "test.csv", "accounts.csv", "merchants.csv", "sample_submission.csv")}


def feature_path(config: dict) -> Path:
    processed = resolve_path(config, "processed_dir", "data/processed")
    cache_name = config.get("features", {}).get("cache_name", config.get("name", "default"))
    return processed / f"features_{cache_name}.parquet"


def build_features(config: dict, force: bool = False, sample_fraction: float | None = None) -> Path:
    raw = resolve_path(config, "raw_dir", "data/raw")
    target = feature_path(config)
    manifest_path = target.with_suffix(".manifest.json")
    hashes = _input_hashes(raw)
    fingerprint = feature_fingerprint(config, hashes)
    if target.exists() and manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and not sample_fraction:
            return target
    train, test, accounts, merchants, _ = read_inputs(raw, sample_fraction=sample_fraction)
    started = time.perf_counter()
    table = build_feature_table(
        train, test, accounts, merchants,
        include_relational=config.get("features", {}).get("relational", True),
        include_history=config.get("features", {}).get("prior_label_history", True),
    )
    if table.txn_id.duplicated().any():
        raise ValueError("duplicate IDs in feature table")
    target.parent.mkdir(parents=True, exist_ok=True)
    if sample_fraction:
        target = target.with_name(target.stem + "_smoke.parquet")
        manifest_path = target.with_suffix(".manifest.json")
    table.to_parquet(target, index=False)
    write_json(manifest_path, {"fingerprint": fingerprint, "rows": len(table),
                               "columns": table.columns.tolist(), "elapsed_seconds": time.perf_counter() - started,
                               "sample_fraction": sample_fraction, "input_hashes": hashes})
    return target


def select_feature_columns(table: pd.DataFrame, config: dict) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.update(config.get("features", {}).get("exclude", []))
    columns = [c for c in table.columns if c not in excluded and pd.api.types.is_numeric_dtype(table[c])]
    if not columns:
        raise ValueError("no numeric model features selected")
    return columns


def _clean_burst_labels(fit_df: pd.DataFrame, y_fit: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Relabel unreported legit rows sitting inside pure training fraud bursts (fit partition only)."""
    y_clean = y_fit.copy()
    t = pd.to_datetime(fit_df.timestamp).to_numpy(dtype="datetime64[s]").astype(np.int64)
    p2p = fit_df.receiver_type.eq("P2P").to_numpy()
    rcv = pd.factorize(fit_df.receiver_id, sort=False)[0].astype(np.int64)
    dev = pd.factorize(fit_df.device_id, sort=False)[0].astype(np.int64)
    snd = pd.factorize(fit_df.sender_id, sort=False)[0].astype(np.int64)

    legit_mask = y_clean == LABEL_TO_ID["legit"]

    # 1. Mule burst: legit P2P row to same receiver within +-1h of >=2 mule_account rows
    mule_id = LABEL_TO_ID["mule_account"]
    mule_ev = Events(rcv, t, mask=(y_fit == mule_id) & p2p)
    near_mule = mule_ev.count(np.where(p2p, rcv, -1), t, -HOUR, HOUR) >= 2
    relabel_mule = legit_mask & p2p & near_mule
    y_clean[relabel_mule] = mule_id

    # 2. Velocity bot burst: legit row on same device within +-5m of >=2 velocity_bot rows
    bot_id = LABEL_TO_ID["velocity_bot"]
    bot_ev = Events(dev, t, mask=(y_fit == bot_id))
    near_bot = bot_ev.count(dev, t, -5 * MINUTE, 5 * MINUTE) >= 2
    relabel_bot = (y_clean == LABEL_TO_ID["legit"]) & near_bot
    y_clean[relabel_bot] = bot_id

    # 3. Refund abuse: legit REFUND row by sender who has >=2 refund_abuse rows in training
    ref_id = LABEL_TO_ID["refund_abuse"]
    is_refund = fit_df.txn_type.eq("REFUND").to_numpy() if "txn_type" in fit_df else np.zeros(len(fit_df), dtype=bool)
    ref_snd_counts = np.bincount(snd[y_fit == ref_id], minlength=len(fit_df))
    relabel_ref = (y_clean == LABEL_TO_ID["legit"]) & is_refund & (ref_snd_counts[snd] >= 2)
    y_clean[relabel_ref] = ref_id

    stats = {
        "relabeled_mule": int(relabel_mule.sum()),
        "relabeled_velocity_bot": int(relabel_bot.sum()),
        "relabeled_refund_abuse": int(relabel_ref.sum()),
    }
    return y_clean, stats


def _new_run_dir(config: dict) -> Path:
    root = resolve_path(config, "runs_dir", "runs")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{config.get('name', 'run')}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def train_validation(config: dict, features: Path | None = None, run_dir: Path | None = None) -> Path:
    root = Path(config["_config_path"]).parents[1]
    features = features or build_features(config)
    table = pd.read_parquet(features)
    table = table[table.split.eq("train") & table.timestamp.lt("2026-10-01")].copy()
    columns = select_feature_columns(table, config)
    dates = config.get("validation", {})
    train_start = pd.Timestamp(dates.get("train_start", "2026-07-01"))
    calibration_start = pd.Timestamp(dates.get("calibration_start", "2026-09-01"))
    assessment_start = pd.Timestamp(dates.get("assessment_start", "2026-09-24"))
    assessment_end = pd.Timestamp(dates.get("assessment_end", "2026-10-01"))
    fit_mask = table.timestamp.ge(train_start) & table.timestamp.lt(calibration_start)
    cal_mask = table.timestamp.ge(calibration_start) & table.timestamp.lt(assessment_start)
    assess_mask = table.timestamp.ge(assessment_start) & table.timestamp.lt(assessment_end)
    if min(int(fit_mask.sum()), int(cal_mask.sum()), int(assess_mask.sum())) == 0:
        raise ValueError("one or more temporal partitions are empty")
    y = table.label.map(LABEL_TO_ID).astype(int).to_numpy()
    y_fit = y[fit_mask]
    cleaning_stats = {}
    if config.get("training", {}).get("clean_burst_labels", False):
        y_fit, cleaning_stats = _clean_burst_labels(table.loc[fit_mask], y_fit)

    model = create_model(config)
    started = time.perf_counter()
    fit_model(model, table.loc[fit_mask, columns], y_fit, table.loc[cal_mask, columns], y[cal_mask])

    cal_probabilities = predict_aligned(model, table.loc[cal_mask, columns])
    assess_probabilities = predict_aligned(model, table.loc[assess_mask, columns])
    probabilities = np.vstack([cal_probabilities, assess_probabilities])
    cal_count = len(cal_probabilities)

    validation_rows = pd.concat([table.loc[cal_mask], table.loc[assess_mask]], axis=0).reset_index(drop=True)
    y_val = np.r_[y[cal_mask], y[assess_mask]]

    stage1_weights = fit_class_weights(cal_probabilities, y[cal_mask])
    stage1_pred = apply_weights(probabilities, stage1_weights)
    stage1_reports = {
        "calibration": classification_report_dict(y[cal_mask], stage1_pred[:cal_count]),
        "assessment": classification_report_dict(y[assess_mask], stage1_pred[cal_count:]),
        "full_september": classification_report_dict(y_val, stage1_pred),
    }

    use_stage2 = config.get("stage2", {}).get("neighbour_propagation", False)
    alpha = np.zeros((len(CHANNEL_NAMES), len(LABELS)), dtype=np.float64)
    weights = stage1_weights
    prediction = stage1_pred
    reports = stage1_reports

    if use_stage2:
        channels = compute_neighbour_channels(validation_rows, probabilities)
        alpha, prop_weights = fit_propagation_and_weights(
            probabilities[:cal_count], channels[:cal_count], y[cal_mask]
        )
        prop_probabilities = apply_propagation(probabilities, channels, alpha)
        prop_pred = apply_weights(prop_probabilities, prop_weights)
        stage2_reports = {
            "calibration": classification_report_dict(y[cal_mask], prop_pred[:cal_count]),
            "assessment": classification_report_dict(y[assess_mask], prop_pred[cal_count:]),
            "full_september": classification_report_dict(y_val, prop_pred),
        }
        if stage2_reports["calibration"]["macro_f1"] >= stage1_reports["calibration"]["macro_f1"]:
            weights = prop_weights
            prediction = prop_pred
            reports = stage2_reports

    run_dir = run_dir or _new_run_dir(config)
    save_model(model, run_dir / "model.pkl")
    np.save(run_dir / "validation_probabilities.npy", probabilities)
    validation = pd.DataFrame({
        "txn_id": validation_rows.txn_id.to_numpy(),
        "timestamp": validation_rows.timestamp.to_numpy(),
        "sender_id": validation_rows.sender_id.to_numpy(),
        "receiver_id": validation_rows.receiver_id.to_numpy(),
        "receiver_type": validation_rows.receiver_type.to_numpy(),
        "device_id": validation_rows.device_id.to_numpy(),
        "label": validation_rows.label.to_numpy(),
        "prediction": np.asarray(LABELS)[prediction],
    })
    validation.to_parquet(run_dir / "validation_predictions.parquet", index=False)

    fallback = int(config.get("model", {}).get("params", {}).get("n_estimators", 1200))
    manifest = {
        "config": config,
        "git": git_revision(root),
        "environment": environment_manifest(),
        "features_path": str(features),
        "feature_columns": columns,
        "feature_file_sha256": sha256_file(features),
        "class_order": LABELS,
        "weights": weights.tolist(),
        "stage1_weights": stage1_weights.tolist(),
        "propagation_alpha": alpha.tolist(),
        "propagation_channels": list(CHANNEL_NAMES),
        "cleaning_stats": cleaning_stats,
        "best_iteration": best_iteration(model, fallback),
        "partitions": {"fit": int(fit_mask.sum()), "calibration": int(cal_mask.sum()),
                       "assessment": int(assess_mask.sum())},
        "reports": reports,
        "stage1_reports": stage1_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def final_fit_predict(config: dict, validation_run: Path, output: Path | None = None) -> Path:
    manifest = json.loads((validation_run / "manifest.json").read_text(encoding="utf-8"))
    features = Path(manifest["features_path"])
    if sha256_file(features) != manifest["feature_file_sha256"]:
        raise ValueError("feature file differs from validation run")
    table = pd.read_parquet(features)
    columns = manifest["feature_columns"]
    train = table[table.split.eq("train")].copy()
    test = table[table.split.eq("test")].copy()
    y = train.label.map(LABEL_TO_ID).astype(int).to_numpy()
    if config.get("training", {}).get("clean_burst_labels", False):
        y, _ = _clean_burst_labels(train, y)

    iterations = max(50, int(manifest["best_iteration"] * config.get("final_fit", {}).get("iteration_multiplier", 1.15)))
    model = create_model(config, n_estimators=iterations)
    fit_model(model, train[columns], y)
    probabilities = predict_aligned(model, test[columns])
    weights = np.asarray(manifest["weights"], dtype=float)
    alpha = np.asarray(manifest.get("propagation_alpha", []), dtype=float)

    effective_probabilities = probabilities
    if alpha.size > 0 and np.any(alpha > 0):
        test_channels = compute_neighbour_channels(test, probabilities)
        effective_probabilities = apply_propagation(probabilities, test_channels, alpha)

    predictions = apply_weights(effective_probabilities, weights)
    save_model(model, validation_run / "final_model.pkl")
    np.save(validation_run / "test_probabilities.npy", probabilities)
    pd.DataFrame({
        "txn_id": test.txn_id.to_numpy(),
        "timestamp": test.timestamp.to_numpy(),
        "sender_id": test.sender_id.to_numpy(),
        "receiver_id": test.receiver_id.to_numpy(),
        "receiver_type": test.receiver_type.to_numpy(),
        "device_id": test.device_id.to_numpy(),
    }).to_parquet(validation_run / "test_ids.parquet", index=False)
    raw = resolve_path(config, "raw_dir", "data/raw")
    sample = pd.read_csv(raw / "sample_submission.csv")
    output = output or resolve_path(config, "submissions_dir", "submissions") / f"{config.get('name', 'model')}.csv"
    submission = write_submission(output, test.txn_id.to_numpy(), predictions, sample)
    write_json(validation_run / "final_fit.json", {"iterations": iterations, "submission": submission})
    return output


def blend_runs(config: dict, run_dirs: list[Path], output: Path | None = None,
               blend_weights: list[float] | None = None) -> Path:
    if len(run_dirs) < 2:
        raise ValueError("blend requires at least two run directories")
    reference = pd.read_parquet(run_dirs[0] / "validation_predictions.parquet")
    reference_ids = reference.txn_id.tolist()
    probability_blocks = []
    for run in run_dirs:
        rows = pd.read_parquet(run / "validation_predictions.parquet")
        probabilities = np.load(run / "validation_probabilities.npy")
        if len(rows) != len(probabilities) or set(rows.txn_id) != set(reference_ids):
            raise ValueError(f"validation rows differ for blend member {run}")
        positions = pd.Series(np.arange(len(rows)), index=rows.txn_id).loc[reference_ids].to_numpy()
        probability_blocks.append(probabilities[positions])

    y = reference.label.map(LABEL_TO_ID).astype(int).to_numpy()
    assessment_start = pd.Timestamp(config.get("validation", {}).get("assessment_start", "2026-09-24"))
    calibration = pd.to_datetime(reference.timestamp).lt(assessment_start).to_numpy()

    if blend_weights is not None:
        weights_array = np.asarray(blend_weights, dtype=float)
        if len(weights_array) != len(run_dirs) or np.any(weights_array < 0) or weights_array.sum() <= 0:
            raise ValueError("blend weights must be non-negative and match run count")
        weights_array /= weights_array.sum()
    else:
        weights_array = np.ones(len(run_dirs), dtype=float) / len(run_dirs)

    probabilities = sum(weight * block for weight, block in zip(weights_array, probability_blocks))
    decision_weights = fit_class_weights(probabilities[calibration], y[calibration])
    effective_val_probs = probabilities

    alpha = np.zeros((len(CHANNEL_NAMES), len(LABELS)), dtype=np.float64)
    has_entity_cols = all(c in reference.columns for c in ("sender_id", "receiver_id", "receiver_type", "device_id"))
    if config.get("stage2", {}).get("neighbour_propagation", False) and has_entity_cols:
        val_channels = compute_neighbour_channels(reference, probabilities)
        alpha, prop_weights = fit_propagation_and_weights(
            probabilities[calibration], val_channels[calibration], y[calibration]
        )
        prop_val = apply_propagation(probabilities, val_channels, alpha)
        if macro_f1_ids(y[calibration], apply_weights(prop_val[calibration], prop_weights)) >= macro_f1_ids(
            y[calibration], apply_weights(probabilities[calibration], decision_weights)
        ):
            effective_val_probs = prop_val
            decision_weights = prop_weights

    prediction = apply_weights(effective_val_probs, decision_weights)
    run_dir = _new_run_dir({**config, "name": config.get("name", "blend") + "_blend"})
    np.save(run_dir / "validation_probabilities.npy", probabilities)
    validation = reference.copy()
    validation["prediction"] = np.asarray(LABELS)[prediction]
    validation.to_parquet(run_dir / "validation_predictions.parquet", index=False)
    reports = {
        "calibration": classification_report_dict(y[calibration], prediction[calibration]),
        "assessment": classification_report_dict(y[~calibration], prediction[~calibration]),
        "full_september": classification_report_dict(y, prediction),
    }
    manifest = {
        "kind": "probability_blend",
        "members": [str(p) for p in run_dirs],
        "blend_weights": weights_array.tolist(),
        "weights": decision_weights.tolist(),
        "propagation_alpha": alpha.tolist(),
        "class_order": LABELS,
        "reports": reports,
        "config": config,
    }

    test_available = all((run / "test_probabilities.npy").exists() and (run / "test_ids.parquet").exists()
                         for run in run_dirs)
    if test_available:
        reference_test = pd.read_parquet(run_dirs[0] / "test_ids.parquet")
        test_blocks = []
        for run in run_dirs:
            ids = pd.read_parquet(run / "test_ids.parquet")
            probs = np.load(run / "test_probabilities.npy")
            if len(ids) != len(probs) or set(ids.txn_id) != set(reference_test.txn_id):
                raise ValueError(f"test rows differ for blend member {run}")
            positions = pd.Series(np.arange(len(ids)), index=ids.txn_id).loc[reference_test.txn_id].to_numpy()
            test_blocks.append(probs[positions])
        test_probabilities = sum(weight * block for weight, block in zip(weights_array, test_blocks))
        np.save(run_dir / "test_probabilities.npy", test_probabilities)
        reference_test.to_parquet(run_dir / "test_ids.parquet", index=False)

        effective_test_probs = test_probabilities
        has_test_entities = all(c in reference_test.columns for c in ("sender_id", "receiver_id", "receiver_type", "device_id"))
        if alpha.size > 0 and np.any(alpha > 0) and has_test_entities:
            test_channels = compute_neighbour_channels(reference_test, test_probabilities)
            effective_test_probs = apply_propagation(test_probabilities, test_channels, alpha)

        sample = pd.read_csv(resolve_path(config, "raw_dir", "data/raw") / "sample_submission.csv")
        output = output or resolve_path(config, "submissions_dir", "submissions") / "blend.csv"
        manifest["submission"] = write_submission(
            output, reference_test.txn_id.to_numpy(), apply_weights(effective_test_probs, decision_weights), sample
        )
    write_json(run_dir / "manifest.json", manifest)
    return run_dir
