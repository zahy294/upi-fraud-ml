"""Forward teacher, fold-specific features, diverse LightGBM fits and final refit.

Training is invoked only by main/run_experiment, never on import.
"""
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
    from .features_v6 import (align_pack, validate_ids, assemble_features, apply_feature_schema,
                              build_label_free_v6, build_temporal_te)
    from .bands import build_bands
    from .clean_labels import clean_training_labels
    from .make_submission import (prepare_rule_context, slice_rule_context, event_groups,
                                  fit_offsets, predict_labels, evaluate_predictions, write_submission,
                                  select_recall_policy)
except ImportError:
    from common import LABELS
    from features_v6 import (align_pack, validate_ids, assemble_features, apply_feature_schema,
                             build_label_free_v6, build_temporal_te)
    from bands import build_bands
    from clean_labels import clean_training_labels
    from make_submission import (prepare_rule_context, slice_rule_context, event_groups,
                                 fit_offsets, predict_labels, evaluate_predictions, write_submission,
                                 select_recall_policy)

PROFILES = {"control": ("v1", "v2", "v3", "v4", "v5"),
            "fusion": ("v1", "v2", "v3", "v4", "v5", "v6"),
            "temporal": ("v1", "v2", "v3", "v4", "v5", "v6", "te"),
            "ensemble": ("v1", "v2", "v3", "v4", "v5", "v6", "te")}


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def ids_hash(ids):
    return hashlib.sha256(json.dumps(np.asarray(ids).astype(str).tolist(),
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    p, classes = np.asarray(proba, float), np.asarray(classes)
    if classes.ndim != 1 or classes.dtype.kind not in "iu" or len(np.unique(classes)) != len(classes) or not np.isin(classes, np.arange(12)).all():
        raise ValueError("classes must be distinct integer labels in 0..11")
    if p.ndim != 2 or p.shape[1] != len(classes) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid probability shape/values")
    result = np.zeros((len(p), 12))
    result[:, classes.astype(int)] = p
    total = result.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("probability row has zero total")
    return result / total[:, None]


def save_probability_artifact(path, ids, proba, *, provenance=None, allow_missing=False):
    raw_ids = pd.Index(np.asarray(ids))
    if raw_ids.has_duplicates or raw_ids.isna().any():
        raise ValueError("artifact IDs must be unique and non-null")
    ids = np.asarray(ids).astype(str)
    index = pd.Index(ids)
    if index.has_duplicates or index.isna().any():
        raise ValueError("artifact IDs must be unique")
    p = np.asarray(proba, float)
    if p.shape != (len(ids), 12):
        raise ValueError("artifact probability shape mismatch")
    missing = np.isnan(p).all(axis=1)
    if missing.any() and not allow_missing:
        raise ValueError("unexpected missing probability rows")
    align_proba(p[~missing], np.arange(12))
    if not np.allclose(p[~missing].sum(axis=1), 1, atol=1e-5):
        raise ValueError("artifact probabilities are not normalized")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, proba=p.astype("float32"), classes=np.asarray(LABELS),
                        provenance=np.asarray(json.dumps(provenance or {}, sort_keys=True)))


def load_probability_artifact(path, expected_ids, *, allow_missing=False, expected_provenance=None):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["classes"], np.asarray(LABELS)):
            raise ValueError("artifact class order mismatch")
        if not np.array_equal(saved["ids"], np.asarray(expected_ids).astype(str)):
            raise ValueError("artifact ID order mismatch")
        p = saved["proba"].astype(float)
        provenance = json.loads(str(saved["provenance"]))
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("artifact provenance mismatch")
    if p.shape != (len(expected_ids), 12):
        raise ValueError("artifact probability shape mismatch")
    missing = np.isnan(p).all(axis=1)
    if missing.any() and not allow_missing:
        raise ValueError("artifact contains unavailable probability rows")
    align_proba(p[~missing], np.arange(12))
    if not np.allclose(p[~missing].sum(axis=1), 1, atol=1e-5):
        raise ValueError("artifact probabilities are not normalized")
    return p


def train_member(X_fit, y_fit, sample_weight, X_stop, y_stop, *,
                 spec: dict, device: str, fixed_rounds: int | None = None) -> tuple[object, int]:
    if device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be cpu, gpu or cuda")
    if fixed_rounds is not None and (X_stop is not None or y_stop is not None):
        raise ValueError("fixed-round fitting requires no evaluation set")
    if fixed_rounds is None and (X_stop is None or y_stop is None or not len(y_stop)):
        raise ValueError("early stopping requires a nonempty stop set")
    # Lazy dependency: synthetic helpers can be imported without LightGBM.
    import lightgbm as lgb
    y_fit = np.asarray(y_fit)
    if not len(y_fit) or not np.isin(y_fit, np.arange(12)).all():
        raise ValueError("training labels must be nonempty and in 0..11")
    classes = np.unique(y_fit)
    if len(classes) < 2:
        raise ValueError("training fold must contain at least two classes")
    # Explicit remapping makes absent-class behavior deterministic.
    encoded = np.searchsorted(classes, y_fit)
    params = dict(objective="multiclass", num_class=len(classes),
                  n_estimators=int(fixed_rounds if fixed_rounds is not None else spec.get("max_rounds", 2000)),
                  learning_rate=spec.get("learning_rate", 0.05),
                  num_leaves=spec["num_leaves"], min_child_samples=spec["min_child_samples"],
                  colsample_bytree=spec["colsample_bytree"], subsample=0.8,
                  subsample_freq=1, reg_lambda=5., n_jobs=spec.get("n_jobs", 8),
                  random_state=spec["seed"], verbosity=-1,
                  deterministic=device == "cpu")
    if params["n_estimators"] < 1:
        raise ValueError("round count must be positive")

    def fit(backend):
        model = lgb.LGBMClassifier(**params, device_type=backend)
        kwargs = {"sample_weight": sample_weight}
        if fixed_rounds is None:
            # Native training API permits stop classes absent from fitting data
            # only after we explicitly reject such a fold.
            if not np.isin(y_stop, classes).all():
                raise ValueError("stop fold contains classes absent from training")
            kwargs.update(eval_set=[(X_stop, np.searchsorted(classes, y_stop))],
                          eval_metric="multi_logloss",
                          callbacks=[lgb.early_stopping(int(spec.get("early_stopping_rounds", 100)), verbose=False)])
        categorical = spec.get("categorical_columns", [])
        if categorical:
            kwargs["categorical_feature"] = categorical
        model.fit(X_fit, encoded, **kwargs)
        model.podium_classes_ = classes
        model.podium_backend_ = backend
        return model

    try:
        model = fit(device)
    except lgb.basic.LightGBMError as exc:
        text = str(exc).lower()
        backend_error = any(s in text for s in ("gpu", "cuda", "opencl", "device"))
        if device == "cpu" or not spec.get("allow_cpu_fallback", False) or not backend_error:
            raise RuntimeError(f"LightGBM {device} fitting failed: {exc}") from exc
        model = fit("cpu")
    rounds = int(model.best_iteration_ or params["n_estimators"])
    return model, rounds


def model_proba(model, X):
    return align_proba(model.predict_proba(X), model.podium_classes_)


def sample_training(ids, y_clean, *, rate=0.25, seed=2026):
    if not 0 < rate <= 1:
        raise ValueError("legitimate sampling rate must be in (0,1]")
    y = np.asarray(y_clean)
    # ID-hashed draws are stable across members, profiles and context row order.
    u = np.array([int(hashlib.sha256(f"{seed}:{txn}".encode()).hexdigest()[:16], 16) / 2**64 for txn in ids])
    keep = (y != 0) | (u < rate)
    weights = np.where(y[keep] == 0, 1 / rate, 1.)
    return keep, weights


def fraud_class_weights(y_fit, *, power=.5, cap=6.):
    """Counts come from the eligible, unsampled fit labels only.

    Keep legit at one; give rarer fraud classes a capped square-root weight.
    These multiply, rather than replace, inverse sampling weights.
    """
    y = np.asarray(y_fit)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in "iu"
            or not np.isin(y, np.arange(12)).all()):
        raise ValueError("class weighting requires nonempty integer fit labels")
    if not np.isfinite(power) or not 0 <= power <= 1 or not np.isfinite(cap) or cap < 1:
        raise ValueError("invalid fraud class weighting parameters")
    counts = np.bincount(y, minlength=12)
    weights = np.ones(12)
    present = np.flatnonzero(counts[1:]) + 1
    if len(present):
        reference = max(float(counts[0]) / len(present), 1.)
        weights[present] = np.clip((reference / counts[present]) ** power, 1., cap)
    return weights, counts


def calendar_dates(base, config):
    if "timestamp" not in base:
        raise ValueError("base.timestamp required: t origin is not assumed to be Unix")
    dates = pd.to_datetime(base.timestamp, errors="raise")
    if dates.isna().any():
        raise ValueError("missing timestamp")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(config.get("calendar_timezone", "Asia/Kolkata")).dt.tz_localize(None)
    # Naive timestamps retain their original calendar dates.
    return dates.reset_index(drop=True)


def split_masks(base, dates, config):
    train = base.split.eq("train").to_numpy()
    years = dates.loc[train & dates.dt.month.isin([7, 8]).to_numpy()].dt.year.unique()
    if len(years) != 1:
        raise ValueError("expected one dataset year with July-August training data")
    year = int(years[0])
    masks = {}
    for name, (start, end) in config["splits"].items():
        lo, hi = pd.Timestamp(f"{year}-{start}"), pd.Timestamp(f"{year}-{end}")
        if lo >= hi:
            raise ValueError("split end must follow start")
        masks[name] = train & (dates >= lo).to_numpy() & (dates < hi).to_numpy()
    masks["full_september"] = train & (dates.dt.year == year).to_numpy() & (dates.dt.month == 9).to_numpy()
    masks["tail"] = train & ~(masks["fit"] | masks["full_september"])
    masks["train"], masks["test"] = train, ~train
    for name in ("fit", "early_stop", "calibration", "assessment"):
        if not masks[name].any():
            raise ValueError(f"empty required partition: {name}")
    selected = np.stack([masks[k] for k in ("fit", "early_stop", "calibration", "assessment")])
    if (selected.sum(axis=0) > 1).any():
        raise ValueError("overlapping supervised partitions")
    if dates[masks["fit"]].max() >= dates[masks["early_stop"]].min():
        raise ValueError("fit rows must precede early-stop rows")
    if dates[masks["early_stop"]].max() >= dates[masks["calibration"]].min() or dates[masks["calibration"]].max() >= dates[masks["assessment"]].min():
        raise ValueError("stopping/calibration/assessment must be chronological")
    return masks, year


def rebuild_chain_features(base, f1, bands):
    """Inherited v5 chain construction, recomputing ch_band per outer context."""
    validate_ids(base)
    f1, bands = align_pack(base, f1), align_pack(base, bands)
    df = base.reset_index(drop=True).copy()
    df["row"] = np.arange(len(df))
    n = len(df)
    q = df[["row", "t", "sender_id", "amount"]].rename(columns={"sender_id": "k"}).sort_values("t")
    inn = df[df.receiver_type == "P2P"][["row", "t", "receiver_id", "amount"]].rename(
        columns={"receiver_id": "k", "row": "p_row", "amount": "p_amt", "t": "p_t"}).sort_values("p_t")
    m = pd.merge_asof(q, inn, left_on="t", right_on="p_t", by="k", direction="backward",
                      allow_exact_matches=False, tolerance=7200).set_index("row")
    ratio = (m.amount / m.p_amt).reindex(df.row).to_numpy()
    par = np.where((ratio >= 0.90) & (ratio <= 1.0), m.p_row.reindex(df.row).fillna(-1).to_numpy(), -1).astype(np.int64)
    q2 = df[df.receiver_type == "P2P"][["row", "t", "receiver_id", "amount"]].rename(columns={"receiver_id": "k"}).sort_values("t")
    out = df[["row", "t", "sender_id", "amount"]].rename(
        columns={"sender_id": "k", "row": "c_row", "amount": "c_amt", "t": "c_t"}).sort_values("c_t")
    m2 = pd.merge_asof(q2, out, left_on="t", right_on="c_t", by="k", direction="forward",
                       allow_exact_matches=False, tolerance=7200).set_index("row")
    r2 = (m2.c_amt / m2.amount).reindex(df.row).to_numpy()
    chi = np.where((r2 >= 0.90) & (r2 <= 1.0), m2.c_row.reindex(df.row).fillna(-1).to_numpy(), -1).astype(np.int64)

    def walk(ptr):
        depth, end, current = np.zeros(n, np.int16), np.arange(n), ptr.copy()
        for _ in range(8):
            has = current >= 0
            depth += has
            end = np.where(has, current, end)
            current = np.where(has, ptr[np.maximum(current, 0)], -1)
        return depth, end

    back, root = walk(par)
    forward, leaf = walk(chi)
    amount = df.amount.to_numpy()
    frame = pd.DataFrame({
        "txn_id": df.txn_id, "ch_depth_back": back, "ch_depth_fwd": forward,
        "ch_len": back + forward + 1,
        "ch_root_snd_vict_in": f1.snd_vict_in_cnt_48h.to_numpy()[root],
        "ch_root_is_102": f1.is_1p02.to_numpy()[root],
        "ch_root_amt_ratio": amount / amount[root],
        "ch_leaf_amt_ratio": amount[leaf] / amount,
        "ch_root_rcv_vict": f1.rcv_vict_cnt_24h.to_numpy()[root],
        "ch_is_root": ((back == 0) & (forward > 0)).astype(np.int8)})
    band = bands.band.to_numpy(dtype=int)
    chain = (back + forward) > 0
    cb = pd.DataFrame({"root": root[chain], "band": band[chain]})
    cb = cb[cb.band > 0].groupby("root").band.agg(["min", "max"])
    one = cb[cb["min"] == cb["max"]]["max"]
    frame["ch_band"] = np.where(chain, pd.Series(root).map(one).fillna(0).to_numpy(), 0)
    numeric = frame.columns.drop("txn_id")
    frame[numeric] = frame[numeric].astype("float32")
    return frame


def _frozen_features(base, packs, schema=None):
    X, candidate = assemble_features(base, packs)
    if schema is None:
        return X, candidate, {}
    if candidate["columns"] != schema["columns"] or schema["class_order"] != list(LABELS):
        raise ValueError("feature schema/class order mismatch")
    unknown = {}
    for col, old_levels in schema["categorical_mappings"].items():
        new_levels = candidate["categorical_mappings"][col]
        lookup = np.array([{v: i for i, v in enumerate(old_levels)}.get(v, -1) for v in new_levels], dtype=int)
        codes = X[col].to_numpy(dtype=int)
        mapped = np.full(len(codes), -1, dtype=np.int32)
        present = codes >= 0
        mapped[present] = lookup[codes[present]]
        unknown[col] = int((present & (mapped < 0)).sum())
        X[col] = mapped
    return X, schema, unknown


def _raw_labels(base, mask):
    y = base.loc[mask, "label"].map({lab: k for k, lab in enumerate(LABELS)})
    if y.isna().any():
        raise ValueError("invalid original label in eligible rows")
    return y.to_numpy(dtype=int)


def _spec(config, member, schema):
    spec = {**config.get("model_defaults", {}), **member,
            "allow_cpu_fallback": config.get("allow_cpu_fallback", False)}
    spec["categorical_columns"] = list(schema["categorical_mappings"])
    return spec


def _dump_model(model, folder, name, rounds, spec, training_ids):
    model.booster_.save_model(str(folder / f"{name}.txt"))
    _json(folder / f"{name}.json", {"spec": spec, "rounds": rounds,
          "backend": model.podium_backend_, "class_order": list(LABELS),
          "observed_classes": model.podium_classes_.tolist(), "training_ids_hash": ids_hash(training_ids)})


def _bootstrap(base, packs, dates, masks, year, config, directory, signature):
    X, schema, _ = _frozen_features(base, packs)
    # Teacher feature packs must never include labels, bands, chain bands or TE.
    forbidden = [c for c in X if "band" in c or c.startswith("v6_te_") or c in {"y_clean", "y_raw"}]
    if forbidden:
        raise ValueError(f"teacher contains prohibited features: {forbidden}")
    _json(directory / "schema.json", schema)
    p = np.full((len(base), 12), np.nan)
    folds = [
        ("july_second_half", f"{year}-07-01", f"{year}-07-15", f"{year}-07-15", f"{year}-08-01"),
        ("august", f"{year}-07-01", f"{year}-08-01", f"{year}-08-01", f"{year}-09-01"),
        ("september", f"{year}-07-01", f"{year}-09-01", f"{year}-09-01", f"{year}-10-01"),
    ]
    # Any late training tails receive predictions from earlier months only.
    for month in sorted(dates[masks["train"] & (dates >= pd.Timestamp(f"{year}-10-01")).to_numpy()].dt.to_period("M").unique()):
        folds.append((str(month), f"{year}-07-01", str(month.start_time.date()),
                      str(month.start_time.date()), str((month + 1).start_time.date())))
    provenance = []
    teacher_spec = _spec(config, config["members"][0], schema)
    for name, lo, hi, qlo, qhi in folds:
        fit = masks["train"] & dates.ge(lo).to_numpy() & dates.lt(hi).to_numpy()
        query = masks["train"] & dates.ge(qlo).to_numpy() & dates.lt(qhi).to_numpy()
        if not query.any():
            continue
        if not fit.any():
            raise ValueError(f"empty teacher fitting fold: {name}")
        model, rounds = train_member(X.loc[fit], _raw_labels(base, fit), np.ones(fit.sum()),
                                     None, None, spec=teacher_spec, device=config["device"], fixed_rounds=500)
        p[query] = model_proba(model, X.loc[query])
        _dump_model(model, directory, f"teacher_{name}", rounds, teacher_spec, base.txn_id[fit])
        provenance.append({"name": name, "fit_start": lo, "fit_end_exclusive": hi,
                           "query_start": qlo, "query_end_exclusive": qhi,
                           "fit_ids_hash": ids_hash(base.txn_id[fit]),
                           "query_ids_hash": ids_hash(base.txn_id[query])})
    save_probability_artifact(directory / "forward_teacher.npz", base.txn_id, p,
                              allow_missing=True, provenance={"signature": signature, "folds": provenance})
    for name, fit in (("validation", masks["fit"]), ("refit", masks["train"])):
        model, rounds = train_member(X.loc[fit], _raw_labels(base, fit), np.ones(fit.sum()),
                                     None, None, spec=teacher_spec, device=config["device"], fixed_rounds=500)
        q = masks["test"]
        test_p = model_proba(model, X.loc[q]) if q.any() else np.empty((0, 12))
        save_probability_artifact(directory / f"{name}_test_teacher.npz", base.txn_id[q], test_p,
                                  provenance={"signature": signature, "fit_ids_hash": ids_hash(base.txn_id[fit])})
        _dump_model(model, directory, f"teacher_{name}_test", rounds, teacher_spec, base.txn_id[fit])
    _json(directory / "manifest.json", {"signature": signature, "folds": provenance, "class_order": list(LABELS)})


def _context_features(base, packs, masks, config, directory, bootstrap, name, signature):
    fit = masks["fit"] if name == "validation" else masks["train"]
    manifest = json.loads((bootstrap / "manifest.json").read_text(encoding="utf-8"))
    if manifest["signature"] != signature or manifest["class_order"] != list(LABELS):
        raise ValueError("bootstrap manifest does not match inputs/config")
    teacher = load_probability_artifact(bootstrap / "forward_teacher.npz", base.txn_id, allow_missing=True,
                                       expected_provenance={"signature": signature, "folds": manifest["folds"]})
    teacher[masks["test"]] = load_probability_artifact(
        bootstrap / f"{name}_test_teacher.npz", base.txn_id[masks["test"]],
        expected_provenance={"signature": signature, "fit_ids_hash": ids_hash(base.txn_id[fit])})
    bands, diagnostic = build_bands(base, teacher, packs["v1"], pd.Index(base.txn_id[fit]))
    chain = rebuild_chain_features(base, packs["v1"], bands)
    cleaned = clean_training_labels(base, teacher, bands, packs["v2"], pd.Index(base.txn_id[fit]))
    context_packs = {**packs, "v3": bands, "v5": chain}
    if any("v6" in PROFILES[p] for p in config["profiles"]):
        context_packs["v6"] = build_label_free_v6(base, packs)
    if any("te" in PROFILES[p] for p in config["profiles"]):
        context_packs["te"] = build_context_te(base, fit)
    directory.mkdir(parents=True, exist_ok=True)
    bands.to_parquet(directory / "features_v3.parquet", index=False)
    chain.to_parquet(directory / "features_v5.parquet", index=False)
    cleaned.to_parquet(directory / "cleaning_rows.parquet", index=False)
    _json(directory / "band_diagnostics.json", diagnostic)
    _json(directory / "cleaning_diagnostics.json", cleaned.reason.value_counts().to_dict())
    return context_packs, cleaned


def build_context_te(base, fit):
    """Eligible training history only; freeze non-fit queries per calendar month."""
    reference = base.loc[fit].copy()
    earlier = build_temporal_te(reference, reference, expanding_months=True)
    future_query = base.loc[~fit].copy()
    future_parts = []
    # An October training tail must not enter an earlier October test encoding.
    # November queries may use eligible October-tail labels.
    for month in sorted(future_query.month.unique()):
        query_month = future_query.loc[future_query.month.eq(month)]
        eligible_reference = reference.loc[reference.month.lt(month)]
        future_parts.append(build_temporal_te(query_month, eligible_reference,
                                              expanding_months=False))
    future = (pd.concat(future_parts, ignore_index=True) if future_parts else
              build_temporal_te(future_query, reference.iloc[:0], expanding_months=False))
    return align_pack(base, pd.concat([earlier, future], ignore_index=True))


def _profile_run(base, packs, cleaned, dates, masks, config, out, profile, name, signature):
    folder = out / profile / name
    folder.mkdir(parents=True, exist_ok=True)
    validation_folder = out / profile / "validation"
    validation = name == "validation"
    fit = masks["fit"] if validation else masks["train"]
    schema = None if validation else json.loads((validation_folder / "schema.json").read_text(encoding="utf-8"))
    X, schema, unknown = _frozen_features(base, {k: packs[k] for k in PROFILES[profile]}, schema)
    _json(folder / "schema.json", schema)
    _json(folder / "unknown_categories.json", unknown)
    y_clean = cleaned.set_index("txn_id").loc[base.txn_id[fit], "y_clean"].to_numpy(dtype=int)
    keep, weights = sample_training(base.txn_id[fit], y_clean, rate=config.get("legit_fraction", .25))
    fit_positions = np.flatnonzero(fit)[keep]
    fit_y = y_clean[keep]
    members = config["members"] if profile == "ensemble" else config["members"][:1]
    members = list(members)
    recall_config = config.get("recall", {}) if profile == "ensemble" else {}
    recall_name = None
    if recall_config.get("enabled", False):
        recall_member = dict(recall_config["member"])
        recall_name = recall_member["name"]
        if recall_name in {m["name"] for m in members}:
            raise ValueError("recall member name must differ from baseline members")
        # This member enters only via the chronological blend gate below.
        recall_member["weight"] = 0.
        members.append(recall_member)
    prior_manifest = None if validation else json.loads((validation_folder / "manifest.json").read_text(encoding="utf-8"))
    if prior_manifest is not None and prior_manifest["signature"] != signature:
        raise ValueError("validation/refit configuration or inputs differ")
    query = masks["full_september"] | masks["tail"] if validation else masks["test"]
    # Validation tail reporting must never include fitted or pre-fit rows.
    query = query & ~fit
    ids = base.txn_id[query].to_numpy()
    ensemble = np.zeros((query.sum(), 12))
    total_weight = sum(float(m["weight"]) for m in members)
    if not np.isfinite(total_weight) or total_weight <= 0 or any(float(m["weight"]) < 0 for m in members):
        raise ValueError("ensemble weights must sum to a positive value")
    round_counts, backends = {}, {}
    recall_proba = None
    provenance = {"signature": signature, "profile": profile, "stage": name,
                  "fit_ids_hash": ids_hash(base.txn_id[fit]),
                  "sampled_fit_ids_hash": ids_hash(base.txn_id.iloc[fit_positions])}
    for member in members:
        spec = _spec(config, member, schema)
        member_weights = weights
        if member["name"] == recall_name:
            class_weights, counts = fraud_class_weights(
                y_clean, power=recall_config.get("class_weight_power", .5),
                cap=recall_config.get("class_weight_cap", 6.))
            member_weights = weights * class_weights[fit_y]
            spec["fraud_class_weights"] = class_weights.tolist()
            spec["unsampled_fit_class_counts"] = counts.tolist()
        fixed = None if validation else int(prior_manifest["round_counts"][member["name"]])
        print(f"{profile}/{name}: fitting {member['name']} on {len(fit_y)} rows", flush=True)
        model, rounds = train_member(
            X.iloc[fit_positions], fit_y, member_weights,
            X.loc[masks["early_stop"]] if validation else None,
            _raw_labels(base, masks["early_stop"]) if validation else None,
            spec=spec, device=config["device"], fixed_rounds=fixed)
        print(f"{profile}/{name}: {member['name']} completed at {rounds} rounds", flush=True)
        p = model_proba(model, X.loc[query]) if query.any() else np.empty((0, 12))
        weight = float(member["weight"]) / total_weight
        ensemble += weight * p
        if member["name"] == recall_name:
            recall_proba = p
        round_counts[member["name"]], backends[member["name"]] = rounds, model.podium_backend_
        _dump_model(model, folder, member["name"], rounds, spec, base.txn_id.iloc[fit_positions])
        save_probability_artifact(folder / f"{member['name']}_proba.npz", ids, p, provenance=provenance)
    meta = base.loc[query, ["txn_id", "receiver_type"]].reset_index(drop=True)
    meta["timestamp"] = dates.loc[query].reset_index(drop=True)
    query_ids = pd.Index(ids)
    f3 = packs["v3"].loc[packs["v3"].txn_id.isin(query_ids)].reset_index(drop=True)
    f5 = packs["v5"].loc[packs["v5"].txn_id.isin(query_ids)].reset_index(drop=True)
    f2 = packs["v2"].loc[packs["v2"].txn_id.isin(query_ids)].reset_index(drop=True)
    f4 = packs["v4"].loc[packs["v4"].txn_id.isin(query_ids)].reset_index(drop=True)
    groups = event_groups(meta, f3, f5, f2, f4)
    context = prepare_rule_context(meta, f3, f5)
    meta.to_parquet(folder / "meta.parquet", index=False)
    align_pack(meta, f3).to_parquet(folder / "features_v3.parquet", index=False)
    align_pack(meta, f5).to_parquet(folder / "features_v5.parquet", index=False)
    np.save(folder / "groups.npy", groups.astype(str))
    _json(folder / "manifest.json", {**provenance, "class_order": list(LABELS),
          "round_counts": round_counts, "backends": backends,
          "config": config, "feature_schema_hash": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
          "partition_boundaries": config["splits"]})
    if validation:
        y = _raw_labels(base, query)
        calibration = masks["calibration"][query]
        baseline_ensemble = ensemble.copy()
        selection = None
        if recall_proba is not None:
            offsets, selection = select_recall_policy(
                ensemble[calibration], recall_proba[calibration], y[calibration], groups[calibration],
                slice_rule_context(context, calibration),
                baseline_config=config.get("calibration", {}), config=recall_config)
            _json(folder / "recall_selection.json", selection)
            alpha = selection["recall_weight"]
            ensemble = (1 - alpha) * ensemble + alpha * recall_proba
        else:
            offsets = fit_offsets(ensemble[calibration], y[calibration], groups[calibration],
                                  slice_rule_context(context, calibration), config=config.get("calibration", {}))
        _json(folder / "offsets.json", offsets)
        # Preserve an exact legacy decision comparator, selected independently
        # of assessment. Both policies are reported on the same raw labels.
        baseline_offsets = fit_offsets(
            baseline_ensemble[calibration], y[calibration], groups[calibration],
            slice_rule_context(context, calibration), config=config.get("calibration", {})) if selection else offsets
        _json(folder / "baseline_offsets.json", baseline_offsets)
        baseline_pred = predict_labels(baseline_ensemble, groups, baseline_offsets, context)
        pred = predict_labels(ensemble, groups, offsets, context)
        pred_global = predict_labels(ensemble, groups, offsets["global_only_offsets"], context)
        raw_argmax = ensemble.argmax(axis=1)
        zero = {"global": [0.] * 12, "conditional": {}, "class_order": list(LABELS)}
        band_only = predict_labels(ensemble, groups, zero, context)
        metrics = {"profile": profile, "label_source": "original",
                   "assessment_reused_project_data": True, "partitions": {}, "recall_selection": selection,
                   "round_counts": round_counts, "corrections": int((pred != raw_argmax).sum()),
                   "band_rule_changes_without_offsets": int((band_only != raw_argmax).sum()),
                   "offset_changes_after_rules": int((pred != band_only).sum()),
                   "cleaning_reasons": cleaned.reason.value_counts().to_dict()}
        for partition in ("early_stop", "calibration", "assessment", "full_september", "tail"):
            mask = masks[partition][query]
            report = evaluate_predictions(y[mask], pred[mask])
            report["legacy_baseline"] = evaluate_predictions(y[mask], baseline_pred[mask])
            report["macro_f1_delta_vs_legacy"] = report["macro_f1"] - report["legacy_baseline"]["macro_f1"]
            report["global_only"] = evaluate_predictions(y[mask], pred_global[mask])
            b = context["band"][mask] > 0
            report["direct_band"] = evaluate_predictions(y[mask][b], pred[mask][b])
            report["no_direct_band"] = evaluate_predictions(y[mask][~b], pred[mask][~b])
            metrics["partitions"][partition] = report
            save_probability_artifact(folder / f"{partition}_proba.npz", ids[mask], ensemble[mask],
                                      provenance={**provenance, "partition": partition})
        _json(folder / "metrics.json", metrics)
        save_probability_artifact(folder / "validation_proba.npz", ids, ensemble, provenance=provenance)
        rows = []
        for partition, report in metrics["partitions"].items():
            for lab, values in report["per_class"].items():
                rows.append({"partition": partition, "label": lab, **values})
        pd.DataFrame(rows).to_csv(folder / "per_class_metrics.csv", index=False)
        print(profile, "raw assessment macro-F1:", metrics["partitions"]["assessment"]["macro_f1"], flush=True)
        return metrics
    offsets = json.loads((validation_folder / "offsets.json").read_text(encoding="utf-8"))
    if recall_proba is not None:
        selection = json.loads((validation_folder / "recall_selection.json").read_text(encoding="utf-8"))
        alpha = float(selection["recall_weight"])
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("invalid saved recall blend weight")
        ensemble = (1 - alpha) * ensemble + alpha * recall_proba
        _json(folder / "recall_selection.json", selection)
    save_probability_artifact(folder / "test_proba.npz", ids, ensemble, provenance=provenance)
    pred = predict_labels(ensemble, groups, offsets, context)
    sample = pd.read_csv(Path(config["_root"]) / "data/raw/sample_submission.csv")
    result = write_submission(ids, pred, sample, folder / "submission.csv")
    _json(folder / "submission_metrics.json", result)
    return result


def run_experiment(root: Path, config: dict, *, stage: str) -> dict:
    if stage not in {"bootstrap", "validate", "refit", "all"}:
        raise ValueError("stage must be bootstrap, validate, refit or all")
    root = Path(root).resolve()
    config = json.loads(json.dumps(config))
    config.setdefault("device", "cpu")
    config.setdefault("profiles", ["ensemble"])
    if not config["profiles"] or len(set(config["profiles"])) != len(config["profiles"]) or any(p not in PROFILES for p in config["profiles"]):
        raise ValueError("unknown/duplicate/empty profile selection")
    tag = config.get("tag", "podium_v6")
    if Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be a single directory name")
    processed = root / "data/processed"
    source_names = ["base_v0.parquet", "ratios_v0.parquet", "features_v1.parquet",
                    "features_v2.parquet", "features_v4.parquet"]
    hashes = {name: file_hash(processed / name) for name in source_names}
    hashes["source"] = {name: file_hash(Path(__file__).parent / name) for name in
                        ("common.py", "features_v6.py", "clean_labels.py", "bands.py",
                         "fit_submit.py", "make_submission.py", "features_v5.py")}
    signature_config = {k: v for k, v in config.items() if k not in {"output_dir", "_root"}}
    signature = hashlib.sha256(json.dumps({"config": signature_config, "inputs": hashes},
                                         sort_keys=True).encode()).hexdigest()
    base = pd.read_parquet(processed / "base_v0.parquet").reset_index(drop=True)
    validate_ids(base)
    ratios = pd.read_parquet(processed / "ratios_v0.parquet")
    if ratios.index.name != "txn_id" or not np.array_equal(ratios.index.to_numpy(), base.txn_id.to_numpy()):
        raise ValueError("ratio txn_id index differs from base: rebuild Card1 base/v1 first")
    del ratios
    if not base.split.isin(["train", "test"]).all():
        raise ValueError("base split must be train/test")
    dates = calendar_dates(base, config)
    # Actual calendar months replace legacy tail folding; retain actual tails.
    base["month"] = dates.dt.strftime("%Y-%m")
    masks, year = split_masks(base, dates, config)
    packs = {key: align_pack(base, pd.read_parquet(processed / f"features_{key}.parquet"))
             for key in ("v1", "v2", "v4")}
    out = Path(config.get("output_dir", root / "runs" / tag))
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    existing_inputs = out / "inputs.json"
    if existing_inputs.exists():
        previous = json.loads(existing_inputs.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("existing run uses different inputs/source/config; use a fresh tag/output directory")
    bootstrap = out / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    _json(out / "inputs.json", {"signature": signature, "hashes": hashes, "dataset_year": year,
          "class_order": list(LABELS), "config": config})
    config["_root"] = str(root)
    if stage in {"bootstrap", "all"}:
        _bootstrap(base, packs, dates, masks, year, config, bootstrap, signature)
    results = {"signature": signature, "run_dir": str(out), "stages": {}}
    for name, selected_stages in (("validation", {"validate", "all"}), ("refit", {"refit", "all"})):
        if stage not in selected_stages:
            continue
        context_packs, cleaned = _context_features(base, packs, masks, config,
                                                   out / "contexts" / name, bootstrap, name, signature)
        results["stages"][name] = {}
        for profile in config["profiles"]:
            results["stages"][name][profile] = _profile_run(
                base, context_packs, cleaned, dates, masks, config, out, profile, name, signature)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/podium.json"))
    ap.add_argument("--stage", choices=["bootstrap", "validate", "refit", "all"], default="all")
    ap.add_argument("--profiles", help="comma-separated control,fusion,temporal,ensemble")
    ap.add_argument("--device", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    path = a.config if a.config.is_absolute() else a.root / a.config
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("device", "tag"):
        if getattr(a, key) is not None:
            config[key] = getattr(a, key)
    if a.profiles:
        config["profiles"] = a.profiles.split(",")
    if a.output_dir is not None:
        config["output_dir"] = str(a.output_dir)
    if a.allow_cpu_fallback:
        config["allow_cpu_fallback"] = True
    result = run_experiment(a.root, config, stage=a.stage)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
