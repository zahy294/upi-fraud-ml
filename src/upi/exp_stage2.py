"""Strict nested month cross-fitting for a probability-neighbour second stage.

Run: python -I src/upi/exp_stage2.py ROOT --out NEW_DIRECTORY
No existing input/output file is overwritten. Months without out-of-fold predictions fall back to a uniform prior,
which is what distinguishes this from ordinary OOF stacking.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import DROP, LABELS, fit_weights, load_xy, macro_f1, per_class  # noqa: E402
from stage2_neighbors import (NeighborFeatures, logical_folds, probability_context,
                              restore_population_prior)  # noqa: E402


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load_inputs(root, feature_files):
    X, y, meta = load_xy(root, feature_files=feature_files, with_traps=False)
    forbidden = DROP | {"app_old", "old", "ip", "target", "y", "is_fraud", "fraud_label"}
    if not X.columns.is_unique or forbidden.intersection(X.columns):
        raise ValueError("Duplicate or forbidden model input columns")
    if not X.index.equals(pd.RangeIndex(len(X))) or not X.index.equals(meta.index):
        raise ValueError("Feature rows must have the original contiguous row index")
    entities = pd.read_parquet(root / "data/processed/base_v0.parquet",
        columns=["txn_id", "sender_id", "receiver_id", "device_id", "t"])
    if not np.array_equal(entities.txn_id.to_numpy(), meta.txn_id.to_numpy()):
        raise ValueError("Entity and feature row orders differ")
    folds = logical_folds(meta)
    train = folds != ""
    if not np.isin(y[train], np.arange(len(LABELS))).all():
        raise ValueError("Missing or unknown training label")
    y = y.copy()
    y[~train] = -1  # Test labels can never become training targets.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X, y, meta, entities, folds


def model_parameters(trees, threads, seed):
    return dict(objective="multiclass", n_estimators=trees, learning_rate=0.05,
                num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, reg_lambda=5.0, verbosity=-1,
                n_jobs=threads, random_state=seed, deterministic=True, force_col_wise=True)


def fit_model(X, y, rows, classes, trees, legit_fraction, threads, seed):
    """Sample fit rows only; every graph/context row was retained beforehand."""
    rows = np.asarray(rows, dtype=np.int64)
    if not 0 < legit_fraction <= 1 or trees < 1:
        raise ValueError("Positive trees and legit_fraction in (0, 1] required")
    if not np.array_equal(np.unique(y[rows]), np.arange(len(classes))):
        raise ValueError("Every fitted month subset must contain all configured classes")
    rng = np.random.RandomState(seed)
    keep = rows[(y[rows] != 0) | (rng.random_sample(len(rows)) < legit_fraction)]
    if not np.array_equal(np.unique(y[keep]), np.arange(len(classes))):
        raise ValueError("Sampling removed an entire class")
    model = lgb.LGBMClassifier(**model_parameters(trees, threads, seed))
    model.fit(X.loc[keep], y[keep])
    if not np.array_equal(model.classes_, np.arange(len(classes))):
        raise ValueError("Unexpected class ordering")
    return model, len(keep)


class Stage1Cache:
    """Caches predictions of each exact training-month subset, at fixed trees."""

    def __init__(self, X, y, folds, classes, out, trees, fraction, threads, seed):
        self.X, self.y, self.folds, self.classes = X, y, folds, classes
        self.out, self.trees, self.fraction = Path(out), trees, fraction
        self.threads, self.seed = threads, seed
        self.cache = {}
        self.months = tuple(sorted(set(folds) - {""}))

    def __call__(self, months):
        months = tuple(sorted(months))
        if not months or not set(months).issubset(self.months):
            raise ValueError("Invalid fitted-month subset")
        if months in self.cache:
            return self.cache[months]
        tag = "_".join(months)
        rows = np.flatnonzero(np.isin(self.folds, months))
        seed = self.seed + sum(1 << self.months.index(m) for m in months)
        print(f"stage1 fit {months}: {len(rows):,} eligible rows", flush=True)
        model, sampled = fit_model(self.X, self.y, rows, self.classes, self.trees,
                                    self.fraction, self.threads, seed)
        path = self.out / f"s1_{tag}_proba.npy"
        if path.exists():
            raise FileExistsError(path)
        p = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                     shape=(len(self.X), len(self.classes)))
        for start in range(0, len(self.X), 100000):
            stop = min(start + 100000, len(self.X))
            p[start:stop] = restore_population_prior(
                model.predict_proba(self.X.iloc[start:stop]), self.fraction)
        p.flush()
        model.booster_.save_model(str(self.out / f"s1_{tag}.txt"), num_iteration=self.trees)
        write_json(self.out / f"s1_{tag}.json", {"fit_months": months, "eligible_rows": len(rows),
            "sampled_rows": sampled, "seed": seed, "trees": self.trees,
            "classes": self.classes, "legit_fraction": self.fraction, "prior_corrected": True})
        del p, model
        self.cache[months] = np.load(path, mmap_mode="r")
        return self.cache[months]


def augmented_features(X, folds, graph, available, rows, cache, classes):
    """No allowed stage-1 fit sees the query month or the source row's month."""
    rows = np.asarray(rows, dtype=np.int64)
    parts, plans = [], []
    for query_month in sorted(set(folds[rows])):
        selected = rows[folds[rows] == query_month]
        p, plan = probability_context(folds, available, query_month, cache, len(classes))
        parts.append(graph.transform(p, classes, selected))
        plans.extend(plan)
        del p
    neighbours = pd.concat(parts).reindex(rows)
    if X.columns.intersection(neighbours.columns).size:
        raise ValueError("Stage-2 feature names collide with existing input columns")
    return pd.concat([X.loc[rows], neighbours], axis=1), plans


def run_stage2(X, y, meta, entities, folds, out, *, classes=LABELS, stage1_trees=500,
               stage2_trees=300, legit_fraction=.25, threads=12, seed=17,
               tune_weights=False, final=True):
    """Run honest outer LOMO evaluation, then optional full-data fit and test inference.

    'out' MUST NOT exist. Raw outer-OOF argmax metrics use no calibration labels.
    Optional weights fitted on pooled OOF are explicitly development-tuned metrics.
    """
    if not 0 < legit_fraction <= 1 or min(stage1_trees, stage2_trees, threads) < 1:
        raise ValueError("Invalid trees, fraction or thread count")
    if len(X) != len(y) or len(X) != len(entities) or len(X) != len(folds):
        raise ValueError("Input lengths differ")
    if not np.array_equal(meta.txn_id.to_numpy(), entities.txn_id.to_numpy()):
        raise ValueError("Transaction IDs must align")
    months = tuple(sorted(set(folds) - {""}))
    if len(months) != 3:
        raise ValueError("Exactly three training folds are required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"row": np.arange(len(X)), "txn_id": meta.txn_id.to_numpy(),
                  "fold": folds}).to_parquet(out / "all_rows.parquet", index=False)
    graph = NeighborFeatures(entities)
    cache = Stage1Cache(X, y, folds, classes, out, stage1_trees, legit_fraction, threads, seed)
    train_rows, test_rows = np.flatnonzero(folds != ""), np.flatnonzero(folds == "")
    oof = np.full((len(X), len(classes)), np.nan, dtype=np.float32)
    base_oof = np.full_like(oof, np.nan)
    plans, scores = {}, {}
    for outer in months:
        available = tuple(m for m in months if m != outer)
        fit_rows = np.flatnonzero(np.isin(folds, available))
        val_rows = np.flatnonzero(folds == outer)
        print(f"stage2 outer holdout {outer}", flush=True)
        train_x, train_plan = augmented_features(X, folds, graph, available, fit_rows, cache, classes)
        model, sampled = fit_model(train_x, y, fit_rows, classes, stage2_trees,
                                   legit_fraction, threads, seed + 100 + months.index(outer))
        del train_x
        gc.collect()
        val_x, val_plan = augmented_features(X, folds, graph, available, val_rows, cache, classes)
        oof[val_rows] = restore_population_prior(model.predict_proba(val_x), legit_fraction)
        base_oof[val_rows] = cache(available)[val_rows]
        scores[outer] = {"stage1_argmax_f1": macro_f1(y[val_rows], base_oof[val_rows].argmax(1), len(classes)),
                        "stage2_argmax_f1": macro_f1(y[val_rows], oof[val_rows].argmax(1), len(classes)),
                        "sampled_stage2_rows": sampled}
        plans[outer] = {"train": train_plan, "validation": val_plan}
        model.booster_.save_model(str(out / f"s2_holdout_{outer}.txt"), num_iteration=stage2_trees)
        print(scores[outer], flush=True)
        del val_x, model
        gc.collect()
    if not np.isfinite(oof[train_rows]).all():
        raise AssertionError("Incomplete OOF coverage")
    np.save(out / "stage2_oof_proba.npy", oof[train_rows])
    np.save(out / "stage1_oof_proba.npy", base_oof[train_rows])
    pd.DataFrame({"txn_id": meta.txn_id.to_numpy()[train_rows], "fold": folds[train_rows]}).to_parquet(
        out / "oof_rows.parquet", index=False)
    scores["pooled"] = {"stage1_argmax_f1": macro_f1(y[train_rows], base_oof[train_rows].argmax(1), len(classes)),
                        "stage2_argmax_f1": macro_f1(y[train_rows], oof[train_rows].argmax(1), len(classes))}
    weights = np.ones(len(classes))
    if tune_weights:
        if list(classes) != LABELS:
            raise ValueError("Existing fit_weights supports the repository's 12 classes only")
        weights = fit_weights(oof[train_rows], y[train_rows])
        scores["pooled"]["development_tuned_f1_NOT_independent"] = macro_f1(
            y[train_rows], (oof[train_rows] * weights).argmax(1))
    np.save(out / "weights.npy", weights)
    if list(classes) == LABELS:
        per_class(y[train_rows], oof[train_rows].argmax(1)).to_csv(out / "oof_per_class.csv")
    if final:
        print("stage2 final fit", flush=True)
        train_x, train_plan = augmented_features(X, folds, graph, months, train_rows, cache, classes)
        model, sampled = fit_model(train_x, y, train_rows, classes, stage2_trees,
                                   legit_fraction, threads, seed + 200)
        feature_names = train_x.columns.tolist()
        del train_x
        gc.collect()
        if len(test_rows):
            test_x, test_plan = augmented_features(X, folds, graph, months, test_rows, cache, classes)
            p = restore_population_prior(model.predict_proba(test_x), legit_fraction)
            np.save(out / "test_proba.npy", p)
            pred = (p * weights).argmax(1)
            pd.DataFrame({"id": meta.txn_id.to_numpy()[test_rows],
                          "label": np.asarray(classes)[pred]}).to_csv(out / "predictions_by_id.csv", index=False)
            pd.DataFrame({"txn_id": meta.txn_id.to_numpy()[test_rows]}).to_parquet(
                out / "test_rows.parquet", index=False)
            del test_x
        else:
            test_plan = []
        model.booster_.save_model(str(out / "s2_final.txt"), num_iteration=stage2_trees)
        plans["final"] = {"train": train_plan, "test": test_plan}
        write_json(out / "final_model.json", {"sampled_rows": sampled, "feature_names": feature_names})
    write_json(out / "fold_provenance.json", plans)
    write_json(out / "metrics.json", scores)
    write_json(out / "config.json", {"classes": list(classes), "months": months,
        "stage1_trees": stage1_trees, "stage2_trees": stage2_trees,
        "legit_fraction": legit_fraction, "seed": seed, "threads": threads,
        "prior_correction": True, "empty_fit_prior": "fixed_uniform",
        "tune_weights_on_pooled_oof": tune_weights, "final_fit": final,
        "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__})
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--features", default="features_v1.parquet")
    parser.add_argument("--stage1-trees", type=int, default=500)
    parser.add_argument("--stage2-trees", type=int, default=300)
    parser.add_argument("--legit-frac", type=float, default=.25)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tune-weights", action="store_true")
    parser.add_argument("--no-final", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    inputs = load_inputs(args.root, tuple(filter(None, args.features.split(","))))
    scores = run_stage2(*inputs, args.out, stage1_trees=args.stage1_trees,
        stage2_trees=args.stage2_trees, legit_fraction=args.legit_frac,
        threads=args.threads, seed=args.seed, tune_weights=args.tune_weights, final=not args.no_final)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
