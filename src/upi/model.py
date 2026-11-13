from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def create_model(config: dict, n_estimators: int | None = None):
    model_cfg = dict(config.get("model", {}))
    kind = model_cfg.pop("kind", "lightgbm")
    params = dict(model_cfg.pop("params", {}))
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        defaults = dict(objective="multiclass", n_estimators=1200, learning_rate=.035,
                        num_leaves=63, min_child_samples=60, subsample=.8,
                        subsample_freq=1, colsample_bytree=.65, reg_lambda=5,
                        n_jobs=-1, random_state=20261113, verbosity=-1)
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        defaults = dict(objective="multi:softprob", tree_method="hist", device="cuda",
                        n_estimators=1600, learning_rate=.04, max_depth=8,
                        min_child_weight=10, subsample=.8, colsample_bytree=.65,
                        max_bin=256, reg_lambda=5, random_state=20261113,
                        eval_metric="mlogloss", early_stopping_rounds=100)
        defaults.update(params)
        if n_estimators is not None:
            defaults["early_stopping_rounds"] = None
        return XGBClassifier(**defaults)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        defaults = dict(loss_function="MultiClass", iterations=1600, learning_rate=.05,
                        depth=8, l2_leaf_reg=5, random_seed=20261113,
                        task_type="GPU", verbose=100, allow_writing_files=False)
        defaults.update(params)
        if n_estimators is not None:
            defaults["iterations"] = defaults.pop("n_estimators", n_estimators)
        return CatBoostClassifier(**defaults)
    raise ValueError(f"unknown model kind: {kind}")


def fit_model(model, x_train, y_train, x_valid=None, y_valid=None):
    module = model.__class__.__module__
    if x_valid is None:
        model.fit(x_train, y_train)
    elif module.startswith("lightgbm"):
        import lightgbm as lgb
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    elif module.startswith("xgboost"):
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    elif module.startswith("catboost"):
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=100)
    else:
        model.fit(x_train, y_train)
    return model


def predict_aligned(model, x, n_classes: int = 12) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float32)
    output = np.zeros((len(x), n_classes), dtype=np.float32)
    classes = np.asarray(model.classes_, dtype=int)
    output[:, classes] = raw
    return output


def best_iteration(model, fallback: int) -> int:
    for name in ("best_iteration_", "best_iteration"):
        value = getattr(model, name, None)
        if value is not None and int(value) >= 0:
            return int(value) + (1 if name == "best_iteration" else 0)
    getter = getattr(model, "get_best_iteration", None)
    if getter:
        value = getter()
        if value is not None and int(value) >= 0:
            return int(value) + 1
    return int(fallback)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)

