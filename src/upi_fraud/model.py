"""Fixed measured LightGBM fit and explicit sampling/column contracts."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .constants import CATEGORICAL, FEATURES, LABELS
from .data import align
from .decisions import validate_probabilities


def fitting_rows(features, targets, cutoff, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("negative_sample_fraction must lie in (0,1]")
    aligned = align(targets[["txn_id", "label"]], features.txn_id)
    y = aligned.label.map({c: i for i, c in enumerate(LABELS)}).fillna(-1).to_numpy(dtype="int32")
    train = (features.split == "train").to_numpy() & (y >= 0)
    if cutoff is not None:
        train &= (features.timestamp < pd.Timestamp(cutoff)).to_numpy()
    retain = np.random.default_rng(seed).random(len(y)) < fraction
    selected = train & ((y > 0) | retain)
    weights = np.where(y[selected] == 0, 1 / fraction, 1.)
    if set(y[selected]) != set(range(12)):
        raise ValueError("Fit requires all twelve training classes")
    return selected, y, weights


def fit_model(features, targets, cutoff, config):
    mask, y, weights = fitting_rows(features, targets, cutoff,
        config["negative_sample_fraction"], config["seed"])
    model = lgb.LGBMClassifier(**config["model"], random_state=config["seed"])
    model.fit(features.loc[mask, FEATURES], y[mask], sample_weight=weights,
              categorical_feature=list(CATEGORICAL))
    if not np.array_equal(model.classes_, np.arange(12)):
        raise ValueError("Unexpected trained class order")
    return model.booster_, {
        "fit_rows": int(mask.sum()), "eligible_rows": int(((features.split == "train") &
            ((features.timestamp < pd.Timestamp(cutoff)) if cutoff else True)).sum()),
        "fit_min_timestamp": str(features.loc[mask, "timestamp"].min()),
        "fit_max_timestamp": str(features.loc[mask, "timestamp"].max()),
        "fit_label_cutoff_exclusive": cutoff,
        "fit_class_counts": dict(zip(LABELS, np.bincount(y[mask], minlength=12).tolist())),
        "negative_sample_fraction": config["negative_sample_fraction"],
        "negative_sampling_weight": 1 / config["negative_sample_fraction"],
        "label_history_features": False,
    }


def predict_proba(booster, features):
    if booster.feature_name() != list(FEATURES):
        raise ValueError("Model feature columns/order differ from baseline schema")
    P = booster.predict(features.loc[:, FEATURES]).astype("float32")
    validate_probabilities(P)
    return P
