"""Separate binary fraud detection from conditional fraud-type classification."""
import numpy as np
try:
    from .fit_submit import train_member, model_proba
except ImportError:
    from fit_submit import train_member, model_proba


def combine_probabilities(fraud, conditional):
    fraud, conditional = np.asarray(fraud, float), np.asarray(conditional, float)
    if (fraud.ndim != 1 or conditional.shape != (len(fraud), 12)
            or not np.isfinite(fraud).all() or not np.isfinite(conditional).all()
            or (fraud < 0).any() or (fraud > 1).any()
            or (conditional < 0).any() or (conditional > 1).any()
            or not np.allclose(conditional[:, 0], 0)
            or not np.allclose(conditional[:, 1:].sum(1), 1)):
        raise ValueError("invalid binary/conditional probabilities")
    result = conditional * fraud[:, None]
    result[:, 0] = 1 - fraud
    return result


def fit_detector(X, y, weights, X_stop, y_stop, *, spec, rounds=None):
    import lightgbm as lgb
    y = np.asarray(y)
    if not np.isin(y, np.arange(12)).all() or len(np.unique(y > 0)) != 2:
        raise ValueError("detector requires both legitimate and fraud fitting rows")
    if rounds is not None and (X_stop is not None or y_stop is not None):
        raise ValueError("refit must not use an evaluation set")
    if rounds is None and (y_stop is None or len(np.unique(np.asarray(y_stop) > 0)) != 2):
        raise ValueError("detector stopping rows require both classes")
    model = lgb.LGBMClassifier(objective="binary", metric="average_precision",
        n_estimators=int(rounds if rounds is not None else spec["max_rounds"]),
        learning_rate=spec["learning_rate"], num_leaves=spec["num_leaves"],
        min_child_samples=spec["min_child_samples"], colsample_bytree=spec["colsample_bytree"],
        subsample=.8, subsample_freq=1, reg_lambda=5., n_jobs=spec["n_jobs"],
        random_state=spec["seed"], verbosity=-1, deterministic=True, device_type="cpu")
    kwargs = {"sample_weight": weights, "categorical_feature": spec["categorical_columns"]}
    if rounds is None:
        kwargs.update(eval_set=[(X_stop, (np.asarray(y_stop) > 0).astype(int))],
                      callbacks=[lgb.early_stopping(spec["early_stopping_rounds"],
                                  first_metric_only=True, verbose=False),
                                 lgb.log_evaluation(100)])
    model.fit(X, (y > 0).astype(int), **kwargs)
    return model, int(model.best_iteration_ or model.n_estimators)


def fit_types(X, y, weights, X_stop, y_stop, *, spec, rounds=None):
    """Only fraud rows teach type discrimination; no held-out label cleaning."""
    y, weights = np.asarray(y), np.asarray(weights)
    fraud = y > 0
    if len(np.unique(y[fraud])) != 11:
        raise ValueError("type fitting requires all eleven fraud classes")
    stop = None if y_stop is None else np.asarray(y_stop) > 0
    return train_member(X.loc[fraud], y[fraud], weights[fraud],
                        None if stop is None else X_stop.loc[stop],
                        None if stop is None else np.asarray(y_stop)[stop],
                        spec=spec, device="cpu", fixed_rounds=rounds)


def hierarchy_proba(detector, types, X):
    classes = np.asarray(detector.classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("binary class mapping differs")
    return combine_probabilities(detector.predict_proba(X)[:, 1], model_proba(types, X))
