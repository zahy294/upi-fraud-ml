"""Shared pieces for the modelling scripts: loading, feature lists, metric, decision weights."""
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover", "phishing_collect",
          "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
K = len(LABELS)
CATS = ["receiver_type", "txn_type", "merchant_category", "app", "kyc_level", "rcv_kyc_level"]
# never used as model inputs: ids, labels, calendar position, and the two trap columns
DROP = {"txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "label", "split", "month", "t", "weekday",
        "app_version", "app_major", "ip_risk_score", "home_city", "rcv_home_city", "mer_city", "mer_category", "device_city",
        "os", "bank_code", "sender_account_age_days", "acct_age_from_opened", "mer_cat_matches_txn", "mer_city_eq_device_city"}
TRAPS = ["ip_risk_score", "app_old"]


def load_xy(root: Path, feature_files=("features_v1.parquet",), with_traps=False):
    """Return (X, y, meta). Rows are in base_v0 order (sorted by time); y is -1 for test rows."""
    P = root / "data/processed"
    df = pd.read_parquet(P / "base_v0.parquet")
    num = [c for c in df.columns if c not in DROP and c not in CATS]
    X = df[num].astype("float32")
    for f in feature_files:
        F = pd.read_parquet(P / f)
        assert (F.txn_id.values == df.txn_id.values).all()
        X = pd.concat([X, F.drop(columns=["txn_id"])], axis=1)
    for c in CATS:
        X[c] = df[c].astype("object").fillna("NA").astype("category")
    if with_traps:
        X["ip_risk_score"] = df.ip_risk_score.astype("float32")
        X["app_old"] = (df.app_major == "3").astype("float32")
    y = df.label.map({lab: i for i, lab in enumerate(LABELS)}).fillna(-1).astype(int).values
    month = df.month.values.copy()
    month[(df.split == "train").values & (month > "2026-09")] = "2026-09"      # October tails of September events
    meta = pd.DataFrame({"txn_id": df.txn_id.values, "split": df.split.values, "month": month,
                         "old": (df.app_major == "3").values, "ip": df.ip_risk_score.values})
    return X, y, meta


def macro_f1(y, pred, k=K):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    return float((2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


def per_class(y, pred, k=K):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": cm.sum(1), "predicted": cm.sum(0)}, index=LABELS)


def fit_weights(P, y, sweeps=3, grid=np.geomspace(0.05, 400, 60), w0=None):
    """Coordinate ascent on class weights w so that argmax(P * w) maximises macro-F1. Legit weight stays 1."""
    k = P.shape[1]
    w = np.ones(k) if w0 is None else w0.copy()
    # rows where no fraud class can beat legit even at the largest weight always predict legit: count them once
    cand = P[:, 1:].max(1) * grid.max() >= P[:, 0]
    base = np.zeros((k, k), dtype=np.int64)
    base[:, 0] = np.bincount(y[~cand], minlength=k)
    Pc, yc = P[cand], y[cand]

    def score(wv):
        cm = base + np.bincount(yc * k + (Pc * wv).argmax(1), minlength=k * k).reshape(k, k)
        tp = np.diag(cm)
        return float((2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())

    for _ in range(sweeps):
        for c in range(1, k):
            s = np.empty(len(grid))
            for i, g in enumerate(grid):
                w2 = w.copy()
                w2[c] = g
                s[i] = score(w2)
            plateau = np.where(s >= s.max() - 0.0003)[0]
            w[c] = grid[plateau[len(plateau) // 2]]
    return w
