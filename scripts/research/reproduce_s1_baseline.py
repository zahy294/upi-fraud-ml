"""Identify whether original EDA model C generated the scored s1_v1.csv.

This compares predictions, not correctness. It never trains on submission
labels and never exports a leaderboard candidate. All outputs stay local.
"""
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eda"))
from common import LABELS, load, add_row_features

ROOT = REPO / "data/processed/research_probe"
OUT = REPO / "models/s1_identity_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

df, accounts, merchants = load(ROOT)
df = add_row_features(df, accounts, merchants).sort_values("t", kind="stable").reset_index(drop=True)
del accounts, merchants
relation = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
assert np.array_equal(df.txn_id, relation.txn_id)
rel_cols = [c for c in relation if c not in ("txn_id", "split", "label", "month")]
row_cols = ["hour", "weekday", "amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee",
    "pin_retries", "login_to_txn_sec", "balance_before", "qr_vpa_mismatch", "amt_to_balance", "away_from_home",
    "home_device", "amt_is_integer", "amt_round_100", "amt_round_500", "amt_round_1000",
    "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10", "rcv_acct_age_days", "mer_age_days"]
cats = ["receiver_type", "txn_type", "merchant_category", "app", "device_city"]
X = pd.concat([df[row_cols], relation[rel_cols]], axis=1)
for c in cats:
    X[c] = df[c].fillna("NA").astype("category")
# Original baseline C orders categoricals BEFORE relation features. This order
# matters because column subsampling selects columns by their positions.
X = X[row_cols + cats + rel_cols]
y = df.label.map({name: i for i, name in enumerate(LABELS)}).fillna(-1).to_numpy(dtype="int32")
fit = (df.split == "train").to_numpy() & (df.month < "2026-09").to_numpy()
valid = (df.split == "train").to_numpy() & (df.month == "2026-09").to_numpy()
test = (df.split == "test").to_numpy()
all_train = (df.split == "train").to_numpy()
test_ids = df.loc[test, "txn_id"].to_numpy()
reference = pd.read_csv(REPO.parent / "s1_v1.csv").set_index("id").loc[test_ids, "label"]
reference_codes = reference.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not reference.isna().any()
del df, relation
gc.collect()


def f1(truth, pred):
    cm = np.bincount(truth * 12 + pred, minlength=144).reshape(12, 12)
    return float((2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


results = {"purpose": "Prediction identity only; no October correctness signal is available",
    "source": "scripts/eda/07_baselines.py model C, original column order and numeric precision",
    "label_order": LABELS, "fits": {}}
for name, train in [("original_july_august", fit), ("original_full_refit", all_train)]:
    started = time.time()
    print("FIT", name, int(train.sum()), flush=True)
    model = lgb.LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=.05,
        num_leaves=63, min_child_samples=40, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, verbose=-1, n_jobs=12)
    model.fit(X.loc[train], y[train])
    model.booster_.save_model(str(OUT / f"{name}.txt"))
    P = model.predict_proba(X.loc[test])
    identity = {}
    for k in [1, 2, 3, 5, 8, 12, 20]:
        weights = np.array([1] + [k] * 11)
        pred = (P * weights).argmax(1)
        identity[str(k)] = {"different_from_s1": int((pred != reference_codes).sum()),
            "exact_match": bool(np.array_equal(pred, reference_codes)), "fraud_predictions": int((pred > 0).sum())}
    result = {"train_rows": int(train.sum()), "identity_by_shared_multiplier": identity,
              "elapsed_seconds": time.time() - started}
    if name == "original_july_august":
        V = model.predict_proba(X.loc[valid])
        result["september_macro_f1_by_multiplier"] = {str(k): f1(y[valid], (V * np.array([1] + [k] * 11)).argmax(1)) for k in [1, 2, 3, 5, 8, 12, 20]}
        np.savez_compressed(OUT / f"{name}_september.npz", P=V, y=y[valid], label_order=np.array(LABELS))
        del V
    np.savez_compressed(OUT / f"{name}_test.npz", P=P, ids=test_ids.astype(str), label_order=np.array(LABELS))
    results["fits"][name] = result
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("RESULT", name, json.dumps(result), flush=True)
    del model, P
    gc.collect()
