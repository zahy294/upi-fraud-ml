"""Research-only ablations; no submission or production-model training.

Run after EDA scripts 00 and 04 against an isolated work root:
  .venv/bin/python -u scripts/research/validate_candidates.py WORK_ROOT
Outputs go to WORK_ROOT. Original EDA reports and raw CSVs stay unchanged.
"""
from pathlib import Path
import sys
import json
import gc
import time
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eda"))
from common import LABELS, load, add_row_features

parser = argparse.ArgumentParser()
parser.add_argument("work_root", type=Path)
parser.add_argument("--context-only", action="store_true")
parser.add_argument("--base-only", action="store_true")
parser.add_argument("--only-month", type=int, choices=[8, 9])
parser.add_argument("--weights-from", type=Path)
parser.add_argument("--seed", type=int, default=137)
ARGS = parser.parse_args()
ROOT = ARGS.work_root
DAY = 86400
SCALE = 2**26
SEED = ARGS.seed
CONTEXT_ONLY = ARGS.context_only


def metrics(y, pred):
    cm = np.bincount(y * 12 + pred, minlength=144).reshape(12, 12)
    tp = cm.diagonal()
    f = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {"macro_f1": float(f.mean()), "f1": dict(zip(LABELS, f.tolist())),
            "predicted": dict(zip(LABELS, cm.sum(0).tolist()))}


def tune(P, y):
    """Small prespecified joint search; August fits, September tests weights."""
    best = (-1, None)
    for k in [1, 2, 3, 5, 8, 12, 20]:
        w = np.array([1] + [k] * 11, dtype=float)
        score = metrics(y, (P * w).argmax(1))["macro_f1"]
        if score > best[0]:
            best = score, w
    shared = best[1].copy()
    w = shared.copy()
    for _ in range(2):
        for c in range(1, 12):
            scores = P * w
            scores[:, c] = -1
            other = scores.argmax(1)
            mx = scores[np.arange(len(y)), other]
            grid = np.unique(np.r_[np.geomspace(0.5, 30, 15), w[c]])
            results = [metrics(y, np.where(P[:, c] * g > mx, c, other))["macro_f1"] for g in grid]
            top = max(results)
            options = np.flatnonzero(np.array(results) >= top - 0.0002)
            # Prefer the smallest departure from the current weight on a plateau.
            chosen = min(options, key=lambda i: abs(np.log(grid[i] / w[c])))
            w[c] = grid[chosen]
    return shared, w


def history(keys, t, y, cutoff):
    """Labels strictly before min(start-of-target-day minus 7d, fold cutoff).

    Training rows use their own historical cutoff; all validation rows are
    additionally frozen at the model's label-training cutoff. No row can see
    its own label, nearby event labels, or any labels from its validation month.
    """
    qt = np.minimum((t // DAY) * DAY - 7 * DAY, cutoff)
    H = {}
    donors = (y >= 0) & (t < cutoff)
    for name, g in keys.items():
        q = g * SCALE + qt
        all_keys = np.sort(g[donors] * SCALE + t[donors])
        starts = np.searchsorted(all_keys, g * SCALE, side="left")
        count = np.searchsorted(all_keys, q, side="left") - starts
        count = np.maximum(count, 0)
        H[name + "_history_n"] = np.log1p(count).astype("float32")
        fraud_n = np.zeros(len(y), dtype="int32")
        for c in range(1, 12):
            d = donors & (y == c)
            k = np.sort(g[d] * SCALE + t[d])
            n = np.searchsorted(k, q, side="left") - np.searchsorted(k, g * SCALE, side="left")
            n = np.maximum(n, 0)
            fraud_n += n
            H[f"{name}_history_{c}"] = (n / (count + 20)).astype("float32")
        H[name + "_history_fraud"] = (fraud_n / (count + 20)).astype("float32")
    return pd.DataFrame(H)


def probes(df, rules, y, t):
    """Validate simple history signal and count unknown neighbours of seeds."""
    records = []
    p2p = (df.receiver_type == "P2P").to_numpy()
    for month in [8, 9]:
        past = (df.timestamp.dt.month.to_numpy() < month) & (y >= 0)
        target = (df.timestamp.dt.month.to_numpy() == month) & (y >= 0)
        if not target.any():
            continue
        for role in ["sender_id", "receiver_id", "device_id"]:
            seen = set(df.loc[past & (y > 0), role])
            hit = target & df[role].isin(seen).to_numpy()
            if role == "receiver_id":
                hit &= p2p
            records.append({"month": month, "role": role, "hits": int(hit.sum()),
                "fraud_precision": float((y[hit] > 0).mean()),
                "fraud_recall": float((hit & (y > 0)).sum() / (target & (y > 0)).sum()),
                "fraud_outside_rules": int((hit & (y > 0) & (rules == 0)).sum())})
    purity = []
    for label, col, window in [(1, "receiver_id", 3600), (2, "device_id", 300)]:
        g = pd.factorize(df[col])[0].astype("int64")
        train = y >= 0
        seed = y == label
        if label == 1:
            seed &= p2p
        k = np.sort(g[seed] * SCALE + t[seed])
        q = g * SCALE + t
        near = (np.searchsorted(k, q + window, side="right") - np.searchsorted(k, q - window, side="left")) > 0
        near &= train
        if label == 1:
            near &= p2p
        purity.append({"label": LABELS[label], "selected_rows": int(near.sum()),
                       "seed_rows": int((near & seed).sum()), "nonseed_rows": int((near & ~seed).sum()),
                       "nonseed_labels": dict(zip(LABELS, np.bincount(y[near & ~seed], minlength=12).tolist()))})
    return {"prior_month": records, "seed_selection_audit": purity}


def context_features(keys, t, amount):
    """Small label-free gap/amount pack motivated by winner writeups."""
    out = {}
    for name, g in keys.items():
        q = g * SCALE + t
        order = np.argsort(q, kind="stable")
        k = q[order]
        sums = np.r_[0., np.cumsum(amount[order], dtype="float64")]
        prev = np.searchsorted(k, q, side="left") - 1
        nxt = np.searchsorted(k, q, side="right")
        for direction, idx in [("prev", prev), ("next", nxt)]:
            safe = np.clip(idx, 0, len(k) - 1)
            gap = np.abs(k[safe] - q)
            ok = (idx >= 0) & (idx < len(k)) & (g[order[safe]] == g) & (gap <= 7 * DAY)
            out[f"{name}_{direction}_gap"] = np.where(ok, gap, np.nan).astype("float32")
        for window, seconds in [("2h", 7200), ("24h", DAY)]:
            left = np.searchsorted(k, q - seconds, side="left")
            right = np.searchsorted(k, q + seconds, side="right")
            count = right - left
            total = sums[right] - sums[left]
            mean = total / np.maximum(count, 1)
            out[f"{name}_sum_{window}"] = total.astype("float32")
            out[f"{name}_amount_over_mean_{window}"] = (amount / np.maximum(mean, .01)).astype("float32")
    return pd.DataFrame(out)


def main():
    start = time.time()
    df, ac, me = load(ROOT)
    df = add_row_features(df, ac, me).sort_values("t", kind="stable").reset_index(drop=True)
    del ac, me
    R = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
    assert np.array_equal(df.txn_id, R.txn_id)
    rel = [c for c in R if c not in ["txn_id", "split", "label", "month"]]
    df = pd.concat([df, R[rel]], axis=1)
    del R
    y = df.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).to_numpy(dtype="int32")
    t = df.t.to_numpy(dtype="int64")
    month = df.timestamp.dt.month.to_numpy()
    p2p = df.receiver_type == "P2P"
    integer = df.amt_is_integer == 1
    # Identical priority and conditions to the saved rules baseline.
    rules = [
        (8, (df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
        (3, integer & p2p & (df.fwd_amt_ratio_30m.between(.8,.9999) | df.prev_in_amt_ratio_30m.between(.8,.9999)) & (df.amount >= 5000)),
        (9, integer & ((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0)))),
        (11, ~p2p & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
        (1, integer & p2p & (df.rcv_uniq_snd_day >= 3) & (df.new_payee == 1)),
        (1, integer & p2p & df.fwd_amt_ratio_30m.between(.8,.9999)),
        (6, integer & (df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
        (7, (df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
        (2, integer & ~p2p & ((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)) & (df.home_device == 0)),
        (4, integer & p2p & (df.sim_age_hours < 168) & (df.new_payee == 1)),
        (5, integer & p2p & (df.device_age_days < 7) & (df.new_payee == 1)),
        (10, integer & (df.merchant_category == "lending") & (df.new_payee == 1)),
    ]
    rp = np.zeros(len(y), dtype="int32")
    extra = {}
    for i, (c, mask) in enumerate(rules):
        extra[f"rule_{i}_{c}"] = mask.fillna(False).to_numpy(dtype="float32")
    for c, mask in reversed(rules):
        rp[mask.fillna(False).to_numpy()] = c
    extra.update({
        "exact_500": (df.amount == 500).astype("float32"),
        "over_balance": (df.amount > df.balance_before).astype("float32"),
        "device_burst_ratio": df.dev_cnt_5m / df.dev_cnt_1h.clip(lower=1),
        "sender_burst_ratio": df.snd_out_cnt_10m / df.snd_out_cnt_24h.clip(lower=1),
        "sim_and_new_device": ((df.sim_age_hours < 168) & (df.device_age_days < 7)).astype("float32"),
        "new_device_away": ((df.device_age_days < 7) & (df.away_from_home == 1)).astype("float32"),
        "fresh_sim_night": ((df.sim_age_hours < 168) & (df.is_night == 1)).astype("float32"),
    })
    cats = ["receiver_type", "txn_type", "merchant_category", "app", "device_city"]
    row = ["hour", "weekday", "amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries", "login_to_txn_sec", "balance_before", "qr_vpa_mismatch", "amt_to_balance", "away_from_home", "home_device", "amt_is_integer", "amt_round_100", "amt_round_500", "amt_round_1000", "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10", "rcv_acct_age_days", "mer_age_days"]
    X = df[row + rel].astype("float32").replace([np.inf, -np.inf], np.nan)
    for c in cats:
        X[c] = df[c].fillna("NA").astype("category").cat.codes.astype("int32")
    E = pd.DataFrame(extra).astype("float32")
    keys = {c: pd.factorize(df[c])[0].astype("int64") for c in ["sender_id", "receiver_id", "device_id"]}
    C = context_features(keys, t, df.amount.to_numpy(dtype="float64")) if CONTEXT_ONLY else None
    result = {"protocol": {"seed": SEED, "negative_sample_fraction": .2, "negative_weight": 5,
        "trees": 200, "leaves": 31, "learning_rate": .05,
        "label_free_context": "all supplied train+test rows, same as existing EDA; no test labels",
        "history": "strictly before min(target day - 7 days, fold training cutoff)",
        "decision_tuning": "August only; frozen for September; full validation populations"},
        "probes": probes(df, rp, y, t), "rules": {}, "experiments": {}}
    for m in [8, 9]:
        ix = (y >= 0) & (month == m)
        if ix.any():
            result["rules"][str(m)] = metrics(y[ix], rp[ix])
    days = (t // DAY).astype("int16")
    ids = df.txn_id.to_numpy()
    del df, rules, extra
    gc.collect()
    # Sampling fixed per row; validation retains all positives and negatives.
    retain = np.random.default_rng(SEED).random(len(y)) < .2
    august_weights = {}
    if ARGS.weights_from:
        previous = json.loads(ARGS.weights_from.read_text())
        for key, value in previous["experiments"].items():
            if key.startswith("8_"):
                august_weights[key[2:]] = (np.array(value["shared_weights"]), np.array(value["joint_weights"]))
    for m in ([ARGS.only_month] if ARGS.only_month else [8, 9]):
        cutoff = int((pd.Timestamp(f"2026-{m:02d}-01") - pd.Timestamp("2026-07-01")).total_seconds())
        train = (y >= 0) & (t < cutoff) & ((y > 0) | retain)
        valid = (y >= 0) & (month == m)
        weights = np.where(y[train] == 0, 5., 1.)
        H = None if (CONTEXT_ONLY or ARGS.base_only) else history(keys, t, y, cutoff)
        families = [("context", [C])] if CONTEXT_ONLY else ([("base", [])] if ARGS.base_only else [("base", []), ("interactions", [E]), ("history_interactions", [E, H])])
        for name, additional in families:
            cols = pd.concat([X] + additional, axis=1)
            print("FIT", m, name, int(train.sum()), cols.shape[1], flush=True)
            model = lgb.LGBMClassifier(objective="multiclass", num_class=12,
                n_estimators=200, learning_rate=.05, num_leaves=31, min_child_samples=40,
                max_bin=127, colsample_bytree=.8, subsample=.8, subsample_freq=1,
                n_jobs=4, random_state=SEED, verbosity=-1)
            model.fit(cols.loc[train], y[train], sample_weight=weights, categorical_feature=cats)
            P = model.predict_proba(cols.loc[valid]).astype("float32")
            if m == 8:
                august_weights[name] = tune(P, y[valid])
            shared, joint = august_weights[name]
            entry = {"raw": metrics(y[valid], P.argmax(1)),
                     "shared_august": metrics(y[valid], (P * shared).argmax(1)),
                     "joint_august": metrics(y[valid], (P * joint).argmax(1)),
                     "shared_weights": shared.tolist(), "joint_weights": joint.tolist()}
            pred = (P * joint).argmax(1)
            entry["new_correct_vs_rules"] = dict(zip(LABELS, np.bincount(y[valid][(pred == y[valid]) & (rp[valid] != y[valid])], minlength=12).tolist()))
            result["experiments"][f"{m}_{name}"] = entry
            np.savez_compressed(ROOT / f"prob_{m}_{name}.npz", P=P, y=y[valid], days=days[valid], ids=ids[valid], rules=rp[valid], weights=joint)
            (ROOT / ("context_results.json" if CONTEXT_ONLY else "candidate_results.json")).write_text(json.dumps(result, indent=2))
            print("RESULT", m, name, {k: entry[k]["macro_f1"] for k in ["raw", "shared_august", "joint_august"]}, flush=True)
            del cols, model, P
            gc.collect()
        del H
    print("elapsed_seconds", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
