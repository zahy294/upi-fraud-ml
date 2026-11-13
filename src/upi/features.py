from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import stable_hash
from .events import Events

MINUTE, HOUR, DAY = 60, 3600, 86400


def _safe_div(numerator, denominator):
    den = pd.Series(denominator).replace(0, np.nan).to_numpy(dtype=np.float64)
    return np.asarray(numerator, dtype=np.float64) / den


def _category_codes(series: pd.Series) -> np.ndarray:
    return pd.factorize(series.astype("string").fillna("__MISSING__"), sort=True)[0].astype(np.int32)


def add_row_features(frame: pd.DataFrame, accounts: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    ts = pd.to_datetime(df.timestamp)
    amount = df.amount.to_numpy(dtype=np.float64)
    floored = np.floor(amount)
    df["month"] = ts.dt.strftime("%Y-%m")
    df["is_tail"] = ((df.split.eq("train") & ts.ge("2026-10-01")) |
                     (df.split.eq("test") & ts.ge("2026-11-01"))).astype(np.int8)
    df["hour"] = ts.dt.hour.astype(np.int8)
    df["weekday"] = ts.dt.weekday.astype(np.int8)
    df["day_of_month"] = ts.dt.day.astype(np.int8)
    df["is_weekend"] = ts.dt.weekday.ge(5).astype(np.int8)
    df["is_night"] = ts.dt.hour.between(0, 5).astype(np.int8)
    df["is_late_evening"] = ts.dt.hour.between(20, 23).astype(np.int8)
    df["log_amount"] = np.log1p(amount).astype(np.float32)
    df["log_balance"] = np.log1p(np.maximum(df.balance_before.to_numpy(dtype=float), 0)).astype(np.float32)
    df["amt_is_integer"] = np.isclose(amount, floored, atol=1e-8, rtol=0).astype(np.int8)
    for multiple in (10, 100, 500, 1000):
        df[f"amt_round_{multiple}"] = np.isclose(np.mod(amount, multiple), 0, atol=1e-8).astype(np.int8)
    gap = np.ceil(amount / 1000) * 1000 - amount
    df["amt_just_under_1000"] = ((gap > 0) & (gap <= 50) & (amount >= 900)).astype(np.int8)
    df["amt_ends_99"] = (np.mod(floored, 100) == 99).astype(np.int8)
    df["amt_tiny_le10"] = (amount <= 10).astype(np.int8)
    df["amt_exact_500"] = np.isclose(amount, 500).astype(np.int8)
    df["amt_to_balance"] = _safe_div(amount, df.balance_before).astype(np.float32)
    df["residual_balance_frac"] = _safe_div(df.balance_before - amount, df.balance_before).astype(np.float32)
    df["amount_above_balance"] = (amount > df.balance_before.to_numpy()).astype(np.int8)
    df["log_sim_age"] = np.log1p(np.maximum(df.sim_age_hours.to_numpy(dtype=float), 0)).astype(np.float32)
    df["log_device_age"] = np.log1p(np.maximum(df.device_age_days.to_numpy(dtype=float), 0)).astype(np.float32)
    df["log_login_delay"] = np.log1p(np.maximum(df.login_to_txn_sec.to_numpy(dtype=float), 0)).astype(np.float32)
    df["fresh_sim_7d"] = df.sim_age_hours.lt(168).astype(np.int8)
    df["new_device_7d"] = df.device_age_days.lt(7).astype(np.int8)
    df["drain_half_balance"] = (df.amt_to_balance.ge(.5) & df.amount.gt(500)).astype(np.int8)
    df["drain_fresh_sim"] = (df.drain_half_balance.eq(1) & df.fresh_sim_7d.eq(1) &
                              df.new_payee.eq(1) & df.receiver_type.eq("P2P") & df.amt_is_integer.eq(1)).astype(np.int8)
    df["drain_new_device"] = (df.drain_half_balance.eq(1) & df.new_device_7d.eq(1) &
                               df.new_payee.eq(1) & df.receiver_type.eq("P2P") & df.amt_is_integer.eq(1)).astype(np.int8)

    acc = accounts.set_index("account_id")
    sender_open = pd.to_datetime(df.sender_id.map(acc.opened_on))
    receiver_open = pd.to_datetime(df.receiver_id.map(acc.opened_on))
    p2p = df.receiver_type.eq("P2P")
    df["home_city"] = df.sender_id.map(acc.home_city)
    df["sender_kyc"] = df.sender_id.map(acc.kyc_level)
    df["sender_os"] = df.sender_id.map(acc.os)
    df["sender_bank"] = df.sender_id.map(acc.bank_code)
    df["receiver_kyc"] = df.receiver_id.map(acc.kyc_level).where(p2p)
    df["receiver_os"] = df.receiver_id.map(acc.os).where(p2p)
    df["receiver_bank"] = df.receiver_id.map(acc.bank_code).where(p2p)
    df["sender_open_age_days"] = (ts - sender_open).dt.days.astype("float32")
    df["receiver_open_age_days"] = (ts - receiver_open).dt.days.where(p2p).astype("float32")
    df["away_from_home"] = df.device_city.ne(df.home_city).astype(np.int8)
    df["home_device"] = df.device_id.eq("DV0" + df.sender_id.str[2:]).astype(np.int8)
    df["same_bank"] = (p2p & df.sender_bank.eq(df.receiver_bank)).astype(np.int8)
    df["same_city_p2p"] = (p2p & df.home_city.eq(df.receiver_id.map(acc.home_city))).astype(np.int8)

    mer = merchants.set_index("merchant_id")
    # reindex is robust for an empty merchant table and preserves row order.
    merchant_open = pd.Series(
        pd.to_datetime(mer.onboarded_on).reindex(df.receiver_id.to_numpy()).to_numpy(),
        index=df.index,
    )
    df["merchant_age_days"] = (ts - merchant_open).dt.days.where(~p2p).astype("float32")
    merchant_city = pd.Series(mer.city.reindex(df.receiver_id.to_numpy()).to_numpy(), index=df.index)
    df["merchant_city_match"] = ((~p2p) & df.device_city.eq(merchant_city)).astype(np.int8)

    categorical = [
        "receiver_type", "txn_type", "merchant_category", "device_city", "app",
        "home_city", "sender_kyc", "sender_os", "sender_bank", "receiver_kyc",
        "receiver_os", "receiver_bank",
    ]
    for column in categorical:
        df[f"cat_{column}"] = _category_codes(df[column])
    # Operational drift traps are intentionally excluded: app_version and ip_risk_score.
    # Keep raw categoricals for relational construction and diagnostics; model
    # selection uses only numeric columns and their stable cat_* encodings.
    df.drop(columns=[c for c in ["app_version", "ip_risk_score"] if c in df], inplace=True)
    return df


def _relational_pass(context: pd.DataFrame, query_ids: set[str]) -> pd.DataFrame:
    df = context.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)
    query = df.txn_id.isin(query_ids).to_numpy()
    t = df.timestamp.to_numpy(dtype="datetime64[s]").astype(np.int64)
    amount = df.amount.to_numpy(dtype=np.float64)
    sender = pd.factorize(df.sender_id, sort=False)[0].astype(np.int64)
    receiver = pd.factorize(df.receiver_id, sort=False)[0].astype(np.int64)
    device = pd.factorize(df.device_id, sort=False)[0].astype(np.int64)
    p2p = df.receiver_type.eq("P2P").to_numpy()
    receiver_account = pd.Index(pd.unique(df.sender_id)).get_indexer(df.receiver_id).astype(np.int64)
    sender_index = pd.Index(pd.unique(df.sender_id)).get_indexer(df.sender_id).astype(np.int64)
    # Use the same account-code namespace for incoming/outgoing flow queries.
    account_index = pd.Index(pd.unique(pd.concat([df.sender_id, df.receiver_id[p2p]], ignore_index=True)))
    sender_account = account_index.get_indexer(df.sender_id).astype(np.int64)
    receiver_account = np.where(p2p, account_index.get_indexer(df.receiver_id), -1).astype(np.int64)
    out_events = Events(sender_account, t, amount)
    in_events = Events(np.where(p2p, receiver_account, receiver + len(account_index)), t, amount)
    device_events = Events(device, t)
    sender_events = Events(sender, t, amount)
    receiver_events = Events(receiver, t, amount)

    f = pd.DataFrame({"txn_id": df.loc[query, "txn_id"].to_numpy()})
    qt, qa = t[query], amount[query]
    qs, qr, qd = sender[query], receiver[query], device[query]
    qsa, qra = sender_account[query], receiver_account[query]
    qp2p = p2p[query]
    for name, events, groups in (("sender", sender_events, qs), ("receiver", receiver_events, qr),
                                  ("device", device_events, qd)):
        for window_name, seconds in (("5m", 5*MINUTE), ("1h", HOUR), ("6h", 6*HOUR), ("24h", DAY), ("7d", 7*DAY)):
            if name == "device" and window_name == "7d":
                continue
            f[f"{name}_cnt_{window_name}"] = events.count(groups, qt, -seconds, seconds)
        f[f"{name}_prev_gap"] = events.previous_gap(groups, qt).astype(np.float32)
        f[f"{name}_next_gap"] = events.next_gap(groups, qt).astype(np.float32)
    f["device_burst_ratio_5m_24h"] = _safe_div(f.device_cnt_5m, f.device_cnt_24h).astype(np.float32)
    f["sender_burst_ratio_1h_24h"] = _safe_div(f.sender_cnt_1h, f.sender_cnt_24h).astype(np.float32)
    f["receiver_burst_ratio_1h_24h"] = _safe_div(f.receiver_cnt_1h, f.receiver_cnt_24h).astype(np.float32)
    f["receiver_amount_1h"] = receiver_events.sum(qr, qt, -HOUR, HOUR).astype(np.float32)
    f["sender_amount_24h"] = sender_events.sum(qs, qt, -DAY, DAY).astype(np.float32)
    f["receiver_out_30m"] = out_events.count(qra, qt, 1, 30*MINUTE)
    f["receiver_out_2h"] = out_events.count(qra, qt, 1, 2*HOUR)
    f["sender_in_30m"] = in_events.count(qsa, qt, -30*MINUTE, -1)
    next_amount = out_events.next_value(qra, qt, 30*MINUTE)
    previous_amount = in_events.previous_value(qsa, qt, 30*MINUTE)
    f["forward_amount_ratio_30m"] = (next_amount / np.maximum(qa, 1e-9)).astype(np.float32)
    f["previous_in_amount_ratio_30m"] = (qa / np.maximum(previous_amount, 1e-9)).astype(np.float32)
    f["flow_forward_compatible"] = (qp2p & (f.forward_amount_ratio_30m >= .8) &
                                     (f.forward_amount_ratio_30m <= 1.0)).astype(np.int8)
    f["flow_previous_compatible"] = ((f.previous_in_amount_ratio_30m >= .8) &
                                      (f.previous_in_amount_ratio_30m <= 1.0)).astype(np.int8)

    pair_index = pd.MultiIndex.from_arrays([df.sender_id, df.receiver_id])
    pair, unique_pairs = pd.factorize(pair_index, sort=False)
    pair = pair.astype(np.int64)
    pair_events = Events(pair, t, amount)
    qp = pair[query]
    f["pair_cnt_24h"] = pair_events.count(qp, qt, -DAY, DAY)
    f["pair_cnt_7d"] = pair_events.count(qp, qt, -7*DAY, 7*DAY)
    f["pair_prev_gap"] = pair_events.previous_gap(qp, qt).astype(np.float32)
    reverse_index = pd.MultiIndex.from_arrays([df.receiver_id, df.sender_id])
    reverse_code = unique_pairs.get_indexer(reverse_index).astype(np.int64)
    qreverse = reverse_code[query]
    f["reverse_pair_cnt_24h"] = pair_events.count(qreverse, qt, -DAY, DAY)
    small = Events(pair, t, amount, mask=amount <= 10)
    big = Events(pair, t, amount, mask=amount >= 100)
    f["reverse_small_before_24h"] = small.count(qreverse, qt, -DAY, -1)
    f["reverse_big_after_24h"] = big.count(qreverse, qt, 1, DAY)

    day = df.timestamp.dt.floor("D")
    daily = df.groupby(["receiver_id", day], observed=True).agg(
        receiver_unique_senders_day=("sender_id", "nunique"),
        receiver_new_payee_share_day=("new_payee", "mean"),
    ).reset_index()
    qkeys = df.loc[query, ["receiver_id"]].copy()
    qkeys["timestamp"] = day[query].to_numpy()
    qkeys.columns = ["receiver_id", "timestamp"]
    daily.columns = ["receiver_id", "timestamp", "receiver_unique_senders_day", "receiver_new_payee_share_day"]
    merged = qkeys.merge(daily, on=["receiver_id", "timestamp"], how="left", sort=False)
    f["receiver_unique_senders_day"] = merged.receiver_unique_senders_day.to_numpy(dtype=np.float32)
    f["receiver_new_payee_share_day"] = merged.receiver_new_payee_share_day.to_numpy(dtype=np.float32)
    f["device_unique_accounts_total"] = df.groupby("device_id", observed=True).sender_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["sender_unique_receivers_day"] = df.groupby(["sender_id", day], observed=True).receiver_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    return f


def add_relational_features(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame.split.eq("train") & frame.timestamp.lt("2026-10-01")].copy()
    test = frame[frame.split.eq("test")].copy()
    train_features = _relational_pass(train, set(train.txn_id))
    # Final target features see historical labelled-period rows without labels, plus all test rows.
    test_context = pd.concat([train.drop(columns=["label"], errors="ignore"),
                              test.drop(columns=["label"], errors="ignore")], ignore_index=True)
    test_features = _relational_pass(test_context, set(test.txn_id))
    features = pd.concat([train_features, test_features], ignore_index=True)
    if features.txn_id.duplicated().any():
        raise ValueError("duplicate txn_id in relational feature table")
    return frame.merge(features, on="txn_id", how="left", validate="one_to_one", sort=False)


def add_prior_label_history(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    train = df[df.split.eq("train") & df.timestamp.lt("2026-10-01")]
    targets = ["2026-08", "2026-09", "2026-10", "2026-11"]
    definitions = [
        ("hist_receiver_anyfraud", "receiver_id", None),
        ("hist_receiver_mule", "receiver_id", "mule_account"),
        ("hist_receiver_layer", "receiver_id", "layering_chain"),
        ("hist_receiver_phishing", "receiver_id", "phishing_collect"),
        ("hist_receiver_qr", "receiver_id", "qr_swap"),
        ("hist_sender_refund", "sender_id", "refund_abuse"),
        ("hist_device_bot", "device_id", "velocity_bot"),
    ]
    for feature, entity, label in definitions:
        values = np.zeros(len(df), dtype=np.float32)
        for month in targets:
            query = df.month.eq(month)
            historical = train.month.lt(month)
            positive = train.label.ne("legit") if label is None else train.label.eq(label)
            counts = train.loc[historical & positive, entity].value_counts()
            values[query] = df.loc[query, entity].map(counts).fillna(0).clip(upper=20).to_numpy(dtype=np.float32)
        df[feature] = values
    df["hist_phishing_collect_interaction"] = ((df.hist_receiver_phishing > 0) &
                                                df.cat_txn_type.eq(_code_for(df, "txn_type", "COLLECT"))).astype(np.int8)
    df["hist_refund_claim_interaction"] = ((df.hist_sender_refund > 0) &
                                            df.cat_txn_type.eq(_code_for(df, "txn_type", "REFUND"))).astype(np.int8)
    return df


def _code_for(df: pd.DataFrame, original: str, value: str) -> int:
    # Codes were generated in sorted lexical order including missing.
    source_name = f"cat_{original}"
    if original == "txn_type":
        known = sorted(["AUTOPAY", "COLLECT", "PAY", "QR", "REFUND", "__MISSING__"])
        return known.index(value)
    raise KeyError(source_name)


def build_feature_table(train: pd.DataFrame, test: pd.DataFrame, accounts: pd.DataFrame,
                        merchants: pd.DataFrame, include_relational: bool = True,
                        include_history: bool = True) -> pd.DataFrame:
    test = test.copy()
    test["label"] = pd.NA
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    featured = add_row_features(combined, accounts, merchants)
    if include_relational:
        featured = add_relational_features(featured)
    if include_history:
        featured = add_prior_label_history(featured)
    for column in featured.columns:
        if column.startswith("cat_"):
            featured[column] = featured[column].fillna(-1).astype(np.int32)
        elif column not in {"txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "label", "split", "month"}:
            if pd.api.types.is_numeric_dtype(featured[column]):
                featured[column] = featured[column].replace([np.inf, -np.inf], np.nan)
                if featured[column].dtype == "float64":
                    featured[column] = featured[column].astype(np.float32)
    return featured


def feature_fingerprint(config: dict, input_hashes: dict[str, str]) -> str:
    source = Path(__file__).read_bytes()
    return stable_hash({"feature_config": config.get("features", {}), "inputs": input_hashes,
                        "source_sha256": hashlib.sha256(source).hexdigest()})

