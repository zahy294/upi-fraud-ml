from __future__ import annotations

import hashlib
from pathlib import Path

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
    df["amt_mod_100"] = np.mod(amount, 100).astype(np.float32)
    df["amt_mod_1000"] = np.mod(amount, 1000).astype(np.float32)
    df["amt_to_balance"] = _safe_div(amount, df.balance_before).astype(np.float32)
    df["residual_balance_frac"] = _safe_div(df.balance_before - amount, df.balance_before).astype(np.float32)
    df["amount_above_balance"] = (amount > df.balance_before.to_numpy()).astype(np.int8)
    df["log_sim_age"] = np.log1p(np.maximum(df.sim_age_hours.to_numpy(dtype=float), 0)).astype(np.float32)
    df["log_device_age"] = np.log1p(np.maximum(df.device_age_days.to_numpy(dtype=float), 0)).astype(np.float32)
    df["log_login_delay"] = np.log1p(np.maximum(df.login_to_txn_sec.to_numpy(dtype=float), 0)).astype(np.float32)
    df["sim_to_device_age_ratio"] = _safe_div(df.sim_age_hours, df.device_age_days * 24.0 + 1.0).astype(np.float32)
    df["fresh_sim_7d"] = df.sim_age_hours.lt(168).astype(np.int8)
    df["fresh_sim_48h"] = df.sim_age_hours.lt(48).astype(np.int8)
    df["new_device_7d"] = df.device_age_days.lt(7).astype(np.int8)
    df["new_device_2d"] = df.device_age_days.lt(2).astype(np.int8)
    df["fast_login_5s"] = df.login_to_txn_sec.le(5).astype(np.int8)
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
    df["qr_p2p_flag"] = (df.txn_type.eq("QR") & p2p).astype(np.int8)
    df["is_fractional_refund"] = (df.txn_type.eq("REFUND") & df.amt_is_integer.eq(0)).astype(np.int8)
    df["round500_sim_new_p2p"] = (df.amt_exact_500.eq(1) & df.fresh_sim_7d.eq(1) & df.new_payee.eq(1) & p2p).astype(np.int8)
    df["round500_dev_new_p2p"] = (df.amt_exact_500.eq(1) & df.new_device_7d.eq(1) & df.new_payee.eq(1) & p2p).astype(np.int8)

    mer = merchants.set_index("merchant_id")
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

    # Unified account index for incoming/outgoing flow queries
    account_index = pd.Index(pd.unique(pd.concat([df.sender_id, df.receiver_id[p2p]], ignore_index=True)))
    sender_account = account_index.get_indexer(df.sender_id).astype(np.int64)
    receiver_account = np.where(p2p, account_index.get_indexer(df.receiver_id), -1).astype(np.int64)

    out_events = Events(sender_account, t, amount)
    in_events = Events(np.where(p2p, receiver_account, receiver + len(account_index)), t, amount)
    device_events = Events(device, t, amount)
    sender_events = Events(sender, t, amount)
    receiver_events = Events(receiver, t, amount)

    f: dict[str, np.ndarray] = {"txn_id": df.loc[query, "txn_id"].to_numpy()}
    qt, qa = t[query], amount[query]
    qs, qr, qd = sender[query], receiver[query], device[query]
    qsa, qra = sender_account[query], receiver_account[query]
    qp2p = p2p[query]

    # 1. Multi-window counts & gaps for sender, receiver, device
    for name, events, groups in (("sender", sender_events, qs), ("receiver", receiver_events, qr),
                                  ("device", device_events, qd)):
        for window_name, seconds in (("1m", MINUTE), ("5m", 5 * MINUTE), ("15m", 15 * MINUTE),
                                     ("1h", HOUR), ("6h", 6 * HOUR), ("24h", DAY),
                                     ("7d", 7 * DAY), ("15d", 15 * DAY)):
            f[f"{name}_cnt_{window_name}"] = events.count(groups, qt, -seconds, seconds)
        f[f"{name}_prev_gap"] = events.previous_gap(groups, qt).astype(np.float32)
        f[f"{name}_next_gap"] = events.next_gap(groups, qt).astype(np.float32)
        f[f"{name}_min_gap"] = np.fmin(f[f"{name}_prev_gap"], f[f"{name}_next_gap"]).astype(np.float32)

    f["sender_cnt_10m"] = sender_events.count(qs, qt, -10 * MINUTE, 10 * MINUTE)

    # Burst ratios (density-robust for October)
    f["device_burst_ratio_1m_1h"] = _safe_div(f["device_cnt_1m"], f["device_cnt_1h"]).astype(np.float32)
    f["device_burst_ratio_5m_1h"] = _safe_div(f["device_cnt_5m"], f["device_cnt_1h"]).astype(np.float32)
    f["device_burst_ratio_5m_24h"] = _safe_div(f["device_cnt_5m"], f["device_cnt_24h"]).astype(np.float32)
    f["device_burst_ratio_1h_24h"] = _safe_div(f["device_cnt_1h"], f["device_cnt_24h"]).astype(np.float32)
    f["sender_burst_ratio_1h_24h"] = _safe_div(f["sender_cnt_1h"], f["sender_cnt_24h"]).astype(np.float32)
    f["sender_burst_ratio_24h_7d"] = _safe_div(f["sender_cnt_24h"], f["sender_cnt_7d"]).astype(np.float32)
    f["receiver_burst_ratio_1h_24h"] = _safe_div(f["receiver_cnt_1h"], f["receiver_cnt_24h"]).astype(np.float32)
    f["receiver_burst_ratio_24h_7d"] = _safe_div(f["receiver_cnt_24h"], f["receiver_cnt_7d"]).astype(np.float32)

    # Amount sums & relative amount ratios (inflation-robust for October)
    f["receiver_amount_1h"] = receiver_events.sum(qr, qt, -HOUR, HOUR).astype(np.float32)
    f["receiver_amount_24h"] = receiver_events.sum(qr, qt, -DAY, DAY).astype(np.float32)
    f["sender_amount_1h"] = sender_events.sum(qs, qt, -HOUR, HOUR).astype(np.float32)
    f["sender_amount_24h"] = sender_events.sum(qs, qt, -DAY, DAY).astype(np.float32)
    f["sender_amount_7d"] = sender_events.sum(qs, qt, -7 * DAY, 7 * DAY).astype(np.float32)
    f["device_amount_1h"] = device_events.sum(qd, qt, -HOUR, HOUR).astype(np.float32)

    snd_mean_amt_15d = _safe_div(sender_events.sum(qs, qt, -15 * DAY, 15 * DAY), f["sender_cnt_15d"])
    rcv_mean_amt_15d = _safe_div(receiver_events.sum(qr, qt, -15 * DAY, 15 * DAY), f["receiver_cnt_15d"])
    f["amt_to_sender_mean_15d"] = _safe_div(qa, snd_mean_amt_15d).astype(np.float32)
    f["amt_to_receiver_mean_15d"] = _safe_div(qa, rcv_mean_amt_15d).astype(np.float32)

    # 2. Money flow / Forwarding / Pass-through (layering_chain & mule_account)
    f["receiver_out_30m"] = out_events.count(qra, qt, 1, 30 * MINUTE)
    f["receiver_out_2h"] = out_events.count(qra, qt, 1, 2 * HOUR)
    f["receiver_out_24h"] = out_events.count(qra, qt, -DAY, DAY)
    f["sender_in_30m"] = in_events.count(qsa, qt, -30 * MINUTE, -1)
    f["sender_in_2h"] = in_events.count(qsa, qt, -2 * HOUR, -1)
    f["sender_in_24h"] = in_events.count(qsa, qt, -DAY, DAY)

    next_amount = out_events.next_value(qra, qt, 30 * MINUTE)
    previous_amount = in_events.previous_value(qsa, qt, 30 * MINUTE)
    f["forward_amount_ratio_30m"] = (next_amount / np.maximum(qa, 1e-9)).astype(np.float32)
    f["previous_in_amount_ratio_30m"] = (qa / np.maximum(previous_amount, 1e-9)).astype(np.float32)
    f["flow_forward_compatible"] = (qp2p & (f["forward_amount_ratio_30m"] >= .8) &
                                     (f["forward_amount_ratio_30m"] <= 1.0)).astype(np.int8)
    f["flow_previous_compatible"] = ((f["previous_in_amount_ratio_30m"] >= .8) &
                                      (f["previous_in_amount_ratio_30m"] <= 1.0)).astype(np.int8)

    next_amount_2h = out_events.next_value(qra, qt, 2 * HOUR)
    prev_amount_2h = in_events.previous_value(qsa, qt, 2 * HOUR)
    f["forward_amount_ratio_2h"] = (next_amount_2h / np.maximum(qa, 1e-9)).astype(np.float32)
    f["previous_in_amount_ratio_2h"] = (qa / np.maximum(prev_amount_2h, 1e-9)).astype(np.float32)

    rcv_out_amt_24h = out_events.sum(qra, qt, -DAY, DAY)
    snd_in_amt_24h = in_events.sum(qsa, qt, -DAY, DAY)
    f["rcv_pass_through_ratio_24h"] = np.where(qp2p, _safe_div(rcv_out_amt_24h, f["receiver_amount_24h"]), np.nan).astype(np.float32)
    f["snd_pass_through_ratio_24h"] = _safe_div(f["sender_amount_24h"], snd_in_amt_24h).astype(np.float32)

    # 3. Pair & Reverse-Pair features (fake_cashback, merchant_collusion, loan_app_extortion, refund_abuse)
    pair_index = pd.MultiIndex.from_arrays([df.sender_id, df.receiver_id])
    pair, unique_pairs = pd.factorize(pair_index, sort=False)
    pair = pair.astype(np.int64)
    pair_events = Events(pair, t, amount)
    qp = pair[query]
    f["pair_cnt_24h"] = pair_events.count(qp, qt, -DAY, DAY)
    f["pair_cnt_7d"] = pair_events.count(qp, qt, -7 * DAY, 7 * DAY)
    f["pair_cnt_30d"] = pair_events.count(qp, qt, -15 * DAY, 15 * DAY)
    f["pair_cnt_total"] = np.bincount(pair)[qp].astype(np.int32)
    f["pair_prev_gap"] = pair_events.previous_gap(qp, qt).astype(np.float32)
    f["pair_next_gap"] = pair_events.next_gap(qp, qt).astype(np.float32)

    reverse_index = pd.MultiIndex.from_arrays([df.receiver_id, df.sender_id])
    reverse_code = unique_pairs.get_indexer(reverse_index).astype(np.int64)
    qreverse = reverse_code[query]
    f["reverse_pair_cnt_24h"] = pair_events.count(qreverse, qt, -DAY, DAY)
    f["reverse_pair_cnt_7d"] = pair_events.count(qreverse, qt, -7 * DAY, 7 * DAY)
    f["rev_pair_cnt_total"] = np.where(
        qreverse >= 0,
        np.bincount(pair, minlength=len(unique_pairs))[np.maximum(qreverse, 0)],
        0,
    ).astype(np.int32)

    small = Events(pair, t, amount, mask=amount <= 10)
    big = Events(pair, t, amount, mask=amount >= 100)
    f["reverse_small_before_24h"] = small.count(qreverse, qt, -DAY, -1)
    f["reverse_big_after_24h"] = big.count(qreverse, qt, 1, DAY)
    f["reverse_small_before_7d"] = small.count(qreverse, qt, -7 * DAY, -1)
    f["reverse_big_after_7d"] = big.count(qreverse, qt, 1, 7 * DAY)

    rev_prev_amt = pair_events.previous_value(qreverse, qt, DAY)
    rev_next_amt = pair_events.next_value(qreverse, qt, DAY)
    f["rev_prev_amount_24h"] = rev_prev_amt.astype(np.float32)
    f["rev_next_amount_24h"] = rev_next_amt.astype(np.float32)
    f["rev_payback_ratio_24h"] = _safe_div(qa, rev_prev_amt).astype(np.float32)
    f["rev_bait_ratio_24h"] = _safe_div(rev_next_amt, qa).astype(np.float32)

    # 4. Refund Abuse, Collect, QR Swap, Merchant Collusion, Loan Extortion specific windows
    is_ref = df.txn_type.eq("REFUND").to_numpy()
    is_col = df.txn_type.eq("COLLECT").to_numpy()
    is_qr = df.txn_type.eq("QR").to_numpy()
    is_r500 = df.amt_round_500.eq(1).to_numpy()
    is_r1000 = df.amt_round_1000.eq(1).to_numpy()
    is_just_under = (df.amt_just_under_1000.eq(1) | df.amt_ends_99.eq(1)).to_numpy()
    is_eve = df.is_late_evening.eq(1).to_numpy()
    is_night = df.is_night.eq(1).to_numpy()
    is_int = df.amt_is_integer.eq(1).to_numpy()
    is_new_payee = df.new_payee.eq(1).to_numpy()
    is_qr_mm = df.qr_vpa_mismatch.eq(1).to_numpy()
    is_lending = df.merchant_category.eq("lending").to_numpy()
    is_nonhome = df.home_device.eq(0).to_numpy()
    is_500 = df.amt_exact_500.eq(1).to_numpy()

    snd_ref_ev = Events(sender, t, mask=is_ref)
    pair_ref_ev = Events(pair, t, mask=is_ref)
    f["snd_refund_cnt_7d"] = snd_ref_ev.count(qs, qt, -7 * DAY, 7 * DAY)
    f["snd_refund_cnt_30d"] = snd_ref_ev.count(qs, qt, -15 * DAY, 15 * DAY)
    f["snd_refund_cnt_total"] = pd.Series(is_ref.astype(np.int32)).groupby(sender).transform("sum")[query].to_numpy(dtype=np.int32)
    f["snd_refund_share_30d"] = _safe_div(f["snd_refund_cnt_30d"], f["sender_cnt_15d"]).astype(np.float32)
    f["pair_refund_cnt_30d"] = pair_ref_ev.count(qp, qt, -15 * DAY, 15 * DAY)

    f["snd_collect_cnt_7d"] = Events(sender, t, mask=is_col).count(qs, qt, -7 * DAY, 7 * DAY)
    f["rcv_collect_cnt_24h"] = Events(receiver, t, mask=is_col).count(qr, qt, -DAY, DAY)
    f["rcv_collect_cnt_7d"] = Events(receiver, t, mask=is_col).count(qr, qt, -7 * DAY, 7 * DAY)
    f["rcv_collect_share_7d"] = _safe_div(f["rcv_collect_cnt_7d"], f["receiver_cnt_7d"]).astype(np.float32)
    f["rcv_just_under_cnt_7d"] = Events(receiver, t, mask=is_just_under).count(qr, qt, -7 * DAY, 7 * DAY)
    f["rcv_just_under_share_7d"] = _safe_div(f["rcv_just_under_cnt_7d"], f["receiver_cnt_7d"]).astype(np.float32)

    f["rcv_qr_mismatch_cnt_7d"] = Events(receiver, t, mask=is_qr_mm).count(qr, qt, -7 * DAY, 7 * DAY)
    f["rcv_qr_p2p_cnt_7d"] = Events(receiver, t, mask=is_qr & p2p).count(qr, qt, -7 * DAY, 7 * DAY)

    f["rcv_round500_cnt_7d"] = Events(receiver, t, mask=is_r500).count(qr, qt, -7 * DAY, 7 * DAY)
    f["rcv_round500_share_7d"] = _safe_div(f["rcv_round500_cnt_7d"], f["receiver_cnt_7d"]).astype(np.float32)
    f["pair_round500_cnt_30d"] = Events(pair, t, mask=is_r500).count(qp, qt, -15 * DAY, 15 * DAY)
    f["pair_round500_cnt_total"] = pd.Series(is_r500.astype(np.int32)).groupby(pair).transform("sum")[query].to_numpy(dtype=np.int32)
    f["pair_round1000_cnt_total"] = pd.Series(is_r1000.astype(np.int32)).groupby(pair).transform("sum")[query].to_numpy(dtype=np.int32)

    f["rcv_cnt_total"] = np.bincount(receiver)[qr].astype(np.int32)
    f["pair_share_of_rcv"] = _safe_div(f["pair_cnt_total"], f["rcv_cnt_total"]).astype(np.float32)
    f["pair_share_of_rcv_7d"] = _safe_div(f["pair_cnt_7d"], f["receiver_cnt_7d"]).astype(np.float32)

    f["evening_pair_cnt_7d"] = Events(pair, t, mask=is_eve).count(qp, qt, -7 * DAY, 7 * DAY)
    f["evening_pair_cnt_30d"] = Events(pair, t, mask=is_eve).count(qp, qt, -15 * DAY, 15 * DAY)
    f["pair_evening_share_30d"] = _safe_div(f["evening_pair_cnt_30d"], f["pair_cnt_30d"]).astype(np.float32)
    f["snd_lending_cnt_30d"] = Events(sender, t, mask=is_lending).count(qs, qt, -15 * DAY, 15 * DAY)
    f["snd_evening_cnt_7d"] = Events(sender, t, mask=is_eve).count(qs, qt, -7 * DAY, 7 * DAY)

    # 5. Drain & Burst Quality features (SIM swap, ATO, Mule, Velocity Bot)
    f["sender_drain_ratio_1h"] = _safe_div(f["sender_amount_1h"], df.loc[query, "balance_before"].to_numpy()).astype(np.float32)
    f["sender_drain_ratio_24h"] = _safe_div(f["sender_amount_24h"], df.loc[query, "balance_before"].to_numpy()).astype(np.float32)
    f["sender_new_payee_cnt_24h"] = Events(sender, t, mask=is_new_payee).count(qs, qt, -DAY, DAY)
    f["sender_new_payee_share_24h"] = _safe_div(f["sender_new_payee_cnt_24h"], f["sender_cnt_24h"]).astype(np.float32)
    f["sender_p2p_cnt_24h"] = Events(sender, t, mask=p2p).count(qs, qt, -DAY, DAY)
    f["sender_int_amt_cnt_24h"] = Events(sender, t, mask=is_int).count(qs, qt, -DAY, DAY)
    f["sender_int_amt_share_24h"] = _safe_div(f["sender_int_amt_cnt_24h"], f["sender_cnt_24h"]).astype(np.float32)
    f["sender_500_cnt_24h"] = Events(sender, t, mask=is_500).count(qs, qt, -DAY, DAY)
    f["sender_night_cnt_24h"] = Events(sender, t, mask=is_night).count(qs, qt, -DAY, DAY)
    f["sender_pin_retry_sum_24h"] = Events(sender, t, df.pin_retries.to_numpy(dtype=np.float64)).sum(qs, qt, -DAY, DAY).astype(np.float32)

    f["rcv_int_amt_cnt_24h"] = Events(receiver, t, mask=is_int).count(qr, qt, -DAY, DAY)
    f["rcv_int_amt_share_24h"] = _safe_div(f["rcv_int_amt_cnt_24h"], f["receiver_cnt_24h"]).astype(np.float32)
    f["rcv_new_payee_cnt_24h"] = Events(receiver, t, mask=is_new_payee).count(qr, qt, -DAY, DAY)
    f["rcv_new_payee_share_24h"] = _safe_div(f["rcv_new_payee_cnt_24h"], f["receiver_cnt_24h"]).astype(np.float32)

    f["device_nonhome_cnt_1h"] = Events(device, t, mask=is_nonhome).count(qd, qt, -HOUR, HOUR)
    f["device_nonhome_cnt_24h"] = Events(device, t, mask=is_nonhome).count(qd, qt, -DAY, DAY)
    f["device_int_amt_cnt_1h"] = Events(device, t, mask=is_int).count(qd, qt, -HOUR, HOUR)
    f["device_int_amt_share_1h"] = _safe_div(f["device_int_amt_cnt_1h"], f["device_cnt_1h"]).astype(np.float32)

    # 6. Daily & total entity cardinality features
    day = df.timestamp.dt.floor("D")
    f["receiver_unique_senders_day"] = df.groupby(["receiver_id", day], observed=True).sender_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["receiver_new_payee_share_day"] = df.groupby(["receiver_id", day], observed=True).new_payee.transform("mean")[query].to_numpy(dtype=np.float32)
    f["receiver_unique_senders_total"] = df.groupby("receiver_id", observed=True).sender_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["device_unique_accounts_total"] = df.groupby("device_id", observed=True).sender_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["device_unique_accounts_day"] = df.groupby(["device_id", day], observed=True).sender_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["device_unique_receivers_day"] = df.groupby(["device_id", day], observed=True).receiver_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["sender_unique_receivers_day"] = df.groupby(["sender_id", day], observed=True).receiver_id.transform("nunique")[query].to_numpy(dtype=np.float32)
    f["acct_uniq_dev_total"] = df.groupby("sender_id", observed=True).device_id.transform("nunique")[query].to_numpy(dtype=np.float32)

    # 7. Explicit Signature (SIG) & Refined Rule (RULES) binary flags + counts
    qdf = df.loc[query].reset_index(drop=True)
    q_int = qdf.amt_is_integer.eq(1).to_numpy()
    q_p2p = qdf.receiver_type.eq("P2P").to_numpy()
    q_new = qdf.new_payee.eq(1).to_numpy()
    q_fwd = (f["forward_amount_ratio_30m"] >= 0.8) & (f["forward_amount_ratio_30m"] <= 0.9999)
    q_prev_in = (f["previous_in_amount_ratio_30m"] >= 0.8) & (f["previous_in_amount_ratio_30m"] <= 0.9999)

    f["sig_mule"] = (q_p2p & (f["receiver_unique_senders_day"] >= 3)).astype(np.int8)
    f["sig_layering"] = (q_fwd | q_prev_in).astype(np.int8)
    f["sig_velocity_bot"] = ((f["device_cnt_5m"] >= 3) | qdf.login_to_txn_sec.le(5).to_numpy()).astype(np.int8)
    f["sig_sim_swap"] = qdf.sim_age_hours.lt(168).to_numpy().astype(np.int8)
    f["sig_ato"] = qdf.device_age_days.lt(7).to_numpy().astype(np.int8)
    f["sig_phishing"] = (qdf.txn_type.eq("COLLECT").to_numpy() & (qdf.amt_just_under_1000.eq(1).to_numpy() | qdf.amt_ends_99.eq(1).to_numpy())).astype(np.int8)
    f["sig_qr_swap"] = (qdf.qr_vpa_mismatch.eq(1).to_numpy() | (qdf.txn_type.eq("QR").to_numpy() & q_p2p)).astype(np.int8)
    f["sig_refund_abuse"] = (qdf.txn_type.eq("REFUND").to_numpy() & (f["snd_refund_cnt_30d"] >= 3)).astype(np.int8)
    f["sig_fake_cashback"] = ((f["reverse_small_before_24h"] > 0) | (qdf.amount.le(10).to_numpy() & (f["reverse_big_after_24h"] > 0))).astype(np.int8)
    f["sig_loan_extortion"] = (qdf.merchant_category.eq("lending").to_numpy() | (qdf.is_late_evening.eq(1).to_numpy() & (f["pair_cnt_7d"] >= 2))).astype(np.int8)
    f["sig_merchant_collusion"] = ((~q_p2p) & qdf.amt_round_500.eq(1).to_numpy() & (f["pair_round500_cnt_total"] >= 2)).astype(np.int8)

    sig_cols = [k for k in f if k.startswith("sig_")]
    f["sig_hit_count"] = sum(f[k] for k in sig_cols).astype(np.int8)

    f["rule_refund_abuse"] = f["sig_refund_abuse"]
    f["rule_layering"] = (q_int & q_p2p & (q_fwd | q_prev_in) & qdf.amount.ge(5000).to_numpy()).astype(np.int8)
    f["rule_fake_cashback"] = (q_int & (f["sig_fake_cashback"] == 1)).astype(np.int8)
    f["rule_merchant_collusion"] = f["sig_merchant_collusion"]
    f["rule_mule"] = (q_int & q_p2p & (((f["receiver_unique_senders_day"] >= 3) & q_new) | q_fwd)).astype(np.int8)
    f["rule_phishing"] = (q_int & (f["sig_phishing"] == 1)).astype(np.int8)
    f["rule_qr_swap"] = f["sig_qr_swap"]
    f["rule_velocity_bot"] = (q_int & (~q_p2p) & (f["sig_velocity_bot"] == 1) & qdf.home_device.eq(0).to_numpy()).astype(np.int8)
    f["rule_sim_swap"] = (q_int & q_p2p & qdf.sim_age_hours.lt(168).to_numpy() & q_new).astype(np.int8)
    f["rule_ato"] = (q_int & q_p2p & qdf.device_age_days.lt(7).to_numpy() & q_new).astype(np.int8)
    f["rule_loan_extortion"] = (q_int & qdf.merchant_category.eq("lending").to_numpy() & q_new).astype(np.int8)

    rule_cols = [k for k in f if k.startswith("rule_")]
    f["rule_hit_count"] = sum(f[k] for k in rule_cols).astype(np.int8)
    return pd.DataFrame(f)


def add_relational_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Compute label-free relational windows across the full combined transaction table
    features = _relational_pass(frame, set(frame.txn_id))
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
        ("hist_receiver_collusion", "receiver_id", "merchant_collusion"),
        ("hist_receiver_cashback", "receiver_id", "fake_cashback"),
        ("hist_receiver_loan", "receiver_id", "loan_app_extortion"),
        ("hist_sender_anyfraud", "sender_id", None),
        ("hist_sender_refund", "sender_id", "refund_abuse"),
        ("hist_sender_sim", "sender_id", "sim_swap"),
        ("hist_sender_ato", "sender_id", "account_takeover"),
        ("hist_sender_loan", "sender_id", "loan_app_extortion"),
        ("hist_sender_cashback", "sender_id", "fake_cashback"),
        ("hist_device_anyfraud", "device_id", None),
        ("hist_device_bot", "device_id", "velocity_bot"),
    ]
    for feature, entity, label in definitions:
        values = np.zeros(len(df), dtype=np.float32)
        for month in targets:
            query = df.month.eq(month)
            if not query.any():
                continue
            historical = train.month.lt(month)
            positive = train.label.ne("legit") if label is None else train.label.eq(label)
            counts = train.loc[historical & positive, entity].value_counts()
            values[query] = df.loc[query, entity].map(counts).fillna(0).clip(upper=20).to_numpy(dtype=np.float32)
        df[feature] = values

    # Prior smoothed fraud rates for receiver, sender, device
    for entity, rate_col in (("receiver_id", "hist_receiver_fraud_rate"),
                             ("sender_id", "hist_sender_fraud_rate"),
                             ("device_id", "hist_device_fraud_rate")):
        rates = np.zeros(len(df), dtype=np.float32)
        for month in targets:
            query = df.month.eq(month)
            if not query.any():
                continue
            hist_rows = train[train.month.lt(month)]
            total_cnt = hist_rows[entity].value_counts()
            fraud_cnt = hist_rows.loc[hist_rows.label.ne("legit"), entity].value_counts()
            q_ent = df.loc[query, entity]
            t_c = q_ent.map(total_cnt).fillna(0).to_numpy(dtype=np.float32)
            f_c = q_ent.map(fraud_cnt).fillna(0).to_numpy(dtype=np.float32)
            rates[query] = np.where(t_c > 0, (f_c + 0.1) / (t_c + 3.0), 0.0)
        df[rate_col] = rates

    df["hist_phishing_collect_interaction"] = ((df.hist_receiver_phishing > 0) &
                                                df.cat_txn_type.eq(_code_for(df, "txn_type", "COLLECT"))).astype(np.int8)
    df["hist_refund_claim_interaction"] = ((df.hist_sender_refund > 0) &
                                            df.cat_txn_type.eq(_code_for(df, "txn_type", "REFUND"))).astype(np.int8)
    return df


def _code_for(df: pd.DataFrame, original: str, value: str) -> int:
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
