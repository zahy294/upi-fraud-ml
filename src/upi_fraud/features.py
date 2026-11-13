"""The measured 61-feature baseline, independent of all target labels.

Counts include the query payment. Windows are closed on both ends. Flow
matching uses the next/previous strictly different timestamp, with loader
order breaking equal-time candidate ties, as in the original EDA.
"""
import numpy as np
import pandas as pd

from .constants import CATEGORICAL, FEATURES, META, RELATIONAL, ROW, T0
from .data import align, require_unique

DAY = 86400
SCALE, OFFSET = 2**26, 2**25
RAW_COLUMNS = (
    "txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "device_id",
    "amount", "txn_type", "merchant_category", "device_city", "sim_age_hours",
    "device_age_days", "sender_account_age_days", "new_payee", "pin_retries",
    "login_to_txn_sec", "balance_before", "app", "qr_vpa_mismatch", "split", "_source_order",
)


class Events:
    """Packed group/time index with explicit span and group-boundary checks."""

    def __init__(self, group, times, values=None, mask=None):
        group, times = np.asarray(group, dtype="int64"), np.asarray(times, dtype="int64")
        values = None if values is None else np.asarray(values)
        if mask is not None:
            group, times = group[mask], times[mask]
            values = None if values is None else values[mask]
        if len(times) and (times.min() < 0 or times.max() >= OFFSET - 16 * DAY):
            raise ValueError("Timestamps outside safe packed-index span")
        if len(group) and (group.min() < 0 or group.max() > (np.iinfo(np.int64).max - SCALE) // SCALE):
            raise ValueError("Event group outside safe packed-index span")
        order = np.lexsort((times, group))
        self.group = group[order]
        self.keys = self.group * SCALE + times[order] + OFFSET
        self.values = None if values is None else values[order]

    def count(self, group, times, lo, hi):
        if lo > hi or min(lo, -hi) < -16 * DAY or max(lo, hi) > 16 * DAY:
            raise ValueError("Unsupported event window")
        group = np.asarray(group, dtype="int64")
        query = group * SCALE + np.asarray(times) + OFFSET
        n = np.searchsorted(self.keys, query + hi, "right") - np.searchsorted(self.keys, query + lo, "left")
        return np.where(group >= 0, n, 0)

    def adjacent_value(self, group, times, within, forward):
        group = np.asarray(group, dtype="int64")
        if self.values is None:
            raise ValueError("Adjacent-value lookup requires values")
        if not len(self.keys):
            return np.full(len(group), np.nan)
        query = group * SCALE + np.asarray(times) + OFFSET
        idx = np.searchsorted(self.keys, query, "right" if forward else "left")
        if not forward:
            idx -= 1
        safe = np.clip(idx, 0, len(self.keys) - 1)
        delta = self.keys[safe] - query
        good = (idx >= 0) & (idx < len(self.keys)) & (group >= 0) & (self.group[safe] == group)
        good &= ((delta > 0) & (delta <= within)) if forward else ((delta < 0) & (delta >= -within))
        return np.where(good, self.values[safe], np.nan)


def numeric_id(series, prefix):
    if not series.str.startswith(prefix).all():
        raise ValueError(f"Expected {prefix} ID prefix")
    values = series.str[2:].astype("int64").to_numpy()
    if (values < 0).any() or (values >= 1_000_000).any():
        raise ValueError("ID suffix outside validated namespace")
    return values


def build_features(context, accounts, merchants, query_ids=None, category_schema=None):
    require_unique(context, "txn_id")
    # Targets, app version and IP risk are never copied into the feature layer.
    df = context.loc[:, RAW_COLUMNS].copy()
    df = df.sort_values(["timestamp", "_source_order"], kind="stable").reset_index(drop=True)
    if not len(df):
        raise ValueError("Empty feature context")
    ts = df.timestamp
    t = ((ts - pd.Timestamp(T0)).dt.total_seconds()).to_numpy(dtype="int64")
    amount = df.amount.to_numpy()
    df["hour"], df["weekday"] = ts.dt.hour, ts.dt.day_of_week
    df["amt_to_balance"] = df.amount / df.balance_before.replace(0, np.nan)
    df["amt_is_integer"] = (df.amount == np.floor(df.amount)).astype("int8")
    for n in [100, 500, 1000]:
        df[f"amt_round_{n}"] = ((df.amount % n) == 0).astype("int8")
    gap = np.ceil(df.amount / 1000) * 1000 - df.amount
    df["amt_just_under_1000s"] = ((gap > 0) & (gap <= 50) & (df.amount >= 900)).astype("int8")
    df["amt_ends_99"] = ((np.floor(df.amount) % 100) == 99).astype("int8")
    df["amt_tiny_le10"] = (df.amount <= 10).astype("int8")
    require_unique(accounts, "account_id")
    require_unique(merchants, "merchant_id")
    ac, me = accounts.set_index("account_id"), merchants.set_index("merchant_id")
    p2p = (df.receiver_type == "P2P").to_numpy()
    df["away_from_home"] = (df.device_city != df.sender_id.map(ac.home_city)).astype("int8")
    df["home_device"] = (df.device_id == "DV0" + df.sender_id.str[2:]).astype("int8")
    df["rcv_acct_age_days"] = (ts - pd.to_datetime(df.receiver_id.map(ac.opened_on))).dt.days.where(p2p)
    df["mer_age_days"] = (ts - pd.to_datetime(df.receiver_id.map(me.onboarded_on))).dt.days.where(~p2p)

    sender = numeric_id(df.sender_id, "AC")
    receiver_number = df.receiver_id.str[2:].astype("int64").to_numpy()
    if (receiver_number < 0).any() or (receiver_number >= 1_000_000).any():
        raise ValueError("Recipient outside validated namespace")
    receiver = np.where(p2p, receiver_number, receiver_number + 1_000_000)
    receiver_account = np.where(p2p, receiver_number, -1)
    device = df.device_id.str[2:].astype("int64").to_numpy()
    day = t // DAY
    outgoing, incoming, devices = Events(sender, t, amount), Events(receiver, t, amount), Events(device, t)
    f = {}
    for name, window in [("1h", 3600), ("24h", DAY)]:
        f[f"rcv_cnt_{name}"] = incoming.count(receiver, t, -window, window)
    f["rcv_cnt_total"] = df.groupby(receiver).txn_id.transform("size").to_numpy()
    f["rcv_uniq_snd_day"] = df.groupby([receiver, day]).sender_id.transform("nunique").to_numpy()
    f["rcv_uniq_snd_total"] = df.groupby(receiver).sender_id.transform("nunique").to_numpy()
    f["rcv_newpayee_share_day"] = df.groupby([receiver, day]).new_payee.transform("mean").to_numpy()
    f["rcv_out_30m"] = outgoing.count(receiver_account, t, 1, 1800)
    f["rcv_out_2h"] = outgoing.count(receiver_account, t, 1, 7200)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["fwd_amt_ratio_30m"] = outgoing.adjacent_value(receiver_account, t, 1800, True) / amount
        f["prev_in_amt_ratio_30m"] = amount / incoming.adjacent_value(sender, t, 1800, False)
    f["snd_in_before_30m"] = incoming.count(sender, t, -1800, -1)
    f["snd_out_cnt_10m"] = outgoing.count(sender, t, -600, 600)
    f["snd_out_cnt_24h"] = outgoing.count(sender, t, -DAY, DAY)
    f["snd_uniq_rcv_day"] = df.groupby([sender, day]).receiver_id.transform("nunique").to_numpy()
    f["snd_in_cnt_24h"] = incoming.count(sender, t, -DAY, DAY)
    f["dev_cnt_5m"] = devices.count(device, t, -300, 300)
    f["dev_cnt_1h"] = devices.count(device, t, -3600, 3600)
    f["dev_uniq_acct_total"] = df.groupby(device).sender_id.transform("nunique").to_numpy()
    f["dev_uniq_acct_day"] = df.groupby([device, day]).sender_id.transform("nunique").to_numpy()
    f["acct_uniq_dev_total"] = df.groupby(sender).device_id.transform("nunique").to_numpy()
    pair = sender * 2_000_000 + receiver
    pair_code, unique_pair = pd.factorize(pair)
    pairs = Events(pair_code, t, amount)
    f["pair_cnt_total"] = np.bincount(pair_code)[pair_code]
    f["pair_cnt_7d"] = pairs.count(pair_code, t, -7 * DAY, 7 * DAY)
    f["pair_cnt_24h"] = pairs.count(pair_code, t, -DAY, DAY)
    reverse = np.where(p2p, receiver_account * 2_000_000 + sender, -1)
    reverse_code = pd.Index(unique_pair).get_indexer(reverse)
    f["rev_pair_cnt_total"] = np.where(reverse_code >= 0, np.bincount(pair_code)[np.maximum(reverse_code, 0)], 0)
    f["rev_small_before_24h"] = Events(pair_code, t, amount, amount <= 10).count(reverse_code, t, -DAY, -1)
    f["rev_big_after_24h"] = Events(pair_code, t, amount, amount >= 100).count(reverse_code, t, 1, DAY)
    refund = (df.txn_type == "REFUND").to_numpy()
    f["snd_refund_cnt_30d"] = Events(sender, t, mask=refund).count(sender, t, -15 * DAY, 15 * DAY)
    f["snd_refund_cnt_total"] = pd.Series(refund).groupby(sender).transform("sum").to_numpy()
    f["snd_collect_cnt_7d"] = Events(sender, t, mask=(df.txn_type == "COLLECT").to_numpy()).count(sender, t, -7 * DAY, 7 * DAY)
    rounded = (df.amt_round_500 == 1).to_numpy()
    f["rcv_round500_cnt_7d"] = Events(receiver, t, mask=rounded).count(receiver, t, -7 * DAY, 7 * DAY)
    f["pair_round500_cnt_total"] = pd.Series(rounded.astype(int)).groupby(pair_code).transform("sum").to_numpy()
    f["pair_share_of_rcv"] = f["pair_cnt_total"] / f["rcv_cnt_total"]
    f["evening_pair_cnt_7d"] = Events(pair_code, t, mask=df.hour.between(20, 23).to_numpy()).count(pair_code, t, -7 * DAY, 7 * DAY)
    relational = pd.DataFrame(f).loc[:, RELATIONAL]
    # Keyed one-to-one join protects against cache/reordering mistakes.
    relational.insert(0, "txn_id", df.txn_id.to_numpy())
    numeric = df.loc[:, ["txn_id", *ROW]].merge(relational, on="txn_id", validate="one_to_one", sort=False)
    rules_input = pd.concat([df[["receiver_type", "txn_type", "merchant_category"]], numeric.drop(columns="txn_id")], axis=1)
    rules = rule_predictions(rules_input)
    frame = df.loc[:, META].merge(numeric, on="txn_id", validate="one_to_one", sort=False)
    frame["rule_code"] = rules
    schema = {} if category_schema is None else category_schema
    for c in CATEGORICAL:
        values = df[c].fillna("NA")
        if category_schema is None:
            schema[c] = sorted(values.unique().tolist())
        frame[c] = pd.Index(schema[c]).get_indexer(values).astype("int32")
    for c in ROW + RELATIONAL:
        frame[c] = frame[c].astype("float32").replace([np.inf, -np.inf], np.nan)
    if len(frame) != len(context) or len(FEATURES) != 61:
        raise ValueError("Unexpected baseline feature shape")
    return (align(frame, query_ids) if query_ids is not None else frame), schema


def rule_predictions(df):
    """Exact priority and conditions of EDA RULES; first condition wins."""
    integer, p2p = df.amt_is_integer == 1, df.receiver_type == "P2P"
    rules = [
        (8, (df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
        (3, integer & p2p & (df.fwd_amt_ratio_30m.between(.8, .9999) | df.prev_in_amt_ratio_30m.between(.8, .9999)) & (df.amount >= 5000)),
        (9, integer & ((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0)))),
        (11, ~p2p & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
        (1, integer & p2p & (df.rcv_uniq_snd_day >= 3) & (df.new_payee == 1)),
        (1, integer & p2p & df.fwd_amt_ratio_30m.between(.8, .9999)),
        (6, integer & (df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
        (7, (df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
        (2, integer & ~p2p & ((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)) & (df.home_device == 0)),
        (4, integer & p2p & (df.sim_age_hours < 168) & (df.new_payee == 1)),
        (5, integer & p2p & (df.device_age_days < 7) & (df.new_payee == 1)),
        (10, integer & (df.merchant_category == "lending") & (df.new_payee == 1)),
    ]
    pred = np.zeros(len(df), dtype="int32")
    for c, mask in reversed(rules):
        pred[mask.fillna(False).to_numpy(dtype=bool)] = c
    return pred
