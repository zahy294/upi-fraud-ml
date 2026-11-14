"""Red-team test of the stealth / ordinary-fraud i.i.d. claim.

Claim under test:
  In data/raw/train.csv, fraud-labelled rows that look ordinary are an i.i.d. sample
  of ordinary rows (~2.3% of them), so nothing available in test.csv (plus accounts.csv
  and merchants.csv, excluding app_version and ip_risk_score) can pick them out.

Definition of ordinary rows:
  Train rows (in the order of data/processed/base_v0.parquet where split == 'train')
  whose out-of-fold prediction from models/s2_a/oof.npy is legit (class 0) after
  multiplying by class weights from colab/bench.py fit_weights_fast against
  models/oof_raw/y_clean.npy, excluding device numbers 423001-428454.

Target rule criterion:
  Built only from columns in test.csv, accounts.csv, and merchants.csv (never
  app_version or ip_risk_score), selecting >= 300 ordinary rows in each of July,
  August, and September with a fraud-label share >= 10% in every month.

Usage:
  python -I scripts/explore/50_redteam_stealth.py
"""
import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_k] = "4"

import sys

sys.dont_write_bytecode = True

import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import roc_auc_score

pa.set_cpu_count(4)
pa.set_io_thread_count(4)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "colab"))
sys.path.insert(0, str(ROOT / "src/upi"))
from bench import fit_weights_fast  # noqa: E402
from common import LABELS  # noqa: E402

MONTHS = ("2026-07", "2026-08", "2026-09")
MIN_N = 300
TARGET_SHARE = 0.10


def min_dt_same_group(g, tt):
    """Minimum time gap (seconds) to another transaction in the same group g."""
    o = np.lexsort((tt, g))
    go, to = g[o], tt[o]
    prev_ok = np.r_[False, go[1:] == go[:-1]]
    next_ok = np.r_[go[:-1] == go[1:], False]
    dt_p = np.where(prev_ok, np.r_[0, to[1:] - to[:-1]], np.inf)
    dt_n = np.where(next_ok, np.r_[to[1:] - to[:-1], 0], np.inf)
    dt_min = np.minimum(dt_p, dt_n)
    out = np.empty(len(tt), dtype=np.float32)
    out[o] = np.where(np.isinf(dt_min), np.nan, dt_min)
    return out


def window_count(group, tt, window_sec):
    """Count of events in the same group within [-window_sec, +window_sec]."""
    scale = np.int64(2**26)
    off = np.int64(2**25)
    o = np.lexsort((tt, group))
    K = group[o].astype(np.int64) * scale + tt[o] + off
    b = group.astype(np.int64) * scale + tt + off
    c = np.searchsorted(K, b + window_sec, "right") - np.searchsorted(K, b - window_sec, "left")
    return np.where(group >= 0, c, 0).astype(np.float32)


def make_qbin(arr, name, q=6):
    """Quantile-bin a numeric array into integer codes 0..K-1 and human-readable bin labels."""
    s = pd.Series(arr)
    try:
        b = pd.qcut(s, q=q, duplicates="drop")
        codes, uniques = pd.factorize(b, sort=True)
        if (codes == -1).any():
            codes = np.where(codes == -1, len(uniques), codes)
            labels = [f"{name} in {u}" for u in uniques] + [f"{name}=NA"]
        else:
            labels = [f"{name} in {u}" for u in uniques]
        return codes.astype(np.int32), labels
    except Exception:
        codes, uniques = pd.factorize(s.fillna(-999), sort=True)
        return codes.astype(np.int32), [f"{name}={u}" for u in uniques]


def make_count_bin(arr, name):
    """Bin a count column into domain-meaningful buckets."""
    v = np.asarray(arr, dtype=np.float64)
    # Buckets: <=0 (or NA), 1, 2, 3, 4-5, 6-10, 11-25, >25
    edges = [-np.inf, 0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 25.5, np.inf]
    lbl_names = ["0", "1", "2", "3", "4..5", "6..10", "11..25", ">=26"]
    idx = np.digitize(np.nan_to_num(v, nan=-1.0), edges) - 1
    idx = np.clip(idx, 0, len(lbl_names) - 1)
    codes, uniques = pd.factorize(idx, sort=True)
    labels = [f"{name}={lbl_names[u]}" for u in uniques]
    return codes.astype(np.int32), labels


def main():
    t0 = time.time()
    print("=== 1. Loading base_v0, s2_a OOF probabilities, and y_clean ===")
    base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet")
    tr_mask = (base.split == "train").values

    P_oof = np.load(ROOT / "models/s2_a/oof.npy").astype(np.float64)
    y_clean = np.load(ROOT / "models/oof_raw/y_clean.npy").astype(np.int64)
    assert len(P_oof) == tr_mask.sum() == len(y_clean)

    w = fit_weights_fast(P_oof, y_clean)
    print("Fitted class weights (fit_weights_fast on s2_a vs y_clean):")
    print(" ", {LABELS[i]: round(float(w[i]), 4) for i in range(len(LABELS))})

    pred_oof = (P_oof * w).argmax(axis=1)
    dev_num_all = base.device_id.str[2:].astype(np.int64).values
    dev_num_tr = dev_num_all[tr_mask]
    in_band_dev = (dev_num_tr >= 423001) & (dev_num_tr <= 428454)

    ordinary_tr = (pred_oof == 0) & (~in_band_dev)

    y_raw_str = base.label.values[tr_mask]
    y_raw_cls = pd.Series(y_raw_str).map({lab: i for i, lab in enumerate(LABELS)}).values.astype(np.int64)
    y_raw_bin = (y_raw_cls > 0).astype(np.int32)
    y_clean_bin = (y_clean > 0).astype(np.int32)

    raw_month_tr = base.month.values[tr_mask].copy()
    bench_month_tr = raw_month_tr.copy()
    bench_month_tr[bench_month_tr > "2026-09"] = "2026-09"

    print(f"Total train rows: {tr_mask.sum():,}")
    print(f"  pred_oof == 0 (legit): {(pred_oof == 0).sum():,}")
    print(f"  device in 423001-428454: {in_band_dev.sum():,} (among pred==0: {((pred_oof == 0) & in_band_dev).sum():,})")
    print(f"  Ordinary train rows: {ordinary_tr.sum():,}")
    print(f"  Ordinary rows with raw month > 2026-09: {(ordinary_tr & (raw_month_tr > '2026-09')).sum():,}")
    print(f"  Ordinary fraud rows (raw label != legit): {y_raw_bin[ordinary_tr].sum():,} ({y_raw_bin[ordinary_tr].mean()*100:.4f}%)")
    print(f"  Ordinary fraud rows (y_clean > 0):        {y_clean_bin[ordinary_tr].sum():,} ({y_clean_bin[ordinary_tr].mean()*100:.4f}%)")
    print(f"  Disagreement between raw fraud and y_clean>0 on ordinary rows: {(y_raw_bin[ordinary_tr] != y_clean_bin[ordinary_tr]).sum():,}")

    for m in MONTHS:
        m_mask = ordinary_tr & (bench_month_tr == m)
        n_m = int(m_mask.sum())
        f_m = int(y_raw_bin[m_mask].sum())
        print(f"  Month {m}: ordinary={n_m:,}, fraud={f_m:,}, share={f_m/n_m*100:.4f}%")

    ord_cls_counts = pd.Series(y_raw_cls[ordinary_tr]).map(dict(enumerate(LABELS))).value_counts()
    print("\nOrdinary rows by raw label:")
    for k, v in ord_cls_counts.items():
        print(f"  {k:<20s}: {v:8,d} ({v/ordinary_tr.sum()*100:.4f}%)")

    # === 2. Build clean feature table from test.csv / accounts.csv / merchants.csv columns ===
    print("\n=== 2. Building test-available feature & cross-row count table (no app_version, no ip_risk_score) ===")
    ac = pd.read_parquet(ROOT / "data/interim/accounts.parquet").set_index("account_id")
    me = pd.read_parquet(ROOT / "data/interim/merchants.parquet").set_index("merchant_id")

    forbidden = {"app_version", "app_major", "ip_risk_score", "label", "split"}

    ts = pd.to_datetime(base.timestamp)
    t_sec = base.t.values.astype(np.int64)
    is_p2p = base.receiver_type.values == "P2P"

    snd_num = base.sender_id.str[2:].astype(np.int64).values
    rnum = base.receiver_id.str[2:].astype(np.int64).values
    rcv_entity = np.where(is_p2p, rnum, rnum + 1_000_000)
    dev_num = dev_num_all
    pair_id = snd_num * 2_000_000 + rcv_entity

    rcv_bank = base.receiver_id.map(ac.bank_code).where(is_p2p)
    rcv_os = base.receiver_id.map(ac.os).where(is_p2p)
    snd_open_days = (pd.to_datetime(base.sender_id.map(ac.opened_on)) - pd.Timestamp("2020-01-01")).dt.days.values
    rcv_open_days = (pd.to_datetime(base.receiver_id.map(ac.opened_on)) - pd.Timestamp("2020-01-01")).dt.days.where(is_p2p).values
    mer_onb_days = (pd.to_datetime(base.receiver_id.map(me.onboarded_on)) - pd.Timestamp("2020-01-01")).dt.days.where(~is_p2p).values

    pcode, _ = pd.factorize(pair_id)
    mcode, _ = pd.factorize(base.month.values)
    rcv_c, _ = pd.factorize(rcv_entity)

    cols = {}
    # Raw & row-level features from test.csv, accounts.csv, merchants.csv
    cols["amount"] = base.amount.values.astype(np.float32)
    cols["balance_before"] = base.balance_before.values.astype(np.float32)
    cols["amt_to_balance"] = base.amt_to_balance.values.astype(np.float32)
    cols["bal_after"] = (base.balance_before.values - base.amount.values).astype(np.float32)
    cols["amt_is_integer"] = base.amt_is_integer.values.astype(np.int8)
    cols["amt_round_100"] = base.amt_round_100.values.astype(np.int8)
    cols["amt_round_500"] = base.amt_round_500.values.astype(np.int8)
    cols["amt_round_1000"] = base.amt_round_1000.values.astype(np.int8)
    cols["amt_just_under_1000s"] = base.amt_just_under_1000s.values.astype(np.int8)
    cols["amt_ends_99"] = base.amt_ends_99.values.astype(np.int8)
    cols["amt_tiny_le10"] = base.amt_tiny_le10.values.astype(np.int8)
    cols["amt_cents"] = np.round((base.amount.values * 100) % 100).astype(np.float32)

    cols["hour"] = base.hour.values.astype(np.int8)
    cols["weekday"] = base.weekday.values.astype(np.int8)
    cols["day_of_month"] = ts.dt.day.values.astype(np.int8)
    cols["minute"] = ts.dt.minute.values.astype(np.int8)
    cols["second"] = ts.dt.second.values.astype(np.int8)
    cols["is_night"] = base.is_night.values.astype(np.int8)
    cols["is_late_evening"] = base.is_late_evening.values.astype(np.int8)

    cols["sim_age_hours"] = base.sim_age_hours.values.astype(np.float32)
    cols["device_age_days"] = base.device_age_days.values.astype(np.float32)
    cols["sender_account_age_days"] = base.sender_account_age_days.values.astype(np.float32)
    cols["acct_age_from_opened"] = base.acct_age_from_opened.values.astype(np.float32)
    cols["acct_age_gap"] = (base.acct_age_from_opened.values - base.sender_account_age_days.values).astype(np.float32)
    cols["rcv_acct_age_days"] = base.rcv_acct_age_days.values.astype(np.float32)
    cols["mer_age_days"] = base.mer_age_days.values.astype(np.float32)
    cols["snd_open_days"] = snd_open_days.astype(np.float32)
    cols["rcv_open_days"] = rcv_open_days.astype(np.float32)
    cols["mer_onb_days"] = mer_onb_days.astype(np.float32)

    cols["new_payee"] = base.new_payee.values.astype(np.int8)
    cols["pin_retries"] = base.pin_retries.values.astype(np.int8)
    cols["login_to_txn_sec"] = base.login_to_txn_sec.values.astype(np.float32)
    cols["qr_vpa_mismatch"] = base.qr_vpa_mismatch.fillna(-1).values.astype(np.float32)
    cols["home_device"] = base.home_device.values.astype(np.int8)
    cols["away_from_home"] = base.away_from_home.values.astype(np.int8)
    cols["mer_cat_matches_txn"] = base.mer_cat_matches_txn.fillna(-1).values.astype(np.float32)
    cols["mer_city_eq_device_city"] = base.mer_city_eq_device_city.fillna(-1).values.astype(np.float32)
    cols["snd_eq_rcv_city"] = np.where(is_p2p, (base.home_city.values == base.rcv_home_city.values).astype(int), -1).astype(np.int8)
    cols["same_bank"] = np.where(is_p2p, (base.bank_code.values == rcv_bank.values).astype(int), -1).astype(np.int8)

    cols["snd_id_num"] = snd_num.astype(np.float32)
    cols["rcv_id_num"] = rnum.astype(np.float32)
    cols["dev_id_num"] = dev_num.astype(np.float32)
    cols["dev_minus_snd"] = (dev_num - snd_num).astype(np.float32)

    # All 33 relational features from base_v0
    rel_cols = [
        "rcv_cnt_1h", "rcv_cnt_24h", "rcv_cnt_total", "rcv_uniq_snd_day", "rcv_uniq_snd_total",
        "rcv_newpayee_share_day", "rcv_out_30m", "rcv_out_2h", "fwd_amt_ratio_30m",
        "snd_in_before_30m", "prev_in_amt_ratio_30m", "snd_out_cnt_10m", "snd_out_cnt_24h",
        "snd_uniq_rcv_day", "snd_in_cnt_24h", "dev_cnt_5m", "dev_cnt_1h",
        "dev_uniq_acct_total", "dev_uniq_acct_day", "acct_uniq_dev_total",
        "pair_cnt_total", "pair_cnt_7d", "pair_cnt_24h", "rev_pair_cnt_total",
        "rev_small_before_24h", "rev_big_after_24h", "snd_refund_cnt_30d", "snd_refund_cnt_total",
        "snd_collect_cnt_7d", "rcv_round500_cnt_7d", "pair_round500_cnt_total",
        "pair_share_of_rcv", "evening_pair_cnt_7d",
    ]
    for c in rel_cols:
        cols[c] = base[c].values.astype(np.float32)

    # Additional cross-row counts per sender, receiver, device, pair
    cols["snd_cnt_total"] = np.bincount(snd_num)[snd_num].astype(np.float32)
    cols["dev_cnt_total"] = np.bincount(dev_num)[dev_num].astype(np.float32)
    cols["snd_in_cnt_total"] = np.bincount(np.where(is_p2p, rnum, 0), minlength=snd_num.max() + 1)[snd_num].astype(np.float32)
    cols["snd_out_cnt_1h"] = window_count(snd_num, t_sec, 3600)
    cols["rcv_cnt_7d"] = window_count(rcv_c, t_sec, 7 * 86400)
    cols["dev_cnt_24h"] = window_count(dev_num, t_sec, 86400)

    snd_m = snd_num * 8 + mcode
    cols["snd_cnt_month"] = np.bincount(snd_m)[snd_m].astype(np.float32)
    rcv_m = rcv_c * 8 + mcode
    cols["rcv_cnt_month"] = np.bincount(rcv_m)[rcv_m].astype(np.float32)
    dev_m = dev_num * 8 + mcode
    cols["dev_cnt_month"] = np.bincount(dev_m)[dev_m].astype(np.float32)
    pair_m = pcode * 8 + mcode
    cols["pair_cnt_month"] = np.bincount(pair_m)[pair_m].astype(np.float32)

    cols["snd_uniq_rcv_total"] = base.groupby("sender_id").receiver_id.transform("nunique").values.astype(np.float32)
    cols["rcv_uniq_dev_total"] = base.groupby("receiver_id").device_id.transform("nunique").values.astype(np.float32)
    cols["dev_uniq_rcv_total"] = base.groupby("device_id").receiver_id.transform("nunique").values.astype(np.float32)

    lamt = np.log1p(cols["amount"])
    lbal = np.log1p(cols["balance_before"])
    for ename, key_arr, cnt_arr in (
        ("snd", snd_num, cols["snd_cnt_total"]),
        ("rcv", rcv_c, cols["rcv_cnt_total"]),
        ("dev", dev_num, cols["dev_cnt_total"]),
        ("pair", pcode, cols["pair_cnt_total"]),
    ):
        s_amt = np.bincount(key_arr, weights=lamt)[key_arr]
        m_amt = np.where(cnt_arr > 1, (s_amt - lamt) / np.maximum(cnt_arr - 1, 1), np.nan)
        cols[f"{ename}_lamt_diff"] = (lamt - m_amt).astype(np.float32)

        s_bal = np.bincount(key_arr, weights=lbal)[key_arr]
        m_bal = np.where(cnt_arr > 1, (s_bal - lbal) / np.maximum(cnt_arr - 1, 1), np.nan)
        cols[f"{ename}_lbal_diff"] = (lbal - m_bal).astype(np.float32)

    cols["snd_dt_nearest"] = min_dt_same_group(snd_num, t_sec)
    cols["rcv_dt_nearest"] = min_dt_same_group(rcv_c, t_sec)
    cols["dev_dt_nearest"] = min_dt_same_group(dev_num, t_sec)
    cols["pair_dt_nearest"] = min_dt_same_group(pcode, t_sec)

    flagged_tr = np.zeros(len(base), dtype=np.int32)
    flagged_tr[tr_mask] = (pred_oof > 0).astype(np.int32)
    cols["snd_flagged_cnt"] = np.bincount(snd_num, weights=flagged_tr)[snd_num].astype(np.float32)
    cols["rcv_flagged_cnt"] = np.bincount(rcv_c, weights=flagged_tr)[rcv_c].astype(np.float32)
    cols["dev_flagged_cnt"] = np.bincount(dev_num, weights=flagged_tr)[dev_num].astype(np.float32)
    cols["pair_flagged_cnt"] = np.bincount(pcode, weights=flagged_tr)[pcode].astype(np.float32)

    cat_cols = [
        "receiver_type", "txn_type", "merchant_category", "device_city", "app",
        "home_city", "kyc_level", "os", "bank_code",
        "rcv_kyc_level", "rcv_home_city", "mer_category", "mer_city",
    ]
    for c in cat_cols:
        cols[c] = base[c].astype("object").fillna("NA").astype("category")
    cols["rcv_bank_code"] = rcv_bank.astype("object").fillna("NA").astype("category")
    cols["rcv_os"] = rcv_os.astype("object").fillna("NA").astype("category")
    cat_cols.extend(["rcv_bank_code", "rcv_os"])

    df_all = pd.DataFrame(cols, index=base.index)
    assert not (set(df_all.columns) & forbidden), f"Forbidden columns present: {set(df_all.columns) & forbidden}"

    ord_idx = np.where(tr_mask)[0][ordinary_tr]
    X_ord = df_all.iloc[ord_idx].reset_index(drop=True)
    y_ord = y_raw_bin[ordinary_tr]
    m_ord = bench_month_tr[ordinary_tr]
    m_masks = [m_ord == m for m in MONTHS]
    y_m = [y_ord[mk] for mk in m_masks]

    print(f"Built X_ord: {X_ord.shape[0]:,} rows x {X_ord.shape[1]} features ({len(cat_cols)} categorical, {X_ord.shape[1] - len(cat_cols)} numeric)")

    # === 3. Cross-row count scan (per sender, receiver, device, pair) ===
    print("\n=== 3. Dedicated Cross-Row Count Scan (Sender, Receiver, Device, Pair) ===")
    cross_count_cols = [
        # Sender counts
        "snd_cnt_total", "snd_cnt_month", "snd_out_cnt_24h", "snd_out_cnt_1h", "snd_out_cnt_10m",
        "snd_in_cnt_total", "snd_in_cnt_24h", "snd_in_before_30m",
        "snd_uniq_rcv_total", "snd_uniq_rcv_day", "acct_uniq_dev_total",
        "snd_refund_cnt_30d", "snd_refund_cnt_total", "snd_collect_cnt_7d", "snd_flagged_cnt",
        # Receiver counts
        "rcv_cnt_total", "rcv_cnt_month", "rcv_cnt_7d", "rcv_cnt_24h", "rcv_cnt_1h",
        "rcv_uniq_snd_total", "rcv_uniq_snd_day", "rcv_uniq_dev_total",
        "rcv_out_30m", "rcv_out_2h", "rcv_round500_cnt_7d", "rcv_flagged_cnt",
        # Device counts
        "dev_cnt_total", "dev_cnt_month", "dev_cnt_24h", "dev_cnt_1h", "dev_cnt_5m",
        "dev_uniq_acct_total", "dev_uniq_acct_day", "dev_uniq_rcv_total", "dev_flagged_cnt",
        # Pair counts
        "pair_cnt_total", "pair_cnt_month", "pair_cnt_7d", "pair_cnt_24h",
        "rev_pair_cnt_total", "rev_small_before_24h", "rev_big_after_24h",
        "pair_round500_cnt_total", "evening_pair_cnt_7d", "pair_flagged_cnt",
    ]

    count_rule_results = []
    for c in cross_count_cols:
        v = X_ord[c].values.astype(np.float64)
        # Test exact buckets, >= thresholds, and <= thresholds
        thresholds = sorted(set(np.nanquantile(v, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)).union({0, 1, 2, 3, 4, 5, 10, 20, 50}))
        for thr in thresholds:
            for op_name, mask in ((f"{c} >= {thr:g}", v >= thr), (f"{c} <= {thr:g}", v <= thr), (f"{c} == {thr:g}", v == thr)):
                ns = [int(mask[mk].sum()) for mk in m_masks]
                if min(ns) < MIN_N:
                    continue
                fs = [int(y_m[i][mask[m_masks[i]]].sum()) for i in range(3)]
                shs = [fs[i] / ns[i] for i in range(3)]
                count_rule_results.append({
                    "rule": op_name,
                    "min_n": min(ns),
                    "min_share": min(shs),
                    "pool_share": sum(fs) / sum(ns),
                    "n_jul": ns[0], "sh_jul": shs[0],
                    "n_aug": ns[1], "sh_aug": shs[1],
                    "n_sep": ns[2], "sh_sep": shs[2],
                })
    count_df = pd.DataFrame(count_rule_results).sort_values("min_share", ascending=False).drop_duplicates(subset=["n_jul", "n_aug", "n_sep", "min_share"])
    print(f"Tested {len(count_rule_results):,} 1D cross-row count threshold/equality rules with min_n >= {MIN_N}.")
    print(f"Rules meeting min_share >= {TARGET_SHARE*100:.1f}%: {(count_df['min_share'] >= TARGET_SHARE).sum()}")
    print("Top 15 1D cross-row count rules by min monthly fraud share (with n >= 300 in every month):")
    print(count_df.head(15).to_string(index=False))

    # === 4. Construct binned columns for 1-way, 2-way, and 3-way scans ===
    print("\n=== 4. Constructing Binned Columns for 1-Way, Pairwise (2-Way), and 3-Way Interaction Scans ===")
    binned_all = {}

    for c in cat_cols:
        codes, uniques = pd.factorize(X_ord[c], sort=True)
        binned_all[c] = (codes.astype(np.int32), [f"{c}={u}" for u in uniques])

    bin_direct = [
        "amt_is_integer", "amt_round_100", "amt_round_500", "amt_round_1000",
        "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10",
        "is_night", "is_late_evening", "new_payee", "pin_retries",
        "home_device", "away_from_home", "snd_eq_rcv_city", "same_bank",
        "weekday", "qr_vpa_mismatch", "mer_cat_matches_txn", "mer_city_eq_device_city",
    ]
    for c in bin_direct:
        codes, uniques = pd.factorize(X_ord[c].values, sort=True)
        binned_all[c] = (codes.astype(np.int32), [f"{c}={u}" for u in uniques])

    cont_cols = [
        "amount", "balance_before", "amt_to_balance", "bal_after", "amt_cents",
        "hour", "day_of_month", "minute", "second",
        "sim_age_hours", "device_age_days", "sender_account_age_days", "acct_age_gap",
        "rcv_acct_age_days", "mer_age_days", "login_to_txn_sec",
        "snd_id_num", "rcv_id_num", "dev_id_num", "dev_minus_snd",
        "pair_share_of_rcv", "rcv_newpayee_share_day", "fwd_amt_ratio_30m", "prev_in_amt_ratio_30m",
        "snd_lamt_diff", "rcv_lamt_diff", "dev_lamt_diff", "pair_lamt_diff",
        "snd_lbal_diff", "rcv_lbal_diff", "dev_lbal_diff", "pair_lbal_diff",
        "snd_dt_nearest", "rcv_dt_nearest", "dev_dt_nearest", "pair_dt_nearest",
    ]
    for c in cont_cols:
        binned_all[c] = make_qbin(X_ord[c].values, c, q=6)

    for c in cross_count_cols:
        binned_all[c] = make_count_bin(X_ord[c].values, c)

    col_names = list(binned_all.keys())
    print(f"Total binned columns: {len(col_names)}")

    # Pre-split binned arrays by month for fast vectorized bincount
    b_m = {c: [binned_all[c][0][mk] for mk in m_masks] for c in col_names}
    b_card = {c: len(binned_all[c][1]) for c in col_names}
    b_lbls = {c: binned_all[c][1] for c in col_names}

    # --- 4a. 1-Way Scan ---
    cells_1way = []
    total_cells_1way = 0
    for c in col_names:
        K = b_card[c]
        total_cells_1way += K
        ns = [np.bincount(b_m[c][i], minlength=K) for i in range(3)]
        fs = [np.bincount(b_m[c][i], weights=y_m[i], minlength=K) for i in range(3)]
        min_n = np.minimum(np.minimum(ns[0], ns[1]), ns[2])
        ok = np.where(min_n >= MIN_N)[0]
        for idx in ok:
            shs = [fs[i][idx] / ns[i][idx] for i in range(3)]
            cells_1way.append({
                "order": 1,
                "cols": c,
                "cell": b_lbls[c][idx],
                "min_n": int(min_n[idx]),
                "min_share": float(min(shs)),
                "pool_share": float((fs[0][idx] + fs[1][idx] + fs[2][idx]) / (ns[0][idx] + ns[1][idx] + ns[2][idx])),
                "n_jul": int(ns[0][idx]), "f_jul": int(fs[0][idx]), "sh_jul": float(shs[0]),
                "n_aug": int(ns[1][idx]), "f_aug": int(fs[1][idx]), "sh_aug": float(shs[1]),
                "n_sep": int(ns[2][idx]), "f_sep": int(fs[2][idx]), "sh_sep": float(shs[2]),
            })
    df_1way = pd.DataFrame(cells_1way).sort_values("min_share", ascending=False)
    print(f"\n--- 4a. 1-Way Binned Scan: {len(col_names)} columns, {total_cells_1way:,} cells ({len(df_1way):,} with min_n >= {MIN_N}) ---")
    print(f"1-way cells with min_share >= {TARGET_SHARE*100:.1f}%: {(df_1way['min_share'] >= TARGET_SHARE).sum()}")
    print("Top 10 1-way cells by min monthly fraud share:")
    print(df_1way[["cell", "min_n", "min_share", "pool_share", "n_jul", "sh_jul", "n_aug", "sh_aug", "n_sep", "sh_sep"]].head(10).to_string(index=False))

    # --- 4b. 2-Way (Pairwise) Scan across all pairs of the 116 binned columns ---
    t_2w = time.time()
    cells_2way = []
    total_pairs = 0
    total_cells_2way = 0
    valid_cells_2way = 0
    for c1, c2 in combinations(col_names, 2):
        total_pairs += 1
        K1, K2 = b_card[c1], b_card[c2]
        K = K1 * K2
        total_cells_2way += K

        comb0 = b_m[c1][0] * K2 + b_m[c2][0]
        n0 = np.bincount(comb0, minlength=K)
        if n0.max() < MIN_N:
            continue
        comb1 = b_m[c1][1] * K2 + b_m[c2][1]
        n1 = np.bincount(comb1, minlength=K)
        comb2 = b_m[c1][2] * K2 + b_m[c2][2]
        n2 = np.bincount(comb2, minlength=K)

        min_n = np.minimum(np.minimum(n0, n1), n2)
        ok_mask = min_n >= MIN_N
        n_ok = int(ok_mask.sum())
        if n_ok == 0:
            continue
        valid_cells_2way += n_ok

        f0 = np.bincount(comb0, weights=y_m[0], minlength=K)
        f1 = np.bincount(comb1, weights=y_m[1], minlength=K)
        f2 = np.bincount(comb2, weights=y_m[2], minlength=K)

        ok_idx = np.where(ok_mask)[0]
        sh0 = f0[ok_idx] / n0[ok_idx]
        sh1 = f1[ok_idx] / n1[ok_idx]
        sh2 = f2[ok_idx] / n2[ok_idx]
        min_sh = np.minimum(np.minimum(sh0, sh1), sh2)

        # Keep any cell with min_sh >= 0.0255 for top-table reporting (and always keep best of each pair if >= 0.025)
        keep_sub = np.where(min_sh >= 0.0250)[0]
        for j in keep_sub:
            idx = int(ok_idx[j])
            i1, i2 = divmod(idx, K2)
            cells_2way.append({
                "order": 2,
                "cols": f"{c1} x {c2}",
                "cell": f"{b_lbls[c1][i1]} & {b_lbls[c2][i2]}",
                "min_n": int(min_n[idx]),
                "min_share": float(min_sh[j]),
                "pool_share": float((f0[idx] + f1[idx] + f2[idx]) / (n0[idx] + n1[idx] + n2[idx])),
                "n_jul": int(n0[idx]), "f_jul": int(f0[idx]), "sh_jul": float(sh0[j]),
                "n_aug": int(n1[idx]), "f_aug": int(f1[idx]), "sh_aug": float(sh1[j]),
                "n_sep": int(n2[idx]), "f_sep": int(f2[idx]), "sh_sep": float(sh2[j]),
            })

    df_2way = pd.DataFrame(cells_2way).sort_values("min_share", ascending=False)
    print(f"\n--- 4b. 2-Way Pairwise Scan: {total_pairs:,} column pairs, {total_cells_2way:,} total cells, {valid_cells_2way:,} cells with min_n >= {MIN_N} ({time.time() - t_2w:.1f}s) ---")
    print(f"2-way cells with min_share >= {TARGET_SHARE*100:.1f}%: {(df_2way['min_share'] >= TARGET_SHARE).sum()}")
    print("Top 15 2-way cells by min monthly fraud share:")
    print(df_2way[["cell", "min_n", "min_share", "pool_share", "n_jul", "sh_jul", "n_aug", "sh_aug", "n_sep", "sh_sep"]].head(15).to_string(index=False))

    # --- 4c. 3-Way Scan across core test / account / merchant / cross-row count columns ---
    t_3w = time.time()
    core_3way_cols = [
        # Raw test.csv attributes
        "receiver_type", "txn_type", "merchant_category", "device_city", "app",
        "amount", "balance_before", "amt_to_balance", "amt_is_integer", "amt_round_500", "amt_cents",
        "hour", "weekday", "day_of_month", "minute", "second",
        "sim_age_hours", "device_age_days", "sender_account_age_days",
        "new_payee", "pin_retries", "login_to_txn_sec", "qr_vpa_mismatch", "home_device", "away_from_home",
        "snd_id_num", "rcv_id_num", "dev_id_num",
        # accounts.csv & merchants.csv attributes
        "home_city", "kyc_level", "os", "bank_code",
        "rcv_kyc_level", "rcv_home_city", "rcv_bank_code", "rcv_os", "rcv_acct_age_days", "same_bank", "snd_eq_rcv_city",
        "mer_category", "mer_city", "mer_age_days", "mer_cat_matches_txn", "mer_city_eq_device_city",
        # Cross-row counts per sender, receiver, device, pair
        "snd_cnt_total", "snd_cnt_month", "snd_out_cnt_24h", "snd_in_cnt_total", "snd_uniq_rcv_total", "acct_uniq_dev_total", "snd_flagged_cnt",
        "rcv_cnt_total", "rcv_cnt_month", "rcv_cnt_24h", "rcv_uniq_snd_total", "rcv_uniq_snd_day", "rcv_flagged_cnt",
        "dev_cnt_total", "dev_cnt_month", "dev_uniq_acct_total", "dev_flagged_cnt",
        "pair_cnt_total", "pair_cnt_month", "pair_cnt_7d", "rev_pair_cnt_total", "pair_share_of_rcv", "pair_flagged_cnt",
    ]
    assert all(c in binned_all for c in core_3way_cols)
    print(f"\n--- 4c. Starting 3-Way Interaction Scan on {len(core_3way_cols)} columns ({len(list(combinations(core_3way_cols, 3))):,} triples) ---")

    cells_3way = []
    total_triples = 0
    total_cells_3way = 0
    valid_cells_3way = 0
    SH_CUT = 0.0280  # track all 3-way cells achieving >= 2.80% in every month with n >= 300

    # Outer loop over pairs (c1, c2) so we can reuse pair codes and skip pairs with max count < 300
    n_core = len(core_3way_cols)
    for i in range(n_core - 2):
        c1 = core_3way_cols[i]
        K1 = b_card[c1]
        b1_0, b1_1, b1_2 = b_m[c1]
        for j in range(i + 1, n_core - 1):
            c2 = core_3way_cols[j]
            K2 = b_card[c2]
            K12 = K1 * K2
            p0 = b1_0 * K2 + b_m[c2][0]
            if np.bincount(p0, minlength=K12).max() < MIN_N:
                continue
            p1 = b1_1 * K2 + b_m[c2][1]
            p2 = b1_2 * K2 + b_m[c2][2]
            for k in range(j + 1, n_core):
                c3 = core_3way_cols[k]
                K3 = b_card[c3]
                K = K12 * K3
                total_triples += 1
                total_cells_3way += K

                comb0 = p0 * K3 + b_m[c3][0]
                n0 = np.bincount(comb0, minlength=K)
                cand0 = n0 >= MIN_N
                if not cand0.any():
                    continue
                f0 = np.bincount(comb0, weights=y_m[0], minlength=K)
                ok_idx = np.where(cand0 & (f0 >= SH_CUT * n0))[0]
                if len(ok_idx) == 0:
                    continue
                comb1 = p1 * K3 + b_m[c3][1]
                n1 = np.bincount(comb1, minlength=K)
                ok_idx = ok_idx[n1[ok_idx] >= MIN_N]
                if len(ok_idx) == 0:
                    continue
                f1 = np.bincount(comb1, weights=y_m[1], minlength=K)
                ok_idx = ok_idx[f1[ok_idx] >= SH_CUT * n1[ok_idx]]
                if len(ok_idx) == 0:
                    continue
                comb2 = p2 * K3 + b_m[c3][2]
                n2 = np.bincount(comb2, minlength=K)
                ok_idx = ok_idx[n2[ok_idx] >= MIN_N]
                if len(ok_idx) == 0:
                    continue
                f2 = np.bincount(comb2, weights=y_m[2], minlength=K)
                ok_idx = ok_idx[f2[ok_idx] >= SH_CUT * n2[ok_idx]]
                if len(ok_idx) == 0:
                    continue
                min_n = np.minimum(np.minimum(n0, n1), n2)
                valid_cells_3way += len(ok_idx)

                for idx in ok_idx:
                    idx = int(idx)
                    i12, i3 = divmod(idx, K3)
                    i1, i2 = divmod(i12, K2)
                    sh0 = float(f0[idx] / n0[idx])
                    sh1 = float(f1[idx] / n1[idx])
                    sh2 = float(f2[idx] / n2[idx])
                    cells_3way.append({
                        "order": 3,
                        "cols": f"{c1} x {c2} x {c3}",
                        "cell": f"{b_lbls[c1][i1]} & {b_lbls[c2][i2]} & {b_lbls[c3][i3]}",
                        "min_n": int(min_n[idx]),
                        "min_share": float(min(sh0, sh1, sh2)),
                        "pool_share": float((f0[idx] + f1[idx] + f2[idx]) / (n0[idx] + n1[idx] + n2[idx])),
                        "n_jul": int(n0[idx]), "f_jul": int(f0[idx]), "sh_jul": sh0,
                        "n_aug": int(n1[idx]), "f_aug": int(f1[idx]), "sh_aug": sh1,
                        "n_sep": int(n2[idx]), "f_sep": int(f2[idx]), "sh_sep": sh2,
                    })

    df_3way = pd.DataFrame(cells_3way).sort_values("min_share", ascending=False) if cells_3way else pd.DataFrame()
    print(f"3-Way Scan finished in {time.time() - t_3w:.1f}s: {total_triples:,} triples, {total_cells_3way:,} total cells, {valid_cells_3way:,} cells with min_n >= {MIN_N} & min_share >= {SH_CUT*100:.2f}%")
    print(f"3-way cells with min_share >= {TARGET_SHARE*100:.1f}%: {int((df_3way['min_share'] >= TARGET_SHARE).sum()) if len(df_3way) else 0}")
    if len(df_3way):
        print("Top 20 3-way cells by min monthly fraud share (with n >= 300 in every month):")
        print(df_3way[["cell", "min_n", "min_share", "pool_share", "n_jul", "sh_jul", "n_aug", "sh_aug", "n_sep", "sh_sep"]].head(20).to_string(index=False))

    # Overall best cell across 1-way, 2-way, and 3-way
    all_cells = pd.concat([df_1way.head(50), df_2way.head(50), df_3way.head(50) if len(df_3way) else pd.DataFrame()], ignore_index=True)
    all_cells = all_cells.sort_values("min_share", ascending=False)
    best_cell = all_cells.iloc[0]
    print("\n=== BEST CELL FOUND ACROSS ALL 1-WAY, 2-WAY, AND 3-WAY INTERACTIONS (n >= 300 per month) ===")
    print(f"  Order      : {best_cell['order']}-way ({best_cell['cols']})")
    print(f"  Rule / Cell: {best_cell['cell']}")
    print(f"  July       : n = {best_cell['n_jul']:,}, fraud = {best_cell['f_jul']:,}, share = {best_cell['sh_jul']*100:.4f}%")
    print(f"  August     : n = {best_cell['n_aug']:,}, fraud = {best_cell['f_aug']:,}, share = {best_cell['sh_aug']*100:.4f}%")
    print(f"  September  : n = {best_cell['n_sep']:,}, fraud = {best_cell['f_sep']:,}, share = {best_cell['sh_sep']*100:.4f}%")
    print(f"  Pooled     : n = {best_cell['n_jul']+best_cell['n_aug']+best_cell['n_sep']:,}, "
          f"fraud = {best_cell['f_jul']+best_cell['f_aug']+best_cell['f_sep']:,}, share = {best_cell['pool_share']*100:.4f}%")
    print(f"  Min-month  : min_n = {best_cell['min_n']:,}, min_share = {best_cell['min_share']*100:.4f}% (target >= 10.0000%)")

    # === 5. One LightGBM with Month-Grouped Cross-Validation (200 trees, n_jobs=4) ===
    print("\n=== 5. One LightGBM (200 trees, n_jobs=4) with 3-Fold Month-Grouped Cross-Validation ===")
    t_lgb = time.time()
    oof_lgb = np.zeros(len(y_ord), dtype=np.float64)
    fold_gains = np.zeros(X_ord.shape[1], dtype=np.float64)

    for fold_idx, held_month in enumerate(MONTHS):
        val_mask = m_ord == held_month
        trn_mask = ~val_mask
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            n_jobs=4,
            random_state=42 + fold_idx,
            verbose=-1,
        )
        clf.fit(X_ord.iloc[ np.where(trn_mask)[0] ], y_ord[trn_mask])
        oof_lgb[val_mask] = clf.predict_proba(X_ord.iloc[ np.where(val_mask)[0] ])[:, 1]
        fold_gains += clf.booster_.feature_importance(importance_type="gain")
        auc_fold = roc_auc_score(y_ord[val_mask], oof_lgb[val_mask])
        print(f"  Fold {fold_idx+1} (held-out {held_month}): val_rows={val_mask.sum():,}, val_fraud={y_ord[val_mask].sum():,}, OOF AUC = {auc_fold:.6f}")

    auc_oof_total = roc_auc_score(y_ord, oof_lgb)
    # Also compute AUC restricted to home_device == 1 (excluding the 45,104 dev_ext_legit rows)
    hd_mask = X_ord["home_device"].values == 1
    auc_oof_hd = roc_auc_score(y_ord[hd_mask], oof_lgb[hd_mask])
    print(f"  Pooled Out-of-Fold AUC across all {len(y_ord):,} ordinary rows: {auc_oof_total:.6f} ({time.time() - t_lgb:.1f}s)")
    print(f"  Pooled Out-of-Fold AUC within home_device == 1 ({hd_mask.sum():,} rows): {auc_oof_hd:.6f}")

    # Per-fraud-class OOF AUC (each class vs ordinary legit)
    y_cls_ord = y_raw_cls[ordinary_tr]
    print("\n  Per-class Out-of-Fold AUC (ordinary class k vs ordinary legit):")
    for k in range(1, len(LABELS)):
        ck_mask = (y_cls_ord == 0) | (y_cls_ord == k)
        auc_k = roc_auc_score((y_cls_ord[ck_mask] == k).astype(int), oof_lgb[ck_mask])
        auc_k_hd = roc_auc_score((y_cls_ord[ck_mask & hd_mask] == k).astype(int), oof_lgb[ck_mask & hd_mask])
        print(f"    {LABELS[k]:<20s} (n={int((y_cls_ord == k).sum()):5,d}): OOF AUC = {auc_k:.6f} | within home_device==1: {auc_k_hd:.6f}")

    imp_series = pd.Series(fold_gains / fold_gains.sum() * 100.0, index=X_ord.columns).sort_values(ascending=False)
    print("\n  Top 15 LightGBM features by mean gain % across 3 month-grouped folds:")
    for fname, gpct in imp_series.head(15).items():
        print(f"    {fname:<26s}: {gpct:6.2f}%")

    # Evaluate top-N per month and global score thresholds on oof_lgb
    print("\n  LightGBM OOF top-K per month performance (selecting top K rows in each month):")
    for K_sel in (300, 500, 1000, 2000, 5000, 10000, 50000):
        ns, fs, shs = [], [], []
        for i, m in enumerate(MONTHS):
            scores_m = oof_lgb[m_masks[i]]
            ym = y_m[i]
            top_idx = np.argsort(scores_m)[-K_sel:]
            ns.append(K_sel)
            fs.append(int(ym[top_idx].sum()))
            shs.append(fs[-1] / K_sel)
        print(f"    Top {K_sel:6,d}/month -> Jul: {fs[0]:4d}/{ns[0]} ({shs[0]*100:5.2f}%), "
              f"Aug: {fs[1]:4d}/{ns[1]} ({shs[1]*100:5.2f}%), Sep: {fs[2]:4d}/{ns[2]} ({shs[2]*100:5.2f}%), "
              f"min_share={min(shs)*100:5.2f}%")

    # Also test global score thresholds on oof_lgb where every month has >= 300 rows
    thr_grid = np.quantile(oof_lgb, np.linspace(0.50, 0.9995, 500))
    best_lgb_thr = None
    for thr in thr_grid:
        sel = oof_lgb >= thr
        ns = [int(sel[mk].sum()) for mk in m_masks]
        if min(ns) < MIN_N:
            break
        fs = [int(y_m[i][sel[m_masks[i]]].sum()) for i in range(3)]
        shs = [fs[i] / ns[i] for i in range(3)]
        rec = {
            "thr": float(thr),
            "min_n": min(ns),
            "min_share": min(shs),
            "pool_share": sum(fs) / sum(ns),
            "n_jul": ns[0], "f_jul": fs[0], "sh_jul": shs[0],
            "n_aug": ns[1], "f_aug": fs[1], "sh_aug": shs[1],
            "n_sep": ns[2], "f_sep": fs[2], "sh_sep": shs[2],
        }
        if best_lgb_thr is None or rec["min_share"] > best_lgb_thr["min_share"]:
            best_lgb_thr = rec

    print("\n  Best global LightGBM OOF probability threshold with >= 300 rows per month:")
    print(f"    oof_lgb >= {best_lgb_thr['thr']:.5f} -> "
          f"Jul: {best_lgb_thr['f_jul']}/{best_lgb_thr['n_jul']} ({best_lgb_thr['sh_jul']*100:.2f}%), "
          f"Aug: {best_lgb_thr['f_aug']}/{best_lgb_thr['n_aug']} ({best_lgb_thr['sh_aug']*100:.2f}%), "
          f"Sep: {best_lgb_thr['f_sep']}/{best_lgb_thr['n_sep']} ({best_lgb_thr['sh_sep']*100:.2f}%), "
          f"min_share={best_lgb_thr['min_share']*100:.2f}%")

    print(f"\n=== Total script runtime: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
