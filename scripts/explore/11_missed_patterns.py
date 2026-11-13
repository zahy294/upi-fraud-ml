"""Profile patterned fraud rows (identified by the trap columns) that the signature rules miss.

Usage: python -I scripts/explore/11_missed_patterns.py <project_root> [class ...]
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
ONLY = sys.argv[2:]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 300)
FRAUD = ["mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover", "phishing_collect",
         "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]

df = pd.read_parquet(ROOT / "data/processed/base_v0.parquet")
p2p = df.receiver_type == "P2P"


def B(m):
    return pd.Series(m, index=df.index).astype("boolean").fillna(False).astype(bool)


SIG = {
    "mule_account": B(p2p & (df.rcv_uniq_snd_day >= 3)) | B(df.fwd_amt_ratio_30m.between(0.8, 0.9999)),
    "layering_chain": B(df.fwd_amt_ratio_30m.between(0.8, 0.9999) | df.prev_in_amt_ratio_30m.between(0.8, 0.9999)),
    "velocity_bot": B((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)),
    "sim_swap": B(df.sim_age_hours < 168),
    "account_takeover": B(df.device_age_days < 7),
    "phishing_collect": B((df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
    "qr_swap": B((df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
    "refund_abuse": B((df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
    "fake_cashback": B((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0))),
    "loan_app_extortion": B((df.merchant_category == "lending") | ((df.is_late_evening == 1) & (df.pair_cnt_7d >= 2))),
    "merchant_collusion": B((~p2p) & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
}
df["old"] = (df.app_major == "3")
df["hi"] = df.ip_risk_score > 0.5
tr = df.split == "train"
NUM = ["amount", "sim_age_hours", "device_age_days", "login_to_txn_sec", "pin_retries", "balance_before", "amt_to_balance",
       "sender_account_age_days", "rcv_acct_age_days", "mer_age_days", "hour",
       "rcv_cnt_1h", "rcv_cnt_24h", "rcv_uniq_snd_day", "rcv_out_30m", "rcv_out_2h", "fwd_amt_ratio_30m",
       "snd_in_before_30m", "prev_in_amt_ratio_30m", "snd_out_cnt_10m", "snd_out_cnt_24h", "snd_uniq_rcv_day", "snd_in_cnt_24h",
       "dev_cnt_5m", "dev_cnt_1h", "dev_uniq_acct_total", "acct_uniq_dev_total", "pair_cnt_total", "pair_cnt_7d", "pair_cnt_24h",
       "rev_pair_cnt_total", "rev_small_before_24h", "rev_big_after_24h", "snd_refund_cnt_30d", "snd_collect_cnt_7d",
       "rcv_round500_cnt_7d", "pair_round500_cnt_total", "evening_pair_cnt_7d"]
FLAG = ["amt_is_integer", "amt_round_500", "amt_round_1000", "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10",
        "new_payee", "home_device", "away_from_home", "is_night", "is_late_evening"]
CAT = ["txn_type", "receiver_type", "merchant_category", "rcv_kyc_level", "kyc_level"]
legit = df[tr & (df.label == "legit")]

for lab in FRAUD:
    if ONLY and lab not in ONLY:
        continue
    c = tr & (df.label == lab)
    pat = c & df.old & df.hi
    hit = df[pat & SIG[lab]]
    miss = df[pat & ~SIG[lab]]
    plain = df[c & ~df.old & (df.ip_risk_score < 0.2) & ~SIG[lab]]      # mostly ordinary-looking rows of the class
    print("\n" + "=" * 110)
    print(f"{lab}: patterned(old&hi)={int(pat.sum())}  hit={len(hit)}  miss={len(miss)}  | ordinary-looking sample={len(plain)}")
    groups = {"hit": hit, "MISS": miss, "plain": plain, "legit": legit}
    fl = pd.DataFrame({k: v[FLAG].mean() for k, v in groups.items()}).round(3)
    print("-- flags (mean) --\n" + fl.T.to_string())
    for col in CAT:
        vc = pd.DataFrame({k: v[col].astype("object").fillna("NA").value_counts(normalize=True) for k, v in groups.items()}).fillna(0).round(3)
        print(f"-- {col} --\n" + vc.sort_values("MISS", ascending=False).head(8).T.to_string())
    q = {}
    for k, v in groups.items():
        qq = v[NUM].quantile([.1, .5, .9]).T
        q[k] = qq.apply(lambda r: f"{r[.1]:.4g}/{r[.5]:.4g}/{r[.9]:.4g}", axis=1)
    print("-- numeric p10/p50/p90 --\n" + pd.DataFrame(q).to_string())
    other = {o: SIG[o][miss.index].mean() for o in FRAUD if o != lab}
    print("-- other signatures firing on MISS:", {k: round(v, 3) for k, v in other.items() if v > 0.03})
