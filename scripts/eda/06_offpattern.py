"""What are the ~70% of fraud rows that do NOT match their type's signature?

(a) are they linked (same sender/receiver/device, short window) to an on-pattern row of the same label?
(b) do they differ from legit on anything other than the two trap features (app_version, ip_risk_score)?

Usage: python -I scripts/eda/06_offpattern.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import LABELS, FRAUD, load, add_row_features, md_table  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
TAB = ROOT / "reports/eda/tables"
SCALE, OFF = np.int64(2 ** 26), np.int64(2 ** 25)
HOUR, DAY = 3600, 86400

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
R = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
df = pd.concat([df, R.drop(columns=["txn_id", "split", "label", "month"])], axis=1)
p2p = df.receiver_type == "P2P"
B = lambda m: pd.Series(m, index=df.index).astype("boolean").fillna(False).astype(bool)  # noqa: E731
SIG = {
    "mule_account": B(p2p & (df.rcv_uniq_snd_day >= 3)),
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
t = df.t.values.astype(np.int64)
G = {"sender ±24h": (df.sender_id.str[2:].astype(np.int64).values, DAY),
     "receiver ±24h": (np.where(p2p, 0, 1_000_000) + df.receiver_id.str[2:].astype(np.int64).values, DAY),
     "device ±1h": (df.device_id.str[2:].astype(np.int64).values, HOUR)}

rows, prof = [], []
lab = df.label.values
for L in FRAUD:
    on = np.where((lab == L) & SIG[L].values)[0]
    off = np.where((lab == L) & ~SIG[L].values)[0]
    r = {"label": L, "n_on": len(on), "n_off": len(off)}
    linked = np.zeros(len(off), bool)
    for k, (g, w) in G.items():
        o = np.lexsort((t[on], g[on])); K = g[on][o] * SCALE + t[on][o] + OFF
        q = g[off] * SCALE + t[off] + OFF
        c = np.searchsorted(K, q + w, "right") - np.searchsorted(K, q - w, "left")
        r[f"off linked via {k} %"] = (c > 0).mean() * 100
        linked |= c > 0
    r["off linked any %"] = linked.mean() * 100
    r["recoverable ceiling % (on + linked off)"] = (len(on) + linked.sum()) / (len(on) + len(off)) * 100
    rows.append(r)
    for part, ix in [("on", on), ("off", off)]:
        d = df.iloc[ix]
        prof.append({"label": L, "part": part, "n": len(ix), "app3.x_%": (d.app_major == "3").mean() * 100,
                     "ip>0.5_%": (d.ip_risk_score > 0.5).mean() * 100, "ip_mean": d.ip_risk_score.mean(),
                     "amt_integer_%": d.amt_is_integer.mean() * 100, "new_payee_%": d.new_payee.mean() * 100,
                     "P2P_%": (d.receiver_type == "P2P").mean() * 100, "amount_median": d.amount.median()})
leg = df[df.label == "legit"]
prof.append({"label": "legit", "part": "-", "n": len(leg), "app3.x_%": (leg.app_major == "3").mean() * 100,
             "ip>0.5_%": (leg.ip_risk_score > 0.5).mean() * 100, "ip_mean": leg.ip_risk_score.mean(),
             "amt_integer_%": leg.amt_is_integer.mean() * 100, "new_payee_%": leg.new_payee.mean() * 100,
             "P2P_%": (leg.receiver_type == "P2P").mean() * 100, "amount_median": leg.amount.median()})
link = pd.DataFrame(rows).set_index("label")
prof = pd.DataFrame(prof).set_index(["label", "part"])
print(link.round(1).to_string()); print(prof.round(2).to_string())

# how separable are OFF-pattern fraud rows from legit WITHOUT the trap features? (univariate AUCs)
off_all = np.zeros(len(df), bool)
for L in FRAUD:
    off_all |= (lab == L) & ~SIG[L].values
d = df[(df.split == "train") & (off_all | (lab == "legit"))]
y = (d.label != "legit").values
cand = ["amt_is_integer", "amount", "amt_to_balance", "new_payee", "rcv_acct_age_days", "mer_age_days",
        "ip_risk_score", "home_device", "device_age_days", "login_to_txn_sec", "sim_age_hours",
        "rcv_cnt_total", "rcv_uniq_snd_total", "pair_share_of_rcv", "rcv_cnt_24h", "snd_out_cnt_24h"]
au = {c: roc_auc_score(y, d[c].fillna(-1)) for c in cand}
au["app_major_is3"] = roc_auc_score(y, (d.app_major == "3").astype(int))
au = pd.Series(au).sort_values(key=lambda s: (s - 0.5).abs(), ascending=False)
print(au.round(3))

out = ["## Off-pattern fraud rows: linked to an on-pattern row of the same label? (train)\n"
       "on = rows matching that type's signature rule, off = the rest.\n" + md_table(link.round(1), ".3g"),
       "\n## Profile of on- vs off-pattern rows (train)\n" + md_table(prof.round(2), ".4g"),
       "\n## Univariate AUC: off-pattern fraud (all types pooled) vs legit\n" + md_table(au.round(3).to_frame("AUC"), ".3g")]
(TAB / "06_offpattern.md").write_text("\n".join(out), encoding="utf-8")
print("wrote")
