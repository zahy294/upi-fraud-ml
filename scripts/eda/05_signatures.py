"""Hand-written signature rule per fraud type: coverage, precision, Sep->Oct firing stability,
plus event connectedness (do a label's rows cluster with other rows of the same label?).

Usage: python -I scripts/eda/05_signatures.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, FRAUD, load, add_row_features, md_table  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
TAB = ROOT / "reports/eda/tables"
SCALE, OFF = np.int64(2 ** 26), np.int64(2 ** 25)
MIN, HOUR, DAY = 60, 3600, 86400

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
R = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
assert (R.txn_id.values == df.txn_id.values).all()
df = pd.concat([df, R.drop(columns=["txn_id", "split", "label", "month"])], axis=1)
p2p = df.receiver_type == "P2P"

# known-fraud receiver: P2P receiver had a fraud inbound row in an EARLIER month (no same-month leakage)
first_fraud_month = df[df.label.notna() & (df.label != "legit")].groupby("receiver_id").month.min()
ffm = df.receiver_id.map(first_fraud_month)
df["known_fraud_rcv"] = (p2p & ffm.notna() & (ffm < df.month)).astype(int)

S = {
    "mule_account": ("P2P & rcv_uniq_snd_day>=3", p2p & (df.rcv_uniq_snd_day >= 3)),
    "layering_chain": ("forwarded 80-100% within 30m (in or out)",
                       df.fwd_amt_ratio_30m.between(0.8, 0.9999) | df.prev_in_amt_ratio_30m.between(0.8, 0.9999)),
    "velocity_bot": ("dev_cnt_5m>=3 | login_to_txn_sec<=5", (df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)),
    "sim_swap": ("sim_age_hours<168", df.sim_age_hours < 168),
    "account_takeover": ("device_age_days<7", df.device_age_days < 7),
    "phishing_collect": ("COLLECT & amount just under round (x950-x999 or ends 99)",
                         (df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
    "qr_swap": ("qr_vpa_mismatch==1 | (QR & P2P)", (df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
    "refund_abuse": ("REFUND & sender refunds in +-15d >= 3", (df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
    "fake_cashback": ("reverse <=10 before | (<=10 & reverse >=100 after)",
                      (df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0))),
    "loan_app_extortion": ("lending merchant | (late evening & pair_cnt_7d>=2)",
                           (df.merchant_category == "lending") | ((df.is_late_evening == 1) & (df.pair_cnt_7d >= 2))),
    "merchant_collusion": ("MERCHANT & round-500 amount & pair has >=2 round-500 txns",
                           (~p2p) & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
}
S["(any) known fraud receiver"] = ("P2P receiver had fraud inbound in an earlier month", df.known_fraud_rcv == 1)
S = {k: (d, pd.Series(m, index=df.index).astype("boolean").fillna(False).astype(bool)) for k, (d, m) in S.items()}

sep = df[(df.split == "train") & (df.month == "2026-09")]
tr = df[df.split == "train"]
te = df[df.split == "test"]
rows = []
for lab, (desc, m) in S.items():
    ms, mt = m[sep.index], m[te.index]
    fired = sep[ms]
    target = lab if lab in LABELS else None
    rows.append({
        "label": lab, "rule": desc,
        "coverage_%": (ms[sep.label == target].mean() * 100) if target else np.nan,
        "precision_%": ((fired.label == target).mean() * 100) if target else np.nan,
        "any_fraud_precision_%": (fired.label != "legit").mean() * 100,
        "fires_Sep_%": ms.mean() * 100, "fires_Oct_%": mt.mean() * 100,
        "top_labels_when_fired": fired.label.value_counts(normalize=True).mul(100).round(1).head(3).to_dict(),
    })
sig = pd.DataFrame(rows).set_index("label")
out = ["## Signature rules evaluated on September (train); firing rate compared with October (test)\n"
       "coverage = % of that label's Sep rows the rule catches; precision = % of fired Sep rows that carry the label.\n",
       md_table(sig, ".3g")]
print(sig.drop(columns=["top_labels_when_fired"]).round(2).to_string())

# share of each label that fires its own vs any signature
own = pd.Series({lab: S[lab][1][tr.index][tr.label == lab].mean() * 100 for lab in FRAUD})
anym = np.zeros(len(df), bool)
for lab in FRAUD:
    anym |= S[lab][1].values
anym = pd.Series(anym, index=df.index)
anysig = tr.assign(a=anym[tr.index]).groupby("label").a.mean().reindex(LABELS) * 100
cov = pd.DataFrame({"own_signature_%": own, "any_signature_%": anysig}).reindex(LABELS)
out.append("\n## How much of each label is 'on-pattern'? (train, all months)\n" + md_table(cov.round(1), ".3g"))
print(cov.round(1))

# ---------- connectedness of each label's rows ----------
t = df.t.values.astype(np.int64)
keys = {
    "sender ±24h": (df.sender_id.str[2:].astype(np.int64).values, DAY),
    "receiver ±24h": (np.where(p2p, 0, 1_000_000) + df.receiver_id.str[2:].astype(np.int64).values, DAY),
    "device ±1h": (df.device_id.str[2:].astype(np.int64).values, HOUR),
}
# money-flow link: my receiver is the sender of a same-label row (or vice versa) within 2h
conn = {}
lab_arr = df.label.values
for lab in FRAUD:
    idx = np.where(lab_arr == lab)[0]
    res = {}
    any_link = np.zeros(len(idx), bool)
    for kname, (g, w) in keys.items():
        gg, tt = g[idx], t[idx]
        o = np.lexsort((tt, gg)); K = gg[o] * SCALE + tt[o] + OFF
        q = gg * SCALE + tt + OFF
        c = np.searchsorted(K, q + w, "right") - np.searchsorted(K, q - w, "left")
        res[kname] = (c >= 2).mean() * 100
        any_link |= c >= 2
    # flow: receiver (as account) of this row sends a same-label row within +-2h
    snd_l = keys["sender ±24h"][0][idx]
    rcv_l = np.where(p2p.values[idx], df.receiver_id.str[2:].astype(np.int64).values[idx], -1)
    o = np.lexsort((t[idx], snd_l)); K = snd_l[o] * SCALE + t[idx][o] + OFF
    q = rcv_l * SCALE + t[idx] + OFF
    c1 = np.where(rcv_l >= 0, np.searchsorted(K, q + 2 * HOUR, "right") - np.searchsorted(K, q - 2 * HOUR, "left"), 0)
    o = np.lexsort((t[idx], rcv_l)); K = rcv_l[o] * SCALE + t[idx][o] + OFF
    q = snd_l * SCALE + t[idx] + OFF
    c2 = np.searchsorted(K, q + 2 * HOUR, "right") - np.searchsorted(K, q - 2 * HOUR, "left")
    res["money-flow ±2h"] = ((c1 + c2) > 0).mean() * 100
    any_link |= (c1 + c2) > 0
    res["any link"] = any_link.mean() * 100
    conn[lab] = res
conn = pd.DataFrame(conn).T
out.append("\n## Connectedness: % of a label's rows linked to ANOTHER row of the same label (train)\n"
           + md_table(conn.round(1), ".3g"))
print(conn.round(1))

# ---------- drift probes for the adaptive types ----------
probe = []
for nm, d in [("Sep legit", sep[sep.label == "legit"]), ("Sep all", sep), ("Oct all", te)]:
    probe.append({"set": nm,
                  "login<=5s_%": (d.login_to_txn_sec <= 5).mean() * 100,
                  "non-home device_%": (d.home_device == 0).mean() * 100,
                  "dev_cnt_5m>=3_%": (d.dev_cnt_5m >= 3).mean() * 100,
                  "dev_cnt_1h>=3_%": (d.dev_cnt_1h >= 3).mean() * 100,
                  "dev_uniq_acct_total>=3_%": (d.dev_uniq_acct_total >= 3).mean() * 100,
                  "sim_age<168h_%": (d.sim_age_hours < 168).mean() * 100,
                  "device_age<7d_%": (d.device_age_days < 7).mean() * 100,
                  "lending_%": (d.merchant_category == "lending").mean() * 100,
                  "lending & 20-23h_%": ((d.merchant_category == "lending") & (d.is_late_evening == 1)).mean() * 100,
                  "late evening & pair_cnt_7d>=2_%": ((d.is_late_evening == 1) & (d.pair_cnt_7d >= 2)).mean() * 100,
                  "COLLECT & just-under_%": ((d.txn_type == "COLLECT") & (d.amt_just_under_1000s == 1)).mean() * 100,
                  "QR & P2P_%": ((d.txn_type == "QR") & (d.receiver_type == "P2P")).mean() * 100,
                  "qr_vpa_mismatch==1_%": (d.qr_vpa_mismatch == 1).mean() * 100})
probe = pd.DataFrame(probe).set_index("set").T
out.append("\n## Single-signal prevalence: Sep legit vs Sep all vs Oct (%)\n"
           "If 'Oct all' sits far below 'Sep all' (toward 'Sep legit'), that fraud pattern got rarer or changed shape in October.\n"
           + md_table(probe.round(3), ".4g"))
print(probe.round(3))

# velocity bots in Sep: how are their bursts shaped? (inter-arrival on the same device)
vb = sep[sep.label == "velocity_bot"].sort_values(["device_id", "t"])
gap = vb.groupby("device_id").t.diff().dropna()
out.append(f"\nvelocity_bot (Sep) gap between consecutive payments on the same device: median {gap.median():.0f}s, "
           f"p75 {gap.quantile(.75):.0f}s, p90 {gap.quantile(.9):.0f}s; rows per device: median "
           f"{vb.groupby('device_id').size().median():.0f}")
nonhome = te[te.home_device == 0].sort_values(["device_id", "t"])
g2 = nonhome.groupby("device_id").t.diff().dropna()
out.append(f"Oct, non-home-device rows: gap between consecutive payments on the same device: median {g2.median():.0f}s, "
           f"share of gaps <=120s {(g2 <= 120).mean()*100:.1f}%  (Sep non-home: "
           f"{(sep[sep.home_device==0].sort_values(['device_id','t']).groupby('device_id').t.diff().dropna() <= 120).mean()*100:.1f}%)")

(TAB / "05_signatures.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[-2:]))
