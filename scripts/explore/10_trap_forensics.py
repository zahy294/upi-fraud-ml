"""Use the trap columns (app_version 3.x, ip_risk_score) as a noisy oracle of "generated fraud" in train.

Questions: what share of each fraud class is generated (patterned) fraud? how much of it do the
signature rules catch? how much generated fraud is labelled legit (unreported)?
Usage: python -I scripts/explore/10_trap_forensics.py <project_root>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "scripts/eda"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import FRAUD, load, add_row_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
R = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
REL = [c for c in R.columns if c not in ("txn_id", "split", "label", "month")]
df = pd.concat([df, R[REL]], axis=1)
df.to_parquet(ROOT / "data/processed/base_v0.parquet", index=False)  # cache for later scripts
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
tr = df[df.split == "train"].copy()
tr["old"] = (tr.app_major == "3").astype(int)
tr["hi"] = (tr.ip_risk_score > 0.5).astype(int)
anysig = np.zeros(len(df), bool)
for lab in FRAUD:
    anysig |= SIG[lab].values
tr["anysig"] = anysig[tr.index]

print("== ip_risk_score quantiles: legit / fraud&anysig / fraud&no-sig / test ==")
qs = [.1, .25, .5, .75, .9, .95, .99]
print(pd.DataFrame({
    "legit": tr[tr.label == "legit"].ip_risk_score.quantile(qs),
    "fraud_sig": tr[(tr.label != "legit") & tr.anysig].ip_risk_score.quantile(qs),
    "fraud_nosig": tr[(tr.label != "legit") & ~tr.anysig].ip_risk_score.quantile(qs),
    "test": df[df.split == "test"].ip_risk_score.quantile(qs)}).round(3).T.to_string())
print("\napp_version share, legit vs fraud&sig:")
print(pd.DataFrame({
    "legit": tr[tr.label == "legit"].app_version.value_counts(normalize=True),
    "fraud_sig": tr[(tr.label != "legit") & tr.anysig].app_version.value_counts(normalize=True)}).sort_index().round(3).T.to_string())

leg = tr[tr.label == "legit"]
pa, pb, pab = leg.old.mean(), leg.hi.mean(), (leg.old & leg.hi).mean()
print(f"\nlegit-labelled: old={pa:.4f} hi={pb:.4f} both={pab:.5f} indep-expect={pa*pb:.5f} ratio={pab/(pa*pb):.3f}")
rows = []
for lab in FRAUD:
    d = tr[tr.label == lab]
    on = SIG[lab][d.index]
    both = (d.hi == 1) & (d.old == 1)
    rows.append({"label": lab, "n": len(d), "sig_cov": on.mean(), "old": d.old.mean(), "hi": d.hi.mean(), "both": both.mean(),
                 "old|on": d.old[on].mean(), "hi|on": d.hi[on].mean(), "old|off": d.old[~on].mean(), "hi|off": d.hi[~on].mean(),
                 "sig|hi": on[d.hi == 1].mean(), "sig|both": on[both].mean(), "anysig|both": tr.anysig[d.index][both].mean()})
T = pd.DataFrame(rows).set_index("label")
print("\n== per class ==\n" + T.round(3).to_string())

print("\n== mixture solve per class: class = g*gen(qa,qb) + (1-g)*plain(pa,pb), independence within parts ==")
for lab in FRAUD:
    d = tr[tr.label == lab]
    a, b, ab = d.old.mean(), d.hi.mean(), (d.old & d.hi).mean()
    best = None
    for g in np.arange(0.15, 0.90, 0.005):
        qa = (a - (1 - g) * pa) / g
        qb = (b - (1 - g) * pb) / g
        if not (0 < qa < 1 and 0 < qb < 1):
            continue
        err = abs(g * qa * qb + (1 - g) * pa * pb - ab)
        if best is None or err < best[0]:
            best = (err, g, qa, qb)
    print(f"{lab:20s} g={best[1]:.3f} qa(old|gen)={best[2]:.3f} qb(hi|gen)={best[3]:.3f} err={best[0]:.5f}")

print("\n== legit-labelled rows: share with old&hi, by any-signature hit ==")
for nm, d in [("legit & anysig", leg[leg.anysig]), ("legit & no sig", leg[~leg.anysig])]:
    print(f"{nm:16s} n={len(d):8d} old={d.old.mean():.4f} hi={d.hi.mean():.4f} both={(d.old & d.hi).mean():.5f} indep={d.old.mean()*d.hi.mean():.5f}")
print("legit-labelled rows hitting each signature: n, old, hi (generated fraud has old~0.6, hi~0.4)")
for lab in FRAUD:
    d = leg[SIG[lab][leg.index]]
    print(f"  {lab:20s} n={len(d):7d} old={d.old.mean():.3f} hi={d.hi.mean():.3f} both={(d.old & d.hi).mean():.4f}")

print("rows hitting the refined RULES patterns, by label group:")
integer = df.amt_is_integer == 1
RULES = {
    "refund_abuse": B((df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
    "layering_chain": B(integer & p2p & (df.fwd_amt_ratio_30m.between(0.8, 0.9999) | df.prev_in_amt_ratio_30m.between(0.8, 0.9999)) & (df.amount >= 5000)),
    "fake_cashback": B(integer & ((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0)))),
    "merchant_collusion": B((~p2p) & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
    "mule_account": B(integer & p2p & (df.rcv_uniq_snd_day >= 3) & (df.new_payee == 1)),
    "phishing_collect": B(integer & (df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
    "qr_swap": B((df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
    "velocity_bot": B(integer & (~p2p) & ((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)) & (df.home_device == 0)),
    "sim_swap": B(integer & p2p & (df.sim_age_hours < 168) & (df.new_payee == 1)),
    "account_takeover": B(integer & p2p & (df.device_age_days < 7) & (df.new_payee == 1)),
    "loan_app_extortion": B(integer & (df.merchant_category == "lending") & (df.new_payee == 1)),
}
for lab, m in RULES.items():
    d = tr[m[tr.index]]
    dl, dc = d[d.label == "legit"], d[d.label == lab]
    do = d[(d.label != "legit") & (d.label != lab)]
    f = lambda x: (x.old & x.hi).mean() if len(x) else float("nan")  # noqa: E731
    o = lambda x: x.old.mean() if len(x) else float("nan")  # noqa: E731
    print(f"  {lab:20s} hits={len(d):6d} own={len(dc):5d} legit={len(dl):5d} other={len(do):5d} | old&hi: own={f(dc):.3f} legit={f(dl):.3f} other={f(do):.3f} | old: own={o(dc):.2f} legit={o(dl):.2f} other={o(do):.2f}")
