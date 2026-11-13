"""Ceiling estimate, rule baseline, trap-feature check in October, and a LightGBM ablation.

Validation protocol everywhere: train on Jul-Aug, score on September.
Usage: python -I scripts/eda/07_baselines.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402
from common import LABELS, FRAUD, load, add_row_features, md_table  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
TAB, SUB = ROOT / "reports/eda/tables", ROOT / "submissions"
SUB.mkdir(exist_ok=True)

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
R = pd.read_parquet(ROOT / "data/processed/relational_v0.parquet")
REL = [c for c in R.columns if c not in ("txn_id", "split", "label", "month")]
df = pd.concat([df, R[REL]], axis=1)
p2p = df.receiver_type == "P2P"
B = lambda m: pd.Series(m, index=df.index).astype("boolean").fillna(False).astype(bool)  # noqa: E731
integer = df.amt_is_integer == 1

# refined rules: signature + the on-pattern profile (integer amounts, P2P etc.), ordered by precision
RULES = [
    ("refund_abuse", B((df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3))),
    ("layering_chain", B(integer & p2p & (df.fwd_amt_ratio_30m.between(0.8, 0.9999) | df.prev_in_amt_ratio_30m.between(0.8, 0.9999)) & (df.amount >= 5000))),
    ("fake_cashback", B(integer & ((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0))))),
    ("merchant_collusion", B((~p2p) & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2))),
    ("mule_account", B(integer & p2p & (df.rcv_uniq_snd_day >= 3) & (df.new_payee == 1))),
    ("mule_account", B(integer & p2p & df.fwd_amt_ratio_30m.between(0.8, 0.9999))),
    ("phishing_collect", B(integer & (df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1)))),
    ("qr_swap", B((df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p))),
    ("velocity_bot", B(integer & (~p2p) & ((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)) & (df.home_device == 0))),
    ("sim_swap", B(integer & p2p & (df.sim_age_hours < 168) & (df.new_payee == 1))),
    ("account_takeover", B(integer & p2p & (df.device_age_days < 7) & (df.new_payee == 1))),
    ("loan_app_extortion", B(integer & (df.merchant_category == "lending") & (df.new_payee == 1))),
]
pred_rule = pd.Series("legit", index=df.index, dtype=object)
for lab, m in reversed(RULES):          # first rule in the list wins
    pred_rule[m] = lab

is_tr = df.split == "train"
sep = is_tr & (df.month == "2026-09")
fit = is_tr & (df.month < "2026-09")
y = df.label
out = []


def report(name, yt, yp):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=LABELS, zero_division=0)
    t = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=LABELS)
    mf = f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
    return mf, t


# 1) oracle ceiling: perfect detection of on-pattern rows of each type, everything else -> legit
SIG = {
    "mule_account": B(p2p & (df.rcv_uniq_snd_day >= 3)) | B(df.fwd_amt_ratio_30m.between(0.8, 0.9999)),
    "layering_chain": B(df.fwd_amt_ratio_30m.between(0.8, 0.9999) | df.prev_in_amt_ratio_30m.between(0.8, 0.9999)),
    "velocity_bot": B((df.dev_cnt_5m >= 3) | (df.login_to_txn_sec <= 5)),
    "sim_swap": B(df.sim_age_hours < 168), "account_takeover": B(df.device_age_days < 7),
    "phishing_collect": B((df.txn_type == "COLLECT") & ((df.amt_just_under_1000s == 1) | (df.amt_ends_99 == 1))),
    "qr_swap": B((df.qr_vpa_mismatch == 1) | ((df.txn_type == "QR") & p2p)),
    "refund_abuse": B((df.txn_type == "REFUND") & (df.snd_refund_cnt_30d >= 3)),
    "fake_cashback": B((df.rev_small_before_24h > 0) | ((df.amount <= 10) & (df.rev_big_after_24h > 0))),
    "loan_app_extortion": B((df.merchant_category == "lending") | ((df.is_late_evening == 1) & (df.pair_cnt_7d >= 2))),
    "merchant_collusion": B((~p2p) & (df.amt_round_500 == 1) & (df.pair_round500_cnt_total >= 2)),
}
oracle = pd.Series("legit", index=df.index, dtype=object)
for lab in FRAUD:
    oracle[(y == lab) & SIG[lab]] = lab
mf, t = report("oracle", y[sep], oracle[sep])
out.append(f"## Oracle ceiling (Sep): perfectly label every on-pattern row, call the rest legit\nmacro-F1 = **{mf:.4f}**\n"
           + md_table(t.round(3), ".3g"))
print("oracle", mf)

# 2) rule baseline
mf, t = report("rules", y[sep], pred_rule[sep])
out.append(f"\n## Hand-written rules only (no model), Sep\nmacro-F1 = **{mf:.4f}**\n" + md_table(t.round(3), ".3g"))
print("rules Sep", mf); print(t.round(3))
pred_rule_test = pred_rule[df.split == "test"]
out.append("\nRule predictions on test (counts): " + str(pred_rule_test.value_counts().to_dict()))
pd.DataFrame({"id": df.txn_id[df.split == "test"], "label": pred_rule_test}).to_csv(SUB / "rules_v0.csv", index=False)

# 3) do the trap features still track fraud signatures in October?
any_rule = pd.Series(False, index=df.index)
for _, m in RULES:
    any_rule |= m
rows = []
for nm, msk in [("Sep", sep), ("Oct (test)", df.split == "test")]:
    d = df[msk]
    f = any_rule[msk]
    rows.append({"set": nm, "rule-hit rows": int(f.sum()),
                 "ip mean | rule hit": d.ip_risk_score[f].mean(), "ip mean | no hit": d.ip_risk_score[~f].mean(),
                 "ip>0.4 % | rule hit": (d.ip_risk_score[f] > 0.4).mean() * 100,
                 "ip>0.4 % | no hit": (d.ip_risk_score[~f] > 0.4).mean() * 100,
                 "oldest app ver % | hit": (d.app_version[f].isin(["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "5.0"])).mean() * 100,
                 "oldest app ver % | no hit": (d.app_version[~f].isin(["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "5.0"])).mean() * 100})
trap = pd.DataFrame(rows).set_index("set")
ver_te = pd.crosstab(any_rule[df.split == "test"], df.app_version[df.split == "test"], normalize="index") * 100
out.append("\n## Do ip_risk_score / app_version still separate rule-hit rows from the rest in October?\n"
           "('oldest app ver' = 3.x in Sep, 5.0 in Oct)\n" + md_table(trap.round(3), ".4g")
           + "\n\nOct app_version share by rule hit (row %):\n" + md_table(ver_te.round(2), ".4g"))
print(trap.round(3)); print(ver_te.round(2))

# 4) LightGBM ablation: Jul-Aug -> Sep
CATS = ["receiver_type", "txn_type", "merchant_category", "app", "device_city"]
ROW = ["hour", "weekday", "amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee",
       "pin_retries", "login_to_txn_sec", "balance_before", "qr_vpa_mismatch", "amt_to_balance", "away_from_home",
       "home_device", "amt_is_integer", "amt_round_100", "amt_round_500", "amt_round_1000",
       "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10", "rcv_acct_age_days", "mer_age_days"]
TRAPS = ["ip_risk_score", "app_ver_num"]
df["app_ver_num"] = df.app_version.astype(float)
X = df[ROW + TRAPS + REL].copy()
for c in CATS:
    X[c] = df[c].fillna("NA").astype("category")
lab2id = {l: i for i, l in enumerate(LABELS)}
yi = y.map(lab2id)


def run(cols, name, sims=None):
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           verbose=-1, n_jobs=12)
    m.fit(X.loc[fit, cols], yi[fit].astype(int))
    res = {}
    for sim_name, Xs in [("Sep as-is", X.loc[sep, cols])] + (sims or []):
        P = m.predict_proba(Xs[cols])
        best = (0, 1)
        for k in [1, 2, 3, 5, 8, 12, 20]:            # one shared up-weight for the fraud classes
            Q = P.copy(); Q[:, 1:] *= k
            f = f1_score(yi[sep].astype(int), Q.argmax(1), labels=range(12), average="macro", zero_division=0)
            best = max(best, (f, k))
        res[sim_name] = best
    print(name, res)
    return m, res


def octoberise(cols):
    """Sep rows with the two trap features made to look like October."""
    Xs = X.loc[sep, cols].copy()
    if "app_ver_num" in cols:
        Xs["app_ver_num"] = np.random.RandomState(0).choice([5.0, 5.1, 5.2], len(Xs))
    if "ip_risk_score" in cols:  # replace with draws from the October marginal (signal gone)
        Xs["ip_risk_score"] = np.random.RandomState(1).choice(df.ip_risk_score[df.split == "test"].values, len(Xs))
    return Xs


abl = []
cols_a = ROW + TRAPS + CATS
_, r = run(cols_a, "A row+traps", [("Sep with Oct-like traps", octoberise(cols_a))])
abl.append(("A: single-row features incl. ip_risk_score & app_version", r))
cols_b = ROW + CATS
_, r = run(cols_b, "B row no traps")
abl.append(("B: single-row features, traps removed", r))
cols_c = ROW + CATS + REL
m_c, r = run(cols_c, "C row+relational no traps")
abl.append(("C: B + relational features", r))
cols_d = ROW + TRAPS + CATS + REL
_, r = run(cols_d, "D everything", [("Sep with Oct-like traps", octoberise(cols_d))])
abl.append(("D: C + traps", r))

lines = ["\n## LightGBM ablation (train Jul-Aug, validate Sep; macro-F1 with best shared fraud up-weight k)"]
for desc, r in abl:
    lines.append(f"- {desc}: " + "; ".join(f"{k}: **{v[0]:.4f}** (k={v[1]})" for k, v in r.items()))
out.append("\n".join(lines))

P = m_c.predict_proba(X.loc[sep, cols_c]); k = dict(abl)["C: B + relational features"]["Sep as-is"][1]
P[:, 1:] *= k
mf, t = report("C", y[sep], pd.Series(np.array(LABELS)[P.argmax(1)], index=y[sep].index))
out.append(f"\n### Model C per-class (Sep, k={k})\n" + md_table(t.round(3), ".3g"))
imp = pd.Series(m_c.booster_.feature_importance("gain"), index=cols_c).sort_values(ascending=False)
out.append("\n### Model C top features by gain %\n" + md_table((imp / imp.sum() * 100).round(2).head(25).to_frame("gain_%"), ".3g"))

(TAB / "07_baselines.md").write_text("\n".join(out), encoding="utf-8")
print("wrote")
