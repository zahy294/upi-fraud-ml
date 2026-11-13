"""Train (Sep) vs test (Oct) drift, per-month signal stability, adversarial validation.

Usage: python -I scripts/eda/03_drift.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import lightgbm as lgb  # noqa: E402
from common import LABELS, FRAUD, BLUE, ORANGE, INK2, load, add_row_features, style, md_table  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
FIG, TAB = ROOT / "reports/eda/figures", ROOT / "reports/eda/tables"
style()
df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
tr = df[df.split == "train"]
sep = tr[tr.month == "2026-09"]
te = df[df.split == "test"]
out = []

NUM = ["amount", "balance_before", "amt_to_balance", "sim_age_hours", "device_age_days",
       "sender_account_age_days", "login_to_txn_sec", "pin_retries", "ip_risk_score", "new_payee",
       "qr_vpa_mismatch", "away_from_home", "home_device", "is_night", "amt_is_integer",
       "amt_round_100", "amt_round_500", "amt_round_1000", "amt_just_under_1000s", "amt_tiny_le10",
       "rcv_acct_age_days", "mer_age_days"]


def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:  # binary-ish: compare value shares
        va = a.value_counts(normalize=True); vb = b.value_counts(normalize=True)
        idx = va.index.union(vb.index)
        pa, pb = va.reindex(idx).fillna(1e-6).values, vb.reindex(idx).fillna(1e-6).values
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        pa = np.histogram(a, qs)[0] / len(a); pb = np.histogram(b, qs)[0] / len(b)
        pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


rows = []
for c in NUM:
    a, b = sep[c], te[c]
    rows.append({"feature": c, "sep_mean": a.mean(), "oct_mean": b.mean(), "sep_median": a.median(),
                 "oct_median": b.median(), "sep_p90": a.quantile(.9), "oct_p90": b.quantile(.9),
                 "KS": ks_2samp(a.dropna(), b.dropna()).statistic, "PSI": psi(a, b)})
drift = pd.DataFrame(rows).set_index("feature").sort_values("PSI", ascending=False)
out.append("## Sep (train) vs Oct (test) marginal drift, all rows\nPSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.\n"
           + md_table(drift, ".4g"))
print(drift)

# fraud-only proxy is impossible for test; compare legit-Sep vs whole Oct as well (Oct is ~96% legit)
rows = []
leg_sep = sep[sep.label == "legit"]
for c in NUM:
    rows.append({"feature": c, "legitSep_mean": leg_sep[c].mean(), "oct_mean": te[c].mean(),
                 "PSI_legitSep_vs_oct": psi(leg_sep[c], te[c])})
out.append("\n## Legit-only September vs all of October\n" +
           md_table(pd.DataFrame(rows).set_index("feature").sort_values("PSI_legitSep_vs_oct", ascending=False), ".4g"))

# categorical shares
for c in ["txn_type", "receiver_type", "merchant_category", "app", "device_city"]:
    s = pd.DataFrame({"sep_%": sep[c].fillna("NA").value_counts(normalize=True) * 100,
                      "oct_%": te[c].fillna("NA").value_counts(normalize=True) * 100})
    s["diff_pp"] = s["oct_%"] - s["sep_%"]
    s = s.sort_values("diff_pp", key=abs, ascending=False).head(12)
    out.append(f"\n## {c}: Sep vs Oct share (top |diff|)\n" + md_table(s.round(3), ".4g"))

# amount by merchant category: festival season
amt = pd.DataFrame({"sep_median": sep.groupby(sep.merchant_category.fillna("P2P")).amount.median(),
                    "oct_median": te.groupby(te.merchant_category.fillna("P2P")).amount.median()})
amt["ratio"] = amt.oct_median / amt.sep_median
out.append("\n## Median amount by merchant category, Sep vs Oct\n" + md_table(amt.round(2), ".4g"))

# per-month univariate AUC (fraud vs legit and one-vs-rest per label) -> is the signal stable?
auc_rows = []
for c in NUM + ["app_major_is3"]:
    r = {"feature": c}
    for m in ["2026-07", "2026-08", "2026-09"]:
        d = tr[tr.month == m]
        x = (d.app_major == "3").astype(int) if c == "app_major_is3" else d[c].fillna(-1)
        r[m] = roc_auc_score(d.label != "legit", x)
    auc_rows.append(r)
auc = pd.DataFrame(auc_rows).set_index("feature")
auc["max_abs_dev"] = (auc - 0.5).abs().max(axis=1)
out.append("\n## Univariate AUC fraud-vs-legit, per month (0.5 = no signal; <0.5 = inverse)\n"
           + md_table(auc.sort_values("max_abs_dev", ascending=False).round(4), ".4g"))

# ip_risk_score deep dive
ip = tr.groupby(["month", tr.label.where(tr.label == "legit", "fraud")]).ip_risk_score.describe()[["mean", "50%", "75%", "max"]]
out.append("\n## ip_risk_score by month and fraud/legit (train)\n" + md_table(ip.round(4), ".4g"))
out.append(f"\nip_risk_score test: mean {te.ip_risk_score.mean():.4f}, median {te.ip_risk_score.median():.4f}, "
           f"p75 {te.ip_risk_score.quantile(.75):.4f}, p99 {te.ip_risk_score.quantile(.99):.4f}")
ipl = tr.groupby("label").ip_risk_score.agg(["mean", "median", lambda s: (s > 0.5).mean() * 100]).reindex(LABELS)
ipl.columns = ["mean", "median", "pct_gt_0.5"]
out.append("\n## ip_risk_score by label (train)\n" + md_table(ipl.round(4), ".4g"))
# per-label one-vs-legit AUC of ip_risk_score per month
r = {}
for l in FRAUD:
    r[l] = {m: roc_auc_score(tr[(tr.month == m) & tr.label.isin(["legit", l])].label == l,
                             tr[(tr.month == m) & tr.label.isin(["legit", l])].ip_risk_score)
            for m in ["2026-07", "2026-08", "2026-09"]}
out.append("\n## ip_risk_score AUC (label vs legit) per month\n" + md_table(pd.DataFrame(r).T.round(3), ".3g"))

fig, ax = plt.subplots(figsize=(8, 3.4))
bins = np.linspace(0, 1, 51)
ax.hist(sep[sep.label == "legit"].ip_risk_score, bins=bins, density=True, histtype="step", color=BLUE, lw=2, label="Sep legit")
ax.hist(sep[sep.label != "legit"].ip_risk_score, bins=bins, density=True, histtype="step", color=INK2, lw=2, ls="--", label="Sep fraud")
ax.hist(te.ip_risk_score, bins=bins, density=True, histtype="step", color=ORANGE, lw=2, label="Oct (test, all)")
ax.set_title("ip_risk_score distribution: Sep legit vs Sep fraud vs October")
ax.set_xlabel("ip_risk_score"); ax.set_ylabel("density"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "06_ip_risk_score_drift.png"); plt.close(fig)

# adversarial validation: can a model tell September from October from row features?
FEATS = [c for c in NUM if c not in ("rcv_acct_age_days", "mer_age_days")] + ["hour", "weekday"]
cats = ["txn_type", "receiver_type", "merchant_category", "app", "device_city"]
X = pd.concat([sep, te])
y = (X.split == "test").astype(int).values
XX = X[FEATS].copy()
for c in cats:
    XX[c] = X[c].fillna("NA").astype("category")
rng = np.random.RandomState(0)
idx = rng.permutation(len(XX)); cut = int(len(idx) * 0.7)
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, verbose=-1, n_jobs=8)
m.fit(XX.iloc[idx[:cut]], y[idx[:cut]])
p = m.predict_proba(XX.iloc[idx[cut:]])[:, 1]
adv_auc = roc_auc_score(y[idx[cut:]], p)
imp = pd.Series(m.booster_.feature_importance("gain"), index=XX.columns).sort_values(ascending=False)
imp = (imp / imp.sum() * 100).round(2)
out.append(f"\n## Adversarial validation (Sep vs Oct, row features excl. app_version & date)\nAUC = {adv_auc:.4f}\n"
           + md_table(imp.head(15).to_frame("gain_%"), ".4g"))
print("adv AUC", adv_auc); print(imp.head(10))
# drop top feature(s) and retest
for drop in [list(imp.index[:1]), list(imp.index[:3])]:
    keep = [c for c in XX.columns if c not in drop]
    m2 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, verbose=-1, n_jobs=8)
    m2.fit(XX.iloc[idx[:cut]][keep], y[idx[:cut]])
    a2 = roc_auc_score(y[idx[cut:]], m2.predict_proba(XX.iloc[idx[cut:]][keep])[:, 1])
    out.append(f"- without {drop}: AUC = {a2:.4f}")
    print("without", drop, a2)

(TAB / "03_drift.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", TAB / "03_drift.md")
