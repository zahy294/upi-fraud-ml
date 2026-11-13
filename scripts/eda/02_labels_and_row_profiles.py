"""Label distribution, per-label row-level profiles, amount patterns, time-of-day.

Usage: python -I scripts/eda/02_labels_and_row_profiles.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import (LABELS, FRAUD, BLUE, ORANGE, INK2, load, add_row_features, style, seq_cmap,  # noqa: E402
                    md_table)

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
FIG, TAB = ROOT / "reports/eda/figures", ROOT / "reports/eda/tables"
style()
pd.set_option("display.width", 250, "display.max_columns", 60)

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
tr = df[df.split == "train"].copy()
te = df[df.split == "test"].copy()
out = []

# ---------- label distribution ----------
vc = tr.label.value_counts().reindex(LABELS)
lab = pd.DataFrame({"count": vc, "pct": (vc / len(tr) * 100).round(3),
                    "pct_of_fraud": (vc / vc[FRAUD].sum() * 100).round(1)})
lab.loc[["legit"], "pct_of_fraud"] = np.nan
for m in ["2026-07", "2026-08", "2026-09"]:
    s = tr[tr.month == m].label.value_counts()
    lab[f"pct_{m}"] = (s / s.sum() * 100).reindex(LABELS).round(3)
out.append("## Label distribution\n" + md_table(lab, ".4g"))
print(lab)

fig, ax = plt.subplots(figsize=(7, 4))
order = vc.sort_values()
ax.barh(order.index, order.values, color=BLUE, height=0.7)
ax.set_xscale("log")
for i, v in enumerate(order.values):
    ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=8, color=INK2)
ax.set_title("Training labels (log scale): legit is 96.3%, smallest class is 1,991 rows")
ax.set_xlabel("rows in train.csv")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "01_label_counts.png"); plt.close(fig)

# ---------- daily volume train vs test ----------
daily = df.groupby([df.timestamp.dt.date, "split"]).size().unstack()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(daily.index, daily["train"], color=BLUE, label="train")
ax.plot(daily.index, daily["test"], color=ORANGE, label="test")
ax.set_title("Transactions per day: train (Jul-Sep) ~17k/day, test (Oct) ~8.4k/day")
ax.set_ylabel("rows"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "02_daily_volume.png"); plt.close(fig)
fraud_daily = tr[tr.label != "legit"].groupby(tr.timestamp.dt.date).size()
out.append(f"\nDaily rows: train median {daily['train'].median():.0f}, test median {daily['test'].median():.0f}. "
           f"Fraud rows/day in train: median {fraud_daily.median():.0f}, min {fraud_daily.min()}, max {fraud_daily.max()}.")

# ---------- numeric profiles ----------
NUM = ["amount", "balance_before", "amt_to_balance", "sim_age_hours", "device_age_days",
       "sender_account_age_days", "acct_age_from_opened", "login_to_txn_sec", "pin_retries",
       "ip_risk_score", "new_payee", "qr_vpa_mismatch", "away_from_home", "home_device",
       "is_night", "is_late_evening", "rcv_acct_age_days", "mer_age_days", "mer_cat_matches_txn",
       "mer_city_eq_device_city"]
med = tr.groupby("label")[NUM].median().reindex(LABELS)
mean = tr.groupby("label")[NUM].mean().reindex(LABELS)
p10 = tr.groupby("label")[NUM].quantile(0.1).reindex(LABELS)
p90 = tr.groupby("label")[NUM].quantile(0.9).reindex(LABELS)
out.append("\n## Numeric medians by label\n" + md_table(med[NUM[:10]], ".4g"))
out.append("\n## Means (binary features => rates) by label\n" + md_table(mean[NUM[10:]], ".3g"))
out.append("\n## p10 by label\n" + md_table(p10[["amount", "sim_age_hours", "device_age_days", "login_to_txn_sec",
                                                 "amt_to_balance", "rcv_acct_age_days"]], ".4g"))
out.append("\n## p90 by label\n" + md_table(p90[["amount", "sim_age_hours", "device_age_days", "login_to_txn_sec",
                                                 "amt_to_balance", "ip_risk_score"]], ".4g"))
print(med); print(mean[NUM[10:]])

# lift heatmap: rate / legit rate for binary-ish, median ratio for continuous
lift = (mean / mean.loc["legit"]).drop(index="legit")
fig, ax = plt.subplots(figsize=(11, 4.6))
show = lift[["new_payee", "qr_vpa_mismatch", "away_from_home", "home_device", "is_night", "is_late_evening",
             "pin_retries", "ip_risk_score", "amt_to_balance", "amount", "sim_age_hours", "device_age_days",
             "login_to_txn_sec"]]
im = ax.imshow(np.log2(show.clip(lower=1 / 64, upper=64)), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
ax.set_xticks(range(show.shape[1]), show.columns, rotation=40, ha="right")
ax.set_yticks(range(show.shape[0]), show.index)
for i in range(show.shape[0]):
    for j in range(show.shape[1]):
        v = show.iat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if abs(np.log2(max(v, 1e-9))) > 3 else "#0b0b0b")
ax.grid(False)
ax.set_title("Mean of each feature relative to legit (x times). Red = higher than legit, blue = lower")
fig.colorbar(im, ax=ax, label="log2(ratio to legit)", shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / "03_feature_lift_vs_legit.png"); plt.close(fig)

# ---------- categorical mixes ----------
for col in ["txn_type", "receiver_type", "merchant_category", "app", "app_major", "kyc_level", "os",
            "pin_retries"]:
    ct = pd.crosstab(tr.label, tr[col].fillna("NA"), normalize="index").reindex(LABELS) * 100
    out.append(f"\n## {col} mix by label (row %)\n" + md_table(ct.round(1), ".3g"))
ct = pd.crosstab(tr.txn_type, tr.qr_vpa_mismatch.fillna(-1), normalize="index") * 100
out.append("\n## qr_vpa_mismatch value by txn_type (-1 = blank), row %\n" + md_table(ct.round(2), ".4g"))
ct = pd.crosstab(tr.label, tr.qr_vpa_mismatch.fillna(-1)).reindex(LABELS)
out.append("\n## qr_vpa_mismatch counts by label (-1 = blank)\n" + md_table(ct, ".6g"))

# ---------- hour of day heatmap ----------
hr = pd.crosstab(tr.label, tr.hour, normalize="index").reindex(LABELS) * 100
fig, ax = plt.subplots(figsize=(11, 4.6))
im = ax.imshow(hr.values, cmap=seq_cmap(), aspect="auto")
ax.set_xticks(range(24)); ax.set_yticks(range(len(LABELS)), LABELS)
ax.set_xlabel("hour of day (IST)"); ax.grid(False)
ax.set_title("Share of each label's transactions by hour: sim_swap peaks at night, loan_app at 20-23h")
fig.colorbar(im, ax=ax, label="% of label's rows", shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / "04_hour_by_label.png"); plt.close(fig)
hr_te = te.hour.value_counts(normalize=True).sort_index() * 100
hr_tr = tr.hour.value_counts(normalize=True).sort_index() * 100
out.append("\n## hour share train vs test (all rows)\n" + md_table(pd.DataFrame({"train": hr_tr, "test": hr_te}).T.round(2), ".3g"))

# ---------- amount patterns ----------
AMT = ["amt_is_integer", "amt_round_100", "amt_round_500", "amt_round_1000", "amt_just_under_1000s",
       "amt_ends_99", "amt_tiny_le10"]
ap = (tr.groupby("label")[AMT].mean().reindex(LABELS) * 100).round(2)
ap.loc["TEST (all rows)"] = (te[AMT].mean() * 100).round(2)
out.append("\n## Amount patterns by label (% of rows)\n" + md_table(ap, ".4g"))
lines = ["\n## Most common amounts by label (top 12, with count)"]
for l in LABELS:
    s = tr.loc[tr.label == l, "amount"].value_counts().head(12)
    lines.append(f"- **{l}** (n={int((tr.label == l).sum())}): " + ", ".join(f"{a:g}x{c}" for a, c in s.items()))
out.append("\n".join(lines))

# amount distribution plot
fig, ax = plt.subplots(figsize=(10, 4.6))
data = [np.log10(tr.loc[tr.label == l, "amount"].clip(lower=0.5)) for l in LABELS]
bp = ax.boxplot(data, vert=False, labels=LABELS, showfliers=False, widths=0.6, patch_artist=True)
for b in bp["boxes"]:
    b.set(facecolor="#cde2fb", edgecolor=BLUE)
for k in ["whiskers", "caps"]:
    for x in bp[k]:
        x.set(color=INK2)
for x in bp["medians"]:
    x.set(color=ORANGE, linewidth=2)
ax.set_xticks(range(0, 6), ["₹1", "₹10", "₹100", "₹1k", "₹10k", "₹100k"])
ax.set_title("Amount by label (box = p25-p75, orange = median, whiskers 1.5 IQR)")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "05_amount_by_label.png"); plt.close(fig)

# ---------- city / bank / sender-level ----------
out.append("\n## Fraud rate by kyc_level / os (train)\n" + md_table(
    pd.concat([tr.groupby("kyc_level").label.apply(lambda s: (s != "legit").mean() * 100).rename("fraud_%"),
               tr.groupby("os").label.apply(lambda s: (s != "legit").mean() * 100).rename("fraud_%")]).to_frame(), ".3g"))
city_rate = tr.groupby("device_city").label.apply(lambda s: (s != "legit").mean() * 100).sort_values()
out.append(f"\nFraud rate by device_city: min {city_rate.min():.2f}% ({city_rate.index[0]}), "
           f"max {city_rate.max():.2f}% ({city_rate.index[-1]}), median {city_rate.median():.2f}%")
bank_rate = tr.groupby("bank_code").label.apply(lambda s: (s != "legit").mean() * 100).sort_values()
out.append(f"Fraud rate by sender bank_code: min {bank_rate.min():.2f}%, max {bank_rate.max():.2f}% ({bank_rate.index[-1]})")

(TAB / "02_labels_and_row_profiles.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", TAB / "02_labels_and_row_profiles.md")
