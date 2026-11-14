"""Does ordinary-looking fraud differ from legit rows in how the generator WROTE the row?

Earlier tests used the model's features. This one uses things the model never sees: how numbers are printed in the raw
CSV (decimal places, string length), digit patterns (seconds, cents, last digits), id numbers, file position, and the
account and merchant tables. Positives are fraud-labelled train rows the model scores as ordinary (fraud probability
< 0.2, not on a band device); negatives are legit-labelled rows scored the same way.

Usage: python -I scripts/explore/30_format_artifacts.py <project_root> [oof_tag]
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
ROOT = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "oof_final_local"

raw = pd.read_csv(ROOT / "data/raw/train.csv", dtype=str, keep_default_na=False)
raw["pos"] = np.arange(len(raw))
base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["txn_id", "split"])
tr = (base.split == "train").values
P = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values[tr]
side = pd.DataFrame({"txn_id": base.txn_id.values[tr], "pf": 1 - P[:, 0], "band": band})
raw = raw.merge(side, on="txn_id", how="left", validate="one_to_one")
assert raw.pf.notna().all()

fraud = (raw.label != "legit").values
ordinary = (raw.pf.values < 0.2) & (raw.band.values == 0)
rng = np.random.RandomState(0)
keep = ordinary & (fraud | (rng.rand(len(raw)) < 0.25))
d = raw[keep].reset_index(drop=True)
y = (d.label != "legit").values.astype(int)
print(f"ordinary-looking rows used: {len(d):,} | fraud-labelled {y.sum():,} | legit-labelled {len(d) - y.sum():,}")


def ndec(s):
    """Number of digits printed after the decimal point (0 when there is no point)."""
    parts = s.str.split(".", n=1, expand=True)
    return parts[1].fillna("").str.len().values if parts.shape[1] > 1 else np.zeros(len(s), dtype=int)


F = pd.DataFrame(index=d.index)
ts = pd.to_datetime(d.timestamp)
F["sec"], F["minute"], F["hour"], F["day"], F["wday"] = ts.dt.second, ts.dt.minute, ts.dt.hour, ts.dt.day, ts.dt.weekday
F["sec_mod5"], F["sec_mod10"], F["min_mod5"] = F.sec % 5, F.sec % 10, F.minute % 5
F["doy"] = ts.dt.dayofyear
for c in ("amount", "balance_before", "sim_age_hours", "device_age_days", "login_to_txn_sec", "sender_account_age_days"):
    v = pd.to_numeric(d[c], errors="coerce")
    F[c] = v
    F[c + "_ndec"] = ndec(d[c])
    F[c + "_len"] = d[c].str.len()
    F[c + "_frac"] = (v * 100).round() % 100
    F[c + "_lastdig"] = pd.to_numeric(d[c].str[-1], errors="coerce")
    F[c + "_int_mod10"] = np.floor(v) % 10
    F[c + "_int_mod100"] = np.floor(v) % 100
F["bal_over_amt"] = F.balance_before / F.amount
F["bal_minus_amt"] = F.balance_before - F.amount
F["pin_retries"] = pd.to_numeric(d.pin_retries, errors="coerce")
F["new_payee"] = pd.to_numeric(d.new_payee, errors="coerce")
F["qr_blank"] = (d.qr_vpa_mismatch == "").astype(int)
F["qr_val"] = pd.to_numeric(d.qr_vpa_mismatch, errors="coerce")
F["qr_len"] = d.qr_vpa_mismatch.str.len()
F["mcat_blank"] = (d.merchant_category == "").astype(int)
F["snd_n"] = d.sender_id.str[2:].astype(np.int64)
F["rcv_n"] = d.receiver_id.str[2:].astype(np.int64)
F["dev_n"] = d.device_id.str[2:].astype(np.int64)
F["dev_minus_snd"] = F.dev_n - F.snd_n
F["rcv_minus_snd"] = F.rcv_n - F.snd_n
F["snd_len"], F["rcv_len"], F["dev_len"] = d.sender_id.str.len(), d.receiver_id.str.len(), d.device_id.str.len()
tx = d.txn_id.str[2:].apply(lambda h: int(h, 16)).astype(np.int64)
F["tx_int"], F["tx_low8"], F["tx_low4"], F["tx_high8"] = tx, tx % 256, tx % 16, tx // 2 ** 24
F["file_pos"] = d.pos
for c in ("txn_type", "merchant_category", "device_city", "app", "receiver_type"):
    F["c_" + c] = d[c].astype("category")

ac = pd.read_csv(ROOT / "data/raw/accounts.csv", dtype=str, keep_default_na=False)
ac["opened_days"] = (pd.to_datetime(ac.opened_on) - pd.Timestamp("2020-01-01")).dt.days
A = ac.set_index("account_id")
for side_, col in (("s", "sender_id"), ("r", "receiver_id")):
    j = A.reindex(d[col].values)
    F[f"{side_}_opened_days"] = j.opened_days.values
    F[f"{side_}_opened_dom"] = pd.to_datetime(j.opened_on.values).day
    for c in ("home_city", "bank_code", "kyc_level", "os"):
        F[f"c_{side_}_{c}"] = pd.Series(j[c].values).fillna("NA").astype("category").values
F["s_city_eq_dev_city"] = (A.home_city.reindex(d.sender_id.values).values == d.device_city.values).astype(int)
me = pd.read_csv(ROOT / "data/raw/merchants.csv", dtype=str, keep_default_na=False).set_index("merchant_id")
jm = me.reindex(d.receiver_id.values)
F["m_onb_days"] = (pd.to_datetime(jm.onboarded_on.values) - pd.Timestamp("2020-01-01")).days
F["c_m_city"] = pd.Series(jm.city.values).fillna("NA").astype("category").values
F["m_cat_eq_txn_cat"] = (jm.category.values == d.merchant_category.values).astype(int)
F["age_gap"] = (ts - pd.to_datetime(A.opened_on.reindex(d.sender_id.values).values)).dt.days - F.sender_account_age_days

print("features:", F.shape[1])
oof = np.zeros(len(y))
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(F, y)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, verbose=-1, n_jobs=12, random_state=f)
    m.fit(F.iloc[a], y[a])
    oof[b] = m.predict_proba(F.iloc[b])[:, 1]
print(f"5-fold AUC, ordinary-looking fraud vs legit, format and id features: {roc_auc_score(y, oof):.4f}")
mon = ts.dt.month.values
for mth in (7, 8, 9):
    k = mon == mth
    print(f"   month {mth}: AUC {roc_auc_score(y[k], oof[k]):.4f} (rows {k.sum():,})")
lab = d.label.values
for name in sorted(set(lab) - {"legit"}):
    k = (lab == name) | (y == 0)
    print(f"   {name:<20} AUC {roc_auc_score(y[k], oof[k]):.4f} (fraud rows {(lab == name).sum():,})")
m.fit(F, y)
imp = pd.Series(m.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
print("top features by gain %:", (imp / imp.sum() * 100).round(1).head(15).to_dict())
auc = {}
for c in F.columns:
    if str(F[c].dtype) == "category":
        continue
    v = F[c].values.astype(float)
    ok = ~np.isnan(v)
    if ok.sum() > 5000 and np.nanstd(v) > 0:
        auc[c] = roc_auc_score(y[ok], v[ok])
s = pd.Series(auc)
s = s[(s - 0.5).abs().sort_values(ascending=False).index]
print("largest single-column AUC gaps:", s.round(4).head(20).to_dict())
top = (oof >= np.quantile(oof, 0.999))
print(f"top 0.1% by score: fraud share {y[top].mean():.3f} (base rate {y.mean():.3f})")
