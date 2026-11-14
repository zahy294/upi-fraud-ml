"""Do ordinary-looking fraud rows deviate from their OWN sender's / receiver's / device's habits?

Row attributes of ordinary-looking fraud match legit rows exactly. This asks a different question: a legit row is
drawn from its account's profile (usual app, balance level, receivers, SIM and device timelines). If the fraud rows
were drawn without that profile, they will disagree with the account's other rows.

Every feature compares a row with the OTHER rows of the same entity (train and test together, no labels).
Usage: python -I scripts/explore/32_profile_deviation.py <project_root> [oof_tag]
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
P = ROOT / "data/processed"
d = pd.read_parquet(P / "base_v0.parquet", columns=[
    "txn_id", "split", "label", "t", "sender_id", "receiver_id", "receiver_type", "device_id", "amount", "balance_before", "txn_type",
    "merchant_category", "device_city", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries",
    "login_to_txn_sec", "app", "app_version", "hour", "home_device", "qr_vpa_mismatch"])
tr = (d.split == "train").values
n = len(d)
F = pd.DataFrame(index=d.index)


def others_mean(key, v):
    """Mean of v over the other rows with the same key (nan when the row is alone)."""
    g = pd.Series(v).groupby(key.values)
    s, c = g.transform("sum").values, g.transform("count").values
    return np.where(c > 1, (s - v) / np.maximum(c - 1, 1), np.nan), c


def others_same(key, cat):
    """Share of the other rows with the same key that have the same value of cat."""
    k2 = key.astype(str) + "|" + cat.astype(str)
    c_all = key.map(key.value_counts()).values
    c_same = k2.map(k2.value_counts()).values
    return np.where(c_all > 1, (c_same - 1) / np.maximum(c_all - 1, 1), np.nan)


def others_std(key, v):
    g = pd.Series(v).groupby(key.values)
    return g.transform("std").values


lbal, lamt = np.log1p(d.balance_before.values), np.log1p(d.amount.values)
sim_t0 = d.t.values / 3600.0 - d.sim_age_hours.values            # hour at which the SIM was last changed, as implied by this row
dev_t0 = d.t.values / 86400.0 - d.device_age_days.values         # day the account first used this device, as implied by this row
acc_t0 = d.t.values / 86400.0 - d.sender_account_age_days.values
sd = d.sender_id + "|" + d.device_id
for name, key in (("snd", d.sender_id), ("rcv", d.receiver_id), ("dev", d.device_id)):
    m, c = others_mean(key, lbal)
    F[f"{name}_n"] = c
    F[f"{name}_lbal_dev"] = lbal - m
    F[f"{name}_lbal_sd"] = others_std(key, lbal)
    m, _ = others_mean(key, lamt)
    F[f"{name}_lamt_dev"] = lamt - m
    F[f"{name}_lamt_sd"] = others_std(key, lamt)
    m, _ = others_mean(key, d.hour.values.astype(float))
    F[f"{name}_hour_dev"] = d.hour.values - m
    for col in ("app", "app_version", "device_city", "txn_type", "merchant_category", "receiver_type"):
        F[f"{name}_same_{col}"] = others_same(key, d[col].fillna("NA"))
    m, _ = others_mean(key, sim_t0)
    F[f"{name}_simt0_dev"] = sim_t0 - m
    F[f"{name}_simt0_sd"] = others_std(key, sim_t0)
    m, _ = others_mean(key, d.login_to_txn_sec.values.astype(float))
    F[f"{name}_login_dev"] = d.login_to_txn_sec.values - m
m, c = others_mean(sd, dev_t0)
F["sd_n"] = c
F["sd_devt0_dev"] = dev_t0 - m
F["sd_devt0_sd"] = others_std(sd, dev_t0)
m, _ = others_mean(d.sender_id, acc_t0)
F["snd_acct0_dev"] = acc_t0 - m
F["snd_same_rcv"] = others_same(d.sender_id, d.receiver_id)
F["snd_same_dev"] = others_same(d.sender_id, d.device_id)
F["rcv_same_snd"] = others_same(d.receiver_id, d.sender_id)
pair = d.sender_id + ">" + d.receiver_id
o = np.lexsort((d.t.values, pair.values))
ps = pair.values[o]
first = np.r_[True, ps[1:] != ps[:-1]]
rank = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))
F["pair_rank"] = pd.Series(rank, index=d.index[o]).reindex(d.index).values        # 0 = first time this sender pays this receiver in the data
F["pair_n"] = pair.map(pair.value_counts()).values
F["new_payee"] = d.new_payee.values
F["np_but_seen"] = ((d.new_payee == 1) & (F.pair_rank > 0)).astype(int)
F["notnp_first"] = ((d.new_payee == 0) & (F.pair_rank == 0)).astype(int)

Pm = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
band = pd.read_parquet(P / "features_v3.parquet", columns=["band"]).band.values
pf = np.full(n, np.nan)
pf[tr] = 1 - Pm[:, 0]
ordinary = tr & (pf < 0.2) & (band == 0)
fraud = (d.label.values != "legit") & tr
rng = np.random.RandomState(0)
keep = ordinary & (fraud | (rng.rand(n) < 0.25))
X, y = F[keep].reset_index(drop=True), fraud[keep].astype(int)
print(f"rows: {len(y):,} | ordinary-looking fraud {y.sum():,} | legit {len(y) - y.sum():,} | features {X.shape[1]}")
oof = np.zeros(len(y))
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, verbose=-1, n_jobs=12, random_state=f)
    m.fit(X.iloc[a], y[a])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
print(f"5-fold AUC, profile-deviation features only: {roc_auc_score(y, oof):.4f}")
m.fit(X, y)
imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("top features by gain %:", (imp / imp.sum() * 100).round(1).head(12).to_dict())
auc = {}
for c in X.columns:
    v = X[c].values.astype(float)
    ok = ~np.isnan(v)
    if ok.sum() > 5000 and np.nanstd(v) > 0:
        auc[c] = roc_auc_score(y[ok], v[ok])
        va = np.abs(v[ok])
        if c.endswith("_dev"):
            auc[c + "|abs"] = roc_auc_score(y[ok], va)
s = pd.Series(auc)
s = s[(s - 0.5).abs().sort_values(ascending=False).index]
print("largest single-column AUC gaps:", s.round(4).head(25).to_dict())
print("\nmeans, fraud vs legit:")
print(pd.DataFrame({"fraud": X[y == 1].mean(), "legit": X[y == 0].mean()}).round(4).to_string())
top = oof >= np.quantile(oof, 0.999)
print(f"\ntop 0.1% by score: fraud share {y[top].mean():.3f} (base rate {y.mean():.3f})")
