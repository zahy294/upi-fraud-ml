"""Is ordinary-looking fraud detectable at all? Device bands give clean labels for rows on brand-new devices:
band device = fraud (of the band type), ordinary new device = legit. Compare the two with every label-free feature."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src/upi")
from common import load_xy
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
X, y, meta = load_xy(__import__("pathlib").Path("."), ("features_v1.parquet", "features_v2.parquet", "features_v4.parquet"))
B = pd.read_parquet("data/processed/features_v3.parquet", columns=["band", "dev_ext_legit"])
P = np.load("models/s1/oof.npy").astype(float)                       # no-band model, leave-one-month-out
tr = (meta.split == "train").values
band = B.band.values[tr].astype(int); extl = B.dev_ext_legit.values[tr] == 1
pf = 1 - P[:, 0]
Xt = X[tr].reset_index(drop=True)
# ordinary fraud = band row (not the bot pool) that the pattern model does NOT flag; legit = row on an ordinary new device
dn = pd.read_parquet("data/processed/base_v0.parquet", columns=["device_id"]).device_id.str[2:].astype(np.int64).values[tr]
pos = (band > 0) & (dn > 423000) & (pf < 0.2)
neg = extl & (pf < 0.2)
print("ordinary fraud rows on band devices:", int(pos.sum()), "by band:", pd.Series(band[pos]).value_counts().sort_index().to_dict(), "| legit rows on new devices:", int(neg.sum()))
m = pos | neg
drop = [c for c in Xt.columns if c.startswith(("dev_", "r_dev_", "d_dev_", "Qs_", "Q_dev", "acct_uniq_dev", "home_device", "device_age", "away_from"))]
Xa = Xt.loc[m].drop(columns=drop); ya = pos[m].astype(int)
oof = np.zeros(len(ya))
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya)):
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, verbose=-1, n_jobs=6, random_state=f)
    mdl.fit(Xa.iloc[a], ya[a]); oof[b] = mdl.predict_proba(Xa.iloc[b])[:, 1]
print("5-fold AUC, ordinary fraud vs legit on new devices (device features removed): %.4f" % roc_auc_score(ya, oof))
mdl.fit(Xa, ya)
imp = pd.Series(mdl.booster_.feature_importance("gain"), index=Xa.columns).sort_values(ascending=False)
print("top features:", (imp / imp.sum() * 100).round(1).head(12).to_dict())
# univariate view of the dropped device-related columns too
num = [c for c in Xt.columns if Xt[c].dtype != "category"]
auc = {}
for c in num:
    v = Xt.loc[m, c].values.astype(float); ok = ~np.isnan(v)
    if ok.sum() > 2000 and ya[ok].sum() > 100 and np.nanstd(v) > 0:
        auc[c] = roc_auc_score(ya[ok], v[ok])
s = pd.Series(auc); s = s[(s - 0.5).abs().sort_values(ascending=False).index]
print("largest univariate AUC gaps:", s.round(3).head(15).to_dict())
