"""Among trap-identified patterned fraud rows, how well can ANY model predict the (noisy) fraud type?
If sibling types stay ~35% confused no matter what, the type noise is random; if a model separates them, a signal exists."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
FRAUD = ["mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover", "phishing_collect",
         "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
df = pd.read_parquet("data/processed/base_v0.parquet")
df["amt_mult10"] = ((df.amount % 10) == 0).astype(int)
df["rcv_is_min_kyc"] = (df.rcv_kyc_level == "min").astype(float)
drop = {"txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "label", "split", "month", "app_version", "app_major",
        "ip_risk_score", "home_city", "rcv_home_city", "mer_city", "t", "weekday", "mer_category", "rcv_kyc_level"}
CATS = ["receiver_type", "txn_type", "merchant_category", "app", "device_city", "kyc_level", "os", "bank_code"]
feats = [c for c in df.columns if c not in drop and c not in CATS]
X = df[feats].astype("float32")
for c in CATS:
    X[c] = df[c].astype("object").fillna("NA").astype("category")
pat = (df.split == "train") & (df.label != "legit") & (df.app_major == "3") & (df.ip_risk_score > 0.4)
y = df.label.map({l: i for i, l in enumerate(FRAUD)})
fit = pat & (df.month < "2026-09"); val = pat & (df.month >= "2026-09")
print("patterned rows: fit", int(fit.sum()), "val", int(val.sum()))
m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=12, random_state=0)
m.fit(X[fit], y[fit].astype(int))
P = m.predict_proba(X[val]); pred = P.argmax(1); yt = y[val].astype(int).values
print("type accuracy on Sep patterned rows: %.3f" % (pred == yt).mean())
cm = pd.crosstab(pd.Series(yt, name="label").map(dict(enumerate(FRAUD))), pd.Series(pred, name="pred").map(dict(enumerate(FRAUD))))
cm = cm.reindex(index=FRAUD, columns=FRAUD, fill_value=0)
short = [f[:6] for f in FRAUD]; cm.index = short; cm.columns = short
print(cm.to_string())
print("row-normalised diagonal:", (np.diag(cm.values) / cm.values.sum(1)).round(2).tolist())
print("col-normalised diagonal (precision):", (np.diag(cm.values) / np.maximum(cm.values.sum(0), 1)).round(2).tolist())
imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("top features:", (imp / imp.sum() * 100).round(1).head(15).to_dict())
