"""For each group of rows the model gives one type, can ANY column tell the true type apart (event-grouped CV)?

Adds columns the main model never sees (calendar position, id numbers, account attributes) so that a generator
artefact would show up. The trap columns are reported separately, only as a yardstick.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "oof_c3"
WNAME = sys.argv[3] if len(sys.argv) > 3 else "weights_s5_c3.npy"
sys.path.insert(0, str(ROOT / "src/upi"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from common import LABELS, load_xy  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = ROOT / "data/processed"
FE = ("features_v1.parquet", "features_v2.parquet", "features_v3.parquet", "features_v4.parquet", "features_v5.parquet")
X, _, meta = load_xy(ROOT, FE)
base = pd.read_parquet(P / "base_v0.parquet")
tr = (base.split == "train").values
b = base[tr].reset_index(drop=True)
Xt = X[tr].reset_index(drop=True)
y = np.load(ROOT / "models/oof_raw/y_clean.npy")
y_raw = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
oof = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
w = np.load(ROOT / "models" / TAG / WNAME)
pred = (oof * w).argmax(1)
band = Xt.band.values.astype(int)

# extra columns
t = b.t.values.astype(np.int64)
snd = b.sender_id.str[2:].astype(np.int64).values
is_p2p = (b.receiver_type == "P2P").values
rnum = b.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
dev = b.device_id.str[2:].astype(np.int64).values
E = pd.DataFrame({
    "x_t": t.astype("float64"), "x_day": (t // 86400).astype("float32"), "x_sec_of_day": (t % 86400).astype("float32"), "x_sec": (t % 60).astype("float32"),
    "x_min": (t // 60 % 60).astype("float32"), "x_weekday": b.weekday.astype("float32").values,
    "x_snd_num": snd.astype("float32"), "x_rcv_num": rnum.astype("float32"), "x_dev_num": dev.astype("float32"),
    "x_txn_int": b.txn_id.str[2:].apply(lambda s: int(s, 16)).astype("float64").values,
    "x_acct_age": b.sender_account_age_days.astype("float32").values,
    "x_amount_frac": (b.amount.values * 100 % 100).astype("float32"), "x_bal_frac": (b.balance_before.values * 100 % 100).astype("float32"),
    "x_login_frac": (b.login_to_txn_sec.values % 1).astype("float32"), "x_sim_frac": (b.sim_age_hours.values % 1).astype("float32"),
})
for c in ("home_city", "rcv_home_city", "mer_city", "device_city", "os", "bank_code", "app"):
    E["x_" + c] = b[c].astype("object").fillna("NA").astype("category").values
TR = pd.DataFrame({"trap_app3": (b.app_major == "3").astype("float32").values, "trap_ip": b.ip_risk_score.astype("float32").values})
XA = pd.concat([Xt, E], axis=1)

# events over detected rows
det = np.where(pred > 0)[0]
s_flag = Xt.S_flag.values == 1


def links(key, window, ok=None):
    idx = det if ok is None else det[ok[det]]
    o = idx[np.lexsort((t[idx], key[idx]))]
    same = (key[o][1:] == key[o][:-1]) & (t[o][1:] - t[o][:-1] <= window)
    return o[:-1][same], o[1:][same]


I, J = [], []
pairkey = np.minimum(snd, rcv) * 2_000_000 + np.maximum(snd, rcv)
for key, window, ok in ((snd, 7 * 86400, None), (rcv, 86400, is_p2p | s_flag), (dev, 600, None), (pairkey, 7 * 86400, is_p2p)):
    i, j = links(key, window, ok)
    I.append(i)
    J.append(j)
I, J = np.concatenate(I), np.concatenate(J)
_, comp = connected_components(coo_matrix((np.ones(len(I)), (I, J)), shape=(len(b), len(b))), directed=False)

CASES = [
    ("pred sim_swap: sim_swap vs account_takeover", [4], 4, [5]),
    ("pred mule: mule vs velocity_bot", [1], 1, [2]),
    ("pred layering: layering vs mule", [3], 3, [1]),
    ("pred cashback: cashback vs phishing", [9], 9, [6]),
    ("pred phishing: phishing vs cashback", [6], 6, [9]),
    ("pred phishing: phishing vs loan", [6], 6, [10]),
    ("pred qr_swap: qr vs collusion/refund", [7], 7, [8, 11]),
    ("pred phishing/cashback: phishing vs cashback", [6, 9], 6, [9]),
]
PRM = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
           reg_lambda=5.0, verbose=-1, n_jobs=4)


def cv_auc(Xs, ys, groups, seed=0):
    o = np.zeros(len(ys))
    for f, (a, bb) in enumerate(GroupKFold(5).split(Xs, ys, groups)):
        m = lgb.LGBMClassifier(random_state=seed + f, **PRM)
        m.fit(Xs.iloc[a], ys[a])
        o[bb] = m.predict_proba(Xs.iloc[bb])[:, 1]
    return roc_auc_score(ys, o), o


for name, predset, a_cls, b_cls in CASES:
    m = np.isin(pred, predset) & (band == 0) & ((y == a_cls) | np.isin(y, b_cls))
    ys = np.isin(y[m], b_cls).astype(int)
    g = comp[m]
    n_ev = (pd.Series(g[ys == 0]).nunique(), pd.Series(g[ys == 1]).nunique())
    print(f"\n=== {name}: rows {int((ys == 0).sum())} vs {int(ys.sum())}; events {n_ev[0]} vs {n_ev[1]}")
    if ys.sum() < 30:
        continue
    Xm = XA.loc[m].reset_index(drop=True)
    auc_model, _ = cv_auc(Xm[Xt.columns], ys, g)
    auc_all, _ = cv_auc(Xm, ys, g)
    auc_extra, _ = cv_auc(Xm[E.columns], ys, g)
    auc_trap, _ = cv_auc(TR.loc[m].reset_index(drop=True), ys, g)
    # the main model's own ranking of the two types
    own = oof[m][:, b_cls].sum(1) / (oof[m][:, b_cls].sum(1) + oof[m][:, a_cls])
    print(f"   AUC: main model's own probabilities {roc_auc_score(ys, own):.3f} | model features {auc_model:.3f} | + extra columns {auc_all:.3f}"
          f" | extra columns only {auc_extra:.3f} | trap columns only (yardstick) {auc_trap:.3f}")
    mdl = lgb.LGBMClassifier(random_state=0, **PRM).fit(Xm, ys)
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=Xm.columns).sort_values(ascending=False)
    print("   top gain:", (imp / imp.sum() * 100).round(1).head(10).to_dict())
    uni = {}
    for c in Xm.columns:
        if str(Xm[c].dtype) == "category":
            continue
        v = Xm[c].values.astype(float)
        ok = ~np.isnan(v)
        if ok.sum() > 0.5 * len(v) and ys[ok].min() != ys[ok].max() and np.nanstd(v) > 0:
            uni[c] = roc_auc_score(ys[ok], v[ok])
    s = pd.Series(uni)
    s = s[(s - 0.5).abs().sort_values(ascending=False).index]
    print("   largest univariate AUC gaps:", s.round(3).head(10).to_dict())
