"""Where does the current best model (oof_c3) lose F1? Per-class table, confusion among types, per-month scores."""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "oof_c3"
WNAME = sys.argv[3] if len(sys.argv) > 3 else "weights_s5_c3.npy"
sys.path.insert(0, str(ROOT / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, macro_f1, per_class  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = ROOT / "data/processed"
base = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "label", "split", "month", "receiver_type", "app_major", "ip_risk_score"])
tr = (base.split == "train").values
y_raw = base.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y = np.load(ROOT / "models/oof_raw/y_clean.npy")
oof = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
w = np.load(ROOT / "models" / TAG / WNAME)
F3 = pd.read_parquet(P / "features_v3.parquet")
F5 = pd.read_parquet(P / "features_v5.parquet", columns=["ch_band"])
band = F3.band.values.astype(int)
p2p_all = (base.receiver_type == "P2P").values


def band_rules(pred, rows):
    b = band[rows]
    S = np.stack([F3[f"snd_band_{k}_3h"].values[rows] for k in range(1, 12)], 1)
    R = np.stack([F3[f"rcv_band_{k}_24h"].values[rows] for k in range(1, 12)], 1)
    sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)
    rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
    cb = F5.ch_band.values[rows].astype(int)
    p2p = p2p_all[rows]
    out = np.where(b > 0, b, pred)
    free = (b == 0) & (pred > 0)
    drain = free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    chain = free & (cb > 0)
    out = np.where(drain, sb, out)
    out = np.where(fanin | shop, rb, out)
    out = np.where(chain, cb, out)
    return out


pred0 = (oof * w).argmax(1)
pred = band_rules(pred0, tr)
print("weights:", dict(zip(LABELS, np.round(w, 2))))
print(f"macro-F1 vs cleaned {macro_f1(y, pred):.4f} | vs raw {macro_f1(y_raw, pred):.4f} | no band rules vs cleaned {macro_f1(y, pred0):.4f}")
T = per_class(y, pred)
bt = band[tr]
T["on_band"] = np.bincount(y[bt > 0], minlength=12)
T["tp_band"] = np.bincount(y[(bt > 0) & (pred == y)], minlength=12)
T["tp_noband"] = np.bincount(y[(bt == 0) & (pred == y)], minlength=12)
T["miss_legit"] = np.bincount(y[(pred == 0) & (y > 0)], minlength=12)
T["miss_type"] = np.bincount(y[(pred > 0) & (y > 0) & (pred != y)], minlength=12)
T["fp_legit"] = np.bincount(pred[(y == 0) & (pred > 0)], minlength=12)
T["fp_type"] = np.bincount(pred[(y > 0) & (pred > 0) & (pred != y)], minlength=12)
print(T.round(3).to_string())

print("\nconfusion among fraud rows predicted as fraud (rows = cleaned label, cols = prediction):")
m = (y > 0) & (pred > 0)
ct = pd.crosstab(pd.Series(np.array(LABELS)[y[m]], name="true"), pd.Series(np.array(LABELS)[pred[m]], name="pred")).reindex(index=LABELS[1:], columns=LABELS[1:], fill_value=0)
ct.columns = [c[:8] for c in ct.columns]
print(ct.to_string())

# upper bounds
det = pred > 0
perfect_type = np.where(det & (y > 0), y, pred)
print(f"\nif every detected fraud row had the right type: {macro_f1(y, perfect_type):.4f}")
no_fp = np.where((y == 0) & det, 0, pred)
print(f"if no legit row were flagged:                   {macro_f1(y, no_fp):.4f}")
both = np.where((y == 0) & det, 0, perfect_type)
print(f"both:                                           {macro_f1(y, both):.4f}")
month = base.month.values[tr].copy()
month[month > "2026-09"] = "2026-09"
for mth in ("2026-07", "2026-08", "2026-09"):
    k = month == mth
    print(mth, f"macro-F1 vs cleaned {macro_f1(y[k], pred[k]):.4f} | vs raw {macro_f1(y_raw[k], pred[k]):.4f}")

# how patterned are the misses? trap columns as a measuring tool only
old = (base.app_major.values[tr] == "3")
ip = base.ip_risk_score.values[tr] > 0.5
for name, mk in (("missed fraud (pred legit)", (pred == 0) & (y > 0)), ("detected fraud", (pred > 0) & (y > 0)), ("flagged legit", (pred > 0) & (y == 0)),
                 ("legit not flagged", (pred == 0) & (y == 0))):
    print(f"{name:28s} n={mk.sum():8d} app3 share {old[mk].mean():.3f} ip>0.5 share {ip[mk].mean():.3f}")
print("\nmissed fraud by class: n, app3 share (0.15 = ordinary, 0.62 = patterned)")
mk = (pred == 0) & (y > 0)
print(pd.DataFrame({"n": np.bincount(y[mk], minlength=12), "app3": [old[mk & (y == k)].mean() if (mk & (y == k)).any() else np.nan for k in range(12)],
                    "ip": [ip[mk & (y == k)].mean() if (mk & (y == k)).any() else np.nan for k in range(12)]}, index=LABELS).round(3).to_string())
