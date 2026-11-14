"""Are ordinary-looking fraud rows tied to anything in TIME? (They are not tied by sender, receiver or device.)

Clean groups from device bands (train): ordinary fraud = band-device row the pattern model does not flag;
legit = row on an ordinary new device. Also the label-defined version on home devices (noisy labels).
H1: gap to the nearest detected fraud row (any type / same type) is shorter for ordinary fraud.
H2: ordinary fraud rows share exact timestamps with each other / with detected rows more than legit rows do.
H3: gap to the nearest other ordinary-fraud row of the same type.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import LABELS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = ROOT / "data/processed"
base = pd.read_parquet(P / "base_v0.parquet", columns=["t", "label", "split", "device_id"])
tr = (base.split == "train").values
b = base[tr].reset_index(drop=True)
y = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
t = b.t.values.astype(np.int64)
dn = b.device_id.str[2:].astype(np.int64).values
B = pd.read_parquet(P / "features_v3.parquet", columns=["band", "dev_ext_legit"])
band = B.band.values[tr].astype(int)
extl = B.dev_ext_legit.values[tr] == 1
oof = np.load(ROOT / "models/oof_c3/oof.npy").astype(float)
pf = 1 - oof[:, 0]
top = oof[:, 1:].argmax(1) + 1
det = pf >= 0.5
pf_s1 = 1 - np.load(ROOT / "models/s1/oof.npy").astype(float)[:, 0]     # the model without band features, for the band-based groups


def nearest_gap(tq, tref, exclude_self=False):
    """Seconds from each query time to the nearest reference time."""
    r = np.sort(tref)
    i = np.searchsorted(r, tq)
    lo = np.abs(tq - r[np.clip(i - 1, 0, len(r) - 1)]).astype(float)
    hi = np.abs(r[np.clip(i, 0, len(r) - 1)] - tq).astype(float)
    if exclude_self:                                  # query is itself in the reference: use the neighbours on both sides
        i0 = np.searchsorted(r, tq, "left")
        i1 = np.searchsorted(r, tq, "right")
        dup = (i1 - i0) > 1
        lo = np.where(i0 - 1 >= 0, np.abs(tq - r[np.clip(i0 - 1, 0, len(r) - 1)]), np.inf)
        hi = np.where(i1 < len(r), np.abs(r[np.clip(i1, 0, len(r) - 1)] - tq), np.inf)
        return np.where(dup, 0.0, np.minimum(lo, hi))
    return np.minimum(lo, hi)


def report(name, pos, neg):
    print(f"\n=== {name}: positives {int(pos.sum())}, negatives {int(neg.sum())}")
    if pos.sum() == 0 or neg.sum() == 0:
        print("  (one group is empty: skipped)")
        return
    m = pos | neg
    lab = pos[m].astype(int)
    g_any = nearest_gap(t[m], t[det & ~m])
    print(f"  H1 gap to nearest detected fraud row: median pos {np.median(g_any[lab == 1]):.0f}s, neg {np.median(g_any[lab == 0]):.0f}s; "
          f"share within 60s pos {np.mean(g_any[lab == 1] <= 60):.3f} neg {np.mean(g_any[lab == 0] <= 60):.3f}; AUC(-gap) {roc_auc_score(lab, -g_any):.3f}")
    cnt_s = pd.Series(t).map(pd.Series(t).value_counts()).values
    print(f"  H2 other rows in the same second: mean pos {cnt_s[pos].mean() - 1:.3f}, neg {cnt_s[neg].mean() - 1:.3f}")


# clean, band-based
pos_b = (band > 0) & (dn > 423000) & (pf_s1 < 0.2)
neg_b = extl & (pf_s1 < 0.2)
report("band-device ordinary fraud vs new-device legit", pos_b, neg_b)
# label-based on home devices
home = (band == 0) & ~extl
pos_l = home & (y > 0) & (pf < 0.2)
rng = np.random.RandomState(0)
neg_l = home & (y == 0) & (pf < 0.2) & (rng.rand(len(y)) < 0.05)
report("home-device ordinary fraud (by label) vs legit sample", pos_l, neg_l)

print("\nH1 by type: gap from an ordinary-fraud row of type k to the nearest DETECTED row predicted as type k, vs the same for legit rows")
rows = []
for k in range(1, 12):
    ref = t[det & (top == k)]
    gp = nearest_gap(t[pos_l & (y == k)], ref)
    gn = nearest_gap(t[neg_l], ref)
    lab = np.r_[np.ones(len(gp)), np.zeros(len(gn))]
    rows.append({"type": LABELS[k], "n_pos": len(gp), "median_pos_s": np.median(gp), "median_neg_s": np.median(gn),
                 "within60_pos": np.mean(gp <= 60), "within60_neg": np.mean(gn <= 60), "auc": roc_auc_score(lab, -np.r_[gp, gn])})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nH3: gap from an ordinary-fraud row of type k to the nearest OTHER ordinary-fraud row of type k, vs a size-matched random legit set")
rows = []
for k in range(1, 12):
    tk = t[pos_l & (y == k)]
    gp = nearest_gap(tk, tk, exclude_self=True)
    tl = rng.choice(t[home & (y == 0)], size=len(tk), replace=False)
    gn = nearest_gap(tl, tl, exclude_self=True)
    rows.append({"type": LABELS[k], "n": len(tk), "median_gap_fraud_s": np.median(gp), "median_gap_legit_s": np.median(gn),
                 "p10_fraud": np.percentile(gp, 10), "p10_legit": np.percentile(gn, 10), "same_second_fraud": np.mean(gp == 0), "same_second_legit": np.mean(gn == 0)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nday-of-period profile: share of ordinary fraud vs legit per week (a drifting share would be usable)")
wk = (t - t.min()) // (7 * 86400)
print(pd.DataFrame({"ord_fraud": np.bincount(wk[pos_l], minlength=wk.max() + 1), "legit_sample": np.bincount(wk[neg_l], minlength=wk.max() + 1)}).assign(
    ratio=lambda d: (d.ord_fraud / d.legit_sample).round(3)).T.to_string())
hr = (t % 86400) // 3600
print(pd.DataFrame({"ord_fraud": np.bincount(hr[pos_l], minlength=24), "legit_sample": np.bincount(hr[neg_l], minlength=24)}).assign(
    ratio=lambda d: (d.ord_fraud / d.legit_sample).round(3)).T.to_string())
